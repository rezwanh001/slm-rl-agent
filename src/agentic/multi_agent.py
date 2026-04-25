#-*- coding: utf-8 -*-
"""
Multi-agent scaffolding for SLM-RL-Agents.

Three roles, each backed by a separately trained SLM from the Pythia/SmolLM2
family:

* **Actor**  ``pi_theta`` - the tool-using policy. Trained with multi-turn PPO.
* **Critic** ``c_phi``    - a regression head + small LM that scores partial
                            trajectories. Trained on (state, return) pairs
                            collected from the actor's own rollouts.
* **Editor** ``e_psi``    - optional. A second actor trained to revise the
                            first actor's finalised answer. Used only during
                            evaluation for test-time improvement.

This module supplies:

1. ``ActorPolicy``     - wraps an SLM as a callable policy that obeys the
                         environment's prompt/action interface.
2. ``CriticModel``     - wraps a second SLM with a regression head.
3. ``CriticDataBuilder`` - converts trajectories into (state_repr, return)
                         supervised samples.
4. ``MultiAgentEnsemble`` - packages actor + critic (+ optional editor) with
                         a single ``rollout`` and ``score`` interface, so
                         ``agentic_ppo.py`` and ``agentic_metrics.py`` do not
                         care how many models are in play.

Nothing here depends on TRL. The actual PPO update is in ``agentic_ppo.py``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn

from src.agentic.environment import (
    AgenticEnvironment,
    State,
    Trajectory,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------

@dataclass
class GenerationConfig:
    max_new_tokens: int = 192
    temperature: float = 0.8
    top_p: float = 0.95
    stop_strings: Tuple[str, ...] = ("</tool>", "</finish>", "</ask>")


class ActorPolicy:
    """Callable that plugs the base HF model into the environment loop.

    The environment calls ``actor(prompt)`` once per turn and receives a
    string continuation. Stop-string handling lets the agent return control
    to the environment as soon as an action sentinel closes.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        generation_config: Optional[GenerationConfig] = None,
        device: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = generation_config or GenerationConfig()
        self.device = device or next(model.parameters()).device
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def __call__(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        new_ids = out[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=False)

        # Soft-early-stop at the first action closing sentinel so the env
        # gets clean action text without trailing hallucination.
        for stop in self.cfg.stop_strings:
            if stop in text:
                text = text.split(stop)[0] + stop
                break
        return text


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

class CriticModel(nn.Module):
    """An SLM backbone with a scalar regression head. Predicts V(s)."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        # Zero-init the output so the critic starts as a constant predictor.
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]  # [B, T, H]

        if attention_mask is not None:
            idx = attention_mask.sum(dim=1) - 1
            last = hidden[torch.arange(hidden.size(0), device=hidden.device), idx]
        else:
            last = hidden[:, -1, :]
        return self.head(last).squeeze(-1)

    @torch.no_grad()
    def score(
        self,
        states_as_text: List[str],
        tokenizer: Any,
        batch_size: int = 8,
    ) -> torch.Tensor:
        """Convenience scorer for lists of state renderings."""
        self.eval()
        device = next(self.parameters()).device
        all_values: List[torch.Tensor] = []
        for i in range(0, len(states_as_text), batch_size):
            batch = states_as_text[i: i + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(device)
            values = self.forward(enc["input_ids"], enc["attention_mask"])
            all_values.append(values.cpu())
        return torch.cat(all_values, dim=0) if all_values else torch.zeros(0)


# ---------------------------------------------------------------------------
# Building critic supervision data from rollouts
# ---------------------------------------------------------------------------

@dataclass
class CriticSample:
    state_text: str
    target_value: float


def build_critic_samples(
    trajectories: List[Trajectory],
    gamma: float = 0.95,
) -> List[CriticSample]:
    """Turn rollouts into ``(s_t, discounted_return_from_t)`` pairs.

    The return used is the trajectory's final reward discounted back to each
    turn. That is a Monte Carlo estimate; it becomes more accurate as the
    actor's policy stabilises.
    """
    samples: List[CriticSample] = []
    for traj in trajectories:
        R = traj.reward_total
        for t_idx, turn in enumerate(traj.turns):
            discount = gamma ** max(0, len(traj.turns) - t_idx - 1)
            target = discount * R
            samples.append(CriticSample(
                state_text=turn.get("prompt", ""),
                target_value=float(target),
            ))
    return samples


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

@dataclass
class MultiAgentEnsemble:
    """Packages the actor, critic, and optional editor for a single interface.

    Usage:

        ensemble = MultiAgentEnsemble(actor=actor, critic=critic)
        traj = ensemble.rollout(env, initial_state)
        v = ensemble.value(state_text)

    The PPO trainer and the eval script both consume this object so neither
    cares whether the critic is a value head on the actor or a separate SLM.
    """
    actor: ActorPolicy
    critic: Optional[CriticModel] = None
    editor: Optional[ActorPolicy] = None
    tokenizer: Any = None

    def rollout(
        self,
        env: AgenticEnvironment,
        initial_state: State,
    ) -> Trajectory:
        return env.run_episode(initial_state, self.actor)

    def value(self, state_text: str) -> float:
        if self.critic is None:
            return 0.0
        values = self.critic.score([state_text], self.tokenizer)
        return float(values[0].item())

    def edited_answer(
        self,
        traj: Trajectory,
        editor_prompt_template: str = "Revise the following answer for clarity and factual accuracy:\n\n{answer}\n\nRevised answer:",
    ) -> Optional[str]:
        if self.editor is None or traj.final_state is None:
            return None
        ans = traj.final_state.finalised_answer or ""
        if not ans:
            return None
        return self.editor(editor_prompt_template.format(answer=ans))


# ---------------------------------------------------------------------------
# Lightweight sanity check
# ---------------------------------------------------------------------------

def _ensemble_contract_test() -> None:
    """Verify the ensemble's interface without loading any HF weights."""

    class FakeActor:
        def __call__(self, prompt):
            return "<finish>ok</finish>"

    class FakeCritic:
        def score(self, texts, tok):
            import torch as _t
            return _t.tensor([0.42] * len(texts))

    ensemble = MultiAgentEnsemble(actor=FakeActor(), critic=FakeCritic())
    assert ensemble.value("state") == 0.42
    assert ensemble.edited_answer(Trajectory(state_initial=State("x"))) is None
    print("[multi_agent] contract test passed")


if __name__ == "__main__":
    _ensemble_contract_test()
