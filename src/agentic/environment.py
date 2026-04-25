#-*- coding: utf-8 -*-
"""
@author: Md Rezwanul Haque
"""

"""
Multi-turn agentic environment for SLM-RL-Agents.

This module defines the MDP the actor policy interacts with during training
and evaluation. It is deliberately thin and composable: the environment holds
external state (observations, tool outputs, budget), the *policy* is any
callable ``generate(prompt) -> str`` that returns a continuation, and the
*reward function* is any callable ``(trajectory, task) -> float``.

The environment does three jobs the original TRL-based pipeline did not do:

1. It maintains a structured ``State`` object across turns of an episode,
   so the "agent" has real state to act on rather than a string prefix.
2. It parses tool-call sentinels out of the policy's raw output, dispatches
   them to the tool registry, and feeds the results back as *observations*
   the policy sees on the next turn.
3. It terminates on an explicit ``FINISH`` action or on budget exhaustion,
   giving the actor a meaningful termination signal the PPO update can use.

Integration with the existing codebase:

- Works with any ``AutoModelForCausalLM`` or ``AutoModelForCausalLMWithValueHead``.
- Does not touch the tokenizer, sampling code, or TRL internals.
- Produces ``Trajectory`` objects that ``agentic_ppo.py`` consumes directly.

Self-test:
    python -m src.agentic.environment --self_test
"""
# from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Action taxonomy
# ---------------------------------------------------------------------------

class ActionType(Enum):
    TOKENS = "tokens"          # free-form generation, no sentinel
    TOOL_CALL = "tool_call"    # <tool name="...">{...}</tool>
    ASK_USER = "ask_user"      # <ask>...</ask>
    FINISH = "finish"          # <finish>...</finish>
    INVALID = "invalid"        # malformed sentinel, counted against budget


@dataclass
class ParsedAction:
    """A single action extracted from the policy's raw output string."""
    type: ActionType
    payload: str = ""
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    raw: str = ""


# ---------------------------------------------------------------------------
# Structured state + budget
# ---------------------------------------------------------------------------

@dataclass
class Budget:
    """Per-episode budget. Every tool call and every turn decrements counters."""
    max_turns: int = 6
    max_tool_calls: int = 10
    max_new_tokens_per_turn: int = 192
    tool_calls_used: int = 0
    turns_used: int = 0

    def turn_allowed(self) -> bool:
        return self.turns_used < self.max_turns

    def tool_allowed(self) -> bool:
        return self.tool_calls_used < self.max_tool_calls

    def consume_turn(self) -> None:
        self.turns_used += 1

    def consume_tool(self) -> None:
        self.tool_calls_used += 1


@dataclass
class Observation:
    """Something appended to the agent's context by the environment."""
    source: str                 # "tool:<name>" | "user" | "system"
    content: str
    turn: int


@dataclass
class State:
    """Structured agent state. Serialised to a prompt by ``render_prompt``."""
    task_prompt: str                                      # user-level request
    observations: List[Observation] = field(default_factory=list)
    scratchpad: str = ""                                  # accumulated tokens
    finalised_answer: Optional[str] = None
    budget: Budget = field(default_factory=Budget)
    task_id: str = "generic"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_terminal(self) -> bool:
        return (
            self.finalised_answer is not None
            or not self.budget.turn_allowed()
        )


@dataclass
class Trajectory:
    """Full episode record consumed by PPO and by evaluation."""
    state_initial: State
    turns: List[Dict[str, Any]] = field(default_factory=list)
    final_state: Optional[State] = None
    reward_total: float = 0.0
    reward_breakdown: Dict[str, float] = field(default_factory=dict)

    def to_json(self) -> str:
        def _default(o):
            if isinstance(o, Enum):
                return o.value
            if hasattr(o, "__dict__"):
                return o.__dict__
            return str(o)
        return json.dumps(asdict(self), default=_default, indent=2)


# ---------------------------------------------------------------------------
# Action parser
# ---------------------------------------------------------------------------

# Sentinels are intentionally boring-looking so they survive BPE reasonably
# well. The agent learns them during SFT warm-up (see ``build_agentic_datasets``).
_TOOL_RE = re.compile(
    r"<tool\s+name=\"(?P<name>[a-z_]+)\">(?P<body>.*?)</tool>",
    flags=re.DOTALL,
)
_ASK_RE = re.compile(r"<ask>(?P<body>.*?)</ask>", flags=re.DOTALL)
_FINISH_RE = re.compile(r"<finish>(?P<body>.*?)</finish>", flags=re.DOTALL)


def parse_action(text: str) -> ParsedAction:
    """Return the *first* action sentinel found, or TOKENS if none."""
    m = _FINISH_RE.search(text)
    if m:
        return ParsedAction(
            type=ActionType.FINISH,
            payload=m.group("body").strip(),
            raw=m.group(0),
        )

    m = _TOOL_RE.search(text)
    if m:
        body = m.group("body").strip()
        try:
            args = json.loads(body) if body else {}
        except json.JSONDecodeError:
            return ParsedAction(
                type=ActionType.INVALID,
                payload=f"bad json in tool call: {body[:80]!r}",
                raw=m.group(0),
            )
        return ParsedAction(
            type=ActionType.TOOL_CALL,
            payload=body,
            tool_name=m.group("name"),
            tool_args=args,
            raw=m.group(0),
        )

    m = _ASK_RE.search(text)
    if m:
        return ParsedAction(
            type=ActionType.ASK_USER,
            payload=m.group("body").strip(),
            raw=m.group(0),
        )

    return ParsedAction(type=ActionType.TOKENS, payload=text, raw=text)


# ---------------------------------------------------------------------------
# Prompt renderer
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = """You are a tool-using language agent. You can:

- Emit plain text to think on a scratchpad.
- Call a tool with <tool name="NAME">{{"arg": "value"}}</tool>. The environment will return <result>...</result> on the next turn.
- Ask the user a clarifying question with <ask>...</ask>.
- Finalise your answer with <finish>YOUR FINAL ANSWER</finish>. This ends the episode.

Available tools: {tool_names}.
Budget: {turns_left} turns, {tools_left} tool calls remaining.

Task: {task_prompt}
"""


def render_prompt(state: State, tool_names: List[str]) -> str:
    """Serialise ``State`` into a string prompt the base SLM can consume."""
    parts = [
        _SYSTEM_TEMPLATE.format(
            tool_names=", ".join(tool_names) if tool_names else "(none)",
            turns_left=state.budget.max_turns - state.budget.turns_used,
            tools_left=state.budget.max_tool_calls - state.budget.tool_calls_used,
            task_prompt=state.task_prompt,
        )
    ]
    for obs in state.observations:
        tag = {"user": "user", "system": "sys"}.get(obs.source, "result")
        parts.append(f"<{tag}>{obs.content}</{tag}>")
    if state.scratchpad:
        parts.append(state.scratchpad)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# The environment itself
# ---------------------------------------------------------------------------

PolicyFn = Callable[[str], str]
"""A callable that consumes a prompt string and returns the next continuation.
Bind your ``AutoModelForCausalLM.generate(...)`` into one of these."""

ToolFn = Callable[[Dict[str, Any], State], str]
"""A callable that consumes ``(args, state)`` and returns a string result.
Tools may read ``state.metadata`` (e.g., the article for Task B) but must
not mutate state directly - the environment handles that."""

UserSimulatorFn = Optional[Callable[[str, State], str]]
"""Optional callable for Task A. Takes the agent's question and current state
and returns the user's reply. ``None`` means ``ASK_USER`` is treated as no-op."""


class AgenticEnvironment:
    """Executes one episode of an agent's interaction with the world.

    The environment is stateless across episodes; call ``run_episode`` each
    time. Thread-safety is not a goal - roll out episodes in separate processes
    if you need parallelism, which is what ``agentic_ppo.py`` does.
    """

    def __init__(
        self,
        tools: Dict[str, ToolFn],
        reward_fn: Callable[[Trajectory], Tuple[float, Dict[str, float]]],
        user_simulator: UserSimulatorFn = None,
        max_invalid_actions: int = 3,
    ):
        self.tools = tools
        self.reward_fn = reward_fn
        self.user_simulator = user_simulator
        self.max_invalid_actions = max_invalid_actions

    def run_episode(
        self,
        initial_state: State,
        policy: PolicyFn,
    ) -> Trajectory:
        """Roll a single episode to termination and return the trajectory."""
        state = initial_state
        traj = Trajectory(state_initial=_clone_state(state))
        invalid_count = 0

        while not state.is_terminal():
            prompt = render_prompt(state, list(self.tools.keys()))
            raw_output = policy(prompt)
            action = parse_action(raw_output)

            turn_record: Dict[str, Any] = {
                "turn": state.budget.turns_used,
                "prompt": prompt,
                "raw_output": raw_output,
                "action_type": action.type.value,
                "action_payload": action.payload[:500],
            }

            if action.type == ActionType.FINISH:
                state.finalised_answer = action.payload
                state.scratchpad += raw_output
                turn_record["event"] = "finish"
                traj.turns.append(turn_record)
                break

            elif action.type == ActionType.TOOL_CALL:
                if not state.budget.tool_allowed():
                    obs = Observation(
                        source="system",
                        content="tool budget exhausted",
                        turn=state.budget.turns_used,
                    )
                    state.observations.append(obs)
                    turn_record["event"] = "tool_budget_exhausted"
                elif action.tool_name not in self.tools:
                    obs = Observation(
                        source="system",
                        content=f"unknown tool: {action.tool_name}",
                        turn=state.budget.turns_used,
                    )
                    state.observations.append(obs)
                    turn_record["event"] = "unknown_tool"
                    invalid_count += 1
                else:
                    try:
                        result = self.tools[action.tool_name](
                            action.tool_args or {}, state
                        )
                    except Exception as exc:
                        result = f"tool error: {type(exc).__name__}: {exc}"
                    state.observations.append(Observation(
                        source=f"tool:{action.tool_name}",
                        content=str(result)[:2000],
                        turn=state.budget.turns_used,
                    ))
                    state.budget.consume_tool()
                    turn_record["event"] = "tool_call"
                    turn_record["tool_name"] = action.tool_name
                    turn_record["tool_result"] = str(result)[:500]
                state.scratchpad += raw_output

            elif action.type == ActionType.ASK_USER:
                if self.user_simulator is None:
                    turn_record["event"] = "ask_ignored"
                else:
                    reply = self.user_simulator(action.payload, state)
                    state.observations.append(Observation(
                        source="user",
                        content=reply,
                        turn=state.budget.turns_used,
                    ))
                    turn_record["event"] = "ask"
                    turn_record["user_reply"] = reply
                state.scratchpad += raw_output

            elif action.type == ActionType.INVALID:
                invalid_count += 1
                state.observations.append(Observation(
                    source="system",
                    content=action.payload,
                    turn=state.budget.turns_used,
                ))
                turn_record["event"] = "invalid"
                if invalid_count >= self.max_invalid_actions:
                    turn_record["event"] = "invalid_budget_exhausted"
                    traj.turns.append(turn_record)
                    break

            else:  # TOKENS
                state.scratchpad += raw_output
                turn_record["event"] = "tokens"

            state.budget.consume_turn()
            traj.turns.append(turn_record)

        reward, breakdown = self.reward_fn(
            _finalise_trajectory(traj, state)
        )
        traj.final_state = state
        traj.reward_total = reward
        traj.reward_breakdown = breakdown
        return traj


def _clone_state(state: State) -> State:
    return State(
        task_prompt=state.task_prompt,
        observations=list(state.observations),
        scratchpad=state.scratchpad,
        finalised_answer=state.finalised_answer,
        budget=Budget(**asdict(state.budget)),
        task_id=state.task_id,
        metadata=dict(state.metadata),
    )


def _finalise_trajectory(traj: Trajectory, state: State) -> Trajectory:
    traj.final_state = state
    return traj


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Smoke test with a dummy policy that finishes after one tool call."""
    logging.basicConfig(level=logging.INFO)

    def dummy_tool(args, state):
        return f"echoed: {args.get('text', '')}"

    def dummy_reward(traj):
        reward = 1.0 if traj.final_state.finalised_answer else 0.0
        return reward, {"completed": reward}

    script = iter([
        '<tool name="echo">{"text": "hello"}</tool>',
        "Let me think about what the echo said.",
        '<finish>The echo said hello.</finish>',
    ])

    def scripted_policy(_prompt):
        return next(script, "<finish>default</finish>")

    env = AgenticEnvironment(
        tools={"echo": dummy_tool},
        reward_fn=dummy_reward,
    )
    init = State(
        task_prompt="Test the environment.",
        task_id="selftest",
    )
    traj = env.run_episode(init, scripted_policy)

    assert traj.final_state.finalised_answer == "The echo said hello.", \
        f"unexpected final answer: {traj.final_state.finalised_answer!r}"
    assert traj.reward_total == 1.0
    assert len(traj.turns) == 3
    tool_turn = traj.turns[0]
    assert tool_turn["event"] == "tool_call"
    assert tool_turn["tool_name"] == "echo"
    print("[environment] self-test passed: 3 turns, reward=1.0, tool echo invoked once")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self_test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
    else:
        print("use --self_test to run the smoke test", file=sys.stderr)
        sys.exit(1)
