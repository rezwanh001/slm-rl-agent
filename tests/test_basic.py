#-*- coding: utf-8 -*-
"""
@author: Md Rezwanul Haque
"""

"""
Basic Tests for SLM-RL-Agents

Run tests with: pytest tests/

These tests verify the basic functionality of the SLM-RL-Agents package.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestImports:
    """Test that all modules can be imported."""
    
    def test_import_main_package(self):
        """Test importing the main package."""
        import src
        assert hasattr(src, "__version__")
    
    def test_import_data_module(self):
        """Test importing data utilities."""
        # পূর্বে: from src.data import load_sft_dataset, load_preference_dataset
        from src.slm_rl_agent.data import load_sft_dataset, load_preference_dataset
        assert callable(load_sft_dataset)
        assert callable(load_preference_dataset)

    def test_import_models_module(self):
        """Test importing model classes."""
        # পূর্বে: from src.models import SLMModel, RewardModel
        from src.slm_rl_agent.models import SLMModel
        from src.slm_rl_agent.rewards import RewardModel
        assert SLMModel is not None
        assert RewardModel is not None

    def test_import_training_module(self):
        """Test importing training utilities."""
        # পূর্বে: from src.training import (...)
        from src.slm_rl_agent.rl import (
            SFTTrainerWrapper,
            DPOTrainerWrapper,
            PPOTrainerWrapper,
            GRPOTrainer,
        )
        assert SFTTrainerWrapper is not None
        assert DPOTrainerWrapper is not None
    
    def test_import_evaluation_module(self):
        """Test importing evaluation metrics."""
        from src.evaluation import (
            compute_perplexity,
            compute_bleu,
            compute_distinct_n,
            EvaluationSuite,
        )
        assert callable(compute_perplexity)
        assert callable(compute_bleu)
    
    def test_import_agent_module(self):
        """Test importing agent classes."""
        from src.agent import SLMAgent, ToolRegistry
        assert SLMAgent is not None
        assert ToolRegistry is not None
    
    def test_import_utils_module(self):
        """Test importing utility functions."""
        # পূর্বে: from src.utils import setup_logging, load_config, save_checkpoint
        from src.slm_rl_agent.utils import setup_logging, load_config, save_checkpoint
        assert callable(setup_logging)
        assert callable(load_config)


class TestDataProcessor:
    """Test data processing utilities."""
    
    def test_data_collator_initialization(self):
        """Test that data collators can be initialized."""
        # পূর্বে: from src.data.data_processor import ...
        from src.slm_rl_agent.data.data_processor import DataCollatorForSFT, DataCollatorForPreference
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        sft_collator = DataCollatorForSFT(tokenizer=tokenizer)
        pref_collator = DataCollatorForPreference(tokenizer=tokenizer)
        
        assert sft_collator is not None
        assert pref_collator is not None


class TestPreferenceDataset:
    """Test preference dataset utilities."""
    
    def test_create_preference_dataset(self):
        """Test creating a preference dataset from lists."""
        # পূর্বে: from src.data.preference_dataset import PreferenceDataset
        from src.slm_rl_agent.data.preference_dataset import PreferenceDataset
        
        dataset = PreferenceDataset.from_comparisons(
            prompts=["What is 2+2?", "What is the capital of France?"],
            chosen=["4", "Paris"],
            rejected=["I don't know", "I'm not sure"],
        )
        
        assert len(dataset) == 2
        assert "prompt" in dataset[0]
        assert "chosen" in dataset[0]
        assert "rejected" in dataset[0]
    
    def test_preference_dataset_statistics(self):
        """Test computing dataset statistics."""
        # পূর্বে: from src.data.preference_dataset import PreferenceDataset
        from src.slm_rl_agent.data.preference_dataset import PreferenceDataset
        
        dataset = PreferenceDataset.from_comparisons(
            prompts=["Question 1", "Question 2"],
            chosen=["Good answer 1", "Good answer 2"],
            rejected=["Bad answer 1", "Bad answer 2"],
        )
        
        stats = dataset.get_statistics()
        assert "num_examples" in stats
        assert stats["num_examples"] == 2


class TestToolRegistry:
    """Test tool calling utilities."""
    
    def test_tool_registration(self):
        """Test registering a tool."""
        from src.agent.tool_calling import ToolRegistry
        
        registry = ToolRegistry()
        
        def dummy_tool(x: str) -> str:
            return f"Result: {x}"
        
        registry.register(
            name="dummy",
            description="A dummy tool",
            parameters={"x": {"type": "string", "description": "Input"}},
            function=dummy_tool,
        )
        
        assert "dummy" in registry.list_tools()
    
    def test_tool_parsing(self):
        """Test parsing tool calls from text."""
        from src.agent.tool_calling import ToolRegistry
        
        registry = ToolRegistry()
        registry.register(
            name="calculator",
            description="Calculator",
            parameters={"expression": {"type": "string", "description": "Expression"}},
            function=lambda expression: str(eval(expression)),
        )
        
        tool_call = registry.parse_tool_call("[TOOL_CALL: calculator(expression='2+2')]")
        
        assert tool_call is not None
        assert tool_call.tool_name == "calculator"
        assert tool_call.arguments.get("expression") == "2+2"
    
    def test_tool_execution(self):
        """Test executing a tool."""
        from src.agent.tool_calling import ToolRegistry
        
        registry = ToolRegistry()
        registry.register(
            name="echo",
            description="Echo",
            parameters={"text": {"type": "string", "description": "Text to echo"}},
            function=lambda text: text,
        )
        
        tool_call = registry.parse_tool_call("[TOOL_CALL: echo(text='hello')]")
        result = registry.execute(tool_call)
        
        assert result.success
        assert result.output == "hello"


class TestEvaluationMetrics:
    """Test evaluation metrics."""
    
    def test_distinct_n(self):
        """Test Distinct-n metric computation."""
        from src.evaluation.metrics import compute_distinct_n
        
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "A different sentence with different words",
        ]
        
        results = compute_distinct_n(texts, n_values=[1, 2])
        
        assert "distinct_1" in results
        assert "distinct_2" in results
        assert 0 <= results["distinct_1"] <= 1
        assert 0 <= results["distinct_2"] <= 1
    
    def test_bleu_basic(self):
        """Test basic BLEU computation."""
        from src.evaluation.metrics import compute_bleu
        
        predictions = ["The cat sat on the mat"]
        references = ["The cat sat on the mat"]
        
        results = compute_bleu(predictions, references)
        
        assert "bleu" in results
        # Identical sentences should have high BLEU
        assert results["bleu"] > 0.5


class TestConfigUtils:
    """Test configuration utilities."""
    
    def test_merge_configs(self):
        """Test merging configuration dictionaries."""
        # পূর্বে: from src.utils.config_utils import merge_configs
        from src.slm_rl_agent.utils.config_utils import merge_configs
        
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 4}, "e": 5}
        
        merged = merge_configs(base, override)
        
        assert merged["a"] == 1
        assert merged["b"]["c"] == 4
        assert merged["b"]["d"] == 3
        assert merged["e"] == 5
    
    def test_get_default_config(self):
        """Test getting default configurations."""
        # পূর্বে: from src.utils.config_utils import get_default_config
        from src.slm_rl_agent.utils.config_utils import get_default_config
        
        sft_config = get_default_config("sft")
        dpo_config = get_default_config("dpo")
        
        assert "learning_rate" in sft_config
        assert "beta" in dpo_config


class TestAgenticScaffolding:
    """Forward-compatibility scaffolding for the multi-turn agentic case."""

    def test_action_parser_recognises_four_types(self):
        from src.agentic import parse_action, ActionType
        assert parse_action('<finish>done</finish>').type == ActionType.FINISH
        assert parse_action('<ask>which one?</ask>').type == ActionType.ASK_USER
        tc = parse_action('<tool name="length_check">{"text": "hi"}</tool>')
        assert tc.type == ActionType.TOOL_CALL
        assert tc.tool_name == "length_check"
        assert tc.tool_args == {"text": "hi"}
        assert parse_action('plain tokens').type == ActionType.TOKENS

    def test_environment_runs_scripted_episode(self):
        from src.agentic import AgenticEnvironment, State

        def echo(args, _state):
            return f"echoed: {args.get('text', '')}"

        def reward(traj):
            return (1.0 if traj.final_state.finalised_answer else 0.0), {}

        script = iter([
            '<tool name="echo">{"text": "hi"}</tool>',
            '<finish>got it</finish>',
        ])
        env = AgenticEnvironment(tools={"echo": echo}, reward_fn=reward)
        traj = env.run_episode(
            State(task_prompt="ok"),
            policy=lambda _p: next(script, "<finish>fallback</finish>"),
        )
        assert traj.final_state.finalised_answer == "got it"
        assert traj.reward_total == 1.0
        assert any(t["event"] == "tool_call" for t in traj.turns)

    def test_story_tools_compute_constraints(self):
        from src.agentic import build_story_tools
        tools = build_story_tools()
        out = tools["length_check"]({"text": "one two three", "max_words": 5}, None)
        assert "within_budget=true" in out
        out = tools["character_check"](
            {"text": "the rabbit ran", "characters": ["rabbit", "fox"]}, None
        )
        assert "all_present=false" in out and "fox" in out

    def test_summarisation_tools_use_bow_fallback(self):
        from src.agentic import ArticleIndex, build_summarisation_tools
        idx = ArticleIndex(
            paragraphs=[
                "The telescope launches in 2026.",
                "Astronomers expect sharper images.",
            ],
            article_id="t",
        )
        tools = build_summarisation_tools(idx)
        out = tools["retrieve"]({"query": "telescope", "k": 1}, None)
        assert "telescope" in out
        out = tools["cite_check"](
            {"sentence": "Telescope launches in 2026.", "paragraph_id": 0}, None
        )
        assert "grounded=" in out and "overlap_f1=" in out

    def test_kb_tools_arithmetic_is_sandboxed(self):
        from src.agentic import KnowledgeBase, ArticleIndex, build_kb_tools
        idx = ArticleIndex(paragraphs=["x"], article_id="t")
        kb = KnowledgeBase(facts=[
            {"entity": "Alan Turing", "relation": "born_in", "value": "1912"},
        ])
        tools = build_kb_tools(kb, idx)
        assert "result=42" in tools["arithmetic"]({"expr": "1954 - 1912"}, None)
        # rejected: any non-arithmetic input
        assert tools["arithmetic"]({"expr": "__import__('os')"}, None).startswith("error:")
        assert "Alan Turing" in tools["entity_lookup"]({"entity": "Alan Turing"}, None)


class TestAgenticDatasetBuilders:
    """Smoke-test the dataset builders for the three forward-compatibility tasks.

    These run only if the upstream SFT splits exist on disk (built by
    ``scripts/prepare_all_datasets.py``). On a fresh checkout without data,
    the tests are skipped rather than failed.
    """

    @pytest.fixture(scope="class")
    def project_root(self):
        return Path(__file__).parent.parent

    def _have(self, root: Path, dataset: str) -> bool:
        return (root / "data" / dataset / "sft_train.json").exists()

    def test_task_b_paragraph_splitter_handles_single_paragraph(self):
        # পূর্বে: একটাই প্যারা থাকলে fallback ট্রিগার হতো না — ফলে Task B
        # খালি output দিত। এই টেস্ট ঐ বাগের রিগ্রেশন গার্ড।
        from scripts.build_agentic_datasets import _cnn_to_paragraphs
        article = "Sentence one. Sentence two. Sentence three. Sentence four."
        paras = _cnn_to_paragraphs(article)
        assert len(paras) >= 2, f"expected >=2 chunks, got {len(paras)}"

    def test_task_c_extracts_facts_from_lifetime_pattern(self):
        from scripts.build_agentic_datasets import _extract_facts
        # "Alan Turing ( 1912 – 1954 )" — wikitext biographical convention.
        facts = _extract_facts(["Alan Turing ( 1912 – 1954 ) was a mathematician."])
        rels = {f["relation"] for f in facts}
        assert "born_in" in rels and "died_in" in rels

    def test_task_a_outputs_required_files(self, project_root):
        if not self._have(project_root, "tinystories"):
            pytest.skip("tinystories sft_train.json not present; run prepare_all_datasets.py first")
        for fname in ("task_prompts.json", "sft_tooluse.json", "trajectory_prefs.json"):
            assert (project_root / "data" / "agentic" / "task_a_tinystories" / fname).exists(), fname


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
