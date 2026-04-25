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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
