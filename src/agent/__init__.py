"""
SLM Agent Implementation for SLM-RL-Agents

This module provides the production-ready agent implementation that can be
deployed for inference. The agent wraps a trained SLM and provides:

Core Capabilities:
    - Text generation with various decoding strategies
    - Tool/function calling for agentic tasks
    - Conversation management with memory
    - Batch inference for throughput

Deployment Options:
    - Python API for direct integration
    - REST API server for microservice deployment
    - CLI for interactive testing
"""

from src.agent.slm_agent import SLMAgent
from src.agent.tool_calling import ToolRegistry, execute_tool
from src.agent.inference_server import start_server, InferenceServer

__all__ = [
    "SLMAgent",
    "ToolRegistry",
    "execute_tool",
    "start_server",
    "InferenceServer",
]
