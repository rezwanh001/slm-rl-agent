#-*- coding: utf-8 -*-
"""
Forward-compatibility scaffolding for multi-turn agentic SLMs.

# README-এর Project Structure ব্লক অনুযায়ী এখানে চারটি মডিউল আছে:
# environment.py / tools.py / multi_agent.py / agentic_ppo.py
# পূর্বে multi_agent ও agentic_ppo ফাইল দুটি tasks/ ফোল্ডারে ছিল — এখন এখানে
# একত্রে আনা হলো যাতে README-এর কাঠামো হুবহু মেলে।

The single-turn pipeline ships in ``src.slm_rl_agent``; this package bridges
that work to the multi-turn agentic case (see README §"Forward Compatibility").
"""

from src.agentic.environment import (
    ActionType,
    AgenticEnvironment,
    Budget,
    Observation,
    ParsedAction,
    State,
    Trajectory,
    parse_action,
    render_prompt,
)
from src.agentic.tools import (
    ArticleIndex,
    KnowledgeBase,
    build_kb_tools,
    build_story_tools,
    build_summarisation_tools,
    tool_call_accuracy,
)

__all__ = [
    "ActionType",
    "AgenticEnvironment",
    "ArticleIndex",
    "Budget",
    "KnowledgeBase",
    "Observation",
    "ParsedAction",
    "State",
    "Trajectory",
    "build_kb_tools",
    "build_story_tools",
    "build_summarisation_tools",
    "parse_action",
    "render_prompt",
    "tool_call_accuracy",
]
