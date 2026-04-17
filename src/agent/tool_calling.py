#-*- coding: utf-8 -*-
"""
@author: Md Rezwanul Haque
"""

"""
Tool Calling Utilities for SLM-RL-Agents

This module provides the infrastructure for enabling Small Language Models to call
external tools and functions. Tool calling (also known as function calling) is a
key capability that transforms a simple chatbot into a useful AI agent.

WHAT IS TOOL CALLING?
Tool calling allows a language model to decide when it needs external capabilities
(like a calculator, web search, or API) and generate the appropriate function call.
The typical flow is:

1. User asks a question requiring external data/computation
2. Model recognizes it needs a tool and generates a tool call
3. System executes the tool with provided arguments
4. Tool result is fed back to the model
5. Model incorporates the result into its response

WHY TOOL CALLING MATTERS FOR SMALL MODELS:
Small language models have limited knowledge and computation capabilities compared
to large models. Tool calling allows them to overcome these limitations by:
- Accessing real-time information (weather, stock prices, etc.)
- Performing reliable computation (math, code execution)
- Interacting with external systems (databases, APIs)
- Reducing hallucination by grounding responses in tool outputs

This module provides:
- ToolRegistry: Register and manage available tools
- Tool execution with error handling
- Output parsing to detect tool calls in model output
- Standard tool implementations (calculator, etc.)
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """
    Definition of a tool that the agent can call.
    
    Attributes:
        name: Unique identifier for the tool
        description: Human-readable description (shown to the model)
        parameters: Dict mapping parameter names to their types/descriptions
        function: The actual Python function to execute
        required_params: List of required parameter names
    """
    name: str
    description: str
    parameters: Dict[str, Dict[str, str]]
    function: Callable
    required_params: List[str] = None
    
    def __post_init__(self):
        if self.required_params is None:
            # By default, all parameters are required
            self.required_params = list(self.parameters.keys())


@dataclass
class ToolCall:
    """
    Represents a parsed tool call from model output.
    
    Attributes:
        tool_name: Name of the tool to call
        arguments: Dictionary of argument name to value
        raw_text: The original text that was parsed
    """
    tool_name: str
    arguments: Dict[str, Any]
    raw_text: str


@dataclass
class ToolResult:
    """
    Result of executing a tool.
    
    Attributes:
        success: Whether the tool executed successfully
        output: The tool's output (if successful)
        error: Error message (if failed)
        tool_call: The original tool call
    """
    success: bool
    output: Any
    error: Optional[str]
    tool_call: ToolCall


class ToolRegistry:
    """
    Registry for managing tools available to the agent.
    
    The ToolRegistry is the central place to define and manage all tools
    that your agent can use. It handles:
    
    - Tool registration with validation
    - Generating tool descriptions for the model
    - Parsing tool calls from model output
    - Executing tools with error handling
    
    DESIGN PRINCIPLES:
    1. Tools should have clear, specific names (calculator, not compute)
    2. Descriptions should explain WHEN to use the tool
    3. Parameter names should be self-explanatory
    4. Functions should handle errors gracefully
    
    Example:
        >>> registry = ToolRegistry()
        >>> registry.register(
        ...     name="calculator",
        ...     description="Evaluates mathematical expressions. Use when you need to compute numbers.",
        ...     parameters={"expression": {"type": "string", "description": "Math expression to evaluate"}},
        ...     function=lambda expression: str(eval(expression))
        ... )
        >>> 
        >>> # Parse a tool call from model output
        >>> call = registry.parse_tool_call("[TOOL_CALL: calculator(expression='25 * 47')]")
        >>> result = registry.execute(call)
        >>> print(result.output)  # "1175"
    """
    
    def __init__(self):
        """Initialize an empty tool registry."""
        self.tools: Dict[str, ToolDefinition] = {}
    
    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Dict[str, str]],
        function: Callable,
        required_params: Optional[List[str]] = None,
    ) -> None:
        """
        Register a new tool.
        
        Args:
            name: Unique name for the tool (used in tool calls)
            description: Clear description of what the tool does and when to use it
            parameters: Dict of parameter definitions. Each parameter should have:
                - "type": The data type (string, number, boolean)
                - "description": What this parameter is for
            function: Python function that implements the tool
            required_params: Which parameters are required (default: all)
        
        Example:
            >>> registry.register(
            ...     name="get_weather",
            ...     description="Get current weather for a location",
            ...     parameters={
            ...         "city": {"type": "string", "description": "City name"},
            ...         "units": {"type": "string", "description": "celsius or fahrenheit"}
            ...     },
            ...     function=get_weather_func,
            ...     required_params=["city"]
            ... )
        """
        if name in self.tools:
            logger.warning(f"Overwriting existing tool: {name}")
        
        tool = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            function=function,
            required_params=required_params,
        )
        
        self.tools[name] = tool
        logger.info(f"Registered tool: {name}")
    
    def unregister(self, name: str) -> bool:
        """
        Remove a tool from the registry.
        
        Args:
            name: Name of the tool to remove
        
        Returns:
            True if the tool was removed, False if it wasn't found
        """
        if name in self.tools:
            del self.tools[name]
            logger.info(f"Unregistered tool: {name}")
            return True
        return False
    
    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool definition by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self.tools.keys())
    
    def get_tool_descriptions(self, format: str = "text") -> str:
        """
        Generate formatted descriptions of all tools for the model.
        
        This description is included in the prompt to tell the model what
        tools are available and how to use them.
        
        Args:
            format: Output format ("text", "json", "markdown")
        
        Returns:
            Formatted string describing all tools
        """
        if format == "json":
            return self._get_tool_descriptions_json()
        elif format == "markdown":
            return self._get_tool_descriptions_markdown()
        else:
            return self._get_tool_descriptions_text()
    
    def _get_tool_descriptions_text(self) -> str:
        """Generate plain text tool descriptions."""
        lines = ["Available tools:"]
        
        for name, tool in self.tools.items():
            lines.append(f"\n{name}: {tool.description}")
            lines.append("  Parameters:")
            for param_name, param_info in tool.parameters.items():
                required = "(required)" if param_name in tool.required_params else "(optional)"
                param_type = param_info.get("type", "any")
                param_desc = param_info.get("description", "")
                lines.append(f"    - {param_name} ({param_type}) {required}: {param_desc}")
        
        lines.append("\nTo call a tool, use this format:")
        lines.append("[TOOL_CALL: tool_name(param1=value1, param2=value2)]")
        
        return "\n".join(lines)
    
    def _get_tool_descriptions_json(self) -> str:
        """Generate JSON-formatted tool descriptions."""
        tools_list = []
        for name, tool in self.tools.items():
            tools_list.append({
                "name": name,
                "description": tool.description,
                "parameters": tool.parameters,
                "required": tool.required_params,
            })
        return json.dumps(tools_list, indent=2)
    
    def _get_tool_descriptions_markdown(self) -> str:
        """Generate markdown-formatted tool descriptions."""
        lines = ["## Available Tools\n"]
        
        for name, tool in self.tools.items():
            lines.append(f"### {name}")
            lines.append(f"{tool.description}\n")
            lines.append("**Parameters:**")
            for param_name, param_info in tool.parameters.items():
                required = "required" if param_name in tool.required_params else "optional"
                param_type = param_info.get("type", "any")
                param_desc = param_info.get("description", "")
                lines.append(f"- `{param_name}` ({param_type}, {required}): {param_desc}")
            lines.append("")
        
        return "\n".join(lines)
    
    def parse_tool_call(self, text: str) -> Optional[ToolCall]:
        """
        Parse a tool call from model output.
        
        This method looks for tool calls in the format:
        [TOOL_CALL: tool_name(arg1=value1, arg2=value2)]
        
        Args:
            text: Text to search for tool calls
        
        Returns:
            ToolCall object if found, None otherwise
        """
        # Pattern: [TOOL_CALL: tool_name(args)]
        pattern = r'\[TOOL_CALL:\s*(\w+)\((.*?)\)\]'
        match = re.search(pattern, text, re.DOTALL)
        
        if not match:
            return None
        
        tool_name = match.group(1)
        args_str = match.group(2).strip()
        
        # Parse arguments
        arguments = {}
        if args_str:
            # Split by comma, but handle commas inside strings
            # Simple approach: split and parse each arg=value pair
            for arg_pair in self._split_arguments(args_str):
                if '=' in arg_pair:
                    key, value = arg_pair.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes from strings
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    
                    # Try to parse as JSON for numbers, bools, etc.
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        pass  # Keep as string
                    
                    arguments[key] = value
        
        return ToolCall(
            tool_name=tool_name,
            arguments=arguments,
            raw_text=match.group(0),
        )
    
    def _split_arguments(self, args_str: str) -> List[str]:
        """Split argument string by commas, respecting quotes."""
        args = []
        current = []
        in_quotes = False
        quote_char = None
        
        for char in args_str:
            if char in '"\'':
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif char == quote_char:
                    in_quotes = False
                    quote_char = None
                current.append(char)
            elif char == ',' and not in_quotes:
                args.append(''.join(current).strip())
                current = []
            else:
                current.append(char)
        
        if current:
            args.append(''.join(current).strip())
        
        return args
    
    def execute(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a tool call.
        
        This method validates the tool call, executes the tool function,
        and returns the result with proper error handling.
        
        Args:
            tool_call: The parsed tool call to execute
        
        Returns:
            ToolResult with output or error
        """
        # Check if tool exists
        tool = self.tools.get(tool_call.tool_name)
        if not tool:
            return ToolResult(
                success=False,
                output=None,
                error=f"Unknown tool: {tool_call.tool_name}",
                tool_call=tool_call,
            )
        
        # Validate required parameters
        missing_params = [
            p for p in tool.required_params
            if p not in tool_call.arguments
        ]
        if missing_params:
            return ToolResult(
                success=False,
                output=None,
                error=f"Missing required parameters: {missing_params}",
                tool_call=tool_call,
            )
        
        # Execute the tool
        try:
            result = tool.function(**tool_call.arguments)
            return ToolResult(
                success=True,
                output=result,
                error=None,
                tool_call=tool_call,
            )
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return ToolResult(
                success=False,
                output=None,
                error=str(e),
                tool_call=tool_call,
            )


def execute_tool(
    tool_call: ToolCall,
    registry: ToolRegistry,
) -> ToolResult:
    """
    Convenience function to execute a tool call.
    
    Args:
        tool_call: The tool call to execute
        registry: The tool registry containing the tool
    
    Returns:
        ToolResult with output or error
    """
    return registry.execute(tool_call)


# =============================================================================
# Standard Tool Implementations
# =============================================================================

def create_calculator_tool() -> Dict[str, Any]:
    """
    Create a calculator tool definition.
    
    The calculator evaluates mathematical expressions safely using a limited
    set of allowed operations. This is safer than using eval() directly.
    
    Returns:
        Dictionary with tool definition ready for registration
    """
    import math
    
    # Safe math functions
    safe_dict = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "pi": math.pi,
        "e": math.e,
    }
    
    def calculator(expression: str) -> str:
        """Safely evaluate a mathematical expression."""
        try:
            # Remove any potentially dangerous characters
            safe_expression = re.sub(r'[^0-9+\-*/().,%\s]', '', expression)
            # But we need to allow function names, so let's be more careful
            # Actually, let's use a simple approach with limited eval
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            return str(result)
        except Exception as e:
            return f"Error: {e}"
    
    return {
        "name": "calculator",
        "description": "Evaluates mathematical expressions. Use for any calculations involving numbers, arithmetic, or math functions like sqrt, sin, cos, log.",
        "parameters": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', '3.14 * 2**2')"
            }
        },
        "function": calculator,
    }


def create_current_time_tool() -> Dict[str, Any]:
    """
    Create a tool that returns the current date and time.
    
    Returns:
        Dictionary with tool definition
    """
    from datetime import datetime
    
    def get_current_time(timezone: str = "UTC") -> str:
        """Get current date and time."""
        try:
            now = datetime.now()
            return now.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            return f"Error: {e}"
    
    return {
        "name": "current_time",
        "description": "Get the current date and time. Use when asked about the current time or date.",
        "parameters": {
            "timezone": {
                "type": "string",
                "description": "Timezone (default: UTC)"
            }
        },
        "function": get_current_time,
        "required_params": [],  # timezone is optional
    }


def create_string_tool() -> Dict[str, Any]:
    """
    Create a string manipulation tool.
    
    Returns:
        Dictionary with tool definition
    """
    def string_ops(text: str, operation: str) -> str:
        """Perform string operations."""
        operations = {
            "upper": lambda t: t.upper(),
            "lower": lambda t: t.lower(),
            "title": lambda t: t.title(),
            "reverse": lambda t: t[::-1],
            "length": lambda t: str(len(t)),
            "words": lambda t: str(len(t.split())),
        }
        
        if operation not in operations:
            return f"Unknown operation. Available: {list(operations.keys())}"
        
        return operations[operation](text)
    
    return {
        "name": "string_ops",
        "description": "Perform string operations like uppercase, lowercase, reverse, count length/words.",
        "parameters": {
            "text": {
                "type": "string",
                "description": "The text to operate on"
            },
            "operation": {
                "type": "string",
                "description": "Operation: upper, lower, title, reverse, length, words"
            }
        },
        "function": string_ops,
    }


def create_standard_tools() -> ToolRegistry:
    """
    Create a registry with standard built-in tools.
    
    Returns:
        ToolRegistry with calculator, time, and string tools registered
    """
    registry = ToolRegistry()
    
    # Register calculator
    calc_def = create_calculator_tool()
    registry.register(**calc_def)
    
    # Register current time
    time_def = create_current_time_tool()
    registry.register(**time_def)
    
    # Register string operations
    string_def = create_string_tool()
    registry.register(**string_def)
    
    return registry
