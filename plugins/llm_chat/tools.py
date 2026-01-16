from typing import List, Dict, Optional, Any
from langchain.tools import BaseTool
import requests
import json
import importlib.util
import sys
import os
from pathlib import Path
import tomli


from .core_tools import CORE_TOOLS


def _get_builtin_tools(config: dict) -> Dict[str, Any]:
    """根据配置返回内置工具的初始化方法字典。"""
    builtin_map: Dict[str, Any] = {}

    # Register Core Tools
    for name, tool_func in CORE_TOOLS.items():
        # Wrap tool_func in lambda to delay instantiation if needed,
        # but since they are already @tool objects, we just return them
        builtin_map[name] = lambda t=tool_func: t

    return builtin_map


def load_tools(
    enabled_tools: Optional[List[str]] = None, tool_paths: Optional[List[str]] = None
) -> List[BaseTool]:
    """
    Load tools.
    Always loads 'Builtin/Core Tools' (defined in config or defaults).
    Then loads 'enabled_tools' (modules) if provided.
    """
    tools_list = []

    # 1. Always load config to get environment setup and default settings
    root_path = Path(__file__).resolve().parents[2]
    config_path = root_path / "config-tools.toml"

    try:
        with open(config_path, "rb") as f:
            config = tomli.load(f)
    except FileNotFoundError:
        # Fallback if config missing, though it should exist
        print(f"Warning: Tools config file not found at {config_path}")
        config = {}

    # 2. Setup Environment (e.g. API Keys)
    # Exa needs no API Key setup here as it uses public MCP endpoint in web_search tool

    # 3. Load Builtin/Core Tools
    # These are tools like 'skill_setup', 'web_search' that should ALWAYS be available
    builtin_tool_factories = _get_builtin_tools(config)

    # We load ALL builtins defined in code (CORE_TOOLS) + defaults in config
    # If config defines 'builtin', use that filter. If not, load all CORE_TOOLS?
    # For safety, let's load what's in config['tools']['builtin'] OR default to all CORE_TOOLS if config missing
    builtin_tool_names = config.get("tools", {}).get("builtin", [])

    # CRITICAL FIX: Ensure 'skill_setup' and 'grep' are ALWAYS loaded if they exist in factories,
    # regardless of config. This supports the new Skill System.
    mandatory_tools = ["skill_setup"]
    for tool_name in mandatory_tools:
        if tool_name not in builtin_tool_names and tool_name in builtin_tool_factories:
            builtin_tool_names.append(tool_name)

    # If config doesn't specify, we default to all keys in builtin_tool_factories
    # This ensures core tools are available even if config is minimal
    if not builtin_tool_names:
        builtin_tool_names = list(builtin_tool_factories.keys())

    for name in builtin_tool_names:
        # Skip Tavily if requested in config but not supported
        if name == "tavily_search" and name not in builtin_tool_factories:
            continue

        factory = builtin_tool_factories.get(name)
        if factory:
            tools_list.append(factory())
        else:
            print(f"Warning: Built-in tool {name} not found")

    # 4. Determine Extension Tools to Load
    # If enabled_tools is passed (dynamic), use it.
    # If None (static/startup), use config.
    if enabled_tools is None:
        enabled_tools = config.get("tools", {}).get("enabled", [])

    search_paths = [str(Path(__file__).resolve().parents[2])]
    if tool_paths:
        search_paths.extend(tool_paths)

    for path in search_paths:
        if path not in sys.path:
            sys.path.insert(0, path)

    for name in enabled_tools:
        try:
            # Try importing as 'tools.xxx' (Legacy)
            try:
                tool_module = importlib.import_module(f"tools.{name.replace('-', '_')}")
            except ModuleNotFoundError:
                # Try importing as full module path (New Skill System)
                tool_module = importlib.import_module(name)

            if hasattr(tool_module, "tools"):
                tools_list.extend(tool_module.tools)  # type: ignore
        except (ModuleNotFoundError, ImportError) as e:
            print(f"Error loading tool {name}: {str(e)}")

    return tools_list
