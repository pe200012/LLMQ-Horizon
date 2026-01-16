from langchain_core.tools import tool
from typing import List, Dict, Optional, Any
import os
import requests
import re
import glob
from bs4 import BeautifulSoup
from langchain_community.tools.tavily_search import TavilySearchResults

from .skills import SKILL_DESCRIPTIONS, get_content_for_skills

# --- Todo Tool ---
# In-memory storage for simplicity. In production, use database/redis.
TODO_STORE: Dict[str, List[str]] = {}

# --- Skill Tool ---
# In-memory storage for active skills per thread
SKILL_STORE: Dict[str, set] = {}


@tool(parse_docstring=True)
def skill_setup(action: str, skill_name: str) -> str:
    """
    Enable or disable a skill. Use this when you need specialized knowledge or tools.

    Args:
        action: 'enable' or 'disable'.
        skill_name: The name of the skill (e.g., 'sgu', 'python-coding').

    Returns:
        Result message.
    """
    thread_id = "default"  # simplified
    if thread_id not in SKILL_STORE:
        SKILL_STORE[thread_id] = set()

    if action == "enable":
        if skill_name not in SKILL_DESCRIPTIONS:
            return f"Error: Skill '{skill_name}' not found. Available: {list(SKILL_DESCRIPTIONS.keys())}"
        SKILL_STORE[thread_id].add(skill_name)
        return f"Skill '{skill_name}' enabled. Knowledge will be injected in the next turn."
    elif action == "disable":
        SKILL_STORE[thread_id].discard(skill_name)
        return f"Skill '{skill_name}' disabled."
    else:
        return "Error: Action must be 'enable' or 'disable'."


@tool(parse_docstring=True)
def todo_write(todos: List[Dict[str, str]]) -> str:
    """
    Create or update the todo list.

    Args:
        todos: A list of todo items, each with 'content' (string) and 'status' ('pending'/'in_progress'/'completed').

    Returns:
        Confirmation message with current count.
    """
    # Simplified: We just replace the list for the default thread
    # Real implementation needs thread_id context
    thread_id = "default"

    # Store as simple strings for now to match previous logic, or complex objects
    # Let's store the raw dicts
    TODO_STORE[thread_id] = todos

    pending = len([t for t in todos if t.get("status") != "completed"])
    return f"Updated todo list. {pending} tasks pending."


@tool(parse_docstring=True)
def todo_read() -> str:
    """
    Read the current todo list.

    Returns:
        JSON string of the todo list.
    """
    thread_id = "default"
    todos = TODO_STORE.get(thread_id, [])
    import json

    return json.dumps(todos, indent=2, ensure_ascii=False)


# --- Web Tools ---


@tool(parse_docstring=True)
def web_search(query: str) -> str:
    """
    Search the web for information.

    Args:
        query: The search query.

    Returns:
        Search results with titles and URLs.
    """
    try:
        tavily = TavilySearchResults(max_results=5)
        results = tavily.invoke(query)
        output = []
        for res in results:
            content = res.get("content", "No content")
            url = res.get("url", "No URL")
            output.append(f"### {url}\n{content}\n")
        return "\n".join(output)
    except Exception as e:
        return f"Search error: {str(e)}"


@tool(parse_docstring=True)
def web_fetch(url: str) -> str:
    """
    Fetch content from a URL and convert to Markdown.

    Args:
        url: The URL to fetch.

    Returns:
        Page content in Markdown format.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 Bot"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        # Simple HTML to Text conversion
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove junk
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        text = soup.get_text(separator="\n\n")

        # Cleanup
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        content = "\n".join(lines)

        return content[:5000] + ("\n...(truncated)" if len(content) > 5000 else "")
    except Exception as e:
        return f"Fetch error: {str(e)}"


# --- Code/Grep Tool ---


@tool(parse_docstring=True)
def grep_tool(pattern: str, path: str = ".", include: str = "*") -> str:
    """
    Search for text patterns in files (like grep).

    Args:
        pattern: Regex pattern to search for.
        path: Directory to search in (relative to root).
        include: Glob pattern for file names (e.g. "*.py").

    Returns:
        Matching lines with file paths and line numbers.
    """
    # Security: Prevent escaping root
    if ".." in path or path.startswith("/"):
        return "Error: Invalid path. Stay in current directory."

    results = []
    try:
        # Resolve files
        search_path = os.path.join(os.getcwd(), path) if path != "." else os.getcwd()

        # Walk and match
        regex = re.compile(pattern)

        # Recursive glob if using python 3.10+
        files = glob.glob(f"{search_path}/**/{include}", recursive=True)

        for file_path in files:
            if os.path.isdir(file_path):
                continue
            if ".git" in file_path or "__pycache__" in file_path:
                continue

            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f):
                        if regex.search(line):
                            # Make path relative for readability
                            rel_path = os.path.relpath(file_path, os.getcwd())
                            results.append(f"{rel_path}:{i + 1}: {line.strip()}")
                            if len(results) > 50:
                                return "\n".join(results) + "\n...(truncated limit 50)"
            except:
                continue

        if not results:
            return "No matches found."

        return "\n".join(results)
    except Exception as e:
        return f"Grep error: {str(e)}"


# Export for registration
CORE_TOOLS: Dict[str, Any] = {
    "websearch": web_search,
    "webfetch": web_fetch,
    "todowrite": todo_write,
    "todoread": todo_read,
    "grep": grep_tool,
    "skill_setup": skill_setup,
}
