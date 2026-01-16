from typing import Any, Dict, List, Tuple
import importlib
import sys
import re
from pathlib import Path

# Registry Storage
SKILL_REGISTRY: Dict[str, List[str]] = {"default": []}  # name -> [tool_module_names]
SKILL_DESCRIPTIONS: Dict[str, str] = {}  # name -> description
SKILL_KEYWORDS: Dict[str, List[str]] = {}  # name -> keywords
SKILL_CONTENT: Dict[str, str] = {}  # name -> markdown_body


import yaml


def _parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """YAML frontmatter parser using PyYAML."""
    meta: Dict[str, Any] = {}
    body = content

    # Check for --- block at start
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
    if match:
        yaml_text = match.group(1)
        body = match.group(2)
        try:
            parsed = yaml.safe_load(yaml_text)
            if isinstance(parsed, dict):
                meta = parsed
        except yaml.YAMLError as e:
            print(f"Error parsing YAML frontmatter: {e}")

    return meta, body


def _discover_skills():
    """
    Dynamically discover skills from 'skills/' directory (SKILL.md architecture).
    """
    root_path = Path(__file__).resolve().parents[2]
    skills_dir = root_path / "skills"

    if not skills_dir.exists():
        print(f"Skills directory not found: {skills_dir}")
        return

    # Add root path to sys.path so we can import 'skills.x.tools'
    if str(root_path) not in sys.path:
        sys.path.insert(0, str(root_path))

    # Scan for SKILL.md
    for skill_path in skills_dir.iterdir():
        if not skill_path.is_dir():
            continue

        md_file = skill_path / "SKILL.md"
        if not md_file.exists():
            continue

        try:
            content = md_file.read_text(encoding="utf-8")
            meta, body = _parse_frontmatter(content)

            skill_name = meta.get("name", skill_path.name)

            # Store Metadata
            SKILL_DESCRIPTIONS[skill_name] = meta.get("description", "")
            SKILL_KEYWORDS[skill_name] = meta.get("keywords", [])
            SKILL_CONTENT[skill_name] = body

            # Look for adjacent tools.py (Optional)
            tools_file = skill_path / "tools.py"
            if tools_file.exists():
                module_name = f"skills.{skill_path.name}.tools"
                if skill_name not in SKILL_REGISTRY:
                    SKILL_REGISTRY[skill_name] = []
                SKILL_REGISTRY[skill_name].append(module_name)
            else:
                # Skill without tools (Knowledge Only)
                # Ensure it exists in registry so it can be activated
                if skill_name not in SKILL_REGISTRY:
                    SKILL_REGISTRY[skill_name] = []

        except Exception as e:
            print(f"Error loading skill {skill_path.name}: {e}")

    # Load default tools from config (legacy support)
    # We can still keep the config-tools.toml logic for 'default' tools if needed
    pass


# Initialize Registry at startup
_discover_skills()

# Log discovered skills
if SKILL_KEYWORDS:
    print(
        f"[Skills] Discovered {len(SKILL_KEYWORDS)} skills: {list(SKILL_KEYWORDS.keys())}"
    )
    for name, kws in SKILL_KEYWORDS.items():
        print(f"  - {name}: keywords={kws[:5]}{'...' if len(kws) > 5 else ''}")
else:
    print("[Skills] No skills discovered. Check skills/ directory.")


def get_tools_for_skills(active_skills: List[str]) -> List[str]:
    """Get unique tool modules."""
    tools = set()
    # Always include default tools (if any configured)
    if "default" in SKILL_REGISTRY:
        tools.update(SKILL_REGISTRY["default"])

    for skill in active_skills:
        if skill in SKILL_REGISTRY:
            tools.update(SKILL_REGISTRY[skill])
    return list(tools)


def get_content_for_skills(active_skills: List[str]) -> str:
    """Get concatenated Markdown content for active skills."""
    content_parts = []
    for skill in active_skills:
        if skill in SKILL_CONTENT:
            content_parts.append(f"--- Skill: {skill} ---\n{SKILL_CONTENT[skill]}")
    return "\n\n".join(content_parts)
