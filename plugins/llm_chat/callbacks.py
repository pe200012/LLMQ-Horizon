from langchain_core.callbacks import BaseCallbackHandler
from typing import Any, Dict, List, Optional, Union
from uuid import UUID
from langchain_core.outputs import LLMResult


class ToolLoggingCallback(BaseCallbackHandler):
    """Callback Handler that logs tool execution."""

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when tool starts running."""
        print(f"\n[Tool Start] 🛠️  {serialized.get('name')} input: {input_str}")

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when tool ends running."""
        # Truncate long output for readability
        print_output = output[:500] + "..." if len(output) > 500 else output
        print(f"[Tool End] ✅ Output: {print_output}\n")

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when tool errors."""
        print(f"[Tool Error] ❌ {error}\n")
