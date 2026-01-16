from typing import Annotated, List, Union, Any, Optional, Dict
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import (
    SystemMessage,
    trim_messages,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_xai import ChatXAI
from langchain_core.language_models import LanguageModelInput
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
from .tools import load_tools
from .config import Config
from .config import plugin_config
import json
from .skills import (
    get_tools_for_skills,
    get_content_for_skills,
    SKILL_DESCRIPTIONS,
)
from .core_tools import SKILL_STORE
from .callbacks import ToolLoggingCallback


groq_models = {"llama3-groq-70b-8192-tool-use-preview", "llama-3.3-70b-versatile"}

xai_models = {
    "grok-3",
    "grok-3-fast",
    "grok-3-mini",
    "grok-3-mini-fast",
    "grok-4",
    "grok-4-fast",
    "grok-code-fast-1",
}

think_oai_models = {
    "o1",
    "o1-2024-12-17",
    "o1-preview",
    "o1-preview-2024-09-12",
    "o3-mini",
    "03-mini-2025-01-31",
}


class MyOpenAI(ChatOpenAI):
    @property
    def _default_params(self) -> Dict[str, Any]:
        params = super()._default_params
        if "max_completion_tokens" in params:
            params["max_tokens"] = params.pop("max_completion_tokens")
        return params

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if "max_completion_tokens" in payload:
            payload["max_tokens"] = payload.pop("max_completion_tokens")
        return payload


async def get_llm(model=None):
    """异步获取适当的 LLM 实例"""
    model = model.lower() if model else plugin_config.llm.model
    print(f"使用模型: {model}")

    # Configure callbacks
    callbacks = []
    if plugin_config.plugin.log_tools:
        callbacks.append(ToolLoggingCallback())

    try:
        if (
            hasattr(plugin_config.llm, "force_openai")
            and plugin_config.llm.force_openai
        ):
            print("强制使用 OpenAI 通道")
            return MyOpenAI(
                model=model,
                temperature=plugin_config.llm.temperature,
                max_tokens=plugin_config.llm.max_tokens,
                api_key=plugin_config.llm.api_key,
                base_url=plugin_config.llm.base_url,
                top_p=plugin_config.llm.top_p,
                callbacks=callbacks,
            )

        if model in think_oai_models:
            print("使用标准openai")
            return ChatOpenAI(
                model=model,
                max_completion_tokens=plugin_config.llm.max_tokens,
                api_key=plugin_config.llm.api_key,
                base_url=plugin_config.llm.base_url,
                callbacks=callbacks,
            )

        if model in groq_models:
            print("使用groq")
            return ChatGroq(
                model=model,
                temperature=plugin_config.llm.temperature,
                max_tokens=plugin_config.llm.max_tokens,
                api_key=plugin_config.llm.groq_api_key,
                callbacks=callbacks,
            )
        elif model in xai_models or model.startswith("grok"):
            print("使用xAI Grok")
            return ChatXAI(
                model=model,
                temperature=plugin_config.llm.temperature,
                max_tokens=plugin_config.llm.max_tokens,
                xai_api_key=plugin_config.llm.xai_api_key,
                callbacks=callbacks,
            )
        elif "/" in model:
            print(f"使用OpenRouter: {model}")
            return ChatOpenAI(
                model=model,
                temperature=plugin_config.llm.temperature,
                max_tokens=plugin_config.llm.max_tokens,
                api_key=plugin_config.llm.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                callbacks=callbacks,
            )
        elif "gemini" in model:
            print("使用google")
            return ChatGoogleGenerativeAI(
                model=model,
                temperature=plugin_config.llm.temperature,
                max_tokens=plugin_config.llm.max_tokens,
                google_api_key=plugin_config.llm.google_api_key,
                top_p=plugin_config.llm.top_p,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                },
                callbacks=callbacks,
            )
        else:
            print("使用 OpenAI")
            return MyOpenAI(
                model=model,
                temperature=plugin_config.llm.temperature,
                max_tokens=plugin_config.llm.max_tokens,
                api_key=plugin_config.llm.api_key,
                base_url=plugin_config.llm.base_url,
                top_p=plugin_config.llm.top_p,
                callbacks=callbacks,
            )
    except Exception as e:
        print(f"模型初始化失败: {str(e)}")
        raise


class State(TypedDict):
    messages: Annotated[list, add_messages]
    active_tools: List[str]


async def build_graph(config: Config, llm):
    """构建并返回对话图"""
    # Load all tools for the ToolNode execution
    # This now loads Core Tools + whatever is in config enabled
    all_enabled_tools = load_tools()

    trimmer = trim_messages(
        strategy="last",
        max_tokens=config.llm.max_context_messages,
        token_counter=len,
        include_system=True,
        allow_partial=False,
        start_on="human",
        end_on=("human", "tool"),
    )

    async def chatbot(state: State):
        messages = state["messages"]

        # 1. Skill Management (LLM Controlled)
        # Fetch active skills from the store (using default thread for now)
        thread_id = "default"
        active_skills = SKILL_STORE.get(thread_id, set())

        # Resolve active tool modules from skills
        active_tool_names = get_tools_for_skills(list(active_skills))

        # Resolve active knowledge content
        skill_content = get_content_for_skills(list(active_skills))

        # 2. Dynamic Tool Binding
        # load_tools() will always include Core Tools (skill_setup, etc.)
        # plus the skill-specific tools we request here.
        current_tools = load_tools(enabled_tools=active_tool_names)
        llm_with_tools = llm.bind_tools(current_tools)

        # Logging for Debugging
        if plugin_config.plugin.debug:
            print(
                f"\n[Graph] Active Skills ({len(active_skills)}): {list(active_skills)}"
            )
            print(
                f"[Graph] Active Tools ({len(current_tools)}): {[t.name for t in current_tools]}"
            )

        # 3. Message Construction
        fixed_messages = []
        if hasattr(config.llm, "system_prompt") and config.llm.system_prompt:
            sp = config.llm.system_prompt

            # Inject Available Skills List (So LLM knows what it can enable)
            sp += "\n\n### Skill System\n"
            sp += "You have access to a 'skill_setup' tool to enable/disable specialized capabilities.\n"
            sp += "Available Skills:\n"
            for name, desc in SKILL_DESCRIPTIONS.items():
                status = "(Active)" if name in active_skills else "(Inactive)"
                sp += f"- {name} {status}: {desc}\n"

            # Stronger Tool Use Instruction
            sp += "\n\n### IMPORTANT INSTRUCTION:\n"
            sp += "If the user asks for something that requires searching, checking, or performing an action, YOU MUST USE THE TOOLS IMMEDIATELY.\n"
            sp += "DO NOT just say 'I will search' or 'Let me check'. CALL THE TOOL directly.\n"
            sp += "If you need specific knowledge (like SGU admission), enable the skill first using 'skill_setup'.\n"

            # Inject Active Skills Knowledge
            if skill_content:
                sp += "\n### Active Skills Knowledge\n" + skill_content

            fixed_messages.append(SystemMessage(content=sp))

        if hasattr(config.llm, "qa_pairs") and config.llm.qa_pairs:
            for user_content, assistant_content in config.llm.qa_pairs:
                fixed_messages.append(HumanMessage(content=user_content))
                fixed_messages.append(AIMessage(content=assistant_content))

        # 修剪
        trimmed_messages = trimmer.invoke(messages)
        if not trimmed_messages:
            return {"messages": []}

        # 合并
        messages = fixed_messages + trimmed_messages
        response = await llm_with_tools.ainvoke(messages)

        if plugin_config.plugin.debug:
            print("AI回复: \n", response)

        return {"messages": [response], "active_tools": active_tool_names}

    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)

    # ToolNode must be able to execute ANY tool that might be called
    tool_node = ToolNode(tools=all_enabled_tools)

    graph_builder.add_node("tools", tool_node)
    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")

    return graph_builder


def format_messages_for_print(
    messages: List[Union[SystemMessage, HumanMessage, AIMessage, ToolMessage]],
) -> str:
    """格式化 LangChain 消息列表"""
    output = []
    for message in messages:
        if isinstance(message, HumanMessage):
            output.append(
                "\n" + "_" * 50 + "\nHumanMessage: \n" + f"{message.content}\n\n"
            )
        elif isinstance(message, AIMessage):
            output.append(f"AIMessage: {message.content}\n")
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    output.append(f"  Tool Name: {tool_call['name']}\n")
                    try:
                        args = json.loads(tool_call["args"])
                    except (json.JSONDecodeError, TypeError):
                        args = tool_call["args"]
                    output.append(f"  Tool Arguments: {args}\n\n")
        elif isinstance(message, ToolMessage):
            output.append(
                f"ToolMessage: \n  Tool Name: {message.name}\n  Tool content: {message.content}\n\n"
            )
    return "".join(output)
