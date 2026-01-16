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
from langchain_core.language_models import LanguageModelInput
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
from .tools import load_tools
from .skills import get_tools_for_skills, get_content_for_skills, SKILL_KEYWORDS

# ... (imports remain)

    async def chatbot(state: State):
        messages = state["messages"]
        
        # 1. Skill Auto-Loading Logic
        active_tool_names = state.get("active_tools", [])
        skill_content = "" # Content to inject
        
        if messages and isinstance(messages[-1], HumanMessage):
            last_msg = messages[-1].content.lower()
            current_skills = set(state.get("active_skills_list", ["default"])) 
            
            # Dynamic Keyword Matching
            for skill_name, keywords in SKILL_KEYWORDS.items():
                if isinstance(keywords, list):
                    for kw in keywords:
                        if kw in last_msg:
                            current_skills.add(skill_name)
                            break
            
            # Get tools and content
            active_tool_names = get_tools_for_skills(list(current_skills))
            skill_content = get_content_for_skills(list(current_skills))
        
        # Determine active tools based on state or default to all enabled
        
        # If no specific tools are active in state, use a default set or all
        if not active_tool_names:
            # Fallback to all enabled tools if no dynamic selection provided
            # Or you could define a "default" skill
            current_tools = all_enabled_tools
        else:
             # Load only the requested tools dynamically
            current_tools = load_tools(enabled_tools=active_tool_names)

        # Dynamic Binding
        if current_tools:
            llm_with_tools = llm.bind_tools(current_tools)
        else:
            llm_with_tools = llm

        if plugin_config.plugin.debug:
            print("传入消息: \n", messages)
        
        # 固定
        fixed_messages = []
        if hasattr(config.llm, "system_prompt") and config.llm.system_prompt:
            sp = config.llm.system_prompt
            # Inject Skill Content into System Prompt
            if skill_content:
                sp += "\n\n### Active Skills Knowledge:\n" + skill_content
            fixed_messages.append(SystemMessage(content=sp))
            
        if hasattr(config.llm, "qa_pairs") and config.llm.qa_pairs:
        if not active_tool_names:
            # Fallback to all enabled tools if no dynamic selection provided
            # Or you could define a "default" skill
            current_tools = all_enabled_tools
        else:
            # Load only the requested tools dynamically
            current_tools = load_tools(enabled_tools=active_tool_names)

        # Dynamic Binding
        if current_tools:
            llm_with_tools = llm.bind_tools(current_tools)
        else:
            llm_with_tools = llm

        if plugin_config.plugin.debug:
            print("传入消息: \n", messages)

        # 固定
        fixed_messages = []
        if hasattr(config.llm, "system_prompt") and config.llm.system_prompt:
            fixed_messages.append(SystemMessage(content=config.llm.system_prompt))
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
        return {"messages": [response]}

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
        # if isinstance(message, SystemMessage):
        #     output.append(f"SystemMessage: {message.content}\n")
        #     output.append("_" * 50 + "\n")
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
