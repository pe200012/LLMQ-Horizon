from nonebot.adapters.onebot.v11 import (
    Message,
    MessageEvent,
    GroupMessageEvent,
    Event,
    MessageSegment,
)
from nonebot.permission import SUPERUSER
from nonebot import on_message, on_command
from nonebot.params import CommandArg, EventMessage, EventPlainText
from nonebot.exception import MatcherException
from nonebot.plugin import PluginMetadata
from nonebot.adapters.onebot.v11.exception import ActionFailed
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from .graph import build_graph, get_llm, format_messages_for_print
from datetime import datetime
from typing import Dict, List
from random import choice
from .config import Config
from .utils import (
    extract_media_urls,
    send_in_chunks,
    get_user_name,
    build_message_content,
    remove_trigger_words,
    filter_sensitive_words,
)
import asyncio
import os
import re
from .config import plugin_config

__plugin_meta__ = PluginMetadata(
    name="LLM Chat",
    description="基于 LangGraph 的chatbot",
    usage="@机器人 或关键词，或使用命令前缀触发对话",
    config=Config,
)

os.environ["OPENAI_API_KEY"] = plugin_config.llm.api_key
os.environ["OPENAI_BASE_URL"] = plugin_config.llm.base_url
os.environ["GOOGLE_API_KEY"] = plugin_config.llm.google_api_key


from .skills import get_tools_for_skills, SKILL_REGISTRY


# 会话模板
class Session:
    def __init__(self, thread_id: str):
        self.thread_id = thread_id
        self.memory = MemorySaver()
        # 最后访问时间
        self.last_accessed = datetime.now()
        self.graph = None
        self.lock = asyncio.Lock()  # 添加会话锁
        self.processing = False  # 添加处理状态标志
        self.processing_start_time = None  # 处理开始时间
        # Default skills - can be customized per user
        self.active_skills = ["sgu", "weather"]

    @property
    def is_expired(self) -> bool:
        """判断会话是否过期"""
        now = datetime.now()
        # 超过CLEANUP_INTERVAL秒未访问则过期
        return (now - self.last_accessed).total_seconds() > CLEANUP_INTERVAL

    @property
    def is_processing_timeout(self) -> bool:
        """判断处理是否超时"""
        if not self.processing or not self.processing_start_time:
            return False
        now = datetime.now()
        # 处理时间超过60秒判定为超时
        return (now - self.processing_start_time).total_seconds() > 60

    def refresh(self):
        """刷新最后访问时间"""
        self.last_accessed = datetime.now()

    def try_acquire_lock(self) -> bool:
        """尝试获取锁"""
        if self.processing:
            if self.is_processing_timeout:
                self.processing = False
                self.processing_start_time = None
            else:
                return False
        return True

    async def start_processing(self):
        """开始处理"""
        self.processing = True
        self.processing_start_time = datetime.now()
        self.refresh()

    async def end_processing(self):
        """结束处理"""
        self.processing = False
        self.processing_start_time = None
        self.refresh()


# "group_123456_789012": Session对象1
sessions: Dict[str, Session] = {}

# 添加异步锁保护sessions字典
sessions_lock = asyncio.Lock()

CLEANUP_INTERVAL = 600  # 会话清理间隔(秒) 例:10分钟
LAST_CLEANUP_TIME = datetime.now()


async def cleanup_sessions():
    """清理过期会话"""
    async with sessions_lock:
        expired_keys = [k for k, s in sessions.items() if s.is_expired]
        for k in expired_keys:
            del sessions[k]
    return len(expired_keys)


async def get_or_create_session(thread_id: str) -> Session:
    """获取或创建会话,同时处理清理"""
    global LAST_CLEANUP_TIME

    # 每隔CLEANUP_INTERVAL秒检查一次过期会话
    now = datetime.now()
    if (now - LAST_CLEANUP_TIME).total_seconds() > CLEANUP_INTERVAL:
        cleaned = await cleanup_sessions()
        if cleaned > 0:
            print(f"已清理 {cleaned} 个过期会话")
        LAST_CLEANUP_TIME = now

    async with sessions_lock:
        if thread_id not in sessions:
            sessions[thread_id] = Session(thread_id)
        session = sessions[thread_id]
        session.refresh()
        return session


# 初始化模型和对话图
llm = None
graph_builder = None


async def initialize_resources():
    global llm, graph_builder
    if llm is None:
        llm = await get_llm()
        graph_builder = await build_graph(plugin_config, llm)


async def _process_llm_response(result: dict, thread_id: str) -> str:
    """处理LLM返回的消息，提取回复内容"""
    if not result["messages"]:
        print("警告: 结果消息列表为空")
        return plugin_config.responses.assistant_empty_reply

    last_message = result["messages"][-1]

    if isinstance(last_message, AIMessage):
        if last_message.invalid_tool_calls:
            if (
                isinstance(last_message.invalid_tool_calls, list)
                and last_message.invalid_tool_calls
            ):
                error_msg = last_message.invalid_tool_calls[0]["error"]
                print(f"工具调用错误: {error_msg}")
                return f"工具调用失败: {error_msg}"
            print("工具调用错误: 未知错误(无错误信息)")
            return "工具调用失败，但没有错误信息"
        if (not last_message.content) or (not last_message.content.strip()):
            # 检查是否是 AI 安全拦截
            finish_reason = getattr(last_message, "response_metadata", {}).get(
                "finish_reason"
            )
            if finish_reason == "SAFETY":
                print("AI消息安全拦截，阻断回复 -> 清理会话")
                async with sessions_lock:
                    if thread_id in sessions:
                        del sessions[thread_id]
                return "AI消息因安全策略被拦截。"

            print("空回复 -> 清理会话")
            async with sessions_lock:
                if thread_id in sessions:
                    del sessions[thread_id]
            return "对不起，我没有理解您的问题。"

        if last_message.content:
            return last_message.content.strip()

        print("警告: AI消息内容为空")
        return "对不起，我没有理解您的问题。"

    if isinstance(last_message, ToolMessage) and last_message.content:
        return (
            last_message.content
            if isinstance(last_message.content, str)
            else str(last_message.content)
        )

    print(f"警告: 未知的消息类型或内容为空: {type(last_message)}")
    return "对不起，我没有理解您的问题。"


async def _handle_langgraph_error(e: Exception, thread_id: str) -> str:
    """处理 LangGraph 调用的异常"""
    error_message = str(e)
    print(f"调用 LangGraph 时发生错误: {error_message}")
    print(f"错误类型: {type(e)}")
    print(f"完整异常信息: {e}")

    print("出现异常 -> 清理会话")
    async with sessions_lock:
        if thread_id in sessions:
            del sessions[thread_id]
    return (
        plugin_config.responses.token_limit_error
        if "'list' object has no attribute 'strip'" in error_message
        else plugin_config.responses.general_error
    )


def _chat_rule(event: Event) -> bool:
    """定义触发规则"""
    trigger_mode = plugin_config.plugin.trigger_mode
    trigger_words = plugin_config.plugin.trigger_words
    msg = str(event.get_message())

    if "at" in trigger_mode and event.is_tome():
        return True
    if "keyword" in trigger_mode:
        for word in trigger_words:
            if word in msg:
                return True
    if "prefix" in trigger_mode:
        for word in trigger_words:
            if msg.startswith(word):
                return True
    if not trigger_mode:
        return event.is_tome()
    return False


chat_handler = on_message(rule=_chat_rule, priority=10, block=True)


@chat_handler.handle()
async def handle_chat(
    # 提取消息全部对象
    event: MessageEvent,
    # 提取各种消息段
    message: Message = EventMessage(),
    # 提取纯文本
    plain_text: str = EventPlainText(),
):
    global llm, graph_builder

    cleaned_message = await remove_trigger_words(message, event)
    if not cleaned_message or cleaned_message.isspace():
        await chat_handler.finish(
            Message(choice(plugin_config.responses.empty_message_replies))
        )
        return

    # 检查输入是否包含敏感词，包含则直接忽略
    if filter_sensitive_words(
        cleaned_message, plugin_config.sensitive_words.input_words
    ):
        print("主消息包含敏感词，忽略处理")
        return

    # 检查引用消息是否包含敏感词
    if event.reply and event.reply.message:
        reply_text = str(event.reply.message)
        if filter_sensitive_words(
            reply_text, plugin_config.sensitive_words.input_words
        ):
            print("引用消息包含敏感词，忽略处理")
            return

    # 确保 llm 已初始化
    if llm is None:
        await initialize_resources()

    # 检查群聊/私聊开关，判断消息对象是否是群聊/私聊的实例
    if (
        isinstance(event, GroupMessageEvent) and not plugin_config.plugin.enable_group
    ) or (
        not isinstance(event, GroupMessageEvent)
        and not plugin_config.plugin.enable_private
    ):
        await chat_handler.finish(plugin_config.responses.disabled_message)

    # ----------------- 对话消息体构建START -----------------
    # 获取用户名
    user_name = await get_user_name(event)

    # 提取媒体链接
    media_urls = await extract_media_urls(
        message, event.reply.message if event.reply else None
    )

    # 构建会话ID，创建或获取Session对象
    if isinstance(event, GroupMessageEvent):
        if plugin_config.plugin.group_chat_isolation:
            thread_id = f"group_{event.group_id}_{event.user_id}"
        else:
            thread_id = f"group_{event.group_id}"
    else:
        thread_id = f"private_{event.user_id}"
    print(f"Current thread: {thread_id}")

    # 构建消息ID，传递给LangGraph
    if isinstance(event, GroupMessageEvent):
        message_id = f"Group_ID: {event.group_id}\nUser_ID: {event.user_id}"
    else:
        message_id = f"User_ID: {event.user_id}"

    # 构建消息内容
    message_content = await build_message_content(
        message, media_urls, event, user_name, message_id
    )

    session = await get_or_create_session(thread_id)

    # ---------- 判断当前会话ID对应的会话是否在处理中，如无则调用langgraph START ----------

    # 处理会话锁
    try:
        if not await asyncio.wait_for(session.lock.acquire(), timeout=1.0):
            await chat_handler.finish(
                Message(plugin_config.responses.session_busy_message)
            )
            return
    except asyncio.TimeoutError:
        await chat_handler.finish(Message(plugin_config.responses.session_busy_message))
        return

    try:
        if not session.try_acquire_lock():
            await chat_handler.finish(
                Message(plugin_config.responses.session_busy_message)
            )
            return

        await session.start_processing()
        # 如果当前会话没有图，则创建一个
        if session.graph is None:
            session.graph = graph_builder.compile(checkpointer=session.memory)

        # 调用 LangGraph

        # Get active tools for this session
        active_tools = get_tools_for_skills(session.active_skills)

        result = await session.graph.ainvoke(
            {
                "messages": [HumanMessage(content=message_content)],
                "active_tools": active_tools,  # Pass dynamic tools to graph
            },
            {"configurable": {"thread_id": thread_id}},
        )
        truncated_messages = result["messages"][-2:]
        print(format_messages_for_print(truncated_messages))

        response = await _process_llm_response(result, thread_id)
    except Exception as e:
        response = await _handle_langgraph_error(e, thread_id)
    finally:
        await session.end_processing()
        # 释放锁
        session.lock.release()

    # 定义媒体类型的正则和处理函数的映射
    MEDIA_PATTERNS = {
        "image": {
            "pattern": r"(?:https?://|file:///)[^\s]+?\.(?:png|jpg|jpeg|gif|bmp|webp)",
            "segment_func": MessageSegment.image,
            "error_msg": "图片",
        },
        "video": {
            "pattern": r"(?:https?://|file:///)[^\s]+?\.(?:mp4|avi|mov|mkv)",
            "segment_func": MessageSegment.video,
            "error_msg": "视频",
        },
        "audio": {
            "pattern": r"(?:https?://|file:///)[^\s]+?\.(?:mp3|wav|ogg|aac|flac)",
            "segment_func": MessageSegment.record,
            "error_msg": "音频",
        },
    }

    async def process_media_message(
        response: str, media_type: str, url: str
    ) -> Message:
        """处理包含媒体的消息"""
        media_info = MEDIA_PATTERNS[media_type]
        if plugin_config.plugin.media_include_text:
            # 清理markdown链接语法
            message_content = re.sub(r"!?\[.*?\]\((.*?)\)", r"\1", response)
            message_content = message_content.replace(url, "").strip()
            try:
                return Message(message_content) + media_info["segment_func"](url)
            except ActionFailed:
                return Message(message_content) + MessageSegment.text(
                    f" ({media_info['error_msg']}发送失败)"
                )
            except MatcherException:
                raise
            except Exception as e:
                return Message(message_content) + MessageSegment.text(
                    f" (未知错误: {e})"
                )
        else:
            # 仅发送媒体
            try:
                return Message(media_info["segment_func"](url))
            except ActionFailed:
                return Message(f"{media_info['error_msg']}发送失败")
            except MatcherException:
                raise
            except Exception as e:
                return Message(f"未知错误: {e}")

    # 在handle_chat函数中替换原来的媒体处理代码:
    for media_type, info in MEDIA_PATTERNS.items():
        if match := re.search(info["pattern"], response, re.IGNORECASE):
            result = await process_media_message(response, media_type, match.group(0))
            await chat_handler.finish(result)

    # 处理纯文本消息
    if plugin_config.plugin.chunk.enable:
        if await send_in_chunks(response, chat_handler):
            return
    await chat_handler.finish(Message(response))


# cmd
chat_command = on_command(
    "chat",
    priority=5,
    block=True,
    permission=SUPERUSER,
)


@chat_command.handle()
async def handle_chat_command(args: Message = CommandArg(), event: Event = None):
    """处理 chat model、chat clear、chat group 等命令"""
    global llm, graph_builder, sessions, plugin_config

    command_args = args.extract_plain_text().strip().split(maxsplit=1)
    if not command_args:
        await chat_command.finish(
            """请输入有效的命令：
            'chat model <模型名字>' 切换模型
            'chat clear' 清理会话
            'chat group <true/false>' 切换群聊会话隔离
            'chat down' 关闭对话功能
            'chat up' 开启对话功能
            'chat chunk <true/false>' 切换分开发送功能"""
        )
    command = command_args[0].lower()
    if not command_args:
        await chat_command.finish(
            """请输入有效的命令：
            'chat model <模型名字>' 切换模型
            'chat clear' 清理会话
            'chat group <true/false>' 切换群聊会话隔离
            'chat down' 关闭对话功能
            'chat up' 开启对话功能
            'chat chunk <true/false>' 切换分开发送功能"""
        )
    command = command_args[0].lower()
    if command == "model":
        # 处理模型切换
        if len(command_args) < 2:
            try:
                current_model = (
                    llm.model_name if hasattr(llm, "model_name") else llm.model
                )
                await chat_command.finish(f"当前模型: {current_model}")
            except MatcherException:
                raise
            except Exception as e:
                await chat_command.finish(f"获取当前模型失败: {str(e)}")
        model_name = command_args[1]
        try:
            new_llm = await get_llm(model_name)
            new_graph_builder = await build_graph(plugin_config, new_llm)
            # 成功创建新实例后才更新全局变量
            llm = new_llm
            graph_builder = new_graph_builder
            # 清理所有会话
            async with sessions_lock:
                sessions.clear()
            await chat_command.finish(f"已切换到模型: {model_name}")
        except MatcherException:
            raise
        except Exception as e:
            await chat_command.finish(f"切换模型失败: {str(e)}")

    elif command == "clear":
        # 处理清理历史会话
        async with sessions_lock:
            sessions.clear()
        await chat_command.finish("已清理所有历史会话。")

    elif command == "group":
        # 处理群聊会话隔离设置
        if len(command_args) < 2:
            current_group = plugin_config.plugin.group_chat_isolation
            await chat_command.finish(f"当前群聊会话隔离: {current_group}")

        isolation_str = command_args[1].strip().lower()
        if isolation_str == "true":
            plugin_config.plugin.group_chat_isolation = True
        elif isolation_str == "false":
            plugin_config.plugin.group_chat_isolation = False
        else:
            await chat_command.finish("请输入 true 或 false")

        # 清理对应会话
        keys_to_remove = []
        if isinstance(event, GroupMessageEvent):
            prefix = f"group_{event.group_id}"
            if plugin_config.plugin.group_chat_isolation:
                keys_to_remove = [
                    key for key in sessions if key.startswith(f"{prefix}_")
                ]
            else:
                keys_to_remove = [key for key in sessions if key == prefix]
        else:
            keys_to_remove = [key for key in sessions if key.startswith("private_")]

        async with sessions_lock:
            for key in keys_to_remove:
                del sessions[key]

        await chat_command.finish(
            f"已{'禁用' if not plugin_config.plugin.group_chat_isolation else '启用'}群聊会话隔离，已清理对应会话"
        )
    elif command == "down":
        plugin_config.plugin.enable_private = False
        plugin_config.plugin.enable_group = False
        await chat_command.finish("已关闭对话功能")
    elif command == "up":
        plugin_config.plugin.enable_private = True
        plugin_config.plugin.enable_group = True
        await chat_command.finish("已开启对话功能")
    elif command == "chunk":
        if len(command_args) < 2:
            await chat_command.finish(
                f"当前分开发送开关: {plugin_config.plugin.chunk.enable}"
            )
        chunk_str = command_args[1].strip().lower()
        if chunk_str == "true":
            plugin_config.plugin.chunk.enable = True
            await chat_command.finish("已开启分开发送回复功能")
        elif chunk_str == "false":
            plugin_config.plugin.chunk.enable = False
            await chat_command.finish("已关闭分开发送回复功能")
        else:
            await chat_command.finish("请输入 true 或 false")

    elif command == "skill":
        # Handle skill management
        if len(command_args) < 2:
            await chat_command.finish(
                "Use: /chat skill list | load <name> | unload <name>"
            )
            return

        subcommand_args = command_args[1].strip().split()
        if not subcommand_args:
            await chat_command.finish(
                "Use: /chat skill list | load <name> | unload <name>"
            )
            return

        action = subcommand_args[0].lower()

        # Get session for current context (Superuser only for now, but applies to context)
        if isinstance(event, GroupMessageEvent):
            if plugin_config.plugin.group_chat_isolation:
                tid = f"group_{event.group_id}_{event.user_id}"
            else:
                tid = f"group_{event.group_id}"
        elif isinstance(event, MessageEvent):  # Private
            tid = f"private_{event.user_id}"
        else:
            await chat_command.finish("Unknown context.")
            return

        session = await get_or_create_session(tid)

        if action == "list":
            active = ", ".join(session.active_skills)
            available = ", ".join(SKILL_REGISTRY.keys())
            await chat_command.finish(f"Active: {active}\nAvailable: {available}")

        elif action == "load":
            if len(subcommand_args) < 2:
                await chat_command.finish("Specify skill name.")
                return
            skill_name = subcommand_args[1]
            if skill_name not in SKILL_REGISTRY:
                await chat_command.finish(f"Skill '{skill_name}' not found.")
                return
            if skill_name not in session.active_skills:
                session.active_skills.append(skill_name)
                await chat_command.finish(f"Skill '{skill_name}' loaded.")
            else:
                await chat_command.finish(f"Skill '{skill_name}' already active.")

        elif action == "unload":
            if len(subcommand_args) < 2:
                await chat_command.finish("Specify skill name.")
                return
            skill_name = subcommand_args[1]
            if skill_name in session.active_skills:
                session.active_skills.remove(skill_name)
                await chat_command.finish(f"Skill '{skill_name}' unloaded.")
            else:
                await chat_command.finish(f"Skill '{skill_name}' wasn't active.")
    else:
        await chat_command.finish(
            "无效的命令，请使用 'chat model <模型名字>'、'chat clear' 或 'chat group <true/false>'。"
        )


about_command = on_command("about", priority=5, block=True)


@about_command.handle()
async def handle_about_command():
    """处理 /about 命令"""
    current_model = "未初始化"
    if llm:
        # Try to get model name from different LC model classes
        current_model = getattr(llm, "model_name", getattr(llm, "model", "未知"))

    msg = (
        f"🤖 {__plugin_meta__.name}\n"
        f"版本: v0.1.1\n"
        f"作者: pe200012 (姆Q)\n"
        f"说明: {__plugin_meta__.description}\n"
        f"当前模型: {current_model}\n"
        f"架构: LangGraph Agentic System\n"
        f"驱动: NoneBot2"
    )
    await about_command.finish(msg)
