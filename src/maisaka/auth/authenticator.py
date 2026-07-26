"""Maisaka 鉴权器。

对 Planner 的决策输出与 Replyer 的回复文本做身份核对，
主要通过一次独立的 LLM 审核判断输出中是否存在用户身份混淆
（归属错误、对象错误、称呼混淆、自我混淆）。
"""

from html import escape
from typing import Any, Dict, List, Optional, Sequence, Tuple

from json_repair import repair_json

import json

from src.chat.message_receive.message import SessionMessage
from src.chat.utils.utils import is_bot_self
from src.common.data_models.llm_service_data_models import LLMGenerationOptions
from src.common.logger import get_logger
from src.common.prompt_i18n import load_prompt
from src.config.config import global_config
from src.llm_models.payload_content.tool_option import ToolCall
from src.maisaka.context.messages import LLMContextMessage, SessionBackedMessage
from src.services.llm_service import LLMServiceClient

from .decision import AuthDecision, IdentityIssue

logger = get_logger("maisaka_auth")

AUTH_TASK_NAME = "auth"
PLANNER_AUTH_REQUEST_TYPE = "maisaka.auth.planner"
REPLYER_AUTH_REQUEST_TYPE = "maisaka.auth.replyer"


def _render_history_message_for_auth(message: LLMContextMessage) -> str:
    """把单条历史消息渲染成带真实身份信息的文本块，供鉴权审核使用。"""

    if not isinstance(message, SessionBackedMessage):
        return ""

    text = message.processed_plain_text.strip()
    if not text:
        return ""

    message_attrs: List[str] = []
    if message.message_id:
        message_attrs.append(f'msg_id="{escape(str(message.message_id), quote=True)}"')

    original = message.original_message
    if original is not None:
        user_info = original.message_info.user_info
        user_id = str(user_info.user_id or "").strip()
        user_name = (user_info.user_cardname or user_info.user_nickname or user_id).strip()
        if user_id and is_bot_self(original.platform, user_id):
            user_name = f"{user_name}(bot自己)" if user_name else "bot自己"
        if user_id:
            message_attrs.append(f'user_id="{escape(user_id, quote=True)}"')
        if user_name:
            message_attrs.append(f'user="{escape(user_name, quote=True)}"')
    else:
        message_attrs.append(f'user="{escape(message.source_kind, quote=True)}"')

    message_attrs.append(f'time="{message.timestamp.strftime("%H:%M:%S")}"')
    return f"<message {' '.join(message_attrs)}>\n{text}\n</message>"


def render_auth_history_context(chat_history: Sequence[LLMContextMessage], *, limit: int) -> str:
    """把最近的真实聊天消息渲染成鉴权审核上下文。

    只渲染真实会话消息（含 bot 自己发送的），跳过参考消息、
    工具结果与 assistant 思考，避免审核上下文混入噪音。
    """

    rendered_blocks: List[str] = []
    for message in chat_history[-max(limit, 1) :]:
        rendered_block = _render_history_message_for_auth(message)
        if rendered_block:
            rendered_blocks.append(rendered_block)
    return "\n".join(rendered_blocks)


def format_tool_calls_for_auth(tool_calls: Sequence[ToolCall]) -> str:
    """把 Planner 的工具调用列表格式化成审核可读的文本。"""

    if not tool_calls:
        return "（无）"
    lines: List[str] = []
    for tool_call in tool_calls:
        args_text = json.dumps(tool_call.args or {}, ensure_ascii=False)
        lines.append(f"- {tool_call.func_name}({args_text})")
    return "\n".join(lines)


def parse_auth_decision(raw_response: str) -> AuthDecision:
    """解析鉴权审核 LLM 的 JSON 响应。

    Raises:
        ValueError: 响应无法解析为合法审核结论时抛出。
    """

    raw = raw_response.strip()
    if not raw:
        raise ValueError("鉴权审核响应为空")

    parsed: Any = None
    try:
        parsed = json.loads(raw)
    except Exception:
        try:
            repaired = repair_json(raw)
            parsed = json.loads(repaired if isinstance(repaired, str) else json.dumps(repaired, ensure_ascii=False))
        except Exception:
            parsed = None

    if not isinstance(parsed, dict) or "passed" not in parsed:
        raise ValueError(f"鉴权审核响应格式非法: {raw[:200]}")

    issues: List[IdentityIssue] = []
    raw_issues = parsed.get("issues")
    if isinstance(raw_issues, list):
        for raw_issue in raw_issues:
            if not isinstance(raw_issue, dict):
                continue
            issue_type = str(raw_issue.get("issue_type") or "").strip()
            detail = str(raw_issue.get("detail") or "").strip()
            if issue_type or detail:
                issues.append(IdentityIssue(issue_type=issue_type or "unknown", detail=detail))

    return AuthDecision(
        passed=bool(parsed.get("passed")),
        reason=str(parsed.get("reason") or "").strip(),
        issues=issues,
    )


class Authenticator:
    """对 Planner 决策与 Replyer 回复进行身份核对的鉴权器。"""

    def __init__(self) -> None:
        self._clients: Dict[str, LLMServiceClient] = {}

    def _get_client(self, request_type: str, session_id: str = "") -> LLMServiceClient:
        """按请求类型惰性创建鉴权审核使用的 LLM 客户端。"""

        client = self._clients.get(request_type)
        if client is None:
            client = LLMServiceClient(task_name=AUTH_TASK_NAME, request_type=request_type, session_id=session_id)
            self._clients[request_type] = client
        return client

    async def _request_audit(self, prompt: str, *, request_type: str, session_id: str) -> str:
        """执行一次鉴权审核 LLM 请求并返回原始文本。"""

        client = self._get_client(request_type, session_id)
        generation_result = await client.generate_response(
            prompt=prompt,
            options=LLMGenerationOptions(temperature=0.1),
            session_id=session_id,
        )
        return (generation_result.response or "").strip()

    async def check_planner_output(
        self,
        *,
        thought_text: str,
        tool_calls: Sequence[ToolCall],
        chat_history: Sequence[LLMContextMessage],
        session_id: str = "",
    ) -> AuthDecision:
        """检查 Planner 输出是否存在用户身份混淆。

        审核服务异常时放行（fail-open）：鉴权器故障不应导致 bot 整体静默，
        异常会以日志形式完整暴露。
        """

        history_text = render_auth_history_context(
            chat_history,
            limit=int(global_config.auth.history_message_limit),
        )
        if not history_text:
            return AuthDecision(passed=True)

        prompt = load_prompt(
            "auth_planner_check",
            bot_name=global_config.bot.nickname,
            chat_history=history_text,
            planner_thought=thought_text.strip() or "（无）",
            planner_tool_calls=format_tool_calls_for_auth(tool_calls),
        )
        try:
            raw_response = await self._request_audit(
                prompt,
                request_type=PLANNER_AUTH_REQUEST_TYPE,
                session_id=session_id,
            )
            return parse_auth_decision(raw_response)
        except Exception as exc:
            logger.exception(f"Planner 输出鉴权审核失败，已放行本次输出: {exc}")
            return AuthDecision(passed=True)

    async def check_replyer_output(
        self,
        *,
        reply_text: str,
        reply_message: Optional[SessionMessage],
        chat_history: Sequence[LLMContextMessage],
        session_id: str = "",
    ) -> AuthDecision:
        """检查待发送的回复文本是否存在用户身份混淆。

        审核服务异常时放行（fail-open），与 Planner 审核策略一致。
        """

        history_text = render_auth_history_context(
            chat_history,
            limit=int(global_config.auth.history_message_limit),
        )
        if not history_text:
            return AuthDecision(passed=True)

        target_message_id, target_user_name, target_user_id, target_text = self._describe_target_message(reply_message)
        prompt = load_prompt(
            "auth_replyer_check",
            bot_name=global_config.bot.nickname,
            chat_history=history_text,
            target_message_id=target_message_id,
            target_user_name=target_user_name,
            target_user_id=target_user_id,
            target_message_text=target_text,
            reply_text=reply_text.strip(),
        )
        try:
            raw_response = await self._request_audit(
                prompt,
                request_type=REPLYER_AUTH_REQUEST_TYPE,
                session_id=session_id,
            )
            return parse_auth_decision(raw_response)
        except Exception as exc:
            logger.exception(f"Replyer 输出鉴权审核失败，已放行本次回复: {exc}")
            return AuthDecision(passed=True)

    @staticmethod
    def _describe_target_message(reply_message: Optional[SessionMessage]) -> Tuple[str, str, str, str]:
        """提取回复目标消息的身份信息。"""

        if reply_message is None:
            return "（无）", "（未知）", "", "（无）"
        user_info = reply_message.message_info.user_info
        user_id = str(user_info.user_id or "").strip()
        user_name = (user_info.user_cardname or user_info.user_nickname or user_id or "未知用户").strip()
        target_text = (reply_message.processed_plain_text or "").strip() or "（无文本）"
        return str(reply_message.message_id or ""), user_name, user_id, target_text


authenticator = Authenticator()
