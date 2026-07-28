"""Maisaka 鉴权器。

对 Planner 的决策输出与 Replyer 的回复文本做身份核对，
主要通过一次独立的 LLM 审核判断输出中是否存在用户身份混淆
（归属错误、对象错误、称呼混淆、自我混淆）。

审核 Prompt 会注入参与者的已知身份与关系（从人物画像快照中
纯存储层读取，不产生额外模型调用），供审核模型交叉核对。
"""

from dataclasses import dataclass
from html import escape
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from json_repair import repair_json

import json
import time

from src.A_memorix.core.utils.profile_text import parse_profile_sections
from src.chat.message_receive.message import SessionMessage
from src.chat.utils.fixed_identity import build_fixed_identity_block
from src.chat.utils.utils import is_bot_self
from src.common.data_models.llm_service_data_models import LLMGenerationOptions
from src.common.logger import get_logger
from src.common.prompt_i18n import load_prompt
from src.config.config import global_config
from src.llm_models.payload_content.tool_option import ToolCall
from src.maisaka.context.messages import LLMContextMessage, SessionBackedMessage
from src.person_info.person_info import resolve_person_id_for_memory
from src.services.llm_service import LLMServiceClient
from src.services.memory_service import memory_service

from .decision import AuthDecision, IdentityIssue

logger = get_logger("maisaka_auth")

AUTH_TASK_NAME = "auth"
PLANNER_AUTH_REQUEST_TYPE = "maisaka.auth.planner"
REPLYER_AUTH_REQUEST_TYPE = "maisaka.auth.replyer"

AUTH_PARTICIPANT_LIMIT = 3
"""注入身份上下文的参与者人数上限。"""

AUTH_PARTICIPANT_LOOKBACK = 12
"""从最近多少条消息中提取参与者。"""

AUTH_PROFILE_SECTION_ITEM_LIMIT = 4
"""身份设定/关系设定每段最多保留的条目数。"""

AUTH_RELATION_EDGE_LIMIT = 12
"""注入的已知人际关系边上限。"""

PROFILE_EMPTY_ITEM_TEXT = "暂无"
"""结构化画像中的空段落占位文本。"""


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


@dataclass(slots=True)
class _ParticipantRef:
    """鉴权参考用的一位聊天参与者。"""

    platform: str
    user_id: str
    display_name: str


@dataclass(slots=True)
class ParticipantIdentityContext:
    """注入审核 Prompt 的参与者身份与人际关系上下文。"""

    person_profiles_text: str = "（无已知画像）"
    entity_relations_text: str = "（无已知人际关系）"


def _extract_auth_participants(
    chat_history: Sequence[LLMContextMessage],
    *,
    limit: int = AUTH_PARTICIPANT_LIMIT,
    lookback: int = AUTH_PARTICIPANT_LOOKBACK,
) -> List[_ParticipantRef]:
    """从最近聊天消息中提取不同的真实发言者（跳过 bot 自己，按时间顺序）。"""

    participants: List[_ParticipantRef] = []
    seen_user_ids: Set[str] = set()
    for message in reversed(chat_history[-max(lookback, 1) :]):
        if not isinstance(message, SessionBackedMessage) or message.original_message is None:
            continue
        original = message.original_message
        user_info = original.message_info.user_info
        user_id = str(user_info.user_id or "").strip()
        if not user_id or user_id in seen_user_ids:
            continue
        platform = str(original.platform or "")
        if is_bot_self(platform, user_id):
            continue
        seen_user_ids.add(user_id)
        participants.append(
            _ParticipantRef(
                platform=platform,
                user_id=user_id,
                display_name=(user_info.user_cardname or user_info.user_nickname or user_id).strip(),
            )
        )
        if len(participants) >= max(1, limit):
            break
    participants.reverse()
    return participants


def _meaningful_profile_items(lines: Sequence[str], *, limit: int) -> List[str]:
    """从结构化画像段落中取出有效条目（去掉占位符和前导符号）。"""

    items: List[str] = []
    for line in lines:
        text = str(line or "").strip().lstrip("-").strip()
        if not text or text == PROFILE_EMPTY_ITEM_TEXT:
            continue
        items.append(text)
        if len(items) >= max(1, limit):
            break
    return items


def _collect_bot_identity_names() -> Set[str]:
    """收集 bot 自己的称呼集合，用于人际关系边过滤与自我混淆核对。"""

    names: Set[str] = set()
    nickname = global_config.bot.nickname.strip()
    if nickname:
        names.add(nickname)
    for alias in global_config.bot.alias_names:
        normalized = str(alias or "").strip()
        if normalized:
            names.add(normalized)
    return names


async def collect_participant_identity_context(
    chat_history: Sequence[LLMContextMessage],
) -> ParticipantIdentityContext:
    """收集参与者的已知身份与关系上下文。

    数据全部来自人物画像快照的纯存储层读取（不触发证据收集与画像重建），
    任何一步失败都降级为空上下文，不影响鉴权主流程。
    """

    participants = _extract_auth_participants(chat_history)
    if not participants:
        return ParticipantIdentityContext()

    profile_blocks: List[str] = []
    # 人名集合用于人际关系边过滤：无论画像快照是否存在，参与者显示名都应计入
    person_names: Set[str] = {participant.display_name for participant in participants if participant.display_name}
    relation_edges_by_hash: Dict[str, Dict[str, Any]] = {}
    bot_names = _collect_bot_identity_names()
    now = time.time()

    for participant in participants:
        try:
            person_id = resolve_person_id_for_memory(
                platform=participant.platform,
                user_id=participant.user_id,
                strict_known=True,
            )
            if not person_id:
                continue
            snapshot = await memory_service.get_person_profile_snapshot(person_id=person_id)
        except Exception as exc:
            logger.debug(f"读取参与者画像快照失败，已跳过: user_id={participant.user_id}, err={exc}")
            continue
        if not snapshot:
            continue

        # 过期快照不作为审核依据
        expires_at = float(snapshot.get("expires_at") or 0)
        if expires_at and expires_at < now:
            continue

        person_name = str(snapshot.get("person_name") or "").strip() or participant.display_name
        aliases = [str(alias or "").strip() for alias in (snapshot.get("aliases") or []) if str(alias or "").strip()]
        person_names.add(person_name)
        person_names.update(aliases)

        # 汇总关系边（按 hash 去重），稍后统一做人-人过滤
        for edge in snapshot.get("relation_edges") or []:
            if not isinstance(edge, dict):
                continue
            edge_hash = str(edge.get("hash") or "").strip()
            if edge_hash:
                relation_edges_by_hash[edge_hash] = edge

        # 只取身份设定与关系设定两个段落，偏好/习惯等其他段落不注入
        profile_text = str(snapshot.get("profile_text") or "")
        sections = parse_profile_sections(profile_text)
        identity_items = _meaningful_profile_items(
            sections.get("身份设定", []), limit=AUTH_PROFILE_SECTION_ITEM_LIMIT
        )
        relation_items = _meaningful_profile_items(
            sections.get("关系设定", []), limit=AUTH_PROFILE_SECTION_ITEM_LIMIT
        )
        if not sections and profile_text.strip():
            # 非结构化画像（如人工覆写）无法按段解析，截断后作为身份设定参考
            identity_items = [" ".join(profile_text.split())[:200]]

        alias_text = f"，别名：{'、'.join(aliases)}" if aliases else ""
        block_lines = [f"- {person_name} (user_id={participant.user_id}{alias_text})"]
        if identity_items:
            block_lines.append(f"  身份设定：{'；'.join(identity_items)}")
        if relation_items:
            block_lines.append(f"  关系设定：{'；'.join(relation_items)}")
        profile_blocks.append("\n".join(block_lines))

    # 只保留参与者之间（含与 bot）的人际关系边，偏好类（人→物）边不注入
    known_entity_names = person_names | bot_names
    relation_lines: List[str] = []
    sorted_edges = sorted(
        relation_edges_by_hash.values(),
        key=lambda edge: float(edge.get("confidence", 0.0) or 0.0),
        reverse=True,
    )
    for edge in sorted_edges:
        subject = str(edge.get("subject") or "").strip()
        predicate = str(edge.get("predicate") or "").strip()
        obj = str(edge.get("object") or "").strip()
        if not subject or not predicate or not obj:
            continue
        if subject not in known_entity_names or obj not in known_entity_names:
            continue
        confidence = float(edge.get("confidence", 1.0) or 1.0)
        relation_lines.append(f"- {subject} →{predicate}→ {obj}（置信度 {confidence:.1f}）")
        if len(relation_lines) >= AUTH_RELATION_EDGE_LIMIT:
            break

    person_profiles_text = "\n".join(profile_blocks) if profile_blocks else "（无已知画像）"
    # 固定身份规则是配置级的权威身份依据，一并提供给审核模型，
    # 使"专属称呼被用于他人"能被身份矛盾/关系矛盾检查明确驳回
    fixed_identity_block = build_fixed_identity_block()
    if fixed_identity_block:
        person_profiles_text = f"{person_profiles_text}\n\n{fixed_identity_block}"

    return ParticipantIdentityContext(
        person_profiles_text=person_profiles_text,
        entity_relations_text="\n".join(relation_lines) if relation_lines else "（无已知人际关系）",
    )


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

        identity_context = await collect_participant_identity_context(chat_history)
        prompt = load_prompt(
            "auth_planner_check",
            bot_name=global_config.bot.nickname,
            person_profiles=identity_context.person_profiles_text,
            entity_relations=identity_context.entity_relations_text,
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
        identity_context = await collect_participant_identity_context(chat_history)
        prompt = load_prompt(
            "auth_replyer_check",
            bot_name=global_config.bot.nickname,
            person_profiles=identity_context.person_profiles_text,
            entity_relations=identity_context.entity_relations_text,
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
