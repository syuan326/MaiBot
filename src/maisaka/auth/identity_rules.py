"""鉴权固定身份规则：系统级 UID 硬比对与规则文案生成。

规则来自配置（auth.identity_rules），将指定用户（platform + user_id）绑定为固定身份。
鉴权审核前由 Python 代码直接做 UID 比对（不依赖 LLM），比对结论与规则文案
动态注入审核 Prompt；结构化结果同时提供给麦麦观察的鉴权事件卡片展示。

名字集合按 platform + user_id 只读查询 person_info 表实时获取
（主名称 + 平台昵称 + 各群群名片），随消息接收自动保持最新，无需配置维护。
正确称呼按群动态解析：同一用户在不同群有不同的群名片，鉴权时按当前群
取该用户的正确称呼，其他群的称呼视为群混淆进入硬性禁止集合。
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple

from sqlmodel import col, select

import json

from src.common.database.database import get_db_session
from src.common.database.database_model import PersonInfo
from src.common.logger import get_logger
from src.config.config import global_config
from src.config.official_configs import AuthIdentityRuleConfig

logger = get_logger("maisaka_auth_identity")

IdentityCheckStage = Literal["planner", "replyer"]

MIN_HARD_CHECK_NAME_LENGTH = 2
"""硬性称呼核对的最短名字长度；低于该长度的称呼不做子串匹配，避免单字误伤（如「明」命中「明天」）。"""


@dataclass(slots=True)
class IdentityCheckResult:
    """一次固定身份核查的结果：Prompt 文案 + 结构化载荷（供监控卡片展示）。"""

    block_text: str = ""
    """注入审核 Prompt 的核查结论文案；未配置规则时为空串。"""

    payload: Optional[Dict[str, Any]] = None
    """结构化的核查结果（发送者、是否目标用户、当前群正确称呼、禁止称呼等）；未配置规则时为 None。"""


def get_identity_rules() -> List[AuthIdentityRuleConfig]:
    """读取配置中的固定身份规则列表。"""
    return list(global_config.auth.identity_rules)


def resolve_known_names(platform: str, user_id: str, *, group_id: Optional[str] = None) -> List[str]:
    """按 platform + user_id 查询 person_info，并集该用户所有已知名字。

    并集来源：麦麦记的主名称、平台昵称、各群群名片；按消息接收自动更新。
    该用户尚未被登记时返回空列表（正常的新用户情况）。

    Args:
        platform: 用户所在平台。
        user_id: 用户ID。
        group_id: 群上下文过滤：
            - None（默认）：返回全量并集（主名称 + 平台昵称 + 所有群群名片）。
            - ""：无群上下文（如私聊），返回主名称 + 平台昵称，不含任何群名片。
            - 非空：返回主名称 + 平台昵称 + 该群的群名片（当前群正确称呼）。
    """
    normalized_platform = platform.strip()
    normalized_user_id = str(user_id).strip()
    if not normalized_platform or not normalized_user_id:
        return []

    names: List[str] = []

    def add_name(raw_name: object) -> None:
        name = str(raw_name or "").strip()
        if name and name not in names:
            names.append(name)

    try:
        with get_db_session(auto_commit=False) as session:
            statement = (
                select(PersonInfo)
                .where(col(PersonInfo.platform) == normalized_platform)
                .where(col(PersonInfo.user_id) == normalized_user_id)
                .limit(1)
            )
            person = session.exec(statement).first()
    except Exception as exc:
        logger.warning(f"查询固定身份规则的已知名字失败: {normalized_platform}:{normalized_user_id}, err={exc}")
        return []

    if person is None:
        return []

    add_name(person.person_name)
    add_name(person.user_nickname)
    if person.group_cardname:
        try:
            for item in json.loads(person.group_cardname):
                if not isinstance(item, dict):
                    continue
                item_group_id = str(item.get("group_id") or "").strip()
                if group_id is not None:
                    if group_id == "":
                        continue
                    if item_group_id != group_id:
                        continue
                add_name(item.get("group_cardname"))
        except (ValueError, TypeError) as exc:
            logger.debug(f"解析群名片 JSON 失败: {normalized_platform}:{normalized_user_id}, err={exc}")
    return names


def _format_aliases(aliases: Sequence[str]) -> str:
    """把称呼列表格式化为「哥哥」「老大」样式。"""
    return "".join(f"「{alias.strip()}」" for alias in aliases if alias.strip())


def _format_known_names(names: Sequence[str]) -> str:
    """把已知名字集合格式化为提示词中的展示文本。"""
    return "、".join(names) if names else "（暂无已知名字）"


def build_identity_check_block(
    platform: str,
    user_id: str,
    *,
    stage: IdentityCheckStage,
    sender_name: str = "",
    group_id: str = "",
    participant_keys: Optional[Set[Tuple[str, str]]] = None,
) -> IdentityCheckResult:
    """对当前消息发送者做固定身份 UID 硬比对，生成审核 Prompt 文案与结构化结果。

    正确称呼按当前群动态解析（不依赖预设称呼）：目标用户在当前群的正确称呼进入
    允许集合，其在其他群的称呼进入禁止集合（群混淆）；其他规则用户的已知名字
    仅在对方参与当前会话时进入禁止集合，避免无关群的同名误伤。

    Args:
        platform: 当前消息发送者所在平台。
        user_id: 当前消息发送者的用户ID。
        stage: 审核阶段；planner 与 replyer 的提示措辞不同。
        sender_name: 当前消息发送者的显示名，仅用于监控卡片展示。
        group_id: 当前会话的群ID；空串表示无群上下文（如私聊）。
        participant_keys: 当前会话参与者（platform, user_id）集合；提供后，
            仅参与会话的规则用户的常规名字进入硬性禁止集合。

    Returns:
        IdentityCheckResult；未配置规则或发送者信息缺失时 block_text 为空、payload 为 None。
    """
    rules = get_identity_rules()
    normalized_user_id = str(user_id).strip()
    if not rules or not normalized_user_id:
        return IdentityCheckResult()

    normalized_platform = platform.strip()
    normalized_group_id = str(group_id or "").strip()
    stage_check_hint = "检查规划器中对该用户的描述" if stage == "planner" else "检查回复中对该用户的称呼"

    matched_rule: Optional[AuthIdentityRuleConfig] = None
    unmatched_rules: List[AuthIdentityRuleConfig] = []
    for rule in rules:
        if rule.user_id.strip() == normalized_user_id and rule.platform.strip() == normalized_platform:
            matched_rule = rule
        else:
            unmatched_rules.append(rule)

    lines: List[str] = ["【固定身份核查】（系统级 UID 硬比对结论，可信，必须执行）"]
    payload: Dict[str, Any] = {
        "sender_user_id": normalized_user_id,
        "sender_name": sender_name.strip(),
        "is_target": matched_rule is not None,
        "group_id": normalized_group_id,
        "allowed_names": [],
        "forbidden_names": [],
        "aliases": [],
        "known_names": [],
        "summary": "",
    }

    if matched_rule is not None:
        # 目标用户：按当前群动态解析正确称呼，不依赖预设；预设专属称呼作为补充允许项
        allowed_names = resolve_known_names(matched_rule.platform, matched_rule.user_id, group_id=normalized_group_id)
        aliases = [alias.strip() for alias in matched_rule.aliases if alias.strip()]
        allowed_names.extend(alias for alias in aliases if alias not in allowed_names)
        all_known_names = resolve_known_names(matched_rule.platform, matched_rule.user_id)
        # 其他群使用的称呼对当前群属于群混淆，进入硬性禁止集合
        forbidden_names = [name for name in all_known_names if name not in allowed_names]
        summary = (
            f"经UID比对，该条消息的发送者是目标用户（user_id={normalized_user_id}），"
            f"在当前聊天中的正确称呼是：{_format_known_names(allowed_names)}。"
        )
        lines.append(summary)
        lines.append("对该发送者使用以上正确称呼是正确的。")
        if forbidden_names:
            lines.append(
                f"注意：该用户在其他群可能显示为{_format_known_names(forbidden_names)}，"
                "在当前聊天中不要用这些称呼称呼ta。"
            )
        payload["allowed_names"] = allowed_names
        payload["forbidden_names"] = forbidden_names
        payload["aliases"] = aliases
        payload["known_names"] = allowed_names
        payload["summary"] = summary
    else:
        # 发送者未命中的规则：其专属称呼与已知名字禁止用于该发送者
        forbidden_aliases: List[str] = []
        forbidden_names: List[str] = []
        for rule in unmatched_rules:
            rule_user_id = rule.user_id.strip()
            known_names = resolve_known_names(rule.platform, rule.user_id)
            aliases = [alias.strip() for alias in rule.aliases if alias.strip()]
            forbidden_aliases.extend(alias for alias in aliases if alias not in forbidden_aliases)
            forbidden_terms = _format_aliases(aliases) + _format_aliases(known_names)
            if forbidden_terms:
                lines.append(
                    f"经UID比对，该条消息的发送者（user_id={normalized_user_id}）不是目标用户"
                    f"（user_id={rule_user_id}，在聊天中可能显示为：{_format_known_names(known_names)}）。"
                    f"注意，{stage_check_hint}，禁止对该用户使用以下称呼或名字：{forbidden_terms}。"
                )
            # 硬性禁止集合：仅收集参与当前会话的规则用户的常规名字，避免无关群里的同名误伤
            if participant_keys is None or (rule.platform.strip(), rule.user_id.strip()) in participant_keys:
                forbidden_names.extend(name for name in known_names if name not in forbidden_names)
        payload["aliases"] = forbidden_aliases
        payload["forbidden_names"] = forbidden_names
        payload["summary"] = (
            f"经UID比对，该条消息的发送者（user_id={normalized_user_id}）不是目标用户，"
            f"禁止对其使用专属称呼：{_format_aliases(forbidden_aliases)}。"
        )

    return IdentityCheckResult(block_text="\n".join(lines), payload=payload)


def scan_forbidden_names_in_text(
    text: str,
    *,
    allowed_names: Sequence[str],
    forbidden_names: Sequence[str],
) -> List[str]:
    """扫描输出文本中出现的硬性禁止称呼（确定性身份核对，不依赖 LLM）。

    命中禁止称呼且文本中没有出现任何正确称呼时视为身份混淆，返回命中的禁止称呼列表；
    未命中或文本中出现正确称呼（无法排除“提及”语义）时返回空列表。

    防误伤规则：
    - 正确称呼命中时，跳过以正确称呼为子串的禁止称呼（如「明」⊂「阿明」）；
    - 长度低于 MIN_HARD_CHECK_NAME_LENGTH 的禁止称呼不做子串匹配（如「明」命中「明天」）。
    """
    normalized_text = (text or "").strip()
    if not normalized_text:
        return []

    allowed = [str(name or "").strip() for name in allowed_names if str(name or "").strip()]
    forbidden = [str(name or "").strip() for name in forbidden_names if str(name or "").strip()]
    if not forbidden:
        return []

    # 文本中出现了目标用户的正确称呼时，跳过整体判定（可能是“提及”而非“称呼错误”）
    if any(name in normalized_text for name in allowed):
        return []

    effective_forbidden = [
        name for name in forbidden if not any(name in allowed_name for allowed_name in allowed)
    ]
    hits: List[str] = []
    for name in effective_forbidden:
        if len(name) < MIN_HARD_CHECK_NAME_LENGTH:
            continue
        if name in normalized_text:
            hits.append(name)
    return hits
