"""鉴权固定身份规则：系统级 UID 硬比对与规则文案生成。

规则来自配置（auth.identity_rules），将指定用户（platform + user_id）与专属称呼绑定。
鉴权审核前由 Python 代码直接做 UID 比对（不依赖 LLM），比对结论与规则文案
动态注入审核 Prompt；结构化结果同时提供给麦麦观察的鉴权事件卡片展示。

名字集合按 platform + user_id 只读查询 person_info 表实时并集
（主名称 + 平台昵称 + 各群群名片），随消息接收自动保持最新，无需配置维护。
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from sqlmodel import col, select

import json

from src.common.database.database import get_db_session
from src.common.database.database_model import PersonInfo
from src.common.logger import get_logger
from src.config.config import global_config
from src.config.official_configs import AuthIdentityRuleConfig

logger = get_logger("maisaka_auth_identity")

IdentityCheckStage = Literal["planner", "replyer"]


@dataclass(slots=True)
class IdentityCheckResult:
    """一次固定身份核查的结果：Prompt 文案 + 结构化载荷（供监控卡片展示）。"""

    block_text: str = ""
    """注入审核 Prompt 的核查结论文案；未配置规则时为空串。"""

    payload: Optional[Dict[str, Any]] = None
    """结构化的核查结果（发送者、是否目标用户、相关称呼等）；未配置规则时为 None。"""


def get_identity_rules() -> List[AuthIdentityRuleConfig]:
    """读取配置中的固定身份规则列表。"""
    return list(global_config.auth.identity_rules)


def resolve_known_names(platform: str, user_id: str) -> List[str]:
    """按 platform + user_id 查询 person_info，并集该用户所有已知名字。

    并集来源：麦麦记的主名称、平台昵称、各群群名片；按消息接收自动更新。
    该用户尚未被登记时返回空列表（正常的新用户情况）。
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
                if isinstance(item, dict):
                    add_name(item.get("group_cardname"))
        except (ValueError, TypeError) as exc:
            logger.debug(f"解析群名片 JSON 失败: {normalized_platform}:{normalized_user_id}, err={exc}")
    return names


def _format_aliases(aliases: List[str]) -> str:
    """把称呼列表格式化为「哥哥」「老大」样式。"""
    return "".join(f"「{alias.strip()}」" for alias in aliases if alias.strip())


def _format_known_names(names: List[str]) -> str:
    """把已知名字集合格式化为提示词中的展示文本。"""
    return "、".join(names) if names else "（暂无已知名字）"


def build_identity_check_block(
    platform: str,
    user_id: str,
    *,
    stage: IdentityCheckStage,
    sender_name: str = "",
) -> IdentityCheckResult:
    """对当前消息发送者做固定身份 UID 硬比对，生成审核 Prompt 文案与结构化结果。

    Args:
        platform: 当前消息发送者所在平台。
        user_id: 当前消息发送者的用户ID。
        stage: 审核阶段；planner 与 replyer 的提示措辞不同。
        sender_name: 当前消息发送者的显示名，仅用于监控卡片展示。

    Returns:
        IdentityCheckResult；未配置规则或发送者信息缺失时 block_text 为空、payload 为 None。
    """
    rules = get_identity_rules()
    normalized_user_id = str(user_id).strip()
    if not rules or not normalized_user_id:
        return IdentityCheckResult()

    normalized_platform = platform.strip()
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
        "known_names": [],
        "aliases": [],
        "summary": "",
    }

    if matched_rule is not None:
        known_names = resolve_known_names(matched_rule.platform, matched_rule.user_id)
        aliases = [alias.strip() for alias in matched_rule.aliases if alias.strip()]
        summary = (
            f"经UID比对，该条消息的发送者是目标用户（user_id={normalized_user_id}，"
            f"在聊天中可能显示为：{_format_known_names(known_names)}），"
            f"该用户的身份具有唯一性：{_format_aliases(aliases)}。"
        )
        lines.append(summary)
        lines.append("对该发送者使用以上专属称呼是正确的。")
        payload["known_names"] = known_names
        payload["aliases"] = aliases
        payload["summary"] = summary

    # 发送者未命中的规则：其专属称呼禁止用于该发送者
    forbidden_aliases: List[str] = []
    for rule in unmatched_rules:
        rule_user_id = rule.user_id.strip()
        known_names = resolve_known_names(rule.platform, rule.user_id)
        aliases = [alias.strip() for alias in rule.aliases if alias.strip()]
        forbidden_aliases.extend(alias for alias in aliases if alias not in forbidden_aliases)
        lines.append(
            f"经UID比对，该条消息的发送者（user_id={normalized_user_id}）不是目标用户"
            f"（user_id={rule_user_id}，在聊天中可能显示为：{_format_known_names(known_names)}）。"
            f"注意，{stage_check_hint}，禁止对该用户进行以下称呼：{_format_aliases(aliases)}。"
        )

    if matched_rule is None:
        payload["aliases"] = forbidden_aliases
        payload["summary"] = (
            f"经UID比对，该条消息的发送者（user_id={normalized_user_id}）不是目标用户，"
            f"禁止对其使用专属称呼：{_format_aliases(forbidden_aliases)}。"
        )

    return IdentityCheckResult(block_text="\n".join(lines), payload=payload)
