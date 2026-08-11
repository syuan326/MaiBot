"""鉴权决策数据模型。"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class IdentityIssue:
    """鉴权审核发现的一项身份问题。"""

    issue_type: str
    """问题类型，如 wrong_attribution / wrong_target / wrong_name / self_confusion。"""

    detail: str
    """问题的具体说明。"""


@dataclass(slots=True)
class AuthDecision:
    """鉴权审核结论。"""

    passed: bool
    """输出是否通过身份核对。"""

    reason: str = ""
    """驳回理由；会作为纠正提示反馈给 Planner / Replyer。"""

    issues: List[IdentityIssue] = field(default_factory=list)
    """驳回时审核发现的具体身份问题列表。"""

    audit_error: bool = False
    """审核模型调用异常后放行（fail-open）时为 True，用于与真正的审核通过区分。"""

    identity_check: Optional[Dict[str, Any]] = None
    """固定身份 UID 核查的结构化结果（发送者、是否目标用户、相关称呼等）；未配置规则时为 None。"""

    identity_check_text: str = ""
    """固定身份 UID 核查的完整文案；驳回反馈时作为规则提醒再次注入，未配置规则时为空。"""
