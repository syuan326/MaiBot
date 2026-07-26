"""鉴权决策数据模型。"""

from dataclasses import dataclass, field
from typing import List


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
