"""Maisaka 鉴权器模块。

在 Planner 决策之后、Replyer 发送之前对输出做身份核对，
防止 bot 认错用户身份（归属错误、对象错误、称呼混淆、自我混淆）。
"""

from .authenticator import Authenticator, authenticator
from .decision import AuthDecision, IdentityIssue

__all__ = [
    "AuthDecision",
    "Authenticator",
    "IdentityIssue",
    "authenticator",
]
