"""用户权限分级解析器。

基于配置（auth.permissions）按平台+用户ID为用户绑定角色，支持五个等级：
owner（拥有者）/ admin（管理员）/ trusted（信任）/ normal（普通）/ blacklist（黑名单）。

角色等级用于：
1. 输入侧拦截：blacklist 用户的消息直接丢弃，不进入对话与记忆。
2. 鉴权审核上下文：把参与者权限注入审核 Prompt，供审核模型判断
   Planner 决策与 Replyer 回复是否存在越权倾向（例如信任普通用户执行管理指令）。
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.common.logger import get_logger
from src.config.config import global_config
from src.config.official_configs import AuthPermissionConfig

logger = get_logger("maisaka_auth_permission")

ROLE_LEVELS: Dict[str, int] = {
    "owner": 100,
    "admin": 80,
    "trusted": 60,
    "normal": 20,
    "blacklist": -100,
}
"""角色等级映射；等级数值仅用于相对比较。"""

ROLE_ORDER: Tuple[str, ...] = ("owner", "admin", "trusted", "normal", "blacklist")
"""配置 UI 中展示的角色顺序。"""

ROLE_LABELS: Dict[str, str] = {
    "owner": "拥有者",
    "admin": "管理员",
    "trusted": "信任用户",
    "normal": "普通用户",
    "blacklist": "黑名单",
}
"""角色中文标签，用于日志与审核上下文。"""

VALID_ROLES = set(ROLE_LEVELS.keys())


@dataclass(slots=True)
class UserPermission:
    """一位用户解析后的权限信息。"""

    platform: str = ""
    user_id: str = ""
    role: str = "normal"
    """解析后的角色；未配置时默认为 normal。"""

    level: int = ROLE_LEVELS["normal"]
    """角色等级数值。"""

    def is_blacklisted(self) -> bool:
        """是否命中黑名单。"""

        return self.role == "blacklist"

    def can(self, required_role: str) -> bool:
        """当前权限是否满足指定角色门槛（等级 >= 该角色等级）。"""

        return self.level >= ROLE_LEVELS.get(required_role, 0)

    def role_label(self) -> str:
        """返回角色的中文展示标签。"""

        return ROLE_LABELS.get(self.role, self.role)


class UserPermissionResolver:
    """用户权限解析器：从配置中按 platform+user_id 解析角色。"""

    def resolve(
        self,
        platform: str,
        user_id: str,
        *,
        refresh: bool = False,
    ) -> UserPermission:
        """解析用户在指定平台上的角色权限。

        每次解析直接读取全局配置（规则量小，开销可忽略），配置热更新后天然生效。

        Args:
            platform: 用户所在平台。
            user_id: 用户ID。
            refresh: 兼容参数，保留用于显式刷新语义。
        """

        del refresh
        normalized_platform = str(platform or "").strip()
        normalized_user_id = str(user_id or "").strip()
        rules = list(getattr(global_config.auth, "permissions", None) or [])
        matched_role = ""
        for rule in rules:
            if not isinstance(rule, AuthPermissionConfig):
                continue
            if rule.platform.strip() == normalized_platform and rule.user_id.strip() == normalized_user_id:
                matched_role = rule.role
                break
        role = matched_role or "normal"
        if role not in VALID_ROLES:
            logger.warning(f"权限角色非法，已按 normal 处理: {role!r}")
            role = "normal"
        return UserPermission(
            platform=normalized_platform,
            user_id=normalized_user_id,
            role=role,
            level=ROLE_LEVELS[role],
        )

    def build_permission_context(
        self,
        participants: List[Tuple[str, str]],
    ) -> str:
        """为鉴权审核构建参与者权限上下文文本。

        Args:
            participants: (platform, user_id) 列表。

        Returns:
            权限上下文文本；所有参与者都是普通用户时返回空串。
        """

        lines: List[str] = []
        for platform, user_id in participants:
            permission = self.resolve(platform, user_id)
            if permission.role == "normal":
                continue
            label = permission.role_label()
            if permission.role == "blacklist":
                lines.append(f"- {user_id}（黑名单用户，不应被正常对待或配合）")
            else:
                lines.append(f"- {user_id}（{label}，可适当响应其管理/权限类请求）")
        if not lines:
            return ""
        return "\n".join(lines)


user_permission_resolver = UserPermissionResolver()
"""用户权限解析器全局单例。"""
