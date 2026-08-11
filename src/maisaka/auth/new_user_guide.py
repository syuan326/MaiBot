"""新用户识别注册引导。

在消息接收链检测陌生新用户（person_info 中尚未被标记为 known 的用户），
将其记录到聊天流会话上；Planner 构建请求时注入引导文本，提醒麦麦
这是陌生新成员、先认识再深聊，避免把陌生人当成老朋友。

同一用户短时间内只提示一次，防止连续消息刷屏重复引导。
"""

from typing import Dict

import time

from src.chat.message_receive.chat_manager import chat_manager
from src.chat.utils.utils import is_bot_self
from src.common.logger import get_logger
from src.person_info.person_info import is_person_known

logger = get_logger("maisaka_new_user_guide")

NEW_USER_GUIDE_COOLDOWN_SECONDS = 3600
"""同一用户两次引导的最小间隔。"""

NEW_USER_GUIDE_MAX_PENDING = 5
"""每个聊天流会话上最多保留的未消费新用户引导数。"""


class NewUserGuide:
    """新用户识别与引导记录。"""

    def __init__(self) -> None:
        self._recently_guided: Dict[str, float] = {}
        """user_id -> 最近一次引导时间戳，用于冷却去重。"""

    async def check_and_record(
        self,
        *,
        platform: str,
        user_id: str,
        user_name: str,
    ) -> bool:
        """检测陌生新用户并记录引导；确认新用户并记录返回 True。

        Args:
            platform: 消息平台。
            user_id: 发送者用户ID。
            user_name: 发送者显示名。

        Returns:
            是否为新用户且已记录引导。
        """

        normalized_user_id = str(user_id or "").strip()
        normalized_platform = str(platform or "").strip()
        if not normalized_user_id or not normalized_platform:
            return False
        if is_bot_self(normalized_platform, normalized_user_id):
            return False

        # 冷却去重：同一用户短时间只引导一次
        now = time.time()
        last_guided_at = self._recently_guided.get(f"{normalized_platform}:{normalized_user_id}", 0.0)
        if now - last_guided_at < NEW_USER_GUIDE_COOLDOWN_SECONDS:
            return False

        try:
            if is_person_known(platform=normalized_platform, user_id=normalized_user_id):
                return False
        except Exception as exc:
            logger.debug(f"新用户引导查询已知状态失败，已跳过: {exc}")
            return False

        self._recently_guided[f"{normalized_platform}:{normalized_user_id}"] = now
        try:
            session = await chat_manager.get_or_create_session(normalized_platform, normalized_user_id)
        except Exception as exc:
            logger.debug(f"新用户引导解析聊天流失败，已跳过: {exc}")
            return True

        pending = getattr(session, "pending_new_user_guides", None)
        if pending is None:
            pending = []
            session.pending_new_user_guides = pending
        pending.append(
            {
                "user_id": normalized_user_id,
                "user_name": str(user_name or "").strip() or normalized_user_id,
                "guided_at": now,
            }
        )
        if len(pending) > NEW_USER_GUIDE_MAX_PENDING:
            del pending[: len(pending) - NEW_USER_GUIDE_MAX_PENDING]
        logger.info(f"[新用户识别] {normalized_platform}:{normalized_user_id} 为陌生新成员，已记录引导")
        return True

    def build_guide_text(self, session_id: str) -> str:
        """从会话取出未消费的新用户引导并生成提示文本；消费后清空。

        供 Planner 构建请求时调用，一次性注入后即消费。
        """

        if not session_id:
            return ""
        try:
            session = chat_manager.get_session_by_session_id(session_id)
        except Exception as exc:
            logger.debug(f"读取新用户引导失败，已跳过: {exc}")
            return ""
        if session is None:
            return ""
        pending = getattr(session, "pending_new_user_guides", None)
        if not pending:
            return ""
        session.pending_new_user_guides = []
        lines = [
            "【新成员提醒】以下用户是聊天流里的陌生新成员，麦麦还不了解 TA，不要当作老朋友熟络："
        ]
        for item in pending[:5]:
            user_name = str(item.get("user_name") or item.get("user_id") or "未知用户")
            lines.append(f"- {user_name}")
        lines.append("可以保持自然礼貌，适度好奇；如果 TA 愿意自我介绍，可以顺势认识并记住。")
        return "\n".join(lines)


new_user_guide = NewUserGuide()
"""新用户引导全局单例。"""
