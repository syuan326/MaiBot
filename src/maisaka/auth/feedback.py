"""鉴权驳回后的反馈构造。

把被驳回的 Planner 输出与驳回理由注入聊天历史（视同一次"虚拟工具结果"），
以及把驳回理由转成 Replyer 重生成约束使用的文本。
"""

from datetime import datetime
from typing import Sequence

import json

from src.llm_models.payload_content.context_item import ContextItem, FunctionCallItem
from src.llm_models.payload_content.tool_option import ToolCall
from src.maisaka.context.messages import (
    ModelOutputContextMessage,
    ReferenceMessage,
    ReferenceMessageType,
    build_model_output_context_messages,
)

from .decision import AuthDecision

MAX_REJECTED_THOUGHT_PREVIEW_CHARS = 600
MAX_REJECTED_TOOL_ARGS_PREVIEW_CHARS = 200


def _truncate_preview(text: str, limit: int) -> str:
    """截断过长的预览文本。"""

    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit]}..."


def build_rejected_output_messages(
    output_items: Sequence[ContextItem],
    *,
    source_kind: str = "assistant",
) -> list[ModelOutputContextMessage]:
    """构造工具调用被清空的被拒模型输出历史条目。

    被拒绝的 Planner 输出不会执行任何工具调用；写入历史前剔除 FunctionCallItem，
    避免历史规范化把这条"没有工具结果配对的调用"整条删除。
    """

    filtered_items = [item for item in output_items if not isinstance(item, FunctionCallItem)]
    return build_model_output_context_messages(filtered_items, source_kind=source_kind)


def build_planner_auth_feedback_message(
    *,
    thought_text: str,
    tool_calls: Sequence[ToolCall],
    decision: AuthDecision,
) -> ReferenceMessage:
    """把 Planner 输出被鉴权驳回的信息构造成参考消息，供重新规划时参考。

    remaining_uses_value=2 表示该反馈可以挺过当前轮次的历史后处理，
    确保下一轮重新规划能看到，随后在再下一轮后处理中被自动清理。
    """

    lines = [
        "【鉴权驳回-必须修正】你的上一轮规划未通过身份核对，已被驳回，其中的工具调用均未执行。"
        "请认真阅读驳回原因并修正身份错误后重新规划，不要重复同样的错误。",
        f"驳回原因：{decision.reason or '输出中存在用户身份混淆'}",
    ]
    if decision.issues:
        lines.append("发现的问题：")
        lines.extend(f"- {issue.issue_type}: {issue.detail}" for issue in decision.issues)

    thought_preview = _truncate_preview(thought_text, MAX_REJECTED_THOUGHT_PREVIEW_CHARS)
    if thought_preview:
        lines.append(f"你上一轮的想法（含身份错误，仅供对照，不要重复其中的错误）：\n{thought_preview}")

    if tool_calls:
        tool_lines = []
        for tool_call in tool_calls:
            args_text = json.dumps(tool_call.args or {}, ensure_ascii=False)
            tool_lines.append(f"- {tool_call.func_name}({_truncate_preview(args_text, MAX_REJECTED_TOOL_ARGS_PREVIEW_CHARS)})")
        lines.append("你上一轮尝试调用的工具（均未执行）：")
        lines.extend(tool_lines)

    # 固定身份规则在驳回反馈中再次提醒，确保重新规划时规则就在眼前
    if decision.identity_check_text.strip():
        lines.append("【固定身份规则-再次提醒，必须遵守】")
        lines.append(decision.identity_check_text.strip())

    return ReferenceMessage(
        content="\n".join(lines),
        timestamp=datetime.now(),
        reference_type=ReferenceMessageType.AUTH_FEEDBACK,
        remaining_uses_value=2,
        display_prefix="[鉴权驳回]",
    )


def build_replyer_auth_reject_reason(decision: AuthDecision) -> str:
    """把鉴权驳回结论转成 Replyer 重生成约束使用的理由文本。"""

    reason = " ".join(decision.reason.split()).strip()
    if decision.issues:
        issue_text = "；".join(
            issue.detail for issue in decision.issues if issue.detail.strip()
        ).strip()
        if issue_text:
            reason = f"{reason}（{issue_text}）" if reason else issue_text
    reason = f"【鉴权驳回-必须修正】身份核对未通过：{reason}" if reason else "【鉴权驳回-必须修正】身份核对未通过，回复中存在用户身份混淆"
    # 固定身份规则在驳回反馈中再次提醒，确保重新生成时规则就在眼前
    if decision.identity_check_text.strip():
        reason = f"{reason}\n【固定身份规则-再次提醒，必须遵守】\n{decision.identity_check_text.strip()}"
    return reason
