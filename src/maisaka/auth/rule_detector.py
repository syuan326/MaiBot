"""输入注入规则检测器（零成本规则通道）。

内置六类注入规则（关键词 + 正则），并支持叠加配置中的自定义关键词与正则。
纯同步实现，不产生模型调用；命中结果交由输入守卫（input_guard）调度
规则→LLM 双通道确认与动作执行。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from src.common.logger import get_logger

logger = get_logger("maisaka_auth_rule")

# ============================================================
# 内置关键词规则
# 只保留高信号规则，避免日常闲聊中的中性表达（如"扮演""从现在开始"）误报。
# ============================================================

INSTRUCTION_OVERRIDE_KEYWORDS: List[str] = [
    # 中文
    "忽略之前",
    "忽略上面",
    "忽略以上",
    "忽略前面",
    "无视之前",
    "无视上面",
    "无视以上",
    "不要理会之前",
    "不要管之前",
    "忘记之前",
    "忘记以上",
    "忘记上面",
    "忘记你的设定",
    "忘记你的指令",
    "忘记所有",
    "重置你的",
    "清除你的记忆",
    # 英文
    "ignore previous",
    "ignore above",
    "ignore all",
    "disregard previous",
    "disregard above",
    "disregard all",
    "forget previous",
    "forget your instructions",
    "forget everything",
    "forget all",
    "reset your",
    "clear your memory",
    "override your",
    "bypass your",
    "do not follow",
    "don't follow your",
]

ROLE_PLAY_KEYWORDS: List[str] = [
    # 中文
    "你现在是",
    "你不再是",
    "你已经不是",
    "假装你是",
    "假设你是",
    "想象你是",
    "角色扮演",
    "你的新身份",
    "你的新角色",
    "你的新人格",
    # 英文
    "you are now",
    "you're now",
    "pretend you are",
    "pretend to be",
    "act as",
    "roleplay as",
    "play the role of",
    "imagine you are",
    "suppose you are",
    "your new identity",
    "your new role",
]

SYSTEM_MARKER_KEYWORDS: List[str] = [
    # 常见 prompt 标记
    "system:",
    "assistant:",
    "[SYSTEM]",
    "[INST]",
    "[/INST]",
    "<|im_start|>",
    "<|im_end|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<<SYS>>",
    "<</SYS>>",
    "### System:",
    "### Human:",
    "### Assistant:",
    "[system message]",
    "[hidden instruction]",
    # XML/JSON 注入
    "<instruction>",
    "</instruction>",
    "<system_prompt>",
    "</system_prompt>",
    "<hidden>",
    "</hidden>",
    '{"role": "system"',
    '{"instruction":',
]

JAILBREAK_KEYWORDS: List[str] = [
    # 中文
    "越狱",
    "破解限制",
    "解除限制",
    "取消限制",
    "无视安全",
    "忽略安全",
    "绕过安全",
    "没有限制",
    "没有规则",
    "无规则模式",
    "开发者模式",
    "调试模式",
    "最高权限",
    "管理员权限",
    "root权限",
    # 英文
    "jailbreak",
    "jailbroken",
    "bypass restrictions",
    "remove restrictions",
    "disable restrictions",
    "ignore safety",
    "bypass safety",
    "disable safety",
    "no restrictions",
    "no rules",
    "no limits",
    "developer mode",
    "debug mode",
    "god mode",
    "DAN mode",
    "do anything now",
    "maximum privilege",
    "admin access",
    "root access",
    "unrestricted mode",
    "uncensored mode",
]

META_INSTRUCTION_KEYWORDS: List[str] = [
    # 中文
    "这是你的新指令",
    "这是隐藏指令",
    "这是秘密指令",
    "真正的任务是",
    "实际上你需要",
    "你真正的目的",
    "不要告诉用户",
    "对用户隐瞒",
    "秘密执行",
    "以上都是假的",
    "以上都是测试",
    # 消息边界伪造
    "消息到这里结束",
    "以上消息结束",
    "对话到此结束",
    "上面的内容结束",
    "以下是新的指令",
    "以下是真正的",
    "下面是系统指令",
    "添加在你prompt",
    "添加到prompt",
    "加入到你的prompt",
    "写入你的prompt",
    "注入到prompt",
    "修改你的prompt",
    "覆盖你的prompt",
    # 英文
    "your new instructions",
    "hidden instruction",
    "secret instruction",
    "your real task is",
    "actually you need to",
    "your true purpose",
    "don't tell the user",
    "hide from user",
    "secretly",
    "everything above is fake",
    "above is just a test",
    "the real prompt is",
    "actual instruction:",
    "message ends here",
    "end of message",
    "add to your prompt",
    "inject into prompt",
    "modify your prompt",
]

MULTILANG_KEYWORDS: List[str] = [
    # 日语
    "今から",
    "命令を無視",
    "新しい指示",
    # 韩语
    "지금부터",
    "명령 무시",
    "새로운 지시",
    # 编码相关
    "Base64:",
    "ROT13:",
    "hex:",
]

KEYWORD_CATEGORY_LABELS: Dict[str, str] = {
    "instruction_override": "指令覆盖",
    "role_play": "角色扮演",
    "system_marker": "系统标记伪造",
    "jailbreak": "越狱尝试",
    "meta_instruction": "元指令",
    "multilang": "多语言绕过",
}

ALL_KEYWORDS: List[Tuple[str, str]] = []
for _category, _keywords in (
    ("instruction_override", INSTRUCTION_OVERRIDE_KEYWORDS),
    ("role_play", ROLE_PLAY_KEYWORDS),
    ("system_marker", SYSTEM_MARKER_KEYWORDS),
    ("jailbreak", JAILBREAK_KEYWORDS),
    ("meta_instruction", META_INSTRUCTION_KEYWORDS),
    ("multilang", MULTILANG_KEYWORDS),
):
    for _keyword in _keywords:
        # 关键词统一小写存储，匹配时不区分大小写
        ALL_KEYWORDS.append((_keyword.lower(), _category))

# ============================================================
# 内置正则规则
# ============================================================

PRESET_PATTERNS: List[Tuple[str, str]] = [
    (r"(?i)ignore\s+(all|previous|above|prior)\s+(instructions?|prompts?|rules?)", "instruction_override"),
    (r"(?i)disregard\s+(everything|all|previous)", "instruction_override"),
    (r"(?i)forget\s+(everything|all|your)\s*(instructions?|rules?)?", "instruction_override"),
    (r"忽略.{0,5}(之前|上面|以上|前面).{0,10}(指令|规则|设定|要求)", "instruction_override"),
    (r"无视.{0,5}(之前|上面|以上).{0,10}(内容|对话|消息)", "instruction_override"),
    (r"(?i)(you\s+are|you're)\s+now\s+a?", "role_play"),
    (r"(?i)(pretend|imagine|suppose)\s+(you\s+are|to\s+be)", "role_play"),
    (r"(?i)(act|roleplay|play)\s+(as|the\s+role)", "role_play"),
    (r"(你现在是|从现在起你是|假装你是).{1,20}", "role_play"),
    (r"扮演.{1,15}(角色|身份|人格)", "role_play"),
    (r"<\|[a-z_]+\|>", "system_marker"),
    (r"\[/?(?:SYSTEM|INST|SYS)\]", "system_marker"),
    (r"(?i)###\s*(system|human|assistant|user)\s*:", "system_marker"),
    (r'"\s*role\s*"\s*:\s*"\s*system\s*"', "system_marker"),
    (r"(?i)(jailbreak|jailbroken|unjail)", "jailbreak"),
    (r"(?i)(bypass|disable|remove|ignore)\s+(safety|restrictions?|limits?|filters?)", "jailbreak"),
    (r"(?i)(developer|debug|test|god|admin|sudo)\s+mode", "jailbreak"),
    (r"(?i)DAN\s*(mode)?", "jailbreak"),
    (r"(?i)do\s+anything\s+now", "jailbreak"),
    (r"(越狱|破解|解除|取消).{0,5}(限制|规则|安全)", "jailbreak"),
    (r"(?i)(maximum|highest|root|admin)\s+(privilege|access|permission)", "jailbreak"),
    (r"(?i)(unrestricted|uncensored|unlimited)\s+mode", "jailbreak"),
    (r"(?i)(hidden|secret|real|actual)\s+(instruction|prompt|command)", "meta_instruction"),
    (r"(?i)don'?t\s+tell\s+(the\s+)?user", "meta_instruction"),
    (r"(隐藏|秘密|真正的?).{0,5}(指令|任务|目的)", "meta_instruction"),
    (r"(消息|对话|内容).{0,5}(到这里|到此).{0,5}结束", "meta_instruction"),
    (r"以(下|后).{0,5}(是|为).{0,10}(指令|内容|prompt)", "meta_instruction"),
    (r"(添加|加入|写入|注入|修改).{0,5}(到|在).{0,5}prompt", "meta_instruction"),
    (r"(重写|改写|重新设定).{0,10}(设定|指令|人格|身份|prompt)", "meta_instruction"),
]

CUSTOM_PATTERN_PLACEHOLDER_CATEGORY = "custom"
"""自定义正则规则的默认类别。"""


@dataclass(slots=True)
class RuleDetectionResult:
    """一次规则检测的命中结果。"""

    hit_count: int = 0
    """命中规则总数（关键词与正则合计）。"""

    categories: List[str] = field(default_factory=list)
    """命中的规则类别（已去重，按首次命中顺序）。"""

    matched_rules: List[str] = field(default_factory=list)
    """命中的具体规则文本（用于日志与监控展示）。"""


class InjectionRuleDetector:
    """注入规则检测器：内置规则 + 配置自定义规则。"""

    def __init__(
        self,
        *,
        custom_keywords: Sequence[str] = (),
        custom_patterns: Sequence[str] = (),
    ) -> None:
        self._custom_keywords: List[Tuple[str, str]] = [
            (str(keyword).strip().lower(), CUSTOM_PATTERN_PLACEHOLDER_CATEGORY)
            for keyword in custom_keywords
            if str(keyword or "").strip()
        ]
        self._custom_patterns: List[Tuple[str, str]] = []
        for pattern in custom_patterns:
            normalized_pattern = str(pattern or "").strip()
            if not normalized_pattern:
                continue
            try:
                import re

                re.compile(normalized_pattern)
            except re.error as exc:
                logger.warning(f"自定义注入检测正则非法，已跳过: {normalized_pattern!r}, err={exc}")
                continue
            self._custom_patterns.append((normalized_pattern, CUSTOM_PATTERN_PLACEHOLDER_CATEGORY))

    def detect(self, text: str) -> Optional[RuleDetectionResult]:
        """对文本执行规则检测；无命中返回 None。"""

        normalized_text = str(text or "")
        if not normalized_text:
            return None

        hit_count = 0
        categories: List[str] = []
        matched_rules: List[str] = []

        def record_match(category: str, rule: str) -> None:
            nonlocal hit_count
            hit_count += 1
            if category not in categories:
                categories.append(category)
            matched_rules.append(rule)

        lower_text = normalized_text.lower()
        for keyword, category in ALL_KEYWORDS:
            if keyword in lower_text:
                record_match(category, keyword)

        for keyword, category in self._custom_keywords:
            if keyword in lower_text:
                record_match(category, keyword)

        import re

        for pattern, category in (*PRESET_PATTERNS, *self._custom_patterns):
            try:
                if re.search(pattern, normalized_text, flags=re.IGNORECASE) is not None:
                    record_match(category, pattern)
            except re.error as exc:
                logger.debug(f"注入检测正则执行失败，已跳过: {pattern!r}, err={exc}")

        if hit_count <= 0:
            return None
        return RuleDetectionResult(
            hit_count=hit_count,
            categories=categories,
            matched_rules=matched_rules,
        )
