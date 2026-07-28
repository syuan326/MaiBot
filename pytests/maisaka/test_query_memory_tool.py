"""query_memory 内置工具检索模式参数处理测试。"""

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from src.core.tooling import ToolInvocation
from src.maisaka.builtin_tool import query_memory as query_memory_module
from src.maisaka.builtin_tool.context import BuiltinToolRuntimeContext
from src.services.memory_service import MemoryHit, MemorySearchResult


def _build_tool_ctx() -> SimpleNamespace:
    """构造最小可用的内置工具上下文；群聊上下文可跳过人物解析。"""

    chat_stream = SimpleNamespace(platform="qq", user_id="u1", group_id="g1")
    runtime = SimpleNamespace(chat_stream=chat_stream, session_id="session-1", log_prefix="[test]")
    return SimpleNamespace(
        runtime=runtime,
        build_success_result=BuiltinToolRuntimeContext.build_success_result,
        build_failure_result=BuiltinToolRuntimeContext.build_failure_result,
    )


class _RecordingMemorySearch:
    """记录调用参数并返回固定结果的 memory_service.search 替身。"""

    def __init__(self, result: MemorySearchResult) -> None:
        self.result = result
        self.calls: List[Dict[str, Any]] = []

    async def __call__(self, query: str, **kwargs: Any) -> MemorySearchResult:
        self.calls.append({"query": query, **kwargs})
        return self.result


@pytest.mark.asyncio
async def test_hybrid_without_time_downgrades_to_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """hybrid 模式缺少时间范围时应降级为 search，而不是触发底层校验失败。"""

    recorder = _RecordingMemorySearch(MemorySearchResult(hits=[MemoryHit(content="主人安排了干活任务")]))
    monkeypatch.setattr(query_memory_module.memory_service, "search", recorder)

    invocation = ToolInvocation(
        tool_name="query_memory",
        arguments={"query": "干活 任务 工作 主人", "mode": "hybrid"},
    )
    result = await query_memory_module.handle_tool(_build_tool_ctx(), invocation)

    assert result.success
    assert len(recorder.calls) == 1
    assert recorder.calls[0]["mode"] == "search"
    assert result.structured_content["mode"] == "hybrid"
    assert result.structured_content["effective_mode"] == "search"
    assert result.structured_content["mode_time_downgraded"] is True
    assert "已自动降级为关键词检索" in result.content


@pytest.mark.asyncio
async def test_time_mode_without_time_fails_before_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """time 模式缺少时间范围时应在调用检索前明确报错。"""

    recorder = _RecordingMemorySearch(MemorySearchResult())
    monkeypatch.setattr(query_memory_module.memory_service, "search", recorder)

    invocation = ToolInvocation(
        tool_name="query_memory",
        arguments={"query": "任务", "mode": "time"},
    )
    result = await query_memory_module.handle_tool(_build_tool_ctx(), invocation)

    assert not result.success
    assert recorder.calls == []
    assert "time_start" in result.error_message
    assert "search 模式" in result.error_message


@pytest.mark.asyncio
async def test_hybrid_with_time_range_keeps_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """hybrid 模式提供了时间范围时不应发生降级。"""

    recorder = _RecordingMemorySearch(MemorySearchResult(hits=[MemoryHit(content="上周的任务安排")]))
    monkeypatch.setattr(query_memory_module.memory_service, "search", recorder)

    invocation = ToolInvocation(
        tool_name="query_memory",
        arguments={"query": "任务", "mode": "hybrid", "time_start": "2026-07-01"},
    )
    result = await query_memory_module.handle_tool(_build_tool_ctx(), invocation)

    assert result.success
    assert len(recorder.calls) == 1
    assert recorder.calls[0]["mode"] == "hybrid"
    assert result.structured_content["mode_time_downgraded"] is False
    assert "已自动降级" not in result.content
