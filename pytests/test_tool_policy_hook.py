"""预调用策略 Hook 适配器与审计链测试。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import asyncio
import json
import threading

import pytest

from src.common.tool_audit import ToolAuditWriter
from src.core.tooling import (
    ToolExecutionContext,
    ToolExecutionResult,
    ToolInvocation,
    ToolRegistry,
    ToolSpec,
)
from src.plugin_runtime.tool_policy import HOOK_NAME, HookToolPolicy, decide, decide_failure


@dataclass(slots=True)
class _DispatchResult:
    """与 HookDispatchResult 字段一致的最小替身，避免测试拉起完整配置与插件运行时。"""

    hook_name: str = HOOK_NAME
    kwargs: dict[str, Any] = field(default_factory=dict)
    aborted: bool = False
    stopped_by: Optional[str] = None
    errors: list[str] = field(default_factory=list)


class _FakeProvider:
    """最小工具 Provider，只记录是否被调用。"""

    provider_name = "fake"
    provider_type = "builtin"

    def __init__(self) -> None:
        self.invocations: list[ToolInvocation] = []

    async def list_tools(self, context: Any = None) -> list[ToolSpec]:
        return [_make_spec()]

    async def invoke(
        self,
        invocation: ToolInvocation,
        context: Optional[ToolExecutionContext] = None,
    ) -> ToolExecutionResult:
        self.invocations.append(invocation)
        return ToolExecutionResult(tool_name=invocation.tool_name, success=True, content="ok")

    async def close(self) -> None:
        return None


class _FakeRuntimeManager:
    """假插件运行时，按调用顺序返回预设的 Hook 结果。"""

    def __init__(
        self,
        result: Optional[_DispatchResult] = None,
        *,
        delay_seconds: float = 0.0,
        results: Optional[list[_DispatchResult]] = None,
    ) -> None:
        self._result = result
        self._delay_seconds = delay_seconds
        self._results = list(results or [])
        self.calls: list[dict[str, Any]] = []

    async def invoke_hook(self, hook_name: str, **kwargs: Any) -> _DispatchResult:
        self.calls.append({"hook_name": hook_name, **kwargs})
        if self._delay_seconds > 0:
            await asyncio.sleep(self._delay_seconds)
        if self._results:
            return self._results.pop(0)
        if self._result is None:
            return _DispatchResult(hook_name=hook_name)
        return self._result


def _install_manager(monkeypatch: pytest.MonkeyPatch, manager: _FakeRuntimeManager) -> None:
    """把适配器的运行时查找替换为假 manager。"""

    monkeypatch.setattr(HookToolPolicy, "_get_runtime_manager", staticmethod(lambda: manager))


def _make_spec() -> ToolSpec:
    return ToolSpec(
        name="echo",
        description="测试工具",
        provider_name="fake",
        provider_type="builtin",
        metadata={"registry_original_name": "echo"},
    )


def _make_context() -> ToolExecutionContext:
    return ToolExecutionContext(session_id="s1", platform="qq", user_id="u1", group_id="g1")


def _make_invocation(call_id: str = "call-1") -> ToolInvocation:
    return ToolInvocation(
        tool_name="echo",
        arguments={"path": "secret.txt"},
        call_id=call_id,
        session_id="s1",
    )


def _read_entries(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@pytest.mark.asyncio
async def test_hook_adapter_maps_abort_to_deny(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _FakeRuntimeManager(
        _DispatchResult(aborted=True, stopped_by="plugin.handler", kwargs={"reason": "blocked"})
    )
    _install_manager(monkeypatch, manager)
    policy = HookToolPolicy(timeout_ms=200, on_error="deny", audit=None)

    decision = await policy.check(_make_invocation(), _make_context(), _make_spec())

    assert decision.allow is False
    assert decision.reason == "blocked"
    assert decision.metadata["decided_by"] == "plugin.handler"
    assert manager.calls[0]["hook_name"] == HOOK_NAME
    assert manager.calls[0]["tool_name"] == "echo"
    assert manager.calls[0]["original_tool_name"] == "echo"
    assert manager.calls[0]["call_id"] == "call-1"


@pytest.mark.asyncio
async def test_hook_adapter_errors_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _FakeRuntimeManager(_DispatchResult(errors=["handler crashed"]))
    _install_manager(monkeypatch, manager)
    policy = HookToolPolicy(timeout_ms=200, on_error="deny", audit=None)
    provider = _FakeProvider()
    registry = ToolRegistry()
    registry.register_provider(provider)
    registry.set_policy(policy)

    result = await registry.invoke(_make_invocation(), _make_context())

    assert result.success is False
    assert result.metadata["policy_denied"] is True
    assert result.metadata["decided_by"] == "error"
    assert "handler crashed" in result.error_message
    assert provider.invocations == []


@pytest.mark.asyncio
async def test_hook_adapter_timeout_denies(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _FakeRuntimeManager(_DispatchResult(), delay_seconds=0.2)
    _install_manager(monkeypatch, manager)
    policy = HookToolPolicy(timeout_ms=10, on_error="deny", audit=None)

    decision = await policy.check(_make_invocation(), _make_context(), _make_spec())

    assert decision.allow is False
    assert "超时" in decision.reason
    assert decision.metadata["decided_by"] == "timeout"


@pytest.mark.asyncio
async def test_hook_adapter_on_error_allow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    error_manager = _FakeRuntimeManager(_DispatchResult(errors=["handler crashed"]))
    _install_manager(monkeypatch, error_manager)
    error_path = tmp_path / "error-audit.jsonl"
    error_policy = HookToolPolicy(timeout_ms=200, on_error="allow", audit=ToolAuditWriter(error_path))

    error_decision = await error_policy.check(_make_invocation(), _make_context(), _make_spec())

    assert error_decision.allow is True
    assert "on_error=allow" in error_decision.reason
    error_entries = _read_entries(error_path)
    assert error_entries[0]["action"] == "allow"
    assert error_entries[0]["decided_by"] == "on_error"
    assert "on_error=allow" in error_entries[0]["reason"]

    timeout_manager = _FakeRuntimeManager(_DispatchResult(), delay_seconds=0.2)
    _install_manager(monkeypatch, timeout_manager)
    timeout_path = tmp_path / "timeout-audit.jsonl"
    timeout_policy = HookToolPolicy(timeout_ms=10, on_error="allow", audit=ToolAuditWriter(timeout_path))

    timeout_decision = await timeout_policy.check(_make_invocation(), _make_context(), _make_spec())

    assert timeout_decision.allow is True
    assert "on_error=allow" in timeout_decision.reason
    timeout_entries = _read_entries(timeout_path)
    assert timeout_entries[0]["action"] == "allow"
    assert timeout_entries[0]["decided_by"] == "on_error"
    assert "on_error=allow" in timeout_entries[0]["reason"]


@pytest.mark.asyncio
async def test_hook_adapter_on_error_allow_reaches_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _FakeRuntimeManager(_DispatchResult(errors=["handler crashed"]))
    _install_manager(monkeypatch, manager)
    policy = HookToolPolicy(timeout_ms=200, on_error="allow", audit=None)
    provider = _FakeProvider()
    registry = ToolRegistry()
    registry.register_provider(provider)
    registry.set_policy(policy)

    result = await registry.invoke(_make_invocation(), _make_context())

    assert result.success is True
    assert [invocation.tool_name for invocation in provider.invocations] == ["echo"]
    assert "policy_denied" not in result.metadata


def test_audit_hash_chain(tmp_path: Path) -> None:
    path = tmp_path / "tool-audit.jsonl"
    writer = ToolAuditWriter(path)
    writer.record(
        action="allow",
        tool="read_file",
        provider_type="plugin",
        platform="qq",
        session_id="s1",
        user_id="u1",
        call_id="c1",
        args={"path": "secret.txt"},
    )
    writer.record(
        action="deny",
        tool="read_file",
        provider_type="plugin",
        platform="qq",
        session_id="s1",
        user_id="u1",
        args={"path": "other.txt"},
    )

    raw = path.read_text(encoding="utf-8")
    entries = _read_entries(path)
    assert len(entries) == 2
    assert [entry["seq"] for entry in entries] == [1, 2]
    assert entries[1]["prev_hash"] == entries[0]["entry_hash"]
    assert entries[0]["call_id"] == "c1"
    assert str(entries[1]["call_id"]).startswith("gen-")
    assert "secret.txt" not in raw


def test_decide_is_pure_sync() -> None:
    aborted = decide(
        _DispatchResult(aborted=True, stopped_by="p:h", kwargs={"reason": "blocked"}),
        on_error="deny",
    )
    assert aborted.allow is False
    assert aborted.reason == "blocked"
    assert aborted.metadata["decided_by"] == "p:h"

    errored = decide(_DispatchResult(errors=["handler crashed"]), on_error="deny")
    assert errored.allow is False
    assert "handler crashed" in errored.reason
    assert errored.metadata["decided_by"] == "error"

    allowed = decide(_DispatchResult(errors=["handler crashed"]), on_error="allow")
    assert allowed.allow is True
    assert "on_error=allow" in allowed.reason
    assert allowed.metadata["decided_by"] == "on_error"

    denied = decide(_DispatchResult(kwargs={"decision": "deny", "reason": "nope"}), on_error="deny")
    assert denied.allow is False
    assert denied.reason == "nope"
    assert denied.metadata["decided_by"] == "hook"

    passed = decide(_DispatchResult(), on_error="deny")
    assert passed.allow is True
    assert passed.metadata["decided_by"] == "hook"

    failure = decide_failure("策略 Hook 超时", on_error="deny", decided_by="timeout")
    assert failure.allow is False
    assert failure.metadata["decided_by"] == "timeout"


@pytest.mark.asyncio
async def test_audit_write_runs_off_event_loop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    writer = ToolAuditWriter(tmp_path / "tool-audit.jsonl")
    thread_ids: list[int] = []
    original_record = writer.record

    def recording_record(**kwargs: Any) -> None:
        thread_ids.append(threading.get_ident())
        original_record(**kwargs)

    monkeypatch.setattr(writer, "record", recording_record)
    manager = _FakeRuntimeManager(_DispatchResult())
    _install_manager(monkeypatch, manager)
    policy = HookToolPolicy(timeout_ms=200, on_error="deny", audit=writer)

    loop_thread = threading.get_ident()
    decision = await policy.check(_make_invocation(), _make_context(), _make_spec())

    assert decision.allow is True
    assert thread_ids
    assert thread_ids[0] != loop_thread


@pytest.mark.asyncio
async def test_audit_seq_and_chain_via_hook(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "tool-audit.jsonl"
    writer = ToolAuditWriter(path)
    manager = _FakeRuntimeManager(
        results=[
            _DispatchResult(aborted=True, stopped_by="plugin.handler", kwargs={"reason": "blocked"}),
            _DispatchResult(),
        ]
    )
    _install_manager(monkeypatch, manager)
    policy = HookToolPolicy(timeout_ms=200, on_error="deny", audit=writer)

    first = await policy.check(_make_invocation("c1"), _make_context(), _make_spec())
    second = await policy.check(_make_invocation("c2"), _make_context(), _make_spec())

    assert first.allow is False
    assert second.allow is True
    raw = path.read_text(encoding="utf-8")
    entries = _read_entries(path)
    assert len(entries) == 2
    assert [entry["action"] for entry in entries] == ["deny", "allow"]
    assert [entry["seq"] for entry in entries] == [1, 2]
    assert entries[1]["prev_hash"] == entries[0]["entry_hash"]
    assert [entry["call_id"] for entry in entries] == ["c1", "c2"]
    assert all("args_hash" in entry for entry in entries)
    assert "secret.txt" not in raw


@pytest.mark.asyncio
async def test_audit_failure_writes_registry_error(tmp_path: Path) -> None:
    path = tmp_path / "audit-failure.jsonl"
    policy = HookToolPolicy(timeout_ms=200, on_error="deny", audit=ToolAuditWriter(path))

    await policy.audit_failure(_make_invocation(), _make_context(), _make_spec(), RuntimeError("boom"))

    entries = _read_entries(path)
    assert len(entries) == 1
    assert entries[0]["action"] == "deny"
    assert entries[0]["decided_by"] == "registry_error"
    assert "RuntimeError" in entries[0]["reason"]


def test_tool_policy_hook_spec_reads_config_timeout_at_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.maisaka import chat_loop_service
    from src.plugin_runtime.host.hook_spec_registry import HookSpecRegistry

    configured_timeout_ms = 4321
    fake_global_config = SimpleNamespace(
        tool_guard=SimpleNamespace(policy_hook_timeout_ms=configured_timeout_ms)
    )
    monkeypatch.setattr(chat_loop_service, "global_config", fake_global_config)
    registry = HookSpecRegistry()

    registered_specs = chat_loop_service.register_maisaka_hook_specs(registry)

    tool_spec = registry.get_hook_spec("maisaka.tool.before_execute")
    assert tool_spec is not None
    assert tool_spec.default_timeout_ms == configured_timeout_ms
    assert tool_spec.allow_abort is True
    assert tool_spec.allow_kwargs_mutation is True
    assert any(spec.name == "maisaka.tool.before_execute" for spec in registered_specs)


def test_tool_guard_timeout_default_is_3000() -> None:
    from src.config.official_configs import ToolGuardConfig

    assert ToolGuardConfig().policy_hook_timeout_ms == 3000
