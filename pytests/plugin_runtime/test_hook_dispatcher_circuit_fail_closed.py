"""真实 HookDispatcher 的熔断 fail-open 回归测试。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import asyncio
import time

import pytest

from src.plugin_runtime.host.circuit_breaker import get_plugin_circuit_breaker
from src.plugin_runtime.host.component_registry import HookHandlerEntry
from src.plugin_runtime.host.hook_dispatcher import HookDispatcher
from src.plugin_runtime.host.hook_spec_registry import HookSpec
from src.plugin_runtime.tool_policy import decide

HOOK_NAME = "test.circuit.before_execute"
_CIRCUIT_OPEN_FAILURES = 3


class _FakeSupervisor:
    """只实现 HookDispatcher 依赖的最小 Supervisor。"""

    group_name = "builtin"

    def __init__(self, entry: HookHandlerEntry, handler: Any) -> None:
        self.component_registry = SimpleNamespace(get_hook_handlers=lambda _hook_name: [entry])
        self._handler = handler
        self.invoke_count = 0

    async def invoke_plugin(
        self,
        action: str,
        plugin_id: str,
        component_name: str,
        args: dict[str, Any],
        *,
        timeout_ms: int = 0,
    ) -> Any:
        del action, plugin_id, component_name, args, timeout_ms
        self.invoke_count += 1
        return await self._handler()


def _make_entry(plugin_id: str) -> HookHandlerEntry:
    return HookHandlerEntry(
        name="policy",
        component_type="HOOK_HANDLER",
        plugin_id=plugin_id,
        metadata={"hook": HOOK_NAME, "mode": "blocking", "order": "normal"},
    )


def _make_dispatcher() -> HookDispatcher:
    dispatcher = HookDispatcher()
    dispatcher.register_hook_spec(
        HookSpec(
            name=HOOK_NAME,
            parameters_schema={},
            default_timeout_ms=5000,
            allow_abort=True,
            allow_kwargs_mutation=True,
        )
    )
    return dispatcher


def _open_circuit(plugin_id: str) -> None:
    breaker = get_plugin_circuit_breaker()
    for _ in range(_CIRCUIT_OPEN_FAILURES):
        permit = breaker.try_acquire(plugin_id, "policy", "hook")
        assert permit.allowed is True
        breaker.record_failure(permit, "test failure")


async def _unreachable_handler() -> Any:
    raise AssertionError("熔断打开时不应调用处理器")


@pytest.mark.asyncio
async def test_circuit_open_is_reported_in_errors_and_denied() -> None:
    plugin_id = "test.hook.circuit_open"
    _open_circuit(plugin_id)
    supervisor = _FakeSupervisor(_make_entry(plugin_id), _unreachable_handler)
    dispatcher = _make_dispatcher()

    result = await dispatcher.invoke_hook(HOOK_NAME, [supervisor], tool_name="reply")

    assert supervisor.invoke_count == 0
    assert result.aborted is False
    assert any("CircuitOpen" in error for error in result.errors)
    decision = decide(result, on_error="deny")
    assert decision.allow is False
    assert decision.metadata["decided_by"] == "error"


@pytest.mark.asyncio
async def test_cancelled_half_open_probe_releases_inflight() -> None:
    plugin_id = "test.hook.circuit_half_open"
    breaker = get_plugin_circuit_breaker()
    _open_circuit(plugin_id)
    state = breaker._states[plugin_id]
    state.opened_until = time.monotonic() - 1.0

    started = asyncio.Event()

    async def blocked_handler() -> Any:
        started.set()
        await asyncio.Event().wait()

    supervisor = _FakeSupervisor(_make_entry(plugin_id), blocked_handler)
    dispatcher = _make_dispatcher()

    task = asyncio.create_task(dispatcher.invoke_hook(HOOK_NAME, [supervisor], tool_name="reply"))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    assert state.half_open_inflight is True

    # 半开探测占用期间的新调用同样必须进 errors，不能静默放行
    busy_supervisor = _FakeSupervisor(_make_entry(plugin_id), _unreachable_handler)
    busy_result = await dispatcher.invoke_hook(HOOK_NAME, [busy_supervisor], tool_name="reply")
    assert busy_supervisor.invoke_count == 0
    assert any("CircuitOpen" in error for error in busy_result.errors)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert state.state == "half_open"
    assert state.half_open_inflight is False

    permit = breaker.try_acquire(plugin_id, "policy", "hook")
    assert permit.allowed is True
    assert permit.half_open is True
    breaker.record_failure(permit, "cleanup")
