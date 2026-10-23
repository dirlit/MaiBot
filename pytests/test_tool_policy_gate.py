"""统一工具注册表的预调用策略 Gate 测试。"""

from __future__ import annotations

from typing import Any, Optional

import pytest

from src.core.tooling import (
    ToolExecutionContext,
    ToolExecutionResult,
    ToolInvocation,
    ToolPolicyDecision,
    ToolRegistry,
    ToolSpec,
)


class _FakeProvider:
    """最小工具 Provider，只记录被调用的次数与参数。"""

    def __init__(
        self,
        name: str = "fake",
        provider_type: str = "builtin",
        tool_names: Optional[list[str]] = None,
    ) -> None:
        self.provider_name = name
        self.provider_type = provider_type
        self._tool_names = tool_names or ["echo"]
        self.invocations: list[ToolInvocation] = []

    async def list_tools(self, context: Any = None) -> list[ToolSpec]:
        return [
            ToolSpec(
                name=tool_name,
                description="测试工具",
                provider_name=self.provider_name,
                provider_type=self.provider_type,
            )
            for tool_name in self._tool_names
        ]

    async def invoke(
        self,
        invocation: ToolInvocation,
        context: Optional[ToolExecutionContext] = None,
    ) -> ToolExecutionResult:
        self.invocations.append(invocation)
        return ToolExecutionResult(tool_name=invocation.tool_name, success=True, content="ok")

    async def close(self) -> None:
        return None


class _DenyPolicy:
    """始终拒绝的策略。"""

    async def check(
        self,
        invocation: ToolInvocation,
        context: Optional[ToolExecutionContext],
        spec: ToolSpec,
    ) -> ToolPolicyDecision:
        return ToolPolicyDecision(allow=False, reason="blocked by test")


class _AllowPolicy:
    """始终放行的策略。"""

    async def check(
        self,
        invocation: ToolInvocation,
        context: Optional[ToolExecutionContext],
        spec: ToolSpec,
    ) -> ToolPolicyDecision:
        return ToolPolicyDecision(allow=True, metadata={"decided_by": "test"})


class _BoomPolicy:
    """判定时直接抛异常，并记录兜底审计调用。"""

    def __init__(self) -> None:
        self.audit_calls: list[tuple[ToolInvocation, Optional[ToolExecutionContext], ToolSpec, BaseException]] = []

    async def check(
        self,
        invocation: ToolInvocation,
        context: Optional[ToolExecutionContext],
        spec: ToolSpec,
    ) -> ToolPolicyDecision:
        raise RuntimeError("boom")

    async def audit_failure(
        self,
        invocation: ToolInvocation,
        context: Optional[ToolExecutionContext],
        spec: ToolSpec,
        exc: BaseException,
    ) -> None:
        self.audit_calls.append((invocation, context, spec, exc))


def _build_registry(provider: _FakeProvider) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_provider(provider)
    return registry


@pytest.mark.asyncio
async def test_no_policy_keeps_behavior() -> None:
    provider = _FakeProvider()
    registry = _build_registry(provider)

    result = await registry.invoke(ToolInvocation(tool_name="echo", arguments={"text": "hi"}))

    assert result.success is True
    assert result.tool_name == "echo"
    assert result.content == "ok"
    assert len(provider.invocations) == 1
    assert provider.invocations[0].tool_name == "echo"


@pytest.mark.asyncio
async def test_policy_deny_skips_provider() -> None:
    provider = _FakeProvider()
    registry = _build_registry(provider)
    registry.set_policy(_DenyPolicy())

    result = await registry.invoke(ToolInvocation(tool_name="echo"))

    assert provider.invocations == []
    assert result.success is False
    assert "blocked by test" in result.error_message
    assert result.metadata["policy_denied"] is True
    assert result.metadata["policy_reason"] == "blocked by test"


@pytest.mark.asyncio
async def test_policy_allow_invokes_provider() -> None:
    provider = _FakeProvider()
    registry = _build_registry(provider)
    registry.set_policy(_AllowPolicy())

    result = await registry.invoke(ToolInvocation(tool_name="echo"))

    assert result.success is True
    assert len(provider.invocations) == 1


@pytest.mark.asyncio
async def test_policy_exception_fails_closed() -> None:
    provider = _FakeProvider()
    registry = _build_registry(provider)
    registry.set_policy(_BoomPolicy())

    result = await registry.invoke(ToolInvocation(tool_name="echo"))

    assert provider.invocations == []
    assert result.success is False
    assert result.metadata["policy_denied"] is True
    assert "策略执行异常" in result.error_message


@pytest.mark.asyncio
async def test_registry_route_prefix_still_works() -> None:
    first = _FakeProvider("provider_a", "plugin", ["echo"])
    second = _FakeProvider("provider_b", "plugin", ["echo"])
    registry = ToolRegistry()
    registry.register_provider(first)
    registry.register_provider(second)
    registry.set_policy(_DenyPolicy())

    names = [spec.name for spec in await registry.list_tools()]
    assert "echo" in names
    prefixed_name = next(name for name in names if name != "echo")
    assert prefixed_name.startswith("plugin_provider_b_")

    result = await registry.invoke(ToolInvocation(tool_name=prefixed_name))

    assert result.success is False
    assert prefixed_name in result.error_message
    assert result.metadata["policy_denied"] is True
    assert first.invocations == []
    assert second.invocations == []


@pytest.mark.asyncio
async def test_registry_fail_closed_writes_audit() -> None:
    provider = _FakeProvider()
    registry = _build_registry(provider)
    policy = _BoomPolicy()
    registry.set_policy(policy)

    result = await registry.invoke(ToolInvocation(tool_name="echo"))

    assert result.success is False
    assert result.metadata["policy_denied"] is True
    assert provider.invocations == []
    assert len(policy.audit_calls) == 1
    assert isinstance(policy.audit_calls[0][3], RuntimeError)
