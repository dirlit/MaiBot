"""工具调用前策略 Hook 适配器。

判定核（build_payload / decide / decide_failure）是同步纯函数，便于脱离事件循环单测；
HookToolPolicy.check 只负责 Hook 分发与审计提交，不在这里做判定。
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional

import asyncio

from src.common.logger import get_logger
from src.common.tool_audit import ToolAuditWriter
from src.core.tooling import (
    ToolExecutionContext,
    ToolInvocation,
    ToolPolicy,
    ToolPolicyDecision,
    ToolSpec,
)

if TYPE_CHECKING:
    from src.plugin_runtime.host.hook_dispatcher import HookDispatchResult

logger = get_logger("plugin_runtime.tool_policy")
HOOK_NAME = "maisaka.tool.before_execute"


def build_payload(
    invocation: ToolInvocation,
    context: Optional[ToolExecutionContext],
    spec: ToolSpec,
) -> dict[str, Any]:
    """同步纯函数：把一次工具调用拍平成 Hook 载荷。"""

    return {
        "tool_name": spec.name,  # 暴露名（可能带来源前缀）
        "original_tool_name": spec.metadata.get("registry_original_name", spec.name),
        "provider_type": spec.provider_type,
        "provider_name": spec.provider_name,
        "arguments": deepcopy(invocation.arguments),
        "platform": context.platform if context else "",
        "user_id": context.user_id if context else "",
        "group_id": context.group_id if context else "",
        "session_id": (context.session_id if context else "") or invocation.session_id,
        "call_id": invocation.call_id,
    }


def decide_failure(message: str, *, on_error: str, decided_by: str) -> ToolPolicyDecision:
    """同步纯函数：Hook 超时/异常等 I/O 层失败的判定；默认拒绝。"""

    if on_error == "allow":
        return ToolPolicyDecision(
            allow=True,
            reason=f"策略 Hook 出错，按 on_error=allow 放行: {message}",
            metadata={"policy_error": message, "decided_by": "on_error"},
        )
    return ToolPolicyDecision(
        allow=False,
        reason=f"策略 Hook 执行出错: {message}",
        metadata={"decided_by": decided_by},
    )


def decide(result: "HookDispatchResult", *, on_error: str) -> ToolPolicyDecision:
    """同步纯函数：Hook 聚合结果 → 判定。"""

    if result.aborted:
        # 显式 abort 或 error_policy=abort；不受 on_error=allow 影响
        return ToolPolicyDecision(
            allow=False,
            reason=str(result.kwargs.get("reason", "") or "策略拒绝"),
            metadata={"decided_by": result.stopped_by or "hook"},
        )
    if result.errors:
        # 默认 error_policy=skip 只把异常收进 errors，不会置 aborted；这里必须 fail-closed
        return decide_failure("；".join(result.errors), on_error=on_error, decided_by="error")
    if str(result.kwargs.get("decision", "")).strip().lower() == "deny":
        return ToolPolicyDecision(
            allow=False,
            reason=str(result.kwargs.get("reason", "") or "策略拒绝"),
            metadata={"decided_by": "hook"},
        )
    return ToolPolicyDecision(allow=True, metadata={"decided_by": "hook"})


class HookToolPolicy(ToolPolicy):
    """把统一工具调用转发到插件 Hook，实现运行期可配置的工具级授权。"""

    def __init__(self, *, timeout_ms: int, on_error: str, audit: ToolAuditWriter | None) -> None:
        """初始化 Hook 策略。

        Args:
            timeout_ms: 单次策略判定的外层等待上限，超时按 on_error 处理。
            on_error: Hook 出错时的处理方式，仅接受 deny / allow，其余按 deny。
            audit: 已配置的审计写入器；None 表示不写审计。
        """

        self._timeout_ms = timeout_ms
        self._on_error = on_error if on_error in {"deny", "allow"} else "deny"
        self._audit = audit

    @staticmethod
    def _get_runtime_manager() -> Any:
        """函数内延迟 import，避免 core ↔ plugin_runtime 循环引用。"""

        from src.plugin_runtime.integration import get_plugin_runtime_manager

        return get_plugin_runtime_manager()

    async def check(
        self,
        invocation: ToolInvocation,
        context: Optional[ToolExecutionContext],
        spec: ToolSpec,
    ) -> ToolPolicyDecision:
        """预调用策略入口：只做 I/O，判定交给纯函数。"""

        payload = build_payload(invocation, context, spec)
        try:
            result = await asyncio.wait_for(
                self._get_runtime_manager().invoke_hook(HOOK_NAME, **payload),
                timeout=self._timeout_ms / 1000,
            )
        except asyncio.TimeoutError:
            return await self._audit_decision(
                decide_failure("策略 Hook 超时", on_error=self._on_error, decided_by="timeout"),
                payload,
            )
        except Exception as exc:
            return await self._audit_decision(
                decide_failure(
                    f"策略 Hook 异常: {exc.__class__.__name__}",
                    on_error=self._on_error,
                    decided_by="exception",
                ),
                payload,
            )

        return await self._audit_decision(decide(result, on_error=self._on_error), payload)

    async def audit_failure(
        self,
        invocation: ToolInvocation,
        context: Optional[ToolExecutionContext],
        spec: ToolSpec,
        exc: BaseException,
    ) -> None:
        """A1 兜底 fail-closed 的审计入口；写一条 decided_by=registry_error 的拒绝记录。"""

        await self._record(
            "deny",
            build_payload(invocation, context, spec),
            reason=f"策略执行异常: {exc.__class__.__name__}: {exc}",
            decided_by="registry_error",
        )

    async def _audit_decision(
        self,
        decision: ToolPolicyDecision,
        payload: dict[str, Any],
    ) -> ToolPolicyDecision:
        """写一条审计（allow 或 deny）后原样返回判定。"""

        await self._record(
            "allow" if decision.allow else "deny",
            payload,
            reason=decision.reason,
            decided_by=str(decision.metadata.get("decided_by", "hook")),
        )
        return decision

    async def _record(
        self,
        action: str,
        payload: dict[str, Any],
        *,
        reason: str = "",
        decided_by: str = "",
    ) -> None:
        """审计落盘统一走线程池，不阻塞事件循环；写失败只记日志，不改变判定。"""

        if self._audit is None:
            return
        try:
            await asyncio.to_thread(
                self._audit.record,
                action=action,
                tool=payload["tool_name"],
                provider_type=payload["provider_type"],
                platform=payload["platform"],
                session_id=payload["session_id"],
                user_id=payload["user_id"],
                call_id=payload["call_id"],
                args=payload["arguments"],
                reason=reason,
                decided_by=decided_by,
            )
        except Exception:
            logger.exception(f"工具审计写入失败（不影响判定）: tool={payload['tool_name']} action={action}")
