"""MCP 每服务器工具白名单/黑名单测试。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import json

import pytest

from src.common.tool_audit import ToolAuditWriter
from src.config.official_configs import MCPServerItemConfig
from src.core.tooling import ToolExecutionContext, ToolExecutionResult, ToolInvocation
from src.mcp_module.config import (
    MCPClientRuntimeConfig,
    MCPServerRuntimeConfig,
    build_mcp_server_runtime_config,
)
from src.mcp_module.manager import MCPManager, is_tool_allowed


def _config(name: str = "x", **overrides: Any) -> MCPServerRuntimeConfig:
    """构造一个只带必要字段的服务器运行时配置。"""

    return MCPServerRuntimeConfig(name=name, transport="stdio", command="python", **overrides)


def _tool(name: str) -> SimpleNamespace:
    """构造连接层返回的最小工具对象。"""

    return SimpleNamespace(
        name=name,
        description="",
        inputSchema={},
        outputSchema=None,
        icons=[],
        annotations=None,
        meta={},
    )


class _FakeConnection:
    """只实现管理器用到的最小连接接口，并记录真实调用。"""

    def __init__(self, config: MCPServerRuntimeConfig, tool_names: list[str]) -> None:
        self.config = config
        self.tools = [_tool(tool_name) for tool_name in tool_names]
        self.prompts: list[Any] = []
        self.resources: list[Any] = []
        self.resource_templates: list[Any] = []
        self.session: Any = None
        self.protocol_version = ""
        self.last_error = ""
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolExecutionResult:
        self.calls.append((tool_name, arguments))
        return ToolExecutionResult(tool_name=tool_name, success=True, content="ok")


def test_empty_allowed_means_unrestricted() -> None:
    config = _config()

    assert is_tool_allowed(config, "read_file")
    assert is_tool_allowed(config, "anything")


def test_disabled_wins_over_allowed() -> None:
    config = _config(allowed_tools=["filesystem_*"], disabled_tools=["filesystem_write_file"])

    assert is_tool_allowed(config, "filesystem_read_file")
    assert not is_tool_allowed(config, "filesystem_write_file")


def test_wildcard_allowed() -> None:
    config = _config(allowed_tools=["filesystem_*"])

    assert is_tool_allowed(config, "filesystem_read_file")
    assert not is_tool_allowed(config, "read_file")


def test_exact_match_is_case_sensitive() -> None:
    config = _config(allowed_tools=["Read_File"])

    assert is_tool_allowed(config, "Read_File")
    assert not is_tool_allowed(config, "read_file")


def test_config_rejects_overlap() -> None:
    with pytest.raises(ValueError, match="同时出现在"):
        MCPServerItemConfig(
            name="x",
            transport="stdio",
            command="python",
            allowed_tools=["a"],
            disabled_tools=["a"],
        )


def test_config_rejects_blank_name() -> None:
    with pytest.raises(ValueError, match="不能为空字符串"):
        MCPServerItemConfig(
            name="x",
            transport="stdio",
            command="python",
            allowed_tools=[" "],
        )


def test_runtime_config_passthrough() -> None:
    server = MCPServerItemConfig(
        name="x",
        transport="stdio",
        command="python",
        allowed_tools=[" read_file ", "read_file", "read_file "],
        disabled_tools=[" write_file ", "write_file"],
    )

    runtime = build_mcp_server_runtime_config(server)

    assert runtime.allowed_tools == ["read_file"]
    assert runtime.disabled_tools == ["write_file"]


def test_disabled_server_skips_new_validation() -> None:
    server = MCPServerItemConfig(
        enabled=False,
        name="x",
        allowed_tools=["a"],
        disabled_tools=["a"],
    )

    assert server.enabled is False


def test_register_skips_denied_tool() -> None:
    pytest.importorskip("mcp")
    config = _config(disabled_tools=["write_file"])
    connection = _FakeConnection(config, ["read_file", "write_file"])
    manager = MCPManager(MCPClientRuntimeConfig())

    registered = manager._register_tools("x", connection)

    assert registered == 1
    assert "read_file" in manager._tool_to_server
    assert "write_file" not in manager._tool_to_server


def test_get_tool_specs_excludes_denied() -> None:
    pytest.importorskip("mcp")
    config = _config(disabled_tools=["write_file"])
    connection = _FakeConnection(config, ["read_file", "write_file"])
    manager = MCPManager(MCPClientRuntimeConfig())
    manager._connections = {"x": connection}

    assert [spec.name for spec in manager.get_tool_specs()] == ["read_file"]


def test_tool_count_excludes_denied() -> None:
    pytest.importorskip("mcp")
    config = _config(allowed_tools=["read_file"])
    connection = _FakeConnection(config, ["read_file", "write_file"])
    manager = MCPManager(MCPClientRuntimeConfig())
    manager._connections = {"x": connection}

    assert len(connection.tools) == 2
    assert manager.tool_count == 1


def test_status_snapshot_counts_exclude_denied() -> None:
    pytest.importorskip("mcp")
    config = _config(allowed_tools=["read_file"])
    connection = _FakeConnection(config, ["read_file", "write_file"])
    manager = MCPManager(MCPClientRuntimeConfig())
    manager._server_configs = {"x": config}
    manager._connections = {"x": connection}

    snapshot = manager.get_status_snapshot()

    assert manager.tool_count == 1
    assert snapshot["tool_count"] == 1
    assert snapshot["servers"][0]["tool_count"] == 1


@pytest.mark.asyncio
async def test_call_tool_invocation_rejects_disabled_with_reason() -> None:
    pytest.importorskip("mcp")
    config = _config(allowed_tools=["read_file"])
    connection = _FakeConnection(config, ["read_file", "write_file"])
    manager = MCPManager(MCPClientRuntimeConfig())
    manager._connections = {"x": connection}

    result = await manager.call_tool_invocation(
        ToolInvocation(tool_name="write_file", metadata={"mcp_server_name": "x"})
    )

    assert result.success is False
    assert "策略禁用" in result.error_message
    assert connection.calls == []


@pytest.mark.asyncio
async def test_denied_and_allowed_audit_write_seq(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("mcp")
    audit_path = tmp_path / "tool-audit.jsonl"
    writer = ToolAuditWriter(audit_path)
    monkeypatch.setattr("src.mcp_module.manager.get_tool_audit_writer", lambda: writer)
    config = _config(allowed_tools=["read_file"], disabled_tools=["write_file"])
    connection = _FakeConnection(config, ["read_file", "write_file"])
    manager = MCPManager(MCPClientRuntimeConfig())
    manager._connections = {"x": connection}
    context = ToolExecutionContext(platform="qq", session_id="s1", user_id="u1")

    denied = await manager.call_tool_invocation(
        ToolInvocation(
            tool_name="write_file",
            arguments={"path": "secret.txt"},
            call_id="call-1",
            metadata={"mcp_server_name": "x"},
        ),
        context,
    )
    allowed = await manager.call_tool_invocation(
        ToolInvocation(
            tool_name="read_file",
            arguments={"path": "public.txt"},
            call_id="call-2",
            metadata={"mcp_server_name": "x"},
        ),
        context,
    )

    assert denied.success is False
    assert "策略禁用" in denied.error_message
    assert allowed.success is True
    assert [call[0] for call in connection.calls] == ["read_file"]

    raw = audit_path.read_text(encoding="utf-8")
    entries = [json.loads(line) for line in raw.splitlines() if line.strip()]
    assert [entry["action"] for entry in entries] == ["deny", "allow"]
    assert [entry["seq"] for entry in entries] == [1, 2]
    assert entries[1]["prev_hash"] == entries[0]["entry_hash"]
    assert [(entry["platform"], entry["session_id"], entry["user_id"]) for entry in entries] == [
        ("qq", "s1", "u1"),
        ("qq", "s1", "u1"),
    ]
    assert [entry["call_id"] for entry in entries] == ["call-1", "call-2"]
    assert all("args_hash" in entry for entry in entries)
    assert "secret.txt" not in raw
    assert "public.txt" not in raw


def test_allowed_only_exposes_listed() -> None:
    pytest.importorskip("mcp")
    config = _config(allowed_tools=["read_file"])
    connection = _FakeConnection(config, ["read_file", "write_file"])
    manager = MCPManager(MCPClientRuntimeConfig())
    manager._connections = {"x": connection}

    assert [spec.name for spec in manager.get_tool_specs()] == ["read_file"]


def test_same_name_on_allowed_server_still_exposed() -> None:
    pytest.importorskip("mcp")
    config_a = _config(name="A", disabled_tools=["x"])
    config_b = _config(name="B", allowed_tools=["x"])
    connection_a = _FakeConnection(config_a, ["x"])
    connection_b = _FakeConnection(config_b, ["x"])
    manager = MCPManager(MCPClientRuntimeConfig())
    manager._connections = {"A": connection_a, "B": connection_b}

    manager._register_tools("A", connection_a)
    manager._register_tools("B", connection_b)

    specs = manager.get_tool_specs()
    assert [spec.name for spec in specs] == ["x"]
    assert specs[0].metadata["server_name"] == "B"
    assert manager._tool_to_server["x"] == "B"
