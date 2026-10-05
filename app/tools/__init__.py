"""Tool gateway package - every agent tool call must go through ToolGateway.call()."""

from app.tools.gateway import ToolError, ToolGateway, ToolResult
from app.tools.registry import build_gateway, get_gateway, set_gateway

__all__ = ["ToolError", "ToolGateway", "ToolResult", "build_gateway", "get_gateway", "set_gateway"]
