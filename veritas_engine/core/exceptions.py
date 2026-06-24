"""Custom exceptions for Veritas Engine.

All exceptions inherit from VeritasError for unified error handling.
Each layer can define its own specific exceptions.
"""

from __future__ import annotations


class VeritasError(Exception):
    """Base exception for all Veritas Engine errors."""

    def __init__(self, message: str, *, code: str = "VERITAS_001", details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}

    def to_dict(self) -> dict:
        return {
            "error": self.message,
            "code": self.code,
            "details": self.details,
        }


# ── 配置相关 ──

class ConfigError(VeritasError):
    """配置错误."""
    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(message, code="CONFIG_001", **kwargs)


# ── 感知层 ──

class SensoriumError(VeritasError):
    """感知层基础错误."""
    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(message, code="SENSORIUM_001", **kwargs)


class SensorConnectionError(SensoriumError):
    """传感器连接失败."""
    def __init__(self, sensor: str, message: str = "", **kwargs) -> None:
        super().__init__(f"传感器连接失败 [{sensor}]: {message}", code="SENSORIUM_002", **kwargs)


class EventBusError(SensoriumError):
    """事件总线错误."""
    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(message, code="SENSORIUM_003", **kwargs)


class AnomalyDetectionError(SensoriumError):
    """异常检测错误."""
    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(message, code="SENSORIUM_004", **kwargs)


# ── 认知层 ──

class NoosphereError(VeritasError):
    """认知层基础错误."""
    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(message, code="NOOSPHERE_001", **kwargs)


class GraphDBError(NoosphereError):
    """图数据库操作错误."""
    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(message, code="NOOSPHERE_002", **kwargs)


class VectorDBError(NoosphereError):
    """向量数据库操作错误."""
    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(message, code="NOOSPHERE_003", **kwargs)


class MemoryCompressionError(NoosphereError):
    """记忆压缩错误."""
    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(message, code="NOOSPHERE_004", **kwargs)


class SchemaError(NoosphereError):
    """图谱Schema错误."""
    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(message, code="NOOSPHERE_005", **kwargs)


# ── 推演层 ──

class HyperionError(VeritasError):
    """推演层基础错误."""
    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(message, code="HYPERION_001", **kwargs)


class SelfPlayError(HyperionError):
    """自博弈引擎错误."""
    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(message, code="HYPERION_002", **kwargs)


class SandboxError(HyperionError):
    """沙盒执行错误."""
    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(message, code="HYPERION_003", **kwargs)


class SandboxTimeoutError(SandboxError):
    """沙盒执行超时."""
    def __init__(self, timeout: int, **kwargs) -> None:
        super().__init__(f"沙盒执行超时 ({timeout}s)", code="HYPERION_004", **kwargs)


class SandboxSecurityError(SandboxError):
    """沙盒安全违规."""
    def __init__(self, violation: str, **kwargs) -> None:
        super().__init__(f"沙盒安全违规: {violation}", code="HYPERION_005", **kwargs)


class PatternMiningError(HyperionError):
    """规律挖掘错误."""
    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(message, code="HYPERION_006", **kwargs)


# ── 情感层 ──

class DaemonError(VeritasError):
    """情感层基础错误."""
    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(message, code="DAEMON_001", **kwargs)


# ── 执行层 ──

class PrometheusError(VeritasError):
    """执行层基础错误."""
    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(message, code="PROMETHEUS_001", **kwargs)


class HITLError(PrometheusError):
    """人机确认错误."""
    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(message, code="PROMETHEUS_002", **kwargs)


class MCPError(PrometheusError):
    """MCP网关错误."""
    def __init__(self, server: str, message: str = "", **kwargs) -> None:
        super().__init__(f"MCP Server [{server}] 错误: {message}", code="PROMETHEUS_003", **kwargs)


class RollbackError(PrometheusError):
    """回滚错误."""
    def __init__(self, strategy_id: str, message: str = "", **kwargs) -> None:
        super().__init__(f"策略 [{strategy_id}] 回滚失败: {message}", code="PROMETHEUS_004", **kwargs)


# ── 元认知层 ──

class OracleError(VeritasError):
    """元认知层基础错误."""
    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(message, code="ORACLE_001", **kwargs)


class GoalDecompositionError(OracleError):
    """目标分解错误."""
    def __init__(self, goal: str, message: str = "", **kwargs) -> None:
        super().__init__(f"目标分解失败 [{goal}]: {message}", code="ORACLE_002", **kwargs)


# ── LLM层 ──

class LLMError(VeritasError):
    """LLM调用错误."""
    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(message, code="LLM_001", **kwargs)


class LLMTimeoutError(LLMError):
    """LLM调用超时."""
    def __init__(self, model: str, timeout: int, **kwargs) -> None:
        super().__init__(f"LLM [{model}] 调用超时 ({timeout}s)", code="LLM_002", **kwargs)


class ToolCallError(VeritasError):
    """工具调用错误."""
    def __init__(self, tool: str, message: str = "", **kwargs) -> None:
        super().__init__(f"工具调用失败 [{tool}]: {message}", code="LLM_003", **kwargs)
