"""Configuration management using Pydantic Settings.

Supports YAML config files, environment variables, and .env files.
Priority: env vars > .env > config file > defaults
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SensoriumConfig(BaseSettings):
    """感知层配置."""
    model_config = SettingsConfigDict(env_prefix="VERITAS_SENSORIUM_")

    nats_embedded: bool = True
    nats_port: int = 4222
    anomaly_zscore_threshold: float = 3.0
    anomaly_contamination: float = 0.1
    anomaly_n_estimators: int = 100
    enable_opcua: bool = False
    enable_mssql: bool = False
    enable_file_watch: bool = True
    opcua_endpoint: str = "opc.tcp://localhost:4840"
    mssql_connection_string: str = ""
    watch_paths: list[str] = Field(default_factory=list)


class NoosphereConfig(BaseSettings):
    """认知层配置."""
    model_config = SettingsConfigDict(env_prefix="VERITAS_NOOSPHERE_")

    kuzu_db_path: str = "./data/kuzu"
    lancedb_path: str = "./data/lancedb"
    working_memory_size: int = 20
    vector_memory_ttl_days: int = 7
    compression_threshold: int = 100_000
    embedding_model: str = "nomic-embed-text"
    embedding_dimensions: int = 768


class HyperionConfig(BaseSettings):
    """推演层配置."""
    model_config = SettingsConfigDict(env_prefix="VERITAS_HYPERION_")

    mcts_iterations: int = 10_000
    mcts_exploration_constant: float = 1.414
    sandbox_timeout: int = 30
    sandbox_max_memory_mb: int = 512
    sandbox_max_cpu_percent: int = 50
    enable_firejail: bool = False
    pattern_miner_population_size: int = 1000
    pattern_miner_generations: int = 20
    causal_test: str = "granger"
    causal_max_lag: int = 5


class DaemonConfig(BaseSettings):
    """情感层配置."""
    model_config = SettingsConfigDict(env_prefix="VERITAS_DAEMON_")

    curiosity_decay: float = 0.95
    urgency_lambda: float = 0.1
    default_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "efficiency": 0.3,
            "cost": 0.25,
            "risk": 0.25,
            "innovation": 0.2,
        }
    )


class PrometheusConfig(BaseSettings):
    """执行层配置."""
    model_config = SettingsConfigDict(env_prefix="VERITAS_PROMETHEUS_")

    default_gray_traffic: float = 0.05
    gray_duration_minutes: int = 30
    hitl_timeout_minutes: int = 60
    sql_whitelist: list[str] = Field(default_factory=lambda: ["SELECT", "INSERT", "UPDATE"])
    sql_blacklist: list[str] = Field(default_factory=lambda: ["DROP", "DELETE", "TRUNCATE"])
    enable_git_versioning: bool = True
    mcp_servers: list[dict[str, Any]] = Field(default_factory=list)


class OracleConfig(BaseSettings):
    """元认知层配置."""
    model_config = SettingsConfigDict(env_prefix="VERITAS_ORACLE_")

    audit_interval_hours: int = 24
    audit_task_threshold: int = 10
    max_plan_depth: int = 5
    enable_auto_reflection: bool = True


class LLMConfig(BaseSettings):
    """LLM配置."""
    model_config = SettingsConfigDict(env_prefix="VERITAS_LLM_")

    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "qwen3.6:35b"
    fallback_model: str = "qwen3.6:7b"
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 120
    embedding_model: str = "nomic-embed-text"
    tool_call_timeout: int = 60


class LoggingConfig(BaseSettings):
    """日志配置."""
    model_config = SettingsConfigDict(env_prefix="VERITAS_LOG_")

    level: str = "INFO"
    format: str = "json"  # json | text
    file: str = "./logs/veritas.log"
    max_size_mb: int = 100
    backup_count: int = 5
    console_output: bool = True


class VeritasConfig(BaseSettings):
    """Veritas Engine 全局配置.

    配置加载优先级（从高到低）:
    1. 环境变量（如 VERITAS_LLM_MODEL=qwen3.6:7b）
    2. .env 文件
    3. 配置文件（config/default.yaml）
    4. 默认值
    """

    model_config = SettingsConfigDict(
        env_prefix="VERITAS_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 系统标识
    name: str = "Veritas Engine"
    version: str = "0.1.0-alpha"

    # 数据路径
    data_dir: str = "./data"
    log_dir: str = "./logs"

    # 子系统配置
    sensorium: SensoriumConfig = Field(default_factory=SensoriumConfig)
    noosphere: NoosphereConfig = Field(default_factory=NoosphereConfig)
    hyperion: HyperionConfig = Field(default_factory=HyperionConfig)
    daemon: DaemonConfig = Field(default_factory=DaemonConfig)
    prometheus: PrometheusConfig = Field(default_factory=PrometheusConfig)
    oracle: OracleConfig = Field(default_factory=OracleConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @field_validator("data_dir", "log_dir")
    @classmethod
    def _ensure_dir(cls, v: str) -> str:
        Path(v).mkdir(parents=True, exist_ok=True)
        return v

    @classmethod
    def from_yaml(cls, path: str = "config/default.yaml") -> "VeritasConfig":
        """从YAML文件加载配置.

        Args:
            path: YAML配置文件路径

        Returns:
            VeritasConfig实例
        """
        config_path = Path(path)
        if not config_path.exists():
            return cls()

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # 处理嵌套结构
        veritas_data = data.get("veritas", data)
        return cls(**veritas_data)

    @classmethod
    def from_env(cls) -> "VeritasConfig":
        """仅从环境变量加载配置."""
        return cls(_env_file=None)

    def set_llm_provider(self, provider: str, api_key: str, model: str | None = None) -> None:
        """设置 LLM 提供商.

        Args:
            provider: 提供商名称 (kimi, deepseek, openai, ollama)
            api_key: API 密钥
            model: 模型名称 (可选)
        """
        if provider == "kimi":
            self.llm.provider = "kimi"
            self.llm.model = model or "moonshot-v1-8k"
            self.llm.api_key = api_key
            self.llm.base_url = "https://api.moonshot.cn/v1"
        elif provider == "deepseek":
            self.llm.provider = "deepseek"
            self.llm.model = model or "deepseek-chat"
            self.llm.api_key = api_key
            self.llm.base_url = "https://api.deepseek.com/v1"
        elif provider == "openai":
            self.llm.provider = "openai"
            self.llm.model = model or "gpt-4o-mini"
            self.llm.api_key = api_key
            self.llm.base_url = "https://api.openai.com/v1"
        else:
            self.llm.provider = "ollama"
            self.llm.model = model or "llama3.1"
            self.llm.base_url = "http://localhost:11434"

    def ensure_directories(self) -> None:
        """确保所有必要的目录存在."""
        dirs = [
            self.data_dir,
            self.log_dir,
            self.noosphere.kuzu_db_path,
            self.noosphere.lancedb_path,
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        """导出为字典（用于序列化）."""
        return self.model_dump()

    def to_yaml(self, path: str) -> None:
        """保存为YAML配置文件."""
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"veritas": self.to_dict()}
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


# ── 全局配置单例 ──
_config: VeritasConfig | None = None


def get_config() -> VeritasConfig:
    """获取全局配置单例.

    首次调用时从配置文件和环境变量加载。
    """
    global _config
    if _config is None:
        _config = VeritasConfig.from_yaml()
        _config.ensure_directories()
    return _config


def reload_config() -> VeritasConfig:
    """重新加载配置."""
    global _config
    _config = VeritasConfig.from_yaml()
    _config.ensure_directories()
    return _config


def set_config(config: VeritasConfig) -> None:
    """设置全局配置（用于测试）."""
    global _config
    _config = config
