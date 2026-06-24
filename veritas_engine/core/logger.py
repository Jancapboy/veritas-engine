"""Structured logging with Rich console output and JSON file logging."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text


class JSONFormatter(logging.Formatter):
    """JSON格式日志格式化器."""

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # 添加额外字段
        if hasattr(record, "layer"):
            log_data["layer"] = record.layer
        if hasattr(record, "event_id"):
            log_data["event_id"] = record.event_id
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        # 异常信息
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False, default=str)


class LayerRichHandler(RichHandler):
    """带层级颜色的Rich日志处理器."""

    LAYER_COLORS = {
        "sensorium": "cyan",
        "noosphere": "violet",
        "hyperion": "yellow",
        "daemon": "magenta",
        "prometheus": "green",
        "oracle": "bright_white",
        "core": "blue",
        "llm": "orange3",
        "system": "white",
    }

    def render_message(self, record: logging.LogRecord, message: str) -> "Text":
        """渲染日志消息，添加层级颜色标记."""
        text = Text()

        # 添加层级标签
        layer = getattr(record, "layer", "system")
        color = self.LAYER_COLORS.get(layer, "white")
        text.append(f"[{layer[:4].upper()}] ", style=f"bold {color}")
        text.append(message)

        return text


def setup_logger(
    name: str = "veritas",
    level: str = "INFO",
    log_format: str = "json",
    log_file: str = "./logs/veritas.log",
    console_output: bool = True,
    max_size_mb: int = 100,
    backup_count: int = 5,
) -> logging.Logger:
    """配置结构化日志.

    Args:
        name: 日志器名称
        level: 日志级别
        log_format: 日志格式 (json|text)
        log_file: 日志文件路径
        console_output: 是否输出到控制台
        max_size_mb: 单个日志文件最大大小(MB)
        backup_count: 保留的备份文件数量

    Returns:
        配置好的Logger实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers = []  # 清除已有处理器
    logger.propagate = False

    # 确保日志目录存在
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # 文件处理器 (JSON格式)
    from logging.handlers import RotatingFileHandler

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_size_mb * 1024 * 1024,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)

    if log_format == "json":
        file_handler.setFormatter(JSONFormatter())
    else:
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    logger.addHandler(file_handler)

    # 控制台处理器 (Rich彩色输出)
    if console_output:
        console_handler = LayerRichHandler(
            console=Console(stderr=True),
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            show_time=True,
            show_path=False,
        )
        console_handler.setLevel(getattr(logging, level.upper()))
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str = "veritas") -> logging.Logger:
    """获取日志器实例.

    如果尚未配置，使用默认配置初始化。
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        from veritas_engine.core.config import get_config
        cfg = get_config().logging
        return setup_logger(
            name=name,
            level=cfg.level,
            log_format=cfg.format,
            log_file=cfg.file,
            console_output=cfg.console_output,
            max_size_mb=cfg.max_size_mb,
            backup_count=cfg.backup_count,
        )
    return logger


# 便捷函数：带层级上下文的日志
def log_with_layer(
    logger: logging.Logger,
    level: str,
    message: str,
    layer: str = "system",
    **extra: Any,
) -> None:
    """带层级标记的日志记录.

    Example:
        log_with_layer(logger, "INFO", "感知层启动", layer="sensorium")
    """
    extra_data = {"layer": layer}
    if extra:
        extra_data["extra"] = extra

    getattr(logger, level.lower(), logger.info)(message, extra=extra_data)
