"""
CityLens 统一日志模块
"""

import logging
import sys
from typing import Optional


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    获取配置好的 logger 实例

    Args:
        name: logger 名称，通常使用 __name__
        level: 日志级别，默认 INFO

    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    logger.setLevel(level or logging.INFO)

    # 控制台输出
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level or logging.INFO)

    # 格式化
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # 防止日志传播到根 logger
    logger.propagate = False

    return logger


# 预配置的 logger 实例
main_logger = get_logger("citylens.main")
llm_logger = get_logger("citylens.llm")
segment_logger = get_logger("citylens.segment")
analysis_logger = get_logger("citylens.analysis")


class LogContext:
    """日志上下文管理器，用于记录操作耗时"""

    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None

    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.info(f"[START] {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        elapsed = time.time() - self.start_time
        if exc_type:
            self.logger.error(
                f"[FAILED] {self.operation} ({elapsed:.2f}s) - {exc_type.__name__}: {exc_val}"
            )
        else:
            self.logger.info(f"[DONE] {self.operation} ({elapsed:.2f}s)")
        return False  # 不抑制异常
