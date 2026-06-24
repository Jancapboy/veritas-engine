"""LanceDB vector memory for Veritas Engine.

Lightweight vector storage for semantic search and working memory.
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from veritas_engine.core.config import get_config
from veritas_engine.core.logger import get_logger
from veritas_engine.core.exceptions import VectorDBError

logger = get_logger("veritas.noosphere")


def _norm_vector(vec: list[float]) -> list[float]:
    """Normalize a vector to unit length."""
    mag = sum(v * v for v in vec) ** 0.5
    if mag == 0:
        return vec
    return [v / mag for v in vec]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class VectorMemory:
    """LanceDB 向量记忆管理器."""

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or get_config().noosphere.lancedb_path
        self._db = None
        self._table = None
        self._mock = False
        self._mock_data: list[dict] = []

    def connect(self) -> None:
        """连接 LanceDB."""
        try:
            import lancedb
        except ImportError:
            logger.warning("lancedb not installed, using mock vector memory")
            self._mock = True
            return

        self._mock = False
        Path(self.db_path).mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(self.db_path)

        # 创建或打开表
        table_name = "vector_memory"
        try:
            self._table = self._db.open_table(table_name)
        except Exception:
            # 创建新表
            import pyarrow as pa
            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("content", pa.string()),
                pa.field("embedding", pa.list_(pa.float32(), 768)),
                pa.field("metadata", pa.string()),
                pa.field("timestamp", pa.timestamp("us")),
                pa.field("source", pa.string()),
            ])
            self._table = self._db.create_table(table_name, schema=schema)

        logger.info("VectorMemory connected to %s", self.db_path, extra={"layer": "noosphere"})

    def close(self) -> None:
        """关闭连接."""
        self._db = None
        self._table = None

    def _generate_embedding(self, text: str) -> list[float]:
        """生成文本嵌入向量（使用简单哈希模拟，实际应调用LLM embedding）."""
        # 模拟 embedding：使用文本哈希生成固定维度向量
        random.seed(int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32))
        vec = [random.gauss(0, 1) for _ in range(768)]
        return _norm_vector(vec)

    def store(self, content: str, metadata: dict[str, Any] | None = None, source: str = "unknown") -> str:
        """存储记忆到向量数据库，返回记忆ID."""
        memory_id = hashlib.md5(f"{content}{datetime.now(timezone.utc).isoformat()}".encode()).hexdigest()
        embedding = self._generate_embedding(content)
        meta_str = json.dumps(metadata or {}) if metadata else "{}"
        timestamp = datetime.now(timezone.utc)

        if self._mock:
            self._mock_data.append({
                "id": memory_id,
                "content": content,
                "embedding": embedding,
                "metadata": meta_str,
                "timestamp": timestamp,
                "source": source,
            })
            return memory_id

        import pyarrow as pa
        self._table.add(pa.table({
            "id": [memory_id],
            "content": [content],
            "embedding": [embedding],
            "metadata": [meta_str],
            "timestamp": [timestamp],
            "source": [source],
        }))
        return memory_id

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """向量语义搜索."""
        query_vec = self._generate_embedding(query)

        if self._mock:
            # 简单模拟：按内容相似度排序
            results = []
            query_vec = self._generate_embedding(query)
            for item in self._mock_data:
                score = _dot(query_vec, item["embedding"])
                results.append({**item, "score": float(score)})
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

        results = self._table.search(query_vec).limit(top_k).to_list()
        return results

    def cleanup_old_memories(self, ttl_days: int | None = None) -> int:
        """清理过期记忆，返回删除数量."""
        ttl = ttl_days or get_config().noosphere.vector_memory_ttl_days
        cutoff = datetime.now(timezone.utc) - timedelta(days=ttl)

        if self._mock:
            old_len = len(self._mock_data)
            self._mock_data = [m for m in self._mock_data if m["timestamp"] > cutoff]
            return old_len - len(self._mock_data)

        # LanceDB 删除支持
        self._table.delete(f"timestamp < timestamp '{cutoff.isoformat()}'")
        return 0  # LanceDB 不返回删除数量

    def get_stats(self) -> dict[str, Any]:
        """获取统计信息."""
        if self._mock:
            return {"total_memories": len(self._mock_data), "mode": "mock"}
        return {"total_memories": self._table.count_rows(), "mode": "lancedb"}


import json
