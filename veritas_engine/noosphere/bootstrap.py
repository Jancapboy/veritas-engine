"""Noosphere bootstrap — initializes graph DB, vector DB, working memory, compressor."""

from __future__ import annotations

from typing import Any

from veritas_engine.core.models import Entity, Observation, PerceptionEvent
from veritas_engine.core.logger import get_logger
from veritas_engine.core.config import get_config
from veritas_engine.noosphere.graph_db import GraphDB
from veritas_engine.noosphere.vector_db import VectorMemory
from veritas_engine.noosphere.working_memory import WorkingMemory
from veritas_engine.noosphere.memory_compressor import MemoryCompressor

logger = get_logger("veritas.noosphere")


class Noosphere:
    """认知层统一入口——管理知识图谱、向量记忆、工作记忆."""

    def __init__(self) -> None:
        self.config = get_config().noosphere
        self.graph = GraphDB()
        self.vector = VectorMemory()
        self.working_memory = WorkingMemory()
        self.compressor = MemoryCompressor(
            threshold=0.75,
            target_ratio=0.30,
            protect_last_n=50,
        )

    def connect(self) -> None:
        """初始化所有认知存储."""
        self.graph.connect()
        self.vector.connect()
        logger.info("Noosphere initialized", extra={"layer": "noosphere"})

    def close(self) -> None:
        """关闭所有认知存储."""
        self.graph.close()
        self.vector.close()

    # ── 实体操作 ──

    def create_entity(self, entity: Entity) -> str:
        """创建实体节点，返回实体ID."""
        return self.graph.create_entity(entity)

    def get_entity(self, entity_id: str) -> Entity | None:
        """按ID获取实体."""
        return self.graph.get_entity(entity_id)

    def query_graph(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """执行 Cypher 查询."""
        return self.graph.query(cypher, params)

    def find_path(self, from_id: str, to_id: str, max_depth: int = 5) -> list[list[str]]:
        """查找两实体间的路径."""
        return self.graph.find_path(from_id, to_id, max_depth)

    def create_relation(
        self, from_id: str, to_id: str, rel_type: str, properties: dict[str, Any] | None = None
    ) -> None:
        """创建实体间关系."""
        self.graph.create_relation(from_id, to_id, rel_type, properties)

    # ── 向量记忆操作 ──

    def store_memory(self, content: str, metadata: dict[str, Any] | None = None, source: str = "unknown") -> str:
        """存储到向量记忆."""
        memory_id = self.vector.store(content, metadata, source)

        # 检查是否需要压缩
        stats = self.vector.get_stats()
        if self.compressor.should_compress(stats.get("total_memories", 0)):
            logger.info("Memory compression threshold reached", extra={"layer": "noosphere"})
            # 实际压缩逻辑可以异步执行

        return memory_id

    def vector_search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """向量语义搜索."""
        return self.vector.search(query, top_k)

    def compress_memories(self, memory_ids: list[str]) -> Entity:
        """压缩多条记忆为知识图谱实体."""
        # 获取记忆内容（mock模式下从内存获取）
        memories = []
        if self.vector._mock:
            memories = [m for m in self.vector._mock_data if m["id"] in memory_ids]
        else:
            # 实际应从LanceDB查询
            pass

        entity = self.compressor.compress(memories)
        self.graph.create_entity(entity)
        return entity

    # ── 工作记忆操作 ──

    def add_to_working_memory(self, event: PerceptionEvent) -> None:
        """添加事件到工作记忆."""
        self.working_memory.add(event)

    def get_working_memory(self) -> WorkingMemory:
        """获取工作记忆窗口."""
        return self.working_memory

    # ── 综合查询 ──

    def recall(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """综合回忆：同时搜索知识图谱和向量记忆."""
        vector_results = self.vector_search(query, top_k)
        graph_results = self.query_graph(
            "MATCH (e:Entity) WHERE e.name CONTAINS $query RETURN e.id, e.name, e.type LIMIT $limit",
            {"query": query, "limit": top_k},
        )
        return {
            "vector_memories": vector_results,
            "graph_entities": graph_results,
        }
