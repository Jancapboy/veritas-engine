"""Memory compressor — compresses multiple vector memories into graph entities."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from veritas_engine.core.models import Entity, EntityType
from veritas_engine.core.logger import get_logger

logger = get_logger("veritas.noosphere")


class MemoryCompressor:
    """记忆压缩器——将多条向量记忆压缩为知识图谱实体.

    参数:
        threshold: 语义相似度阈值（0-1），高于此值的记忆会被聚类压缩
        target_ratio: 压缩目标比例，压缩后保留的记忆数量比例
        protect_last_n: 保护最近 N 条记忆不被压缩
    """

    def __init__(self, threshold: float = 0.75, target_ratio: float = 0.30, protect_last_n: int = 50) -> None:
        self.threshold = threshold
        self.target_ratio = target_ratio
        self.protect_last_n = protect_last_n

    def should_compress(self, memory_count: int) -> bool:
        """判断是否需要压缩——当记忆数量超过保护数量时触发."""
        return memory_count > self.protect_last_n

    def compress(self, memories: list[dict[str, Any]]) -> Entity:
        """压缩记忆列表为知识实体.

        压缩策略:
        1. 保护最近 protect_last_n 条记忆
        2. 对剩余记忆按语义相似度聚类（threshold=0.75）
        3. 每个聚类压缩为一个知识实体
        4. 目标压缩比例 target_ratio=0.30

        Args:
            memories: 向量记忆列表，每项包含 content, metadata, source, embedding 等

        Returns:
            压缩后的知识图谱实体
        """
        if not memories:
            return Entity(
                type=EntityType.CONCEPT,
                name="empty_compression",
                attributes={"summary": "No memories to compress"},
            )

        total = len(memories)
        # 保护最近 N 条
        protected = memories[-self.protect_last_n:] if total > self.protect_last_n else memories
        to_compress = memories[:-self.protect_last_n] if total > self.protect_last_n else []

        if not to_compress:
            return Entity(
                type=EntityType.CONCEPT,
                name="no_compression_needed",
                attributes={
                    "summary": f"Only {total} memories, all protected (protect_last_n={self.protect_last_n})",
                    "memory_count": total,
                    "protected_count": len(protected),
                },
            )

        # 计算目标压缩后数量
        target_count = max(int(total * self.target_ratio), self.protect_last_n)
        clusters_to_create = max(1, target_count - len(protected))

        # 简单聚类：按 embedding 余弦相似度分组
        clusters = self._cluster_memories(to_compress, clusters_to_create)

        # 为每个聚类生成摘要
        all_topics = []
        all_sources = set()
        cluster_summaries = []
        for cluster in clusters:
            contents = [m.get("content", "") for m in cluster]
            sources = set(m.get("source", "unknown") for m in cluster)
            topics = self._extract_topics(contents)
            all_topics.extend(topics)
            all_sources.update(sources)
            cluster_summaries.append({
                "count": len(cluster),
                "topics": topics[:5],
                "sample": contents[0][:100] if contents else "",
            })

        summary = (
            f"Compressed {total} memories -> {len(protected)} protected + {len(clusters)} clusters. "
            f"Target ratio: {self.target_ratio}, Threshold: {self.threshold}. "
            f"Sources: {', '.join(all_sources)}."
        )

        entity = Entity(
            type=EntityType.CONCEPT,
            name=f"compressed_memory_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            attributes={
                "summary": summary,
                "original_count": total,
                "protected_count": len(protected),
                "cluster_count": len(clusters),
                "target_ratio": self.target_ratio,
                "threshold": self.threshold,
                "protect_last_n": self.protect_last_n,
                "topics": list(set(all_topics))[:10],
                "sources": list(all_sources),
                "cluster_summaries": cluster_summaries,
                "compressed_at": datetime.now(timezone.utc).isoformat(),
            },
            confidence=0.8,
        )

        logger.info(
            "Compressed %d memories -> %d protected + %d clusters (target_ratio=%.2f, threshold=%.2f)",
            total,
            len(protected),
            len(clusters),
            self.target_ratio,
            self.threshold,
            extra={"layer": "noosphere"},
        )
        return entity

    def _cluster_memories(self, memories: list[dict[str, Any]], cluster_count: int) -> list[list[dict[str, Any]]]:
        """按语义相似度聚类记忆.

        简单实现：使用贪心聚类，将相似记忆分到同一组。
        """
        if not memories:
            return []
        if len(memories) <= cluster_count:
            return [[m] for m in memories]

        clusters: list[list[dict[str, Any]]] = []
        unassigned = list(memories)

        # 计算两两相似度
        def _similarity(a: dict, b: dict) -> float:
            emb_a = a.get("embedding", [])
            emb_b = b.get("embedding", [])
            if not emb_a or not emb_b or len(emb_a) != len(emb_b):
                # 无 embedding 时按内容 Jaccard 相似度
                set_a = set(a.get("content", "").split())
                set_b = set(b.get("content", "").split())
                inter = len(set_a & set_b)
                union = len(set_a | set_b)
                return inter / union if union > 0 else 0.0
            # 余弦相似度
            dot = sum(x * y for x, y in zip(emb_a, emb_b))
            mag_a = sum(x * x for x in emb_a) ** 0.5
            mag_b = sum(x * x for x in emb_b) ** 0.5
            return dot / (mag_a * mag_b) if mag_a * mag_b > 0 else 0.0

        while unassigned and len(clusters) < cluster_count:
            # 以第一个未分配记忆为种子创建新聚类
            seed = unassigned[0]
            cluster = [seed]
            unassigned.remove(seed)

            # 贪婪添加相似记忆
            to_add = []
            for m in unassigned:
                sim = _similarity(seed, m)
                if sim >= self.threshold:
                    to_add.append((m, sim))

            # 按相似度排序，优先添加最相似的
            to_add.sort(key=lambda x: x[1], reverse=True)
            for m, _ in to_add:
                if m in unassigned:
                    cluster.append(m)
                    unassigned.remove(m)

            clusters.append(cluster)

        # 剩余未分配的记忆各自成类或合并到最后一个聚类
        if unassigned:
            clusters.append(unassigned)

        return clusters

    def _extract_topics(self, contents: list[str]) -> list[str]:
        """从内容中提取主题关键词（简单实现）."""
        from collections import Counter
        import re

        # 提取所有中文和英文单词
        words = []
        for text in contents:
            # 英文单词
            words.extend(re.findall(r'\b[a-zA-Z]{4,}\b', text.lower()))
            # 中文词汇（简单按2-4字切分）
            words.extend(re.findall(r'[\u4e00-\u9fff]{2,4}', text))

        # 过滤停用词
        stopwords = {"this", "that", "with", "from", "have", "been", "will", "would", "could", "should"}
        filtered = [w for w in words if w not in stopwords]

        counter = Counter(filtered)
        return [word for word, count in counter.most_common(10)]
