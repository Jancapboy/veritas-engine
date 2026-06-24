"""Kuzu graph database manager for Veritas Engine.

Kuzu is an embedded graph database (no server needed).
We use its Python API directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from veritas_engine.core.models import Entity, Observation
from veritas_engine.core.config import get_config
from veritas_engine.core.logger import get_logger
from veritas_engine.core.exceptions import GraphDBError

logger = get_logger("veritas.noosphere")


class GraphDB:
    """Kuzu 知识图谱管理器."""

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or get_config().noosphere.kuzu_db_path
        self._conn = None
        self._db = None

    def connect(self) -> None:
        """连接/初始化 Kuzu 数据库."""
        try:
            import kuzu
        except ImportError:
            logger.warning("kuzu not installed, using mock graph DB")
            self._mock = True
            self._mock_entities: dict[str, dict] = {}
            self._mock_relations: list[dict] = []
            return

        self._mock = False
        Path(self.db_path).mkdir(parents=True, exist_ok=True)
        self._db = kuzu.Database(self.db_path)
        self._conn = kuzu.Connection(self._db)
        self._init_schema()
        logger.info("GraphDB connected to %s", self.db_path, extra={"layer": "noosphere"})

    def close(self) -> None:
        """关闭数据库连接."""
        if not self._mock and self._db:
            # Kuzu doesn't have explicit close, but we clear references
            self._conn = None
            self._db = None

    def _init_schema(self) -> None:
        """初始化图谱 Schema."""
        if self._mock:
            return
        from veritas_engine.core.constants import KUZU_SCHEMA_CYPHER
        for statement in KUZU_SCHEMA_CYPHER.split(";"):
            stmt = statement.strip()
            if stmt and stmt.startswith("CREATE"):
                try:
                    self._conn.execute(stmt)
                except Exception as e:
                    # Schema may already exist
                    logger.debug("Schema init: %s", e, extra={"layer": "noosphere"})

    def create_entity(self, entity: Entity) -> str:
        """创建实体节点，返回实体ID."""
        if self._mock:
            self._mock_entities[entity.id] = entity.model_dump()
            return entity.id

        props = entity.to_cypher_properties()
        cypher = """
            CREATE (e:Entity {
                id: $id,
                type: $type,
                name: $name,
                attributes: $attributes,
                created_at: $created_at,
                confidence: $confidence
            })
            RETURN e.id
        """
        result = self._conn.execute(cypher, {
            "id": props["id"],
            "type": props["type"],
            "name": props["name"],
            "attributes": props["attributes"],
            "created_at": props["created_at"],
            "confidence": props["confidence"],
        })
        row = result.get_next()
        return row[0] if row else props["id"]

    def get_entity(self, entity_id: str) -> Entity | None:
        """按ID获取实体."""
        if self._mock:
            data = self._mock_entities.get(entity_id)
            if data:
                return Entity(**data)
            return None

        cypher = "MATCH (e:Entity {id: $id}) RETURN e"
        result = self._conn.execute(cypher, {"id": entity_id})
        row = result.get_next()
        if not row:
            return None
        node = row[0]
        return Entity(
            id=node["id"],
            type=node["type"],
            name=node["name"],
            attributes=json.loads(node["attributes"]) if isinstance(node["attributes"], str) else node["attributes"],
            confidence=node["confidence"],
        )

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """执行 Cypher 查询."""
        if self._mock:
            return []

        result = self._conn.execute(cypher, params or {})
        rows = []
        while result.has_next():
            row = result.get_next()
            rows.append({f"col_{i}": v for i, v in enumerate(row)})
        return rows

    def create_relation(
        self, from_id: str, to_id: str, rel_type: str, properties: dict[str, Any] | None = None
    ) -> None:
        """创建实体间关系."""
        if self._mock:
            self._mock_relations.append({
                "from": from_id, "to": to_id, "type": rel_type, "props": properties or {}
            })
            return

        props = properties or {}
        prop_str = ", ".join([f"{k}: ${k}" for k in props.keys()])
        if prop_str:
            prop_str = ", " + prop_str

        cypher = f"""
            MATCH (a:Entity {{id: $from_id}}), (b:Entity {{id: $to_id}})
            CREATE (a)-[:{rel_type}{{{prop_str}}}]->(b)
        """
        params = {"from_id": from_id, "to_id": to_id, **props}
        self._conn.execute(cypher, params)

    def find_path(self, from_id: str, to_id: str, max_depth: int = 5) -> list[list[str]]:
        """查找两实体间的路径."""
        if self._mock:
            return []

        cypher = """
            MATCH path = (a:Entity {id: $from_id})-[:CAUSES|DEPENDS_ON|SIMILAR_TO*1..$max_depth]->(b:Entity {id: $to_id})
            RETURN [node in nodes(path) | node.id] AS path_ids
        """
        result = self._conn.execute(cypher, {"from_id": from_id, "to_id": to_id, "max_depth": max_depth})
        paths = []
        while result.has_next():
            row = result.get_next()
            paths.append(row[0])
        return paths
