"""Noosphere package initialization."""

from __future__ import annotations

from veritas_engine.noosphere.bootstrap import Noosphere
from veritas_engine.noosphere.graph_db import GraphDB
from veritas_engine.noosphere.vector_db import VectorMemory
from veritas_engine.noosphere.working_memory import WorkingMemory
from veritas_engine.noosphere.memory_compressor import MemoryCompressor

__all__ = [
    "Noosphere",
    "GraphDB",
    "VectorMemory",
    "WorkingMemory",
    "MemoryCompressor",
]
