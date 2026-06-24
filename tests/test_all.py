"""Tests for Veritas Engine core components."""

from __future__ import annotations

import asyncio
import pytest

from veritas_engine.core.models import (
    PerceptionEvent,
    Entity,
    Strategy,
    EmotionalState,
    EventType,
    EntityType,
    StrategySource,
    StrategyStatus,
)
from veritas_engine.core.config import VeritasConfig


class TestModels:
    """测试核心数据模型."""

    def test_perception_event_creation(self):
        event = PerceptionEvent(
            source="test_sensor",
            event_type=EventType.DATA_CHANGE,
            payload={"value": 42},
        )
        assert event.source == "test_sensor"
        assert event.confidence == 1.0
        assert event.id is not None

    def test_entity_creation(self):
        entity = Entity(
            type=EntityType.DEVICE,
            name="TEMP_001",
            attributes={"location": "A1"},
        )
        assert entity.type == EntityType.DEVICE
        assert entity.confidence == 1.0

    def test_strategy_creation(self):
        strategy = Strategy(
            name="test_strategy",
            description="Test strategy",
            actions=[{"type": "config", "params": {"x": 1}}],
        )
        assert strategy.status == StrategyStatus.PENDING
        assert strategy.source == StrategySource.SELF_PLAY

    def test_emotional_state(self):
        state = EmotionalState(
            curiosity=0.8,
            urgency=0.5,
            frustration=0.1,
            achievement=0.3,
        )
        assert state.epsilon > 0
        assert state.exploration_bias > 0
        assert state.exploitation_bias > 0

    def test_value_evaluation(self):
        state = EmotionalState()
        metrics = {"efficiency": 0.8, "cost": 0.6, "risk": 0.4, "innovation": 0.7}
        value = state.evaluate_value(metrics)
        assert 0 <= value <= 1


class TestConfig:
    """测试配置管理."""

    def test_default_config(self):
        config = VeritasConfig()
        assert config.name == "Veritas Engine"
        assert config.version == "0.1.0-alpha"
        assert config.sensorium.nats_port == 4222

    def test_config_to_dict(self):
        config = VeritasConfig()
        data = config.to_dict()
        assert "name" in data
        assert "sensorium" in data


class TestSensorium:
    """测试感知层."""

    @pytest.mark.asyncio
    async def test_event_bus(self):
        from veritas_engine.sensorium.event_bus import EventBus

        bus = EventBus()
        await bus.start()

        received = []
        async def handler(event):
            received.append(event)

        sub_id = await bus.subscribe(["data_change"], handler)
        event = PerceptionEvent(source="test", event_type=EventType.DATA_CHANGE, payload={"v": 1})
        await bus.emit(event)
        await asyncio.sleep(0.1)

        assert len(received) == 1
        assert received[0].source == "test"

        await bus.unsubscribe(sub_id)
        await bus.stop()

    @pytest.mark.asyncio
    async def test_anomaly_detector(self):
        from veritas_engine.sensorium.anomaly_detector import AnomalyDetector

        detector = AnomalyDetector(zscore_threshold=2.0)
        for i in range(15):
            detector.update(50.0 + (i % 3))

        result = detector.update(200.0)
        assert result["is_anomaly"] is True
        assert result["method"] == "zscore"

    def test_normalizer(self):
        from veritas_engine.sensorium.normalizer import DataNormalizer

        event = DataNormalizer.normalize_human_command("test command", user="admin")
        assert event.source == "human"
        assert event.event_type == EventType.HUMAN_COMMAND


class TestNoosphere:
    """测试认知层."""

    def test_working_memory(self):
        from veritas_engine.noosphere.working_memory import WorkingMemory

        wm = WorkingMemory(max_size=5)
        wm.add(PerceptionEvent(source="test", payload={"v": 1}))
        wm.add(PerceptionEvent(source="test", payload={"v": 2}))

        assert len(wm.get_all()) == 2
        summary = wm.summary()
        assert summary["total_events"] == 2

    def test_memory_compressor(self):
        from veritas_engine.noosphere.memory_compressor import MemoryCompressor

        compressor = MemoryCompressor(threshold=0.75, target_ratio=0.30, protect_last_n=50)
        assert compressor.threshold == 0.75
        assert compressor.target_ratio == 0.30
        assert compressor.protect_last_n == 50

        # 少量记忆不应压缩
        assert compressor.should_compress(30) is False
        assert compressor.should_compress(100) is True

        memories = [
            {"content": f"Memory {i}", "source": "test", "embedding": [0.1] * 768}
            for i in range(60)
        ]
        entity = compressor.compress(memories)
        assert entity.attributes["original_count"] == 60
        assert entity.attributes["protected_count"] == 50

    def test_graph_db_mock(self):
        from veritas_engine.noosphere.graph_db import GraphDB

        graph = GraphDB()
        graph.connect()

        entity = Entity(type=EntityType.DEVICE, name="TEST_001", attributes={"temp": 25.5})
        eid = graph.create_entity(entity)
        assert eid is not None

        retrieved = graph.get_entity(eid)
        assert retrieved is not None
        assert retrieved.name == "TEST_001"


class TestHyperion:
    """测试推演层."""

    @pytest.mark.asyncio
    async def test_sandbox(self):
        from veritas_engine.hyperion.bootstrap import Sandbox

        sandbox = Sandbox()
        strategy = Strategy(name="test", description="Test", actions=[{"type": "noop"}])
        result = await sandbox.execute(strategy)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_pattern_miner(self):
        from veritas_engine.hyperion.bootstrap import PatternMiner

        miner = PatternMiner()
        data = [
            {"temp": 20, "efficiency": 0.8},
            {"temp": 25, "efficiency": 0.75},
            {"temp": 30, "efficiency": 0.6},
        ]
        patterns = await miner.mine(data, "efficiency")
        assert len(patterns) > 0
        assert any(p["type"] == "correlation" for p in patterns)

    @pytest.mark.asyncio
    async def test_exhaustive_search(self):
        from veritas_engine.hyperion.bootstrap import ExhaustiveExplorer

        explorer = ExhaustiveExplorer()
        config_space = [
            {"name": "a", "values": [1, 2]},
            {"name": "b", "values": [10, 20]},
        ]
        def evaluator(config):
            return config["a"] * 10 + config["b"]

        strategies = await explorer.search(config_space, evaluator)
        assert len(strategies) == 4
        assert strategies[0].expected_outcome["score"] == 40  # a=2, b=20


class TestDaemon:
    """测试情感层."""

    def test_curiosity_reward(self):
        from veritas_engine.daemon.bootstrap import Daemon

        daemon = Daemon()
        reward = daemon.compute_curiosity_reward(0.01)
        assert reward > 0
        assert daemon.state.curiosity > 0.5

    def test_urgency_discount(self):
        from veritas_engine.daemon.bootstrap import Daemon

        daemon = Daemon()
        discount = daemon.compute_urgency_discount(10.0)
        assert 0 < discount < 1

    def test_frustration_penalty(self):
        from veritas_engine.daemon.bootstrap import Daemon

        daemon = Daemon()
        penalty = daemon.compute_frustration_penalty(0.8, 0.5)
        assert penalty < 0

    def test_strategy_selection(self):
        from veritas_engine.daemon.bootstrap import Daemon

        daemon = Daemon()

        class MockStrategy:
            def __init__(self, score):
                self.expected_outcome = {"efficiency": score}

        strategies = [MockStrategy(0.9), MockStrategy(0.5), MockStrategy(0.7)]
        selected = daemon.select_strategy(strategies)
        assert selected is not None


class TestPrometheus:
    """测试执行层."""

    @pytest.mark.asyncio
    async def test_execution_pipeline(self):
        from veritas_engine.prometheus.bootstrap import ExecutionPipeline
        from veritas_engine.hyperion.bootstrap import Sandbox

        pipeline = ExecutionPipeline()
        sandbox = Sandbox()
        strategy = Strategy(name="test", description="Test", actions=[{"type": "noop"}])

        result = await pipeline.execute(strategy, sandbox.execute, require_hitl=False)
        assert result["success"] is True
        assert result["stage"] == "executed"

    def test_solidify_knowledge(self):
        from veritas_engine.prometheus.bootstrap import ExecutionPipeline

        pipeline = ExecutionPipeline()
        strategy = Strategy(name="test", description="Test", actions=[{"type": "noop"}])
        entity = pipeline.solidify_knowledge(strategy)
        assert entity.type.value == "proven_strategy"


class TestOracle:
    """测试元认知层."""

    @pytest.mark.asyncio
    async def test_goal_decomposition(self):
        from veritas_engine.oracle.bootstrap import GoalDecomposer

        decomposer = GoalDecomposer()
        plan = await decomposer.decompose("优化MES排产效率")
        assert len(plan) > 0
        assert plan[0]["id"] == "t1"

    @pytest.mark.asyncio
    async def test_reflection_audit(self):
        from veritas_engine.oracle.bootstrap import Reflector
        from veritas_engine.core.models import EmotionalState

        reflector = Reflector()
        strategies = []
        emotional = EmotionalState()
        wm_summary = {"total_events": 10}

        report = await reflector.conduct_audit(strategies, emotional, wm_summary)
        assert report.tasks_reviewed == 0

    def test_direction_controller(self):
        from veritas_engine.oracle.bootstrap import DirectionController

        controller = DirectionController()
        adjustments = controller.adjust({"success_rate": 0.2, "novelty": 0.05})
        assert "risk" in adjustments
        assert controller.get_direction() == "cautious"


class TestLLM:
    """测试LLM层."""

    def test_tool_registry(self):
        from veritas_engine.llm.ollama_client import ToolRegistry

        registry = ToolRegistry()

        def add_tool(x: int, y: int) -> int:
            return x + y

        registry.register(
            name="add",
            description="Add two numbers",
            parameters={"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}},
            handler=add_tool,
        )

        assert "add" in registry.list_tools()
        assert len(registry.get_tools()) == 1

    @pytest.mark.asyncio
    async def test_tool_call(self):
        from veritas_engine.llm.ollama_client import ToolRegistry

        registry = ToolRegistry()
        registry.register(
            name="multiply",
            description="Multiply",
            parameters={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
            handler=lambda a, b: a * b,
        )

        result = await registry.call("multiply", {"a": 3, "b": 4})
        assert result == 12


class TestEngine:
    """测试核心引擎."""

    @pytest.mark.asyncio
    async def test_engine_lifecycle(self):
        from veritas_engine.core.engine import Engine

        engine = Engine()
        await engine.start()
        assert engine._running is True

        status = await engine.get_status()
        assert status.layer_status["sensorium"] == "running"

        await engine.stop()
        assert engine._running is False

    @pytest.mark.asyncio
    async def test_run_goal(self):
        from veritas_engine.core.engine import Engine

        engine = Engine()
        await engine.start()
        try:
            result = await engine.run_goal("测试目标")
            assert result["success"] is True
            assert "strategy_id" in result
            assert "emotional_state" in result
        finally:
            await engine.stop()
