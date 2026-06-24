"""Constants used across Veritas Engine."""

from __future__ import annotations

# ── 系统 ──
VERSION = "0.1.0-alpha"
SYSTEM_NAME = "Veritas Engine"
SYSTEM_NAME_CN = "真理引擎"

# ── 认知闭环步骤 ──
COGNITION_LOOP = [
    "感知现实",
    "推演规律",
    "情感驱动",
    "试错执行",
    "验证固化",
    "反哺认知",
]

# ── 六层架构 ──
LAYERS = {
    "sensorium": {"name_en": "Sensorium", "name_cn": "感知层", "color": "#00F0FF"},
    "noosphere": {"name_en": "Noosphere", "name_cn": "认知层", "color": "#8B5CF6"},
    "hyperion": {"name_en": "Hyperion", "name_cn": "推演层", "color": "#FFD700"},
    "daemon": {"name_en": "Daemon", "name_cn": "情感层", "color": "#FF2E9A"},
    "prometheus": {"name_en": "Prometheus", "name_cn": "执行层", "color": "#10B981"},
    "oracle": {"name_en": "Oracle", "name_cn": "元认知层", "color": "#FFD700"},
}

# ── 情感公式 ──
EMOTION_FORMULAS = {
    "curiosity": "reward = -log(p)",
    "urgency": "discount = e^(-λt)",
    "frustration": "penalty = -|expected - actual|",
    "achievement": "bonus = Σ(confirmed_knowledge)",
    "value": "V = w₁×efficiency + w₂×cost + w₃×risk + w₄×innovation",
}

# ── 数据库 ──
KUZU_SCHEMA_CYPHER = """
// 创建实体节点表
CREATE NODE TABLE Entity (
    id STRING PRIMARY KEY,
    type STRING,
    name STRING,
    attributes STRING,
    created_at TIMESTAMP,
    confidence FLOAT
);

// 创建观测节点表
CREATE NODE TABLE Observation (
    id STRING PRIMARY KEY,
    raw_data STRING,
    source STRING,
    timestamp TIMESTAMP
);

// 创建关系表
CREATE REL TABLE HAS_OBSERVATION (
    FROM Entity TO Observation,
    MANY_MANY
);

CREATE REL TABLE CAUSES (
    FROM Entity TO Entity,
    strength FLOAT,
    evidence_count INT
);

CREATE REL TABLE SIMILAR_TO (
    FROM Entity TO Entity,
    score FLOAT
);

CREATE REL TABLE DEPENDS_ON (
    FROM Entity TO Entity,
    MANY_MANY
);
"""

# ── 沙盒 ──
SANDBOX_SQL_BLACKLIST = ["DROP", "DELETE", "TRUNCATE"]
SANDBOX_SQL_WHITELIST = ["SELECT", "INSERT", "UPDATE"]

# ── 默认超时 ──
DEFAULT_TIMEOUTS = {
    "sandbox": 30,
    "llm": 120,
    "tool_call": 60,
    "hitl": 3600,
    "mcp": 30,
}

# ── 里程碑 ──
MILESTONES = [
    {
        "phase": 1,
        "title": "认知基建",
        "duration_weeks": 2,
        "color": "#00F0FF",
        "tasks": [
            "搭建Kuzu知识图谱Schema，导入现有MES表结构元数据",
            "实现感知层：连接MSSQL变更追踪 + Node-RED日志流",
            "实现基础MCP Server：数据库查询、文件读取",
            "接入Ollama，完成本地LLM工具调用链路",
        ],
    },
    {
        "phase": 2,
        "title": "推演引擎",
        "duration_weeks": 3,
        "color": "#8B5CF6",
        "tasks": [
            "实现沙盒环境（SQLite内存副本 + Firejail）",
            "实现穷举探索器（SSRS报表布局优化POC）",
            "实现数据规律挖掘器（TabNet + 符号回归）",
            "实现自博弈引擎v1（单Agent排产参数优化）",
        ],
    },
    {
        "phase": 3,
        "title": "情感与闭环",
        "duration_weeks": 2,
        "color": "#FF2E9A",
        "tasks": [
            "实现价值函数和好奇心机制",
            "实现试错-验证-固化流水线",
            "实现HITL确认机制（飞书消息推送）",
            "实现元认知审计（每日自动反思报告）",
        ],
    },
    {
        "phase": 4,
        "title": "多Agent与开放",
        "duration_weeks": 3,
        "color": "#FFD700",
        "tasks": [
            "多Agent对抗/协作（推演层支持2+ Agent）",
            "开放外部MCP Server接入（GitHub、Slack等）",
            "知识图谱可视化（Web UI查看认知网络）",
            "文档与内部培训",
        ],
    },
]

# ── 成功标准 ──
SUCCESS_CRITERIA = [
    {
        "title": "自主性",
        "description": "连续72小时无需人类干预，自动完成「监控→发现异常→推演→验证→固化」闭环",
    },
    {
        "title": "发现力",
        "description": "在MES数据中至少发现1条人类专家未意识到的规律（如隐藏的设备关联性）",
    },
    {
        "title": "可解释性",
        "description": "任何决策都能在3步内追溯到知识图谱中的证据链",
    },
    {
        "title": "可复制性",
        "description": "新部署一套真理引擎，导入知识图谱快照，24小时内达到原系统80%的认知水平",
    },
]
