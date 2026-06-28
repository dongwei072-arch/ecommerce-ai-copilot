<div align="center">

# 🛒 E-Commerce AI Copilot

基于大语言模型（LLM）与 Agentic Workflow 构建的企业级电商数据运营与风控工作台

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/Architecture-Multi--Agent-orange.svg" alt="Multi-Agent">
  <img src="https://img.shields.io/badge/Database-FAISS-brightgreen.svg" alt="FAISS">
</p>

[**项目简介**](#-项目简介) • [**系统架构**](#-系统架构) • [**核心特性**](#-核心特性) • [**项目结构**](#-项目结构) • [**效果展示**](#-效果展示) • [**快速上手**](#-快速上手) • [**配置说明**](#-配置说明)

</div>

---

## 📖 项目简介

在真实的电商数据中台中，直接让 LLM 编写 SQL 或计算数据往往面临极高的“幻觉”风险。**E-Commerce AI Copilot** 摒弃了传统的纯对话模式，采用 **“调度大脑与执行组件解耦”** 的 Multi-Agent 架构。

系统通过引入 **指标语义层 (Metric Store)** 与 **动态熔断机制**，实现“自然语言查询 -> 数据沙箱计算 -> 阈值风控阻断 -> SOP 知识库联动”的完整业务闭环。极大提升了电商运营报表分析、售后流程处理及业务决策的效率与准确性。

---

## 🧠 系统架构 (Workflow)

> **💡 提示**: 本系统采用 Planner (全局规划) + Reasoner (逻辑推理) 双脑架构，所有业务指标计算由底层硬编码的 Python 沙箱严格接管，实现 AI 与严谨业务数据的完美隔离。

┌──────────────────────────────────────────────────┐
│                  Web UI (Streamlit)              │
│                  main_app.py (:8501)             │
│          交互反馈 · 任务追踪 · 实时风控预警      │
└─────────────────────────┬────────────────────────┘
                          │ 异步事件流
┌─────────────────────────▼────────────────────────┐
│            智能体编排层 (Core Agents)            │
│  PlannerAgent (任务拆解) + ReasonerAgent (决策)  │
│        状态机管理 · 动态工具链调度 · 防幻觉      │
└───────┬─────────────────────────┬────────────────┘
        │ 调用工具                 │ 检索知识
┌───────▼──────────┐     ┌─────────▼───────────────┐
│  工具算子库      │     │ 知识库检索 (RAG)        │
│  Data/Chart/     │     │ FAISS 向量库 + SOP      │
│  Report Tool     │     │ (多平台售后规则)        │
└───────┬──────────┘     └─────────────────────────┘
        │ 交互
┌───────▼──────────────────────────────────────────┐
│  业务数据沙箱 (Sandbox)                          │
│  CSV/SQL 处理 · 指标计算 · 熔断判定              │
└──────────────────────────────────────────────────┘

---

## ✨ 核心特性

* 🤖 **双擎 Agent 架构 (ReAct)**
  * **Planner Agent**：意图路由与任务拆解。
  * **Reasoner Agent**：执行状态机，防死循环设计，精准编排工具链。
* 🛡️ **工业级动态风控熔断**
  * 内置时序漂移监控，拦截陈旧历史数据注入（防止 GIGO）。
  * 实时大盘监控（如退货率超标），触发异常后强制阻断常规对话，拉响红色预警。
* 📚 **SOP 知识库智能联动 (RAG)**
  * 触发熔断后，自动检索 FAISS 向量库中的企业应急预案（支持抖音、天猫、京东多平台售后规则）。
* 👁️ **全链路执行追踪 (Traceable UI)**
  * 基于 Streamlit 打造，告别黑盒。清晰展示 Agent 思考链 (Thought)、工具调用 (Tool Use) 与响应耗时。

---

## 📂 项目结构

```text
ecommerce-ai-copilot/
├── core/                   # Agent 核心逻辑层
│   ├── agents/             # 包含 PlannerAgent 与 ReasonerAgent
│   ├── tools/              # 工具算子 (Data, Chart, Knowledge, Report)
│   └── morning_report_agent.py # 晨报生成模块
├── data/                   # 本地数据源与知识库
│   ├── knowledge/          # Markdown 格式的各平台 SOP 文件及 FAISS 索引
│   └── *.csv               # 模拟电商明细与 RFM 画像数据
├── init_knowledge.py       # 向量数据库初始化脚本
├── main_app.py             # Streamlit 前端交互主入口
└── requirements.txt        # 依赖清单

```

---

## 📸 效果展示

### 1. 数据指标查询与可视化生成

### 2. 触发风控熔断与 SOP 自动检索

---

## 🛠️ 技术栈与系统要求

* **核心语言**: Python 3.9+ (兼容 Windows / macOS / Linux)
* **大模型服务**: 阿里云百炼 (兼容 OpenAI API 格式，默认适配 Qwen 模型)
* **关键依赖**: `Streamlit` (前端界面) / `FAISS` (向量检索) / `Pandas` (数据处理)

---

## 🚀 快速上手

### 1. 克隆与环境准备

建议使用虚拟环境（Virtual Environment）隔离项目依赖：

```bash
# 克隆项目
git clone [https://github.com/dongwei072-arch/ecommerce-ai-copilot.git](https://github.com/dongwei072-arch/ecommerce-ai-copilot.git)
cd ecommerce-ai-copilot

# 创建并激活虚拟环境 (Windows示例)
python -m venv .venv
.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

```

### 2. 初始化知识库

首次运行前，需要构建本地 SOP 知识库的 FAISS 索引：

```bash
python init_knowledge.py

```

*(注：执行成功后，`data/knowledge/` 目录下将生成向量索引文件 `sop_index.faiss`)*

### 3. 启动工作台

```bash
streamlit run main_app.py

```

> 服务启动后，浏览器将自动打开 `http://localhost:8501`。

---

## 🔧 配置说明

### API 配置

本项目运行主要依赖大语言模型服务，目前默认适配 **阿里云百炼 (DashScope)** 平台模型（兼容 OpenAI API 格式）。

* **配置方式**：在项目根目录新建 `.env` 文件，并输入您的 SK 键。

```env
DASHSCOPE_API_KEY=sk-你的_api_key_填写在这里

```

*(提示：您可以前往阿里云百炼控制台获取免费或付费的 API Key。)*

### 核心处理参数说明

核心系统参数硬编码在 `core/tools/data_tool.py` 等文件中，供有需要的开发者二次调整。

| 参数分类 | 参数说明 | 默认值 |
| --- | --- | --- |
| **风控阈值** | 触发熔断的退货率红线 | `15.0%` (可调) |
| **向量检索** | FAISS RAG 返回的文本片段数量 | `top_k=3` |
| **图表输出** | Plotly 可视化默认图表尺寸 | `800x400` |
| **数据沙箱** | 模拟读取 CSV 的限制行数 (防内存溢出) | 无限制 |

---

## 🧪 开发与扩展指南

* **热更新知识库**: 直接将新的业务线 Markdown 规则文件放入 `data/knowledge/platform/` 目录，重新运行 `init_knowledge.py` 即可。
* **接入真实数据库**: 核心数据算子位于 `core/tools/data_tool.py`，开发者可在此处替换模拟的 CSV 读取逻辑，直连 MySQL / ClickHouse / Doris 等真实业务库。

---

## 📄 许可证

本项目采用 **MIT 许可证**。详细信息请参阅 `LICENSE` 文件。

## 🤝 贡献

非常欢迎大家参与到这个项目中来！无论你是想修复 Bug、添加新的业务算子，还是优化现有逻辑，我们都期待你的声音。欢迎提交 **Issue** 和 **Pull Request**！
