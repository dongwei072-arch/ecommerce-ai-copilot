import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="E-Commerce AI Copilot", page_icon="⚙️", layout="wide")

st.title("⚙️ E-Commerce AI Copilot")
st.markdown("##### 专注电商运营与数据决策的效率工作台")
st.markdown("---")

with st.sidebar:
    st.markdown("### 🖥️ 系统状态")
    st.success("🟢 调度大脑：运行中")
    st.success("🟢 上下文缓存：已激活")
    st.markdown("---")
    st.markdown("### 🧰 核心算子")
    st.checkbox("📊 report_tool", value=True, disabled=True)
    st.checkbox("🧠 knowledge_tool", value=True, disabled=True)
    st.checkbox("📈 data_tool", value=True, disabled=True)
    st.checkbox("📉 chart_tool", value=True, disabled=True)

    if st.button("🗑️ 清空工作流缓存"):
        st.session_state.messages = []
        st.rerun()

# 1. 初始化业务上下文缓存
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "系统已就绪。请输入具体的运营指令或数据查询需求。"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 2. 工作台输入口
user_task = st.chat_input("输入运营指令（如：查询近7日GMV走势 / 检索退款异常SOP）")

if user_task:
    st.session_state.messages.append({"role": "user", "content": user_task})
    with st.chat_message("user"):
        st.markdown(user_task)

    # 提取最近上下文用于业务追问
    chat_history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-4:-1]])

    with st.chat_message("assistant"):
        status_container = st.container()

        with status_container:
            from core.agents.reasoner_agent import ReasonerAgent

            reasoner = ReasonerAgent()

            global_context = ""
            max_steps = 4
            step_count = 0

            with st.status("⚙️ 工作流执行中...", expanded=True) as status:
                while step_count < max_steps:
                    step_count += 1

                    full_goal = f"【历史业务上下文】:\n{chat_history_str}\n\n【最新指令】:\n{user_task}"

                    decision = reasoner.think(user_goal=full_goal, current_context=global_context)
                    st.info(f"⚡ **内部执行流**: {decision.get('thought')}")

                    next_tool = decision.get("next_tool")
                    tool_intent = decision.get("tool_intent")

                    if next_tool == "None" or not next_tool:
                        break

                    st.write(f"🛠️ **调用算子**: `{next_tool}` -> {tool_intent}")

                    if next_tool == "data_tool":
                        if 'data_tool_instance' not in st.session_state:
                            from core.tools.data_tool import DataTool

                            st.session_state.data_tool_instance = DataTool(raw_data_path="data/data.csv",
                                                                           rfm_data_path="data/rfm_results.csv")
                        result = st.session_state.data_tool_instance.query(tool_intent)
                        global_context += f"\n【DataTool 返回】:\n{result}\n"

                    elif next_tool == "knowledge_tool":
                        if 'knowledge_tool_instance' not in st.session_state:
                            from core.tools.knowledge_tool import KnowledgeTool

                            st.session_state.knowledge_tool_instance = KnowledgeTool()
                        result = st.session_state.knowledge_tool_instance.query(tool_intent)
                        global_context += f"\n【KnowledgeTool 返回】:\n{result}\n"

                    elif next_tool == "chart_tool":
                        if 'chart_tool_instance' not in st.session_state:
                            from core.tools.chart_tool import ChartTool

                            st.session_state.chart_tool_instance = ChartTool()
                        chart_result = st.session_state.chart_tool_instance.query(tool_intent)
                        if chart_result.get("type") == "line_chart":
                            st.line_chart(chart_result.get("raw_data"))
                            global_context += f"\n【ChartTool 渲染完毕】: {chart_result.get('text_summary')}\n"
                        else:
                            global_context += f"\n【ChartTool 报错】: 图表生成失败。\n"

                    elif next_tool == "report_tool":
                        global_context += "\n【ReportTool】: 请求生成晨报。\n"

                status.update(label="算子调用完毕", state="complete", expanded=False)

            # 3. 极简、冷酷的业务总结约束
            final_summary_prompt = f"""
            你是一个严格的电商运营数据终端。请基于【底层工具查询到的事实】和【历史业务上下文】，直接响应用户的【最新指令】。

            【底层事实】:
            {global_context}

            【最新指令】: "{user_task}"

            严格执行以下规则：
            1. 拒绝任何客套话、语气词（如“好的”、“没问题”、“Hi”），直接输出核心结论。
            2. 如果用户的指令与电商运营（数据分析、SOP规则、库存、大促等）无关，请直接回复：“⚠️ 错误：该指令超出运营工作台支持范围。”，不要尝试解答。
            3. 结论必须以 Bullet points（要点）形式列出，包含“数据发现”和“操作建议”。
            """

            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),
                            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

            response_placeholder = st.empty()
            full_response = ""

            for chunk in client.chat.completions.create(
                    model="qwen-plus",
                    messages=[{"role": "user", "content": final_summary_prompt}],
                    temperature=0.1,  # 极低温度，确保输出高度确定性和机械感
                    stream=True
            ):
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})