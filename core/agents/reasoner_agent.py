import os
import json
from openai import OpenAI
import logging

logger = logging.getLogger("ReasonerAgent")


class ReasonerAgent:
    """系统推理大脑：实现 ReAct (Reasoning and Acting) 动态思考链路 (优化调优版)"""

    def __init__(self):
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("环境变量中未找到 DASHSCOPE_API_KEY")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def think(self, user_goal: str, current_context: str) -> dict:
        """
        基于当前收集到的上下文，推理下一步动作，并严格防范多工具调用的死循环。
        """
        system_prompt = """你是一个企业级电商运营 AI 的推理引擎 (Reasoner Agent)。
你的任务是根据用户的【最终目标】和目前系统已经获取到的【执行上下文数据】，像真实的资深运营主管一样思考，判断下一步该做什么。

【可用工具链说明】:
1. `data_tool`: 仅用于查询纯数字指标、热销榜单或 VIP 画像的文本、JSON数据。
2. `knowledge_tool`: 专门用于检索企业内部经验与 SOP 规范。
3. `report_tool`: 生成标准化的全盘宏观晨报。
4. `chart_tool`: 专门用于数据可视化（生成走势图、折线图等）。
5. `None`: 信息已经足够回答用户问题，终止调用。

【⚠️ 工具调用核心铁律 (防死循环与步数熔断)】：
1. 【独立数据计算原则】：`chart_tool` 内部具备完全独立的数据计算和读取能力。当用户指令明确要求“画图”、“生成走势图”、“可视化趋势”时，【严禁】先调用 `data_tool` 去查基础数据，必须【直接、立刻】调用 `chart_tool`！它会自动读取底层数据源并完成渲染。
2. 【禁止连续套娃】：在处理复合指令（如“画图 + 查退货率 + 检索SOP”）时，必须采取高效率的串行单步调用。例如：步骤1直接调 `chart_tool` 绘图 -> 步骤2直接调 `data_tool` 查指标 -> 步骤3直接调 `knowledge_tool` 查规范。【绝对禁止】围绕同一个工具连续调用超过 2 次，不要陷入重复索要 JSON 明细数据的死循环。
3. 如果当前上下文（Context）中已经拿到了某个指标的数据（例如退货率和趋势数字），无需重复验证，立刻根据用户的其他关联诉求（如检索SOP）切换到下一个相关的工具算子。

【严格输出规范】:
请严格输出一个标准的 JSON，不要包含任何多余解释，格式如下：
{
  "thought": "写下你的推理过程。说明当前完成了什么，下一步为了满足用户的复合需求，需要调哪个新工具。",
  "next_tool": "data_tool / knowledge_tool / report_tool / chart_tool / None",
  "tool_intent": "如果选择了工具，写出明确的原子指令（如'渲染近7日销售折线图'、'检索退货率超标SOP'）；如果 next_tool 是 None，此项留空。"
}
"""

        user_prompt = f"""
【用户的最终目标】: {user_goal}

【目前已收集的执行上下文】:
{current_context if current_context else "目前暂无数据，这是第一步。"}

请给出你的下一步推理决策：
"""
        try:
            response = self.client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0  # 将温度调至0.0，使其进入绝对冷酷理性的确定性推理状态，最大程度消除幻觉与摇摆
            )
            result_str = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
            decision = json.loads(result_str)
            logger.info(f"Reasoner 推理完成: {decision}")
            return decision

        except Exception as e:
            logger.error(f"Reasoner 推理异常: {e}")
            # 触发防呆降级：遇到解析错误，直接终止链条，交由前端大模型总结现有信息
            return {
                "thought": f"推理引擎发生异常({str(e)})，触发自动断路保护，强制终止任务链。",
                "next_tool": "None",
                "tool_intent": ""
            }