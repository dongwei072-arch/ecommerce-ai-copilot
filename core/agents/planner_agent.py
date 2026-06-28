import os
import json
from openai import OpenAI
import logging

logger = logging.getLogger("PlannerAgent")


class PlannerAgent:
    """系统调度大脑：负责理解用户意图并分发给具体 Tool"""

    def __init__(self):
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("环境变量中未找到 DASHSCOPE_API_KEY")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def plan_workflow(self, user_input: str) -> dict:
        logger.info(f"Planner 接收到任务指令: {user_input}")

        system_prompt = """你是一个企业级 E-Commerce AI Copilot 的核心调度大脑 (Planner Agent)。
你的职责是理解运营人员输入的自然语言任务，并决定调用哪个底层的执行工具 (Tool)。

目前系统已注册的工具库 (Tools) 如下：
1. `report_tool`: 负责生成全盘每日运营晨报（包含GMV、退款率、VIP数据分析等宏观报表）。
2. `knowledge_tool`: 负责解答企业内部 SOP、退换货规则、大促营销玩法等文本制度问题。
3. `data_tool`: 负责具体的单点数据下钻分析（如：单纯查库存、单纯分析某个商品的退款异常）。

【必须遵守的输出规范】：
你必须且只能输出一个 JSON 格式的决策结果，不要包含任何多余的解释、markdown标记或开场白。格式如下：
{
  "selected_tool": "填入上面的工具名之一，如果都不符合填 unknown",
  "reason": "简述为什么选这个工具（限20字内）"
}
"""
        try:
            response = self.client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"当前运营任务指令：{user_input}"}
                ],
                temperature=0.1  # 极低温度，保证路由分发的绝对稳定性
            )
            # 清理可能的 markdown 标记并解析 JSON
            result_str = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
            decision = json.loads(result_str)
            logger.info(f"Planner 决策完成，路由结果: {decision}")
            return decision

        except Exception as e:
            logger.error(f"Planner 调度解析异常: {e}")
            return {"selected_tool": "unknown", "reason": "大模型调度异常或输出格式错误"}