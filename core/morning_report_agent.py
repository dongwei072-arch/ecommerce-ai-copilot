import pandas as pd
from openai import OpenAI
import datetime
import logging
import os
import json
from dotenv import load_dotenv

# 基础日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MorningReportAgent")


class DataEngine:
    """数据引擎：负责底层业务数据清洗与核心指标计算"""

    def __init__(self, raw_data_path: str, rfm_data_path: str):
        self.raw_data_path = raw_data_path
        self.rfm_data_path = rfm_data_path

    def calculate_daily_metrics(self) -> dict:
        """抽取大盘数据，计算核心运营指标"""
        logger.info(f"开始加载底层业务流数据: {self.raw_data_path}")
        try:
            # 1. 交易流水清洗
            df_raw = pd.read_csv(self.raw_data_path, encoding='ISO-8859-1')
            df_raw['InvoiceDate'] = pd.to_datetime(df_raw['InvoiceDate'])
            df_raw['TotalAmount'] = df_raw['Quantity'] * df_raw['UnitPrice']

            # 获取最新交易日作为T-1基准
            max_date = df_raw['InvoiceDate'].max()
            target_date = max_date.date()

            df_daily = df_raw[df_raw['InvoiceDate'].dt.date == target_date]

            # 分离正向销售与逆向退款
            sales_df = df_daily[df_daily['Quantity'] > 0]
            refunds_df = df_daily[df_daily['Quantity'] < 0]

            daily_gmv = sales_df['TotalAmount'].sum()
            daily_orders = sales_df['InvoiceNo'].nunique()
            refund_amount = abs(refunds_df['TotalAmount'].sum())
            refund_rate = (refund_amount / daily_gmv) if daily_gmv > 0 else 0

            # 2. 融合 RFM 模型数据评估私域健康度
            logger.info("关联 RFM 分层模型数据")
            df_rfm = pd.read_csv(self.rfm_data_path, encoding='utf-8-sig')

            vip_count = len(df_rfm[df_rfm['Customer_Label'].str.contains('VIP', na=False)])
            risk_count = len(df_rfm[df_rfm['Customer_Label'].str.contains('挽留', na=False)])

            metrics = {
                "report_date": str(target_date),
                "daily_gmv": round(daily_gmv, 2),
                "daily_orders": daily_orders,
                "refund_rate": f"{round(refund_rate * 100, 2)}%",
                "customer_health": {
                    "vip_users": vip_count,
                    "at_risk_users": risk_count
                }
            }
            logger.info("核心指标提取完成")
            return metrics

        except Exception as e:
            logger.error(f"DataEngine 运行异常: {str(e)}")
            raise e


class MorningReportCopilot:
    """决策副驾：基于业务指标生成运营洞察"""

    def __init__(self, api_key: str = None):
        # 生产环境走环境变量获取密钥
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("Missing DASHSCOPE_API_KEY in environment variables")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def generate_report(self, business_metrics: dict) -> str:
        """调用 LLM 生成业务战报"""
        logger.info("请求 Qwen 引擎进行深度分析")

        system_prompt = """你是一个顶级的 AI 商业运营副驾 (Business Copilot)。
你的任务是将干瘪的数据库统计指标，翻译为极具商业洞察的【每日运营晨报】。
你的语气必须像一位资深的数据分析专家，客观、尖锐且注重行动落地（Action-oriented）。"""

        user_prompt = f"""
请基于底层 DataEngine 提取的今日业务核心指标，生成一份 Markdown 格式的晨报。

【输入指标 JSON】：
{json.dumps(business_metrics, ensure_ascii=False, indent=2)}

【输出排版要求（必须严格遵守）】：
1. 报表标题：`# 📊 AI 智能运营晨报 ({business_metrics['report_date']})`
2. **核心数据速览**：用 Markdown 的引用语法（>）简述 GMV、订单数和退款率。如果退款率大于 5%，请标记 ⚠️ 警告。
3. **客群健康度诊断**：分析 VIP 客户与流失风险客户的比例，判断当前私域流量池的健康状态。
4. **Agent 自动执行建议**：给出 3 条今天最应该做的运营 SOP 操作。

限制：不要任何开场白，直接输出 Markdown 报告主体。
"""
        try:
            response = self.client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3  # 降低温度控制幻觉
            )
            report_content = response.choices[0].message.content
            logger.info("晨报生成完毕")
            return report_content

        except Exception as e:
            logger.error(f"LLM API 请求失败: {str(e)}")
            raise e


# ==========================================
# 本地联调测试入口
# ==========================================
if __name__ == "__main__":
    # 加载环境变量
    load_dotenv()

    # 定义测试数据路径
    RAW_DATA = '../data/data.csv'
    RFM_DATA = '../data/rfm_results.csv'

    try:
        # 初始化数据引擎
        engine = DataEngine(raw_data_path=RAW_DATA, rfm_data_path=RFM_DATA)
        metrics = engine.calculate_daily_metrics()

        # 初始化 AI 副驾并生成报告
        copilot = MorningReportCopilot()
        report = copilot.generate_report(metrics)

        print("\n" + "=" * 50)
        print(report)
        print("=" * 50 + "\n")

    except Exception as err:
        logger.error(f"联调测试崩溃: {err}")