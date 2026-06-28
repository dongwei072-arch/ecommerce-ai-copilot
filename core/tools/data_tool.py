import pandas as pd
import json
import logging
import os
from openai import OpenAI

logger = logging.getLogger("DataTool")


class DataTool:
    """指标语义层数据引擎：基于配置驱动的企业级数据计算与风控中心"""

    def __init__(self, raw_data_path: str = "data/data.csv", rfm_data_path: str = "data/rfm_results.csv"):
        self.raw_data_path = raw_data_path
        self.rfm_data_path = rfm_data_path

        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("未找到 DASHSCOPE_API_KEY 环境变量")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        self._load_data()
        self._register_metric_store()

    def _load_data(self):
        """装载底层数据"""
        logger.info(f"DataTool 正在装载底层流水...")
        try:
            self.df = pd.read_csv(self.raw_data_path, encoding='ISO-8859-1')
            if 'InvoiceDate' in self.df.columns:
                self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
            if 'Quantity' in self.df.columns and 'UnitPrice' in self.df.columns:
                self.df['TotalAmount'] = self.df['Quantity'] * self.df['UnitPrice']

            if os.path.exists(self.rfm_data_path):
                self.rfm_df = pd.read_csv(self.rfm_data_path)
            else:
                self.rfm_df = pd.DataFrame()
        except Exception as e:
            logger.error(f"数据装载失败: {e}")
            self.df = pd.DataFrame()
            self.rfm_df = pd.DataFrame()

    def _register_metric_store(self):
        """建立企业级指标语义层 (Semantic Metric Store)"""
        self.metric_store = {
            "top_refunds": {
                "name": "高退款商品预警",
                "calc": self._calc_top_refunds
            },
            "top_sales": {
                "name": "热销商品榜单",
                "calc": self._calc_top_sales
            },
            "sales_trend": {
                "name": "近7日大盘GMV走势",
                "calc": self._calc_sales_trend
            },
            "return_rate": {
                "name": "大盘整体退货率",
                "calc": self._calc_return_rate
            },
            "atv": {
                "name": "大盘客单价(ATV)",
                "calc": self._calc_atv
            },
            "vip_churn": {
                "name": "VIP高风险流失名单",
                "calc": self._calc_vip_churn
            }
        }

    # ================= 指标计算公式定义区 =================

    def _calc_top_refunds(self):
        refunds = self.df[self.df['Quantity'] < 0].copy()
        refunds['RefundAmount'] = abs(refunds['TotalAmount'])
        top_refunds = refunds.groupby('Description')['RefundAmount'].sum().sort_values(ascending=False).head(5)
        return [{"商品名称": str(k), "退款金额": round(v, 2)} for k, v in top_refunds.items()]

    def _calc_top_sales(self):
        sales = self.df[self.df['Quantity'] > 0]
        top_sales = sales.groupby('Description')['TotalAmount'].sum().sort_values(ascending=False).head(5)
        return [{"商品名称": str(k), "销售金额": round(v, 2)} for k, v in top_sales.items()]

    def _calc_sales_trend(self):
        sales = self.df[self.df['Quantity'] > 0].copy()
        sales['Date'] = sales['InvoiceDate'].dt.date
        trend = sales.groupby('Date')['TotalAmount'].sum().sort_index().tail(7)
        return [{"日期": str(k), "GMV": round(v, 2)} for k, v in trend.items()]

    def _calc_return_rate(self):
        """带熔断阈值的退货率计算"""
        total_orders = self.df[self.df['Quantity'] > 0]['InvoiceNo'].nunique()
        refund_orders = self.df[self.df['Quantity'] < 0]['InvoiceNo'].nunique()
        if total_orders == 0: return {"error": "总订单量为0"}

        rate_value = (refund_orders / total_orders) * 100
        threshold = 8.0
        is_alert = rate_value >= threshold

        return {
            "总单量": int(total_orders),
            "退单量": int(refund_orders),
            "退货率": f"{round(rate_value, 2)}%",
            "系统状态": f"🚨【风控警报】：退货率已突破 {threshold}% 红线！" if is_alert else "🟢 健康"
        }

    def _calc_atv(self):
        """带熔断阈值的客单价计算"""
        sales = self.df[self.df['Quantity'] > 0]
        total_gmv = sales['TotalAmount'].sum()
        total_orders = sales['InvoiceNo'].nunique()
        if total_orders == 0: return {"error": "有效订单为0"}

        atv_value = total_gmv / total_orders
        threshold = 50.0
        is_alert = atv_value < threshold

        return {
            "总GMV": round(total_gmv, 2),
            "有效单量": int(total_orders),
            "客单价": round(atv_value, 2),
            "系统状态": f"🚨【风控警报】：客单价低于 {threshold} 元警戒线！" if is_alert else "🟢 健康"
        }

    def _calc_vip_churn(self):
        if self.rfm_df.empty: return {"error": "RFM源为空"}
        if 'Monetary' in self.rfm_df.columns and 'Frequency' in self.rfm_df.columns:
            m_th, f_th = self.rfm_df['Monetary'].quantile(0.8), self.rfm_df['Frequency'].quantile(0.8)
            vips = self.rfm_df[(self.rfm_df['Monetary'] >= m_th) & (self.rfm_df['Frequency'] >= f_th)]
        else:
            return {"error": "缺失MF字段"}
        if vips.empty or 'Recency' not in vips.columns: return {"error": "条件不足"}

        churn_risk = vips[vips['Recency'] > vips['Recency'].mean()].sort_values(by='Monetary', ascending=False).head(5)
        return [{"客户ID": str(row.get('CustomerID', row.name)), "贡献": round(row['Monetary'], 2),
                 "未消费天数": int(row['Recency'])} for _, row in churn_risk.iterrows()]

    # ================= 核心路由与执行引擎 =================

    def query(self, question: str) -> str:
        if self.df.empty:
            return "❌ 系统异常：底层数据表未成功挂载，所有指标查询中止。"

        logger.info(f"DataTool 收到查询请求: {question}")

        supported_metrics_str = "\n".join([f'- "{k}": 查询{v["name"]}' for k, v in self.metric_store.items()])

        intent_prompt = f"""
        你是一个数据分析路由引擎。请分析用户的查询意图，严格输出一个 JSON。

        【当前语义层已注册的指标】:
        {supported_metrics_str}

        用户问题: "{question}"

        输出格式: {{"action": "上述指标key之一，若无匹配项必须填 unknown"}}
        """

        try:
            response = self.client.chat.completions.create(
                model="qwen-plus",
                messages=[{"role": "user", "content": intent_prompt}],
                temperature=0.0
            )
            action = json.loads(
                response.choices[0].message.content.replace("```json", "").replace("```", "").strip()).get("action",
                                                                                                           "unknown")
        except Exception:
            return "❌ 数据请求解析失败，路由引擎熔断。"

        if action in self.metric_store:
            metric_config = self.metric_store[action]
            raw_data = {
                "query_type": metric_config["name"],
                "data": metric_config["calc"]()
            }
        else:
            return f"【系统拦截】：底层指标库未注册针对此维度的数据。当前已配置指标：{', '.join(self.metric_store.keys())}。请重新调整查询粒度。"

        insight_prompt = f"""
        你是一个严谨且冷酷的商业风控分析师。以下是底层执行的标准化数据输出：
        {json.dumps(raw_data, ensure_ascii=False)}

        用户原问题: "{question}"

        严格执行以下响应规范：
        1. 检查数据中是否包含“🚨【风控警报】”。
        2. 如果【没有警报】：简述数据事实，用Bullet points给出一到两条常规实操建议，拒绝废话。
        3. 如果【包含警报】：必须在开头用粗体标出“⚠️ 业务大盘触发熔断预警”。直接列出红线数据，并以命令口吻强制要求运营介入排查，拒绝一切寒暄。
        """
        try:
            final_response = self.client.chat.completions.create(
                model="qwen-plus",
                messages=[{"role": "user", "content": insight_prompt}],
                temperature=0.1
            )
            return final_response.choices[0].message.content
        except Exception as e:
            return f"❌ 洞察生成失败: {e}"