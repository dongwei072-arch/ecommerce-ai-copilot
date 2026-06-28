import pandas as pd
import logging
import os

logger = logging.getLogger("ChartTool")


class ChartTool:
    """图表可视化工具：严格受控的本地渲染引擎，拒绝 LLM 幻觉画图"""

    def __init__(self, raw_data_path="data/data.csv"):
        self.raw_data_path = raw_data_path

        logger.info(f"ChartTool 正在装载可视化基础数据...")
        try:
            self.df = pd.read_csv(self.raw_data_path, encoding='ISO-8859-1')
            if 'InvoiceDate' in self.df.columns:
                self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
            if 'Quantity' in self.df.columns and 'UnitPrice' in self.df.columns:
                self.df['TotalAmount'] = self.df['Quantity'] * self.df['UnitPrice']
        except Exception as e:
            logger.error(f"图表数据装载失败: {e}")
            self.df = pd.DataFrame()

    def query(self, intent: str) -> dict:
        """
        拦截大模型的画图请求，直接在本地计算并返回结构化图表数据。
        """
        logger.info(f"ChartTool 收到图表渲染请求: {intent}")

        if self.df.empty or 'InvoiceDate' not in self.df.columns:
            return {"type": "error", "message": "图表数据源不可用"}

        # 默认执行近 7 日 GMV 走势计算
        sales = self.df[self.df['Quantity'] > 0].copy()
        sales['Date'] = sales['InvoiceDate'].dt.date
        trend = sales.groupby('Date')['TotalAmount'].sum().sort_index().tail(7)

        # 构建 Streamlit 可以直接识别的 DataFrame 格式
        df_trend = pd.DataFrame({
            'GMV': trend.values
        }, index=pd.to_datetime(trend.index))

        # 返回字典格式：既包含给前端画图用的 raw_data，也包含喂给大模型做上下文的 text
        return {
            "type": "line_chart",
            "title": "近7日大盘 GMV 走势",
            "raw_data": df_trend,
            "text_summary": f"已成功生成近7日GMV折线图，起始日GMV为 {trend.iloc[0]:.2f}，最新日GMV为 {trend.iloc[-1]:.2f}。"
        }
