import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger("KnowledgeTool")


class KnowledgeTool:
    """企业内部知识库检索工具 (基于 FAISS + SentenceTransformer)"""

    def __init__(self, index_path="data/knowledge/sop_index.faiss", texts_path="data/knowledge/sop_texts.pkl"):
        self.index_path = index_path
        self.texts_path = texts_path
        self.is_ready = False

        logger.info("KnowledgeTool 正在挂载本地知识库...")
        try:
            # 加载 FAISS 索引和原始文本
            if os.path.exists(self.index_path) and os.path.exists(self.texts_path):
                self.index = faiss.read_index(self.index_path)
                with open(self.texts_path, 'rb') as f:
                    self.sop_texts = pickle.load(f)

                # 加载与初始化时相同的 Embedding 模型
                self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                self.is_ready = True
                logger.info("✅ 知识库挂载成功。")
            else:
                logger.warning("⚠️ 未找到知识库文件，请先运行初始化脚本。")
        except Exception as e:
            logger.error(f"知识库装载失败: {e}")

    def query(self, intent: str, top_k: int = 2) -> str:
        """根据大模型的意图，检索最相关的 SOP 规则"""
        logger.info(f"KnowledgeTool 收到检索指令: {intent}")

        if not self.is_ready:
            return "❌ 知识库未初始化，无法检索。请联系管理员先执行知识库构建流程。"

        try:
            # 1. 将用户的查询意图向量化
            query_vector = self.model.encode([intent])

            # 2. 在 FAISS 库中进行相似度检索
            distances, indices = self.index.search(np.array(query_vector), top_k)

            # 3. 提取最相关的结果
            results = []
            for idx in indices[0]:
                if idx != -1 and idx < len(self.sop_texts):
                    results.append(self.sop_texts[idx])

            if not results:
                return "知识库中未匹配到相关的 SOP 规范。"

            # 格式化输出，喂给上层的大模型
            formatted_result = "【已检索到以下相关企业内部SOP规范】:\n" + "\n".join([f"- {res}" for res in results])
            return formatted_result

        except Exception as e:
            return f"检索过程中发生异常: {e}"