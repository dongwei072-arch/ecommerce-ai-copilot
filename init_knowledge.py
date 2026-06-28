import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer


def init_sop_database():
    print("🧠 正在构建企业级 SOP 知识库...")

    # 1. 准备企业内部的真实 SOP 语料（这里模拟了几条核心规则）
    sop_documents = [
        "【退款异常SOP - 触发熔断】当日大盘退货率突破 8% 或单一 SKU 退货率突破 15% 时，立即触发三级风控熔断。动作：1. 暂停该商品全渠道广告投放；2. 冻结该 SKU 库存发货；3. 人工抽检近50单退款归因。",
        "【退款异常SOP - 物流归因】若退款归因TOP1为'包装破损'或'物流延迟'，操作建议：立即调取仓储出库录像，核查近期纸箱/缓冲材批次质量；同步向合作物流（如顺丰/中通）发起工单预警。",
        "【退款异常SOP - 描述不符】若退款归因TOP1为'尺寸不符'或'材质与描述不符'，操作建议：运营立刻下线商品详情页进行整改，客服团队在拦截退款时发放 10 元无门槛补偿券进行挽留。",
        "【客单价挽回SOP】当大盘客单价低于 50 元警戒线时，禁止单品打折，强制启动关联营销。客服话术必须加入'满99减20'的凑单推荐。"
    ]

    # 2. 加载本地开源的轻量级 Embedding 模型（将文本转为向量）
    # 首次运行会自动从 HuggingFace 下载模型，请保持网络通畅
    print("⏳ 正在加载向量化模型 (SentenceTransformer)...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # 3. 将 SOP 文本编码为高维向量
    print("🔢 正在对 SOP 文档进行特征提取与向量化...")
    embeddings = model.encode(sop_documents)

    # 4. 构建 FAISS 索引
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    # 5. 保存持久化文件
    os.makedirs('data/knowledge', exist_ok=True)
    faiss.write_index(index, 'data/knowledge/sop_index.faiss')

    # 将原始文本也存下来，方便检索出向量后能展示给人看
    with open('data/knowledge/sop_texts.pkl', 'wb') as f:
        pickle.dump(sop_documents, f)

    print("✅ 知识库初始化成功！FAISS 索引与文段已写入 data/knowledge/ 目录。")


if __name__ == "__main__":
    init_sop_database()