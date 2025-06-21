import ray
import torch
from ray import serve
from typing import List, Dict
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ray.serve.config import HTTPOptions
import time
from qwen3_embedding_transformers import Qwen3Embedding
from qwen3_reranker_transformers import Qwen3Reranker

# 模型路径配置
model_name_or_path_reranker = "/root/autodl-tmp/Qwen/Qwen3-Reranker-4B"
model_name_or_path_embedding = "/root/autodl-tmp/Qwen/Qwen3-Embedding-4B"

# 部署配置
NUM_REPLICAS = 2  # 部署的模型数量
NUM_GPUS = 0.9    # 每个模型占用的GPU占比（0.9表示每个GPU使用90%的GPU资源）

# 示例配置说明：
# 1. 当前配置：启动2个GPU，每个使用0.9个GPU
# 2. 示例配置1：NUM_REPLICAS=6, NUM_GPUS=0.5 → 6个模型，每2个共享一个GPU
# 3. 示例配置2：NUM_REPLICAS=2, NUM_GPUS=0.6 → 2个模型，分别在不同的GPU上占0.6

# 定义输入数据模型
class RerankerInput(BaseModel):
    """重排序服务的输入模型"""
    questions: List[str]  # 问题列表
    texts: List[str]      # 待排序文本列表

class EmbeddingInput(BaseModel):
    """嵌入服务的输入模型"""
    input: List[str]  # 需要嵌入的文本列表
    is_query: bool    # 是否为查询（True表示查询，False表示文档）

# class EmbeddingInput2(BaseModel):
#     """嵌入服务的输入模型"""
#     questions: List[str]  # 需要嵌入的文本列表
#     texts: List[str]    # 待排序文本列表

# 创建FastAPI应用并配置CORS
app = FastAPI(title="Qwen3 Embedding & Reranker Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

@serve.deployment(
    num_replicas=NUM_REPLICAS,  # 设置模型数量
    ray_actor_options={"num_gpus": NUM_GPUS}  # 设置GPU资源
)
@serve.ingress(app)  # 将FastAPI应用作为服务的入口
class BatchCombineInferModel:
    """组合嵌入和重排序模型的服务类"""
    
    def __init__(
        self, 
        model_name_or_path_reranker: str, 
        model_name_or_path_embedding: str
    ):
        """初始化模型"""
        # 初始化嵌入模型
        self.model_embedding = Qwen3Embedding(
            model_name_or_path=model_name_or_path_embedding,max_length=2048
        )
        # 初始化重排序模型
        self.model_reranker = Qwen3Reranker(
            model_name_or_path=model_name_or_path_reranker,
            instruction="Retrieval document that can answer user's query",
            max_length=2048,
        )

    @app.post("/emb/api", summary="获取文本嵌入向量")
    async def embedding(self, texts: EmbeddingInput) -> List[List[float]]:
        """
        处理文本嵌入请求
        参数:
            texts: 包含输入文本和是否为查询的标志
        返回:
            文本的嵌入向量列表
        """
        with torch.inference_mode():  # 禁用梯度计算以提高性能
            # 获取嵌入向量并转换为列表格式
            embeddings = self.model_embedding.encode(
                texts.input, 
                is_query=texts.is_query
            )
            return embeddings.cpu().detach().numpy().tolist()


    # @app.post("/emb/api", summary="获取文本嵌入向量")
    # async def embedding(self, texts: EmbeddingInput2) -> List[List[float]]:
    #     """
    #     处理文本嵌入请求
    #     参数:
    #         texts: 包含输入文本和是否为查询的标志
    #     返回:
    #         文本的嵌入向量列表
    #     """
    #     with torch.inference_mode():  # 禁用梯度计算以提高性能
    #         # 获取嵌入向量并转换为列表格式
    #         query_embeddings = self.model_embedding.encode(
    #             texts.questions, 
    #             is_query=True
    #         )

    #         documents_embeddings = self.model_embedding.encode(
    #             texts.texts, 
    #             is_query=False
    #         )
    #         return (query_embeddings @ documents_embeddings.T).cpu().detach().numpy().tolist()

    @app.post("/reranker/api", summary="对文本进行重排序")
    async def reranker(self, texts: RerankerInput) -> List[float]:
        """
        处理重排序请求
        参数:
            texts: 包含问题和待排序文本
        返回:
            文本的相关性分数列表
        """
        with torch.inference_mode():
            # 将问题和文本配对
            pairs = list(zip(texts.questions, texts.texts))
            
            # 计算相关性分数
            instruction = "Given the user query, retrieval the relevant passages"
            scores = self.model_reranker.compute_scores(pairs, instruction)
            return scores

# 启动Ray Serve服务
def start_service():
    """启动模型服务"""
    # 配置HTTP选项
    http_options = HTTPOptions(host="0.0.0.0", port=1080)
    
    # 启动Ray Serve
    serve.start(http_options=http_options)
    
    # 部署服务
    serve.run(
        BatchCombineInferModel.bind(
            model_name_or_path_reranker, 
            model_name_or_path_embedding
        ),
        route_prefix="/",  # 设置路由前缀
    )
    
    # 保持服务运行
    while True:
        time.sleep(1000)

if __name__ == "__main__":
    start_service()


