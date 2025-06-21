# coding:utf8
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import torch
from vllm import LLM, PoolingParams
from vllm.distributed.parallel_state import destroy_model_parallel

app = FastAPI()

class EmbeddingRequest(BaseModel):
    sentences: List[str]  # 输入的句子列表
    is_query: bool = False  # 是否为查询（True 会添加指令）
    instruction: Optional[str] = None  # 自定义指令（可选）
    dim: int = -1  # 嵌入维度（默认自动选择）

class Qwen3EmbeddingVllm:
    def __init__(self, model_name_or_path: str, instruction: str = None, max_length: int = 8192):
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        self.instruction = instruction
        self.model = LLM(model=model_name_or_path, task="embed", hf_overrides={"is_matryoshka": True})

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        if task_description is None:
            task_description = self.instruction
        return f'Instruct: {task_description}\nQuery:{query}'

    def encode(self, sentences: List[str], is_query: bool = False, instruction: str = None, dim: int = -1):
        if is_query:
            sentences = [self.get_detailed_instruct(instruction, sent) for sent in sentences]
        if dim > 0:
            output = self.model.embed(sentences, pooling_params=PoolingParams(dimensions=dim))
        else:
            output = self.model.embed(sentences)
        return torch.tensor([o.outputs.embedding for o in output])

    def stop(self):
        destroy_model_parallel()

# 全局加载模型
model_path = "/root/autodl-tmp/Qwen/Qwen3-Embedding-4B"
embedding_model = Qwen3EmbeddingVllm(model_path)

@app.post("/embed")
async def embed(request: EmbeddingRequest):
    """获取句子的嵌入向量"""
    embeddings = embedding_model.encode(
        sentences=request.sentences,
        is_query=request.is_query,
        instruction=request.instruction,
        dim=request.dim
    )
    return {"embeddings": embeddings.tolist()}  # 转换为列表返回

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)