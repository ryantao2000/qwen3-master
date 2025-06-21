# coding:utf8
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Tuple, Optional
import torch
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from transformers import AutoTokenizer
import math
import os

app = FastAPI()

class RerankerRequest(BaseModel):
    pairs: List[Tuple[str, str]]
    instruction: Optional[str] = None
    max_length: int = 2048

class RerankerResponse(BaseModel):
    scores: List[float]

class Qwen3Rerankervllm:
    def __init__(self, model_name_or_path: str, instruction: str = None, **kwargs):
        self.instruction = instruction or "Given the user query, retrieval the relevant passages"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.max_length = kwargs.get('max_length', 8192)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        self.true_token = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
        self.false_token = self.tokenizer("no", add_special_tokens=False).input_ids[0]
        
        # 初始化LLM (兼容最新vLLM版本)
        self.lm = LLM(
            model=model_name_or_path,
            tensor_parallel_size=torch.cuda.device_count(),
            max_model_len=self.max_length,
            gpu_memory_utilization=0.5,
            enforce_eager=True,
            dtype="float16",
            distributed_executor_backend='ray'
        )
        
        self.sampling_params = SamplingParams(
            temperature=0,
            top_p=0.95,
            max_tokens=1,
            logprobs=20,
            allowed_token_ids=[self.true_token, self.false_token],
        )

    def format_prompt(self, instruction: str, query: str, doc: str) -> str:
        """将对话格式转换为纯文本提示"""
        messages = [
            {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Answer only \"yes\" or \"no\"."},
            {"role": "user", "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"}
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,  # 返回字符串而非token
            add_generation_prompt=False
        ) + self.suffix

    def compute_scores(self, pairs: List[Tuple[str, str]], **kwargs) -> List[float]:
        """直接使用字符串作为输入 (兼容最新vLLM)"""
        instruction = kwargs.get("instruction", self.instruction)
        prompts = [
            self.format_prompt(instruction, query, doc)
            for query, doc in pairs
        ]
        
        outputs = self.lm.generate(prompts, self.sampling_params)
        
        scores = []
        for output in outputs:
            logprobs = output.outputs[0].logprobs[-1]
            true_score = math.exp(logprobs.get(self.true_token, -10).logprob)
            false_score = math.exp(logprobs.get(self.false_token, -10).logprob)
            scores.append(true_score / (true_score + false_score))
        return scores

    def stop(self):
        destroy_model_parallel()

# 初始化模型
model_path = "/root/autodl-tmp/Qwen/Qwen3-Reranker-4B"
reranker = Qwen3Rerankervllm(
    model_name_or_path=model_path,
    instruction="Retrieval document that can answer user's query",
    max_length=2048
)

@app.post("/rerank", response_model=RerankerResponse)
async def rerank(request: RerankerRequest):
    scores = reranker.compute_scores(
        pairs=request.pairs,
        instruction=request.instruction
    )
    return {"scores": scores}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)