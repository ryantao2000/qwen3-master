#coding:utf8
import os
from typing import Dict, Optional, List, Union
import torch
from torch import nn
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from transformers.utils import is_flash_attn_2_available
import numpy as np
from collections import defaultdict

class Qwen3Embedding():
    def __init__(self, model_name_or_path, instruction=None,  use_fp16: bool = True, use_cuda: bool = True, max_length=8192):
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        self.instruction = instruction
        if is_flash_attn_2_available() and use_cuda:
            self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, attn_implementation="flash_attention_2", torch_dtype=torch.float16)
        else:
            self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.float16)
        if use_cuda:
            self.model = self.model.cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, padding_side='left')
        self.max_length=max_length
    
    def last_token_pool(self, last_hidden_states: Tensor,
        attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        if task_description is None:
            task_description = self.instruction
        return f'Instruct: {task_description}\nQuery:{query}'

    def encode(self, sentences: Union[List[str], str], is_query: bool = False, instruction=None, dim: int = -1):
        if isinstance(sentences, str):
            sentences = [sentences]
        if is_query:
            sentences = [self.get_detailed_instruct(instruction, sent) for sent in sentences]
        inputs = self.tokenizer(sentences, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
        inputs.to(self.model.device)
        with torch.no_grad():
            model_outputs = self.model(**inputs)
            output = self.last_token_pool(model_outputs.last_hidden_state, inputs['attention_mask'])
            if dim != -1:
                output = output[:, :dim]
            output  = F.normalize(output, p=2, dim=1)
        return output

if __name__ == "__main__":
    model_path = "D:/AI/models_datasets/qwen3-embedding-4b/Qwen/Qwen3-Embedding-4B"
    model = Qwen3Embedding(model_path,use_cuda=False)
    queries = ['What is the capital of China?', 'Explain gravity']
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
    ]
    dim = 1024
    query_outputs = model.encode(queries, is_query=True, dim=dim)
    doc_outputs = model.encode(documents, dim=dim)
    print('query outputs', query_outputs)
    print('doc outputs', doc_outputs)
    scores = (query_outputs @ doc_outputs.T) * 100
    print(scores.tolist())