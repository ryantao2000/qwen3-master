import json
from torch.utils.data import Dataset
import torch

class MedicalQANoPaddingDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len):
        """
        不填充的医学问答数据集处理类
        
        参数:
            data_path: JSON数据文件路径
            tokenizer: 预训练的分词器
            max_seq_len: 最大序列长度（超长部分会被截断）
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.eos_token_id = tokenizer.eos_token_id
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        
        # 1. 构建对话结构
        
        cot = item["cot"] if item["type"] == "reason" else ""
        answer = item["answer"]
        question = f"{item["question"]} /think" if item["type"] == "reason" else f"{item["question"]} /no_think"
        # 2. 生成模型输入
        messages = [
            {"role": "system", "content": "你是一个专业的医疗助手。"},
            {"role": "user", "content": question}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 3. 生成完整序列（不添加特殊token）
        response = f"<think>{cot}</think>\n{answer}" if cot else f"<think></think>\n{answer}"
        full_text = prompt + response
        input_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
        
        # 4. 生成labels（prompt部分用-100忽略）
        prompt_len = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        labels = [-100] * prompt_len + self.tokenizer.encode(response, add_special_tokens=False)

        # 5. 添加EOS token
        input_ids.append(self.eos_token_id)
        labels.append(self.eos_token_id)
        
        # 6. 严格截断（不填充）
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
        
        attention_mask = [1] * len(input_ids)
        
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels)
        }