from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

model_name = "zero-dpo"
# model_name = "/root/autodl-tmp/merged_model_finetune_1_7b"
# peft_model_path = "zero-dpo"
# 保存路径
save_path = "merged_finetune_dpo_model"  # 你可以修改为你想要保存的路径


# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} 
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device_map,
    attn_implementation="flash_attention_2",
    
)

# 加载微调后的模型
# model = PeftModel.from_pretrained(model, peft_model_path)
# model = model.merge_and_unload()

# 保存合并后的模型和tokenizer
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"模型和tokenizer已成功保存到 {save_path}")