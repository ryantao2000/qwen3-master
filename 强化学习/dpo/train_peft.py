import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
import deepspeed
from trl import DPOTrainer,DPOConfig
from peft import LoraConfig, TaskType, get_peft_model
from swanlab.integration.transformers import SwanLabCallback

DS_CONFIG = "ds_z2_offload_config.json"


model_name = "/root/autodl-tmp/merged_model_finetune_1_7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)


device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} 
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device_map,
    attn_implementation="flash_attention_2",
    use_cache=False
)
model.enable_input_require_grads() 


config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=64,  # Lora 秩
    lora_alpha=128,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1,  # Dropout 比例
)

train_data = load_dataset("json", data_files="train_preference_data.json")['train']

# 设置SwanLab回调
swanlab_callback = SwanLabCallback(
    project="Qwen3-1.7B-peft",
    experiment_name="Qwen3-1.7B-dpo-peft",
    description="使用通义千问Qwen3-1.7B模型在nuriyev/medical-question-answering-rl-labeled-qwen-0.5B-binarized和Morefreedai/medical-dpo-v1数据集上微调。",
    config={
        "model": "Qwen/Qwen3-1.7B",
        "dataset": "https://huggingface.co/datasets/nuriyev/medical-question-answering-rl-labeled-qwen-0.5B-binarized",
        "train_data_number": len(train_data),
        "lora_rank": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.1,
    }
)


# Train the model
training_args = DPOConfig(
    output_dir="./zero-dpo-peft",
    # bf16=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    logging_steps=5,
    save_steps=500,
    learning_rate=2e-4,
    logging_first_step=5,
    max_grad_norm=2.0,
    beta=0.8,
    
)


trainer = DPOTrainer(
     model=model,
     ref_model=None, # 如果是相对优化，可以提供一个 reference model,如果用了peft，就设置为None
     train_dataset=train_data,
     eval_dataset=None,
     tokenizer=tokenizer,
     peft_config=config,
     callbacks=[swanlab_callback],
     args=training_args
)
     
trainer.train()
trainer.save_model('./zero-dpo-peft')
trainer.save_state()