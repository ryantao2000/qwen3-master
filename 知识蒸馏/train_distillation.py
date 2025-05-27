from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments,DataCollatorForSeq2Seq
# from dataset import SFTDataset
from utils import compute_fkl
import os
import json
from datasets import load_dataset,Dataset
from bitsandbytes.optim import AdamW8bit
from transformers import get_cosine_schedule_with_warmup
import deepspeed
DS_CONFIG = "ds_zero2_no_offload.json"
from dataset_med import MedicalQANoPaddingDataset
from torch.serialization import add_safe_globals
import numpy._core.multiarray


class KGTrainer(Trainer):
    def __init__(
        self,
        model=None,
        teacher_model=None,
        if_use_entropy=False,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.teacher_model = teacher_model
        self.if_use_entropy = if_use_entropy
    def _load_rng_state(self, checkpoint):
        rng_file = os.path.join(checkpoint, "rng_state.pth")
        if os.path.exists(rng_file):
            # 强制禁用 weights_only
            checkpoint_rng_state = torch.load(rng_file, weights_only=False)
            torch.set_rng_state(checkpoint_rng_state["torch"])
            np.random.set_state(checkpoint_rng_state["numpy"])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(checkpoint_rng_state["cuda"])

    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # 注意这里的缩进

        device = next(model.parameters()).device
        # 将所有张量移动到指定的设备上
        for key, value in inputs.items():
            inputs[key] = value.to(device)
        
        outputs = model(**inputs)
        with torch.no_grad():
            device = next(self.teacher_model.parameters()).device
            # 将所有张量移动到指定的设备上
            for key, value in inputs.items():
                inputs[key] = value.to(device)
            teacher_outputs = self.teacher_model(**inputs)
        
        loss = outputs.loss/4
        
        labels = inputs['labels']
        logits = outputs.logits

        
        teacher_logits = teacher_outputs.logits

        
        if logits.shape[-1] != teacher_logits.shape[-1]:
            teacher_logits = teacher_logits[:, :, :logits.shape[-1]]
        
        
        labels.to(device)
        kl = compute_fkl(logits, teacher_logits, labels, padding_id=-100, temp=1.0)
        # print(f"kl=={kl} loss=={loss}\n\n")
        if self.if_use_entropy:
            loss_total = 0.5 * kl + 0.5 * loss
        else:
            loss_total = kl

            
        return (loss_total, outputs) if return_outputs else loss_total

if __name__ == '__main__':
    add_safe_globals([numpy._core.multiarray._reconstruct])
    
    # 学生模型
    student_name = "/root/autodl-tmp/Qwen/Qwen3-1.7B"
    # teacher_name = "merged_model_4b"
    teacher_name = "merged_model"
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} 
    model = AutoModelForCausalLM.from_pretrained(
        student_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        # device_map="cuda:1",
        attn_implementation="flash_attention_2"
    )
    model.enable_input_require_grads()
    
    lora_config = LoraConfig(
    r=8,  
    lora_alpha=32,  
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1, 
    task_type=TaskType.CAUSAL_LM)
    # 使用lora方法训练
    model = get_peft_model(model, lora_config)
    
    print(model.print_trainable_parameters())
    
    tokenizer = AutoTokenizer.from_pretrained(student_name,use_cache=False)
    
    # 教师模型
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        # device_map="cuda:0",
        attn_implementation="flash_attention_2"
    )

    teacher_model.eval()
    
    args = TrainingArguments(output_dir='./results', 
                            num_train_epochs=6, 
                            do_train=True, 
                            per_device_train_batch_size=1,
                            gradient_accumulation_steps=4,
                            gradient_checkpointing=True,
                            logging_steps=10,
                            report_to='tensorboard',
                            save_strategy='epoch',
                            save_total_limit=1,
                            bf16=True,
                            learning_rate=2e-4,
                            lr_scheduler_type='cosine',
                            warmup_ratio=0.1,
                            max_grad_norm=1.0,
                            optim = "adamw_8bit",
                            fp16=False,
                            dataloader_num_workers=2,
                            gradient_checkpointing_kwargs={"use_reentrant": False},
                            deepspeed=DS_CONFIG,
                            dataloader_pin_memory=True
                            )
    # data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)


    dataset = MedicalQANoPaddingDataset(
        data_path="combined_medical_data.json",
        tokenizer=tokenizer,
        max_seq_len=2048
    )
    data_collator = DefaultDataCollator()
    # 计算总训练步数
    total_steps = len(dataset) // args.per_device_train_batch_size * args.num_train_epochs
    




    trainer = KGTrainer(model=model,
                        teacher_model=teacher_model, 
                        if_use_entropy = True,
                        args=args, 
                        train_dataset=dataset, 
                        tokenizer=tokenizer, 
                        # optimizers=(optimizer,lr_scheduler),
                        data_collator=data_collator)
    # 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('./results')
    trainer.save_state()
    
      
    