from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import evaluate
import langid
import numpy as np
import json
from tqdm import tqdm
from collections import defaultdict
import torch
import os

from bert_score import score

# 初始化评估指标
rouge = evaluate.load("./metric/rouge.py")
bleu = evaluate.load("./metric/bleu.py")
# bertscore = evaluate.load("./metric/bertscore.py")

# 1. 加载基础模型
base_model_name = "merged_finetune_dpo_model"

# 加载tokenizer和基础模型
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} 
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map=device_map,
    attn_implementation="flash_attention_2"
)


model.eval()

# 2. 加载测试数据
def load_test_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

test_data = load_test_data("test_qa_data.json")

# 3. 语言检测函数
def detect_language(text):
    lang, _ = langid.classify(text)
    return lang

# 4. 生成预测（支持思考模式开关）
def generate_predictions(questions, enable_thinking=True):
    predictions = []
    thinking_contents = []
    
    for q in tqdm(questions, desc=f"Generating (thinking={enable_thinking})"):
        messages = [
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": q}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                pad_token_id=tokenizer.eos_token_id
            )
        
        output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)  # </think> token
            thinking = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            answer = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        except ValueError:
            thinking = ""
            answer = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        
        thinking_contents.append(thinking)
        predictions.append(answer)
    
    return predictions, thinking_contents

# 5. 评估函数
def evaluate_metrics(predictions, references):
    # 计算ROUGE
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    
    # 计算BLEU
    bleu_results = bleu.compute(
        predictions=predictions,
        references=references  # BLEU需要references是列表的列表
    )
    
    # 计算BERTScore（按语言分组）
    def calculate_bertscore(preds, refs):
        lang_groups = defaultdict(list)
        for p, r in zip(preds, refs):
            lang = detect_language(r)
            lang_groups[lang].append((p, r))
        
        scores = {'precision': [], 'recall': [], 'f1': []}
        for lang, group in lang_groups.items():
            if not group:
                continue
            g_preds, g_refs = zip(*group)
            try:
                P, R, F1 = score(
                    g_preds, 
                    g_refs,
                    model_type = "/root/autodl-tmp/rl/metric/bert-base-chinese" if lang == 'zh' else "/root/autodl-tmp/rl/metric/roberta-large",
                    lang='en' if lang == 'en' else 'zh', 
                    verbose=True,
                    num_layers=24 if lang == 'en' else 12,
                )
                # 将numpy数组转换为Python列表
                scores['precision'].extend(P.tolist())
                scores['recall'].extend(R.tolist())
                scores['f1'].extend(F1.tolist())
            except Exception as e:
                print(f"BERTScore计算错误 ({lang}): {str(e)}")
        
        return {
            k: float(np.mean(v)*100) if v else 0  # 显式转换为Python float
            for k, v in scores.items()
        }
    
    bert_scores = calculate_bertscore(predictions, references)
    
    return {
        "rouge1": round(rouge_scores["rouge1"], 2),
        "rouge2": round(rouge_scores["rouge2"], 2),
        "rougeL": round(rouge_scores["rougeL"], 2),
        "bleu": round(bleu_results["bleu"] * 100, 2),
        "bertscore_precision": round(bert_scores['precision'], 2),
        "bertscore_recall": round(bert_scores['recall'], 2),
        "bertscore_f1": round(bert_scores['f1'], 2)
    }

# 6. 主评估流程
def main():
    questions = [d["question"] for d in test_data]
    references = [d["answer"] for d in test_data]
    
    # 添加总进度条
    with tqdm(total=2, desc="Overall Progress") as pbar:
        # 分别测试两种模式
        results = {}
        for thinking_mode in [True, False]:
            preds, thoughts = generate_predictions(questions, enable_thinking=thinking_mode)
            metrics = evaluate_metrics(preds, references)
            
            results[f"thinking_{thinking_mode}"] = {
                "predictions": preds,
                "thinking_contents": thoughts,
                "metrics": metrics
            }
            pbar.update(1)
    
    # 打印对比结果
    print("\n=== 微调模型评估结果（思考模式 vs 非思考模式） ===")
    for mode in [True, False]:
        data = results[f"thinking_{mode}"]
        m = data["metrics"]
        print(f"\n◆ 模式: {'思考模式' if mode else '非思考模式'}")
        print(f"ROUGE-1: {m['rouge1']}% | ROUGE-2: {m['rouge2']}% | ROUGE-L: {m['rougeL']}%")
        print(f"BLEU: {m['bleu']}% | BERTScore-F1: {m['bertscore_f1']}%")
        
        # 打印首条示例
        print("\n示例：")
        print(f"问题: {questions[0]}")
        if mode:
            print(f"思考过程: {data['thinking_contents'][0]}")
        print(f"生成回答: {data['predictions'][0]}")
        print(f"参考答案: {references[0]}")
    
    # 保存完整结果
    with open("evaluation_1_7b_finetune_dpo.json", "w") as f:
        json.dump({
            "test_data": test_data,
            "results": results
        }, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()