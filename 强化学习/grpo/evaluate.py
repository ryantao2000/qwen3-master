import os
import re
import json
import nltk  # 新增英文分词库
from datasets import load_from_disk
from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import pandas as pd

# 下载nltk数据（第一次运行时需要）
nltk.download('punkt_tab')
SYSTEM_PROMPT = {
    "en": """Respond with <think>your reasoning process</think> followed by <answer>your final answer</answer>. 
    The response part should be as concise as possible: 
    - For math problems, only provide the numerical result
    - For judgment questions, only answer yes/no/maybe
    - For multiple-choice questions, only provide the letter (A/B/C/D)
    When not in thinking mode, only provide the content within <answer> tags""",
    
    "zh": """请用<think>你的思考过程</think>和<answer>最终答案</answer>的格式回答。
    其中response部分填写尽量简短：
    - 数学题只需要回答数字结果
    - 判断题只需要回答yes/no/maybe
    - 选择题只需要填写A/B/C/D
    非思考模式时，只需要提供<answer>标签包含的内容"""
}

THREE_SHOT_EXAMPLES = {
    "gsm8k": [
        # 思考模式示例
        {
            "role": "user", 
            "content": "A bakery sells 12 cupcakes for $18. How much does one cupcake cost? /think"
        },
        {
            "role": "assistant",
            "content": "<think>Unit price = Total / Quantity = 18 / 12 = 1.5</think>\n<answer>1.5</answer>"
        },
        {
            "role": "user",
            "content": "A laptop originally priced at $800 is on sale for 25% off. What's the sale price? /think"
        },
        {
            "role": "assistant",
            "content": "<think>Discount = 800 * 0.25 = 200; Sale price = 800 - 200 = 600</think>\n<answer>600</answer>"
        },
        {
            "role": "user",
            "content": "John has 3 times as many books as Mary. If Mary has 8 books, how many books do they have together? /think"
        },
        {
            "role": "assistant",
            "content": "<think>John's books = 3 * 8 = 24; Total = 24 + 8 = 32</think>\n<answer>32</answer>"
        },
        
        # 非思考模式示例
        {
            "role": "user",
            "content": "If 5 apples cost $10, what's the price per apple? /no_think"
        },
        {
            "role": "assistant",
            "content": "<answer>2</answer>"
        },
        {
            "role": "user",
            "content": "A car travels 240 miles in 4 hours. What's its speed? /no_think"
        },
        {
            "role": "assistant",
            "content": "<answer>60</answer>"
        },
        {
            "role": "user",
            "content": "What is 15% of 200? /no_think"
        },
        {
            "role": "assistant",
            "content": "<answer>30</answer>"
        }
    ],
    
    "pubmedqa": [
        # 思考模式示例
        {
            "role": "user",
            "content": "Based on the study: 'Omega-3 supplementation reduces depression symptoms by 30% in elderly patients.'\nQuestion: Should elderly patients with depression take omega-3 supplements? /think"
        },
        {
            "role": "assistant",
            "content": "<think>Study shows significant symptom reduction, benefits outweigh risks</think>\n<answer>yes</answer>"
        },
        {
            "role": "user",
            "content": "Study states: 'No significant difference in mortality found between Treatment A and placebo.'\nQuestion: Is Treatment A effective for improving survival? /think"
        },
        {
            "role": "assistant",
            "content": "<think>Study shows no mortality benefit compared to placebo</think>\n<answer>no</answer>"
        },
        {
            "role": "user",
            "content": "Meta-analysis reports: 'Evidence on vitamin D preventing fractures is inconclusive.'\nQuestion: Should postmenopausal women take vitamin D for fracture prevention? /think"
        },
        {
            "role": "assistant",
            "content": "<think>Current evidence is not conclusive, may depend on individual risk factors</think>\n<answer>maybe</answer>"
        },
        
        # 非思考模式示例
        {
            "role": "user",
            "content": "Study shows 'Regular exercise lowers blood pressure.' Should hypertensives exercise? /no_think"
        },
        {
            "role": "assistant",
            "content": "<answer>yes</answer>"
        },
        {
            "role": "user",
            "content": "'Antibiotics show no benefit for viral colds.' Should colds be treated with antibiotics? /no_think"
        },
        {
            "role": "assistant",
            "content": "<answer>no</answer>"
        },
        {
            "role": "user",
            "content": "'Probiotics may help some IBS patients.' Should all IBS patients take probiotics? /no_think"
        },
        {
            "role": "assistant",
            "content": "<answer>maybe</answer>"
        }
    ],
    
    "med_mc": [
        # 思考模式示例
        {
            "role": "user",
            "content": "What is the first-line treatment for uncomplicated UTI?\nA. Amoxicillin\nB. Ciprofloxacin\nC. Nitrofurantoin\nD. Vancomycin /think"
        },
        {
            "role": "assistant",
            "content": "<think>Nitrofurantoin is first-line for uncomplicated UTIs per guidelines</think>\n<answer>C</answer>"
        },
        {
            "role": "user",
            "content": "Which medication is contraindicated in pregnancy?\nA. Acetaminophen\nB. Warfarin\nC. Levothyroxine\nD. Insulin /think"
        },
        {
            "role": "assistant",
            "content": "<think>Warfarin is teratogenic, others are safe</think>\n<answer>B</answer>"
        },
        {
            "role": "user",
            "content": "A patient with penicillin allergy needs antibiotic for pneumonia. Which is safest?\nA. Amoxicillin\nB. Ceftriaxone\nC. Azithromycin\nD. Penicillin /think"
        },
        {
            "role": "assistant",
            "content": "<think>Azithromycin is macrolide with no cross-reactivity</think>\n<answer>C</answer>"
        },
        
        # 非思考模式示例
        {
            "role": "user",
            "content": "Preferred pain reliever for mild headache?\nA. Morphine\nB. Ibuprofen\nC. Warfarin\nD. Digoxin /no_think"
        },
        {
            "role": "assistant",
            "content": "<answer>B</answer>"
        },
        {
            "role": "user",
            "content": "Which drug causes birth defects?\nA. Folic acid\nB. Thalidomide\nC. Prenatal vitamins\nD. Iron /no_think"
        },
        {
            "role": "assistant",
            "content": "<answer>B</answer>"
        },
        {
            "role": "user",
            "content": "Safe for G6PD deficiency?\nA. Sulfa drugs\nB. Primaquine\nC. Acetaminophen\nD. Aspirin /no_think"
        },
        {
            "role": "assistant",
            "content": "<answer>C</answer>"
        }
    ]
}
# --- 2. 数据加载 ---
def load_local_datasets(data_dir="./"):
    """从同级目录加载数据集"""
    if os.path.exists(data_dir):
        dataset = load_from_disk(data_dir)
        return dataset
    raise FileNotFoundError("Local dataset not found, please confirm the dataset is saved in current directory")

# --- 3. 答案匹配逻辑（重点修改部分）---
def is_gsm8k_correct(prediction: str, true_answer: str) -> bool:
    """GSM8K严格数字匹配"""
    pred_numbers = re.findall(r'\d+\.?\d*', prediction)  # 匹配整数和小数
    return true_answer in pred_numbers

def is_soft_correct(prediction: str, true_answer: str) -> bool:
    """PubMedQA宽松匹配"""
    prediction = prediction.lower().strip()
    true_answer = true_answer.lower().strip()
    return true_answer in prediction

def is_med_mc_correct(prediction: str, true_answer: str) -> bool:
    """医学选择题增强匹配逻辑（英文版）"""
    # 预处理
    prediction = prediction.lower().strip()
    true_answer = true_answer.lower().strip()
    
    # 1. 直接匹配完整答案（如"A"或"ACE inhibitors"）
    # if true_answer in prediction:
    #     return True
    
    # 2. 英文分词后匹配
    tokens = nltk.word_tokenize(prediction)
    
    # 3. 匹配选项字母（A/B/C/D）
    option_letters = re.findall(r'\b[a-d]\b', prediction)
    if true_answer in option_letters:
        return True
    
    # 4. 匹配选项关键词（如"ace inhibitors"）
    if len(true_answer) > 1:  # 如果真实答案是文本而非单个字母
        # 检查预测中是否包含答案的所有关键词
        answer_keywords = set(nltk.word_tokenize(true_answer))
        pred_keywords = set(tokens)
        if answer_keywords.issubset(pred_keywords):
            return True
    
    return False

# --- 4. 批量模型推理 ---
def batch_generate_with_shots(model, tokenizer, prompts, dataset_type, batch_size=96):
    """批量处理多条数据"""
    all_thinkings = []
    all_answers = []
    
    for i in tqdm(range(0, len(prompts), batch_size), desc=f"Processing {dataset_type}"):
        batch_prompts = prompts[i:i+batch_size]
        batch_messages = []
        
        for prompt in batch_prompts:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT["en"]},
                # *THREE_SHOT_EXAMPLES[dataset_type],
                prompt[-1]
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            batch_messages.append(text)
        
        inputs = tokenizer(batch_messages, return_tensors="pt", padding=True, truncation=True).to(model.device)
        # inputs = tokenizer(batch_messages, return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            # do_sample=True,
            # temperature=0.7,
            # pad_token_id=tokenizer.eos_token_id
        )
        
        for j in range(len(batch_prompts)):
            input_length = len(inputs.input_ids[j])
            output_ids = generated_ids[j][input_length:].tolist()
            
            try:
                index = len(output_ids) - output_ids[::-1].index(151668)  # </think>标记
            except ValueError:
                index = 0
            
            thinking = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            answer = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            
            all_thinkings.append(thinking)
            all_answers.append(answer)
    
    return all_thinkings, all_answers

# --- 5. 主流程 ---
if __name__ == "__main__":
    # 初始化输出文件
    output_json = "raw_model_no_think_evaluate_results.json"
    if os.path.exists(output_json):
        os.remove(output_json)
    
    # 加载数据
    print("Loading dataset...")
    test_dataset = load_local_datasets("test_dataset")
    
    # 按数据集类型拆分
    datasets = {
        "med_mc": test_dataset.filter(lambda x: x["db_set"] == "med_mc"),
        "gsm8k": test_dataset.filter(lambda x: x["db_set"] == "gsm8k"),
        "pubmedqa": test_dataset.filter(lambda x: x["db_set"] == "pubmedqa")
        
    }
    
    # 加载模型
    print("Loading model...")
    model_name = "./grpo_model"
    # model_name = "Qwen/Qwen3-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_cache=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
        use_cache=False
    )
    tokenizer.padding_side = 'left'
    
    # 运行评测
    final_results = {}
    for name, dataset in datasets.items():
        print(f"\nEvaluating {name}...")
        correct = 0
        results = []
        
        all_prompts = [item["prompt"] for item in dataset]
        true_answers = [str(item["answer"]) for item in dataset]
        
        thinkings, answers = batch_generate_with_shots(model, tokenizer, all_prompts, name, batch_size=128)
        
        for i in tqdm(range(len(dataset)), desc=f"Evaluating {name}"):
            if name == "gsm8k":
                is_correct = is_gsm8k_correct(answers[i], true_answers[i])
            elif name == "med_mc":
                is_correct = is_med_mc_correct(answers[i], true_answers[i])
            else:
                is_correct = is_soft_correct(answers[i], true_answers[i])
            
            correct += int(is_correct)
            results.append({
                "dataset": name,
                "question": all_prompts[i][-1]["content"],
                "true_answer": true_answers[i],
                "thinking": thinkings[i],
                "answer": answers[i],
                "is_correct": is_correct
            })
        
        # 保存结果
        with open(output_json, "a", encoding="utf-8") as f:
            for record in results:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        final_results[name] = {
            "correct": correct,
            "total": len(dataset),
            "accuracy": f"{correct/len(dataset):.2%}",
            "eval_mode": "strict" if name == "gsm8k" else "enhanced" if name == "med_mc" else "lenient"
        }
    
    # 打印并保存结果
    print("\nEvaluation Summary:")
    print(pd.DataFrame(final_results).T)
    
    with open("raw_model_think_no_evaluate_scores.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nDetailed results saved to {output_json}")
    print("Summary saved to raw_model_no_think_evaluate_scores.json")