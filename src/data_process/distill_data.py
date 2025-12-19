import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
import random

# ================= 配置区 =================
MODEL_PATH = "/root/autodl-tmp/ZhangWuji_Project/models/base/Qwen/Qwen2.5-7B-Instruct"
INPUT_FILE = "data/processed/wuji_chunks.jsonl"
OUTPUT_FILE = "data/processed/train_dataset_dialogue.json"

# 🔥 核心策略：只有包含以下关键词的片段，才会被视为“成年/教主剧情”
# 这些人只在张无忌长大后才大量与他产生对手戏
ADULT_KEYWORDS = ["赵敏", "周芷若", "小昭", "杨逍", "范遥", "乾坤大挪移", "九阳神功", "太极拳", "郡主", "敏敏"]

# 限制提取数量 (设为 50 条验证，验证通过后设为 0 跑全量)
MAX_EXTRACT_COUNT = 50 
# =========================================

def build_extraction_prompt(chunk_text):
    return f"""
你是《倚天屠龙记》的剧本专家。请提取【张无忌】与他人的对话。

**规则：**
1. **Input**: 他人对张无忌说的话（去掉“XX道：”等前缀）。
2. **Output**: 张无忌的回答。
3. 必须是一问一答。
4. 排除旁白。

**片段：**
{chunk_text}

请输出 JSON List (key: instruction, output)。
"""

def main():
    print(f"🚀 Loading Model: {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        all_chunks = [json.loads(line) for line in f]
    
    dataset = []
    processed_count = 0
    
    print(f"🎯 启动智能雷达，寻找关键词: {ADULT_KEYWORDS}")
    print("⚗️ 开始扫描并提取...")
    
    # 遍历所有片段
    for item in tqdm(all_chunks):
        text = item['text']
        
        # ⚡️ 雷达扫描：如果没有关键词，直接跳过！
        if not any(kw in text for kw in ADULT_KEYWORDS):
            continue
            
        # 命中关键词，开始让 LLM 提取
        prompt = build_extraction_prompt(text)
        messages = [{"role": "user", "content": prompt}]
        text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)
        
        try:
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512, # 对话通常不长，512够了，加快速度
                temperature=0.7,
                do_sample=True
            )
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            response = response.replace("", "").strip()
            extracted_pairs = json.loads(response)
            
            if isinstance(extracted_pairs, list):
                for pair in extracted_pairs:
                    if len(pair.get('output', '')) < 2: continue
                    if len(pair.get('instruction', '')) < 2: continue
                    
                    final_data = {
                        "instruction": pair['instruction'],
                        "input": "",
                        "output": pair['output'],
                        "system": "你现在是张无忌，请以明教教主的身份，用武侠风格回答。"
                    }
                    dataset.append(final_data)
                    processed_count += 1
                    
        except Exception:
            continue
            
        # 达到目标数量就停止
        if MAX_EXTRACT_COUNT > 0 and processed_count >= MAX_EXTRACT_COUNT:
            break

    # 保存
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        
    print("="*50)
    print(f"🎉 提取完成！")
    print(f"   已扫描片段: {len(all_chunks)}")
    print(f"   成功提取对话: {len(dataset)} 条")
    print(f"   结果保存至: {OUTPUT_FILE}")
    print("="*50)
    
    if dataset:
        print("\n👀 逻辑检查 (必须是成年无忌):")
        for i, d in enumerate(dataset[:3]):
            print(f"[{i+1}]")
            print(f"   👤 {d['instruction']}")
            print(f"   🤖 {d['output']}")

if __name__ == "__main__":
    main()
