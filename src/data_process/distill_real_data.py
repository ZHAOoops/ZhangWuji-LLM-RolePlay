from unsloth import FastLanguageModel
import torch
import json
from tqdm import tqdm
import os
import re

# ================= 配置区 =================
MODEL_PATH = "/root/autodl-tmp/ZhangWuji_Project/models/base/Qwen/Qwen2.5-7B-Instruct"
INPUT_FILE = "data/processed/wuji_chunks.jsonl"
OUTPUT_FILE = "data/processed/train_dataset_dialogue.json"

# 核心人物名单 (只有跟这些人聊天才是有营养的数据)
TARGET_ROLES = ["赵敏", "周芷若", "小昭", "殷离", "蛛儿", "谢逊", "义父", "张三丰", "太师父", "杨逍", "韦一笑", "范遥", "灭绝"]

# 提取目标数量 (设为 200 条高质量对话就足够训练出非常好的效果了，设为 0 则跑全本)
TARGET_COUNT = 300
# =========================================

def build_extraction_prompt(chunk_text):
    return f"""
你是《倚天屠龙记》的原著分析师。请阅读下面的小说片段，提取【张无忌】的对话。

**提取标准（严格）：**
1. 仅提取 **"对方说话 -> 张无忌回答"** 的交互。
2. 必须保留原著的武侠风味（不要改写成现代白话）。
3. 如果对方是赵敏、周芷若，保留张无忌那种纠结、温柔或无奈的语气。
4. 如果对方是长辈（如张三丰、谢逊），保留恭敬的语气。

**格式要求：**
输出 JSON List，包含：
- "instruction": 对方说的话（包含对方的名字，如：赵敏笑道：“...”）
- "output": 张无忌的回答

**小说片段：**
{chunk_text}

请直接输出 JSON。
"""

def main():
    print(f"🚀 Loading Unsloth Model: {MODEL_PATH} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_PATH,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        all_chunks = [json.loads(line) for line in f]
    
    # 读取已有的数据（支持断点续传）
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            try:
                dataset = json.load(f)
                print(f"📂 发现已有数据 {len(dataset)} 条，将追加保存...")
            except:
                dataset = []
    else:
        dataset = []

    print(f"🎯 开始挖掘高质量对话...")
    print(f"   筛选条件：必须包含引号 + 必须出现 {TARGET_ROLES[:5]}... 等核心人物")
    
    valid_chunks_processed = 0
    
    # 进度条
    pbar = tqdm(total=len(all_chunks), desc="扫描原文")
    
    for item in all_chunks:
        if len(dataset) >= TARGET_COUNT and TARGET_COUNT > 0:
            break
            
        text = item['text']
        pbar.update(1)

        # ====================
        # 1. 规则清洗 (Rule-based Filtering)
        # ====================
        
        # 过滤1：必须包含引号（没有对话的片段直接扔掉，节省大量时间）
        if "“" not in text:
            continue
            
        # 过滤2：必须包含核心人物名字
        if not any(role in text for role in TARGET_ROLES):
            continue
            
        # 过滤3：必须包含张无忌（显式出现）
        if "无忌" not in text and "教主" not in text:
            continue

        # ====================
        # 2. LLM 提取
        # ====================
        prompt = build_extraction_prompt(text)
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        
        try:
            outputs = model.generate(input_ids=inputs, max_new_tokens=512, temperature=0.6, use_cache=True)
            response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
            
            # 清洗JSON
            response = response.replace("", "").strip()
            # 尝试修复常见的 JSON 结尾错误
            if not response.endswith("]"):
                 # 寻找最后一个 ]
                 idx = response.rfind("]")
                 if idx != -1: response = response[:idx+1]

            extracted_pairs = json.loads(response)
            
            if isinstance(extracted_pairs, list):
                new_items_count = 0
                for pair in extracted_pairs:
                    # 质量极简校验：回答不能为空，且要有一定长度
                    if len(pair.get('output', '')) < 3: continue
                    if len(pair.get('instruction', '')) < 3: continue
                    
                    final_data = {
                        "instruction": pair['instruction'],
                        "input": "",
                        "output": pair['output'],
                        "system": "你现在是张无忌，身处倚天屠龙记的武侠世界中。请以张无忌的口吻、性格和记忆来回答。"
                    }
                    dataset.append(final_data)
                    new_items_count += 1
                
                # 每提取到一个有效片段，就立即保存文件！
                # 这样你看到数据涨了就可以随时停
                if new_items_count > 0:
                    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                        json.dump(dataset, f, ensure_ascii=False, indent=2)
                    pbar.set_postfix({"有效对话": len(dataset)})
                    
        except Exception as e:
            continue

    pbar.close()
    print("="*50)
    print(f"🎉 挖掘完成！")
    print(f"   最终获得高质量对话: {len(dataset)} 条")
    print(f"   已保存至: {OUTPUT_FILE}")
    print("="*50)
    
    if dataset:
        print("\n👀 看看真正的‘原著味’数据 (最后一条):")
        print(json.dumps(dataset[-1], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
