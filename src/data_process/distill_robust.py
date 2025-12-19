import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import os

# ================= 配置区 =================
# 你的本地模型路径
MODEL_PATH = "/root/autodl-tmp/ZhangWuji_Project/models/base/Qwen/Qwen2.5-7B-Instruct"
INPUT_FILE = "data/processed/wuji_chunks.jsonl"
OUTPUT_FILE = "data/processed/train_dataset_dialogue.json"

# 核心人物 (只提取跟这些人聊天的片段)
TARGET_ROLES = ["赵敏", "周芷若", "小昭", "殷离", "蛛儿", "谢逊", "义父", "张三丰", "太师父", "杨逍", "韦一笑", "范遥", "灭绝"]

# 提取目标数量 (设为 300 条)
TARGET_COUNT = 300
# =========================================

def build_extraction_prompt(chunk_text):
    return f"""
你是原著分析师。请阅读片段，提取【张无忌】的对话。

**规则：**
1. 仅提取 "他人说话 -> 张无忌回答" 的对话。
2. 保持原著武侠语气。
3. 排除旁白，只留口语。

**片段：**
{chunk_text}

请输出 JSON List (instruction, output)。
"""

def main():
    print(f"🚀 使用原生 Transformers 加载模型 (BF16)...")
    
    # 强制离线模式，防止联网卡死
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    # 4090 显存大，直接用 bfloat16 加载，既快又稳
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        all_chunks = [json.loads(line) for line in f]
    
    dataset = []
    # 如果文件已存在，读取旧数据继续跑
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            print(f"📂 继承已有数据: {len(dataset)} 条")
        except:
            pass

    print(f"🎯 开始挖掘 (目标: {TARGET_COUNT} 条)...")
    
    pbar = tqdm(total=len(all_chunks), desc="扫描进度")
    
    for item in all_chunks:
        if len(dataset) >= TARGET_COUNT:
            break
            
        text = item['text']
        pbar.update(1)

        # === 快速过滤 (不费显卡) ===
        if "“" not in text: continue # 没对话，跳过
        if not any(r in text for r in TARGET_ROLES): continue # 没熟人，跳过
        if "无忌" not in text and "教主" not in text: continue # 没主角，跳过

        # === LLM 提取 ===
        prompt = build_extraction_prompt(text)
        messages = [{"role": "user", "content": prompt}]
        text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)
        
        try:
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True
            )
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            response = response.replace("", "").strip()
            # 简单修复 JSON 尾部
            if not response.endswith("]"):
                 idx = response.rfind("]")
                 if idx != -1: response = response[:idx+1]

            extracted = json.loads(response)
            
            if isinstance(extracted, list):
                saved_count = 0
                for pair in extracted:
                    if len(pair.get('output', '')) < 2: continue
                    
                    dataset.append({
                        "instruction": pair['instruction'],
                        "input": "",
                        "output": pair['output'],
                        "system": "你现在是张无忌，请以明教教主的身份，用武侠风格回答。"
                    })
                    saved_count += 1
                
                # 实时保存
                if saved_count > 0:
                    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                        json.dump(dataset, f, ensure_ascii=False, indent=2)
                    pbar.set_postfix({"已提取": len(dataset)})
                    
        except:
            continue

    pbar.close()
    print(f"\n🎉 完成！共提取 {len(dataset)} 条数据。")
    print(f"💾 文件保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
