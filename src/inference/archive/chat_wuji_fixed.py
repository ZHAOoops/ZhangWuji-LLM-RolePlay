import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import os
import datetime
import re

# ================= 配置区 =================
BASE_MODEL_PATH = "/root/autodl-tmp/ZhangWuji_Project/models/base/Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = "models/lora/zhangwuji_v1_fixed"
LOG_FILE = "logs/chat_history_final.jsonl"
SYSTEM_PROMPT = "你现在是张无忌。请以明教教主的身份，用简练、侠义的口吻回答。"
# =========================================

def smart_truncate(text):
    """
    ✂️ 智能剪刀：只保留精华，切除废话
    """
    # 1. 如果包含“User:”或“Instruction:”，说明模型开始自言自语了，直接切断
    if "User:" in text: text = text.split("User:")[0]
    if "Instruction:" in text: text = text.split("Instruction:")[0]
    
    # 2. 按标点符号切分句子
    # 匹配 。！？... 
    sentences = re.split(r'(。|！|？|\n)', text)
    
    # 重新组合，只保留前 3 个完整句子
    # (sentences 列表里是 [句1, 标点1, 句2, 标点2, ...])
    keep_count = 6 # 3句话 * 2 (内容+标点)
    
    if len(sentences) > keep_count:
        # 看看后面是不是废话，如果是，就只取前几句
        truncated = "".join(sentences[:keep_count])
        return truncated
    else:
        return text

def main():
    print(f"🚀 Loading Base Model: {BASE_MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    print(f"🔗 Loading LoRA Adapter: {LORA_PATH}...")
    model = PeftModel.from_pretrained(model, LORA_PATH)
    
    print("="*50)
    print("🎉 张无忌（精修版）已上线！")
    print("="*50)
    
    history = []
    
    while True:
        user_input = input("\n👤 你 (User): ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input:
            continue
            
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": user_input}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]
        
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=256,
                temperature=0.6, # 温度再低一点，更稳
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True,
                eos_token_id=stop_token_ids
            )
        
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        raw_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 🔥🔥🔥 调用智能剪刀 🔥🔥🔥
        final_response = smart_truncate(raw_response)
        
        print(f"🤖 张无忌: {final_response}")
        
        if len(history) > 6: history = history[-6:]
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": final_response}) # 存入历史的是剪切后的干净版本
        
        log_entry = {"input": user_input, "output": final_response, "raw": raw_response}
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()