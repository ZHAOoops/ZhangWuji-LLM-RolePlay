import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import os
import datetime
import re

# ================= 配置区 =================
BASE_MODEL_PATH = "/root/autodl-tmp/ZhangWuji_Project/models/base/Qwen/Qwen2.5-7B-Instruct"

# 🔥 指向最新的 V2 混合模型
LORA_PATH = "models/lora/zhangwuji_v2_mixed"

LOG_FILE = "logs/chat_history_v2.jsonl"
SYSTEM_PROMPT = "你现在是张无忌。请以明教教主的身份，用简练、侠义的口吻回答。面对现代概念要表现出好奇或用武侠逻辑理解。"
# =========================================

def smart_truncate(text):
    """智能剪刀：只保留前3句"""
    if "User:" in text: text = text.split("User:")[0]
    if "Instruction:" in text: text = text.split("Instruction:")[0]
    sentences = re.split(r'(。|！|？|\n)', text)
    keep_count = 6 
    if len(sentences) > keep_count:
        return "".join(sentences[:keep_count])
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
    print("🎉 张无忌 V2 (完全体) 已上线！")
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
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True,
                eos_token_id=stop_token_ids
            )
        
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        raw_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        final_response = smart_truncate(raw_response)
        
        print(f"🤖 张无忌: {final_response}")
        
        if len(history) > 6: history = history[-6:]
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": final_response})
        
        log_entry = {"input": user_input, "output": final_response}
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
