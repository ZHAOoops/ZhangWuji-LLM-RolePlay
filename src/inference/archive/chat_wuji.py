import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import os
import datetime

# ================= 配置区 =================
BASE_MODEL_PATH = "/root/autodl-tmp/ZhangWuji_Project/models/base/Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = "models/lora/zhangwuji_v1_native"
LOG_FILE = "logs/chat_history.jsonl"

SYSTEM_PROMPT = "你现在是张无忌，身处倚天屠龙记的武侠世界。请以明教教主的身份，用谦和、侠义但偶尔纠结的口吻回答。回答要简短有力，不要长篇大论。"
# =========================================

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
    print("🎉 张无忌已上线！(输入 'exit' 退出)")
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
        
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=256, # 🔥 缩短最大生成长度，强制它精简
                temperature=0.6,    # 🔥稍微降低温度，减少胡言乱语
                top_p=0.9,
                repetition_penalty=1.2, # 🔥 核心救命药：如果它开始复读，立刻重罚，强行打断
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id # 确保它知道怎么结束
            )
        
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"🤖 张无忌: {response}")
        
        # 简单的历史记录管理，防止上下文过长导致复读
        if len(history) > 10: 
            history = history[-10:]
            
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
        
        log_entry = {
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input": user_input,
            "output": response
        }
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()