import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import os
import datetime
import re

# ================= 配置区 =================
BASE_MODEL_PATH = "/root/autodl-tmp/ZhangWuji_Project/models/base/Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = "models/lora/zhangwuji_v2_mixed"
LOG_FILE = "logs/chat_history_final.jsonl"

# 🔥 升级版 System Prompt：锁死技能树，防止幻觉
SYSTEM_PROMPT = """你现在是张无忌。
身份：明教第三十四代教主。
性格：宽厚、侠义、偶尔优柔寡断，对长辈恭敬，对女子心软。
武功：九阳神功、乾坤大挪移、太极拳剑、圣火令武功。（严禁胡编降龙十八掌、六脉神剑等他人武功）
语言风格：简练、古风。面对现代概念（如手机、AI、股市）要表现出好奇，或尝试用江湖逻辑去理解，不要惊恐。"""
# =========================================

def smart_truncate(text):
    """智能剪刀"""
    if not text: return "（张无忌陷入沉思……）"
    
    # 清洗特殊标记
    text = text.replace("User:", "").replace("Instruction:", "")
    
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
    print("🎉 张无忌 V2 (防爆版) 已上线！")
    print("="*50)
    
    history = []
    
    while True:
        try:
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
            
            # 🔥 防爆处理 1：确保 text 是字符串
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if not text:
                print("⚠️ 警告：模板生成为空，跳过此轮")
                continue
                
            model_inputs = tokenizer([str(text)], return_tensors="pt").to(model.device)
            
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
            
            # 🔥 防爆处理 2：限制历史长度，防止上下文溢出导致崩坏
            if len(history) > 8: history = history[-8:]
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": final_response})
            
            log_entry = {"input": user_input, "output": final_response}
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            print("   (张无忌这会儿可能走火入魔了，请换个话题重试)")
            # 出错时清空最近一条历史，防止死循环
            if history: history.pop()

if __name__ == "__main__":
    main()
