import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import os
import re
import argparse
import readline  # 🔥 关键修复 1：引入 readline，让 input() 支持退格和历史

# ================= 命令行参数 =================
parser = argparse.ArgumentParser()
parser.add_argument("--version", type=str, default="v5_dpo", help="默认加载 v5_dpo，也可以切回 v4_family_fix")
args = parser.parse_args()

BASE_MODEL_PATH = "/root/autodl-tmp/ZhangWuji_Project/models/base/Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = f"models/lora/zhangwuji_{args.version}"
LOG_FILE = f"logs/chat_history_{args.version}.jsonl"

SYSTEM_PROMPT = """你现在是张无忌。
身份：明教第三十四代教主，武当张翠山与天鹰教殷素素之子，谢逊的义子。
性格：宽厚、侠义、重情重义。
关系：赵敏是爱人（朝廷郡主），周芷若是青梅竹马（峨嵋掌门），张三丰是太师父。
语言风格：简练、古风。"""

def clean_input_text(text):
    """🔥 关键修复 2：清洗输入，去掉不可见字符和控制码"""
    if not text: return ""
    # 去掉像 \x08 (Backspace) 这样的控制字符
    # 只保留可打印字符，或者汉字
    cleaned = "".join(ch for ch in text if ch.isprintable() or '\u4e00' <= ch <= '\u9fff')
    return cleaned.strip()

def smart_truncate(text):
    if not text: return "（张无忌正在沉思……）"
    text = text.replace("User:", "").replace("Instruction:", "").strip()
    sentences = re.split(r'(。|！|？|\n)', text)
    keep_count = 6 
    if len(sentences) > keep_count:
        return "".join(sentences[:keep_count])
    else:
        return text

def main():
    if not os.path.exists(LORA_PATH):
        print(f"❌ 错误：找不到路径 {LORA_PATH}")
        return

    print(f"🚀 Loading Base Model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    print(f"🔗 Loading LoRA: {args.version} ...")
    model = PeftModel.from_pretrained(model, LORA_PATH)
    
    print("="*50)
    print(f"🎉 张无忌 [{args.version}] 终极体验版已上线！")
    print("💡 提示：现在可以放心使用退格键修改错误了。")
    print("="*50)
    
    history = []
    
    while True:
        try:
            # 原生 input 在 import readline 后会自动变强
            raw_input = input("\n👤 你 (User): ")
            
            # 再次清洗，双重保险
            user_input = clean_input_text(raw_input)
            
            if user_input.lower() in ["exit", "quit"]: break
            if not user_input: continue
            
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history, {"role": "user", "content": user_input}]
            
            # 转 Tensor
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
            
            if len(history) > 8: history = history[-8:]
            history.append({"role": "user", "content": str(user_input)})
            history.append({"role": "assistant", "content": str(final_response)})
            
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps({"input": user_input, "output": final_response}, ensure_ascii=False) + "\n")
                
        except Exception as e:
            print(f"\n❌ 异常: {e}")
            if history: history.pop()

if __name__ == "__main__":
    main()
