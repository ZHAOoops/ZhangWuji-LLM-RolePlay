import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import os
import re
import argparse

# ================= 命令行参数 =================
parser = argparse.ArgumentParser()
parser.add_argument("--version", type=str, default="v3_final", help="默认加载 v3_final")
args = parser.parse_args()

BASE_MODEL_PATH = "/root/autodl-tmp/ZhangWuji_Project/models/base/Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = f"models/lora/zhangwuji_{args.version}"
LOG_FILE = f"logs/chat_history_{args.version}.jsonl"

# 🔥 System Prompt 再次加强：把亲属关系写死在这里，作为最后一道防线
SYSTEM_PROMPT = """你现在是张无忌。
身份：明教第三十四代教主，武当张翠山与天鹰教殷素素之子，谢逊的义子。
性格：宽厚、侠义、偶尔优柔寡断。
关系：赵敏是爱人（朝廷郡主），周芷若是青梅竹马（峨嵋掌门），殷离是表妹。
语言风格：简练、古风。面对现代概念要表现出好奇。"""

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
    print(f"🎉 张无忌 [{args.version}] 稳定防爆版已上线！")
    print("="*50)
    
    history = []
    
    while True:
        try:
            user_input = input("\n👤 你 (User): ").strip()
            if user_input.lower() in ["exit", "quit"]: break
            if not user_input: continue
            
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history, {"role": "user", "content": user_input}]
            
            # 🔥 修复核心：先转字符串，打印 debug，再转 Tensor
            # 这样绝对不会报 TypeError，因为我们手动控制了流程
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # 打印一下看看模型到底吃了什么（调试用，稳定后可注释掉）
            # print(f"\n[Debug] 输入模型的文本:\n{text[-100:]}...\n") 
            
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
            # 打印堆栈以便排查
            import traceback
            traceback.print_exc()
            if history: history.pop()

if __name__ == "__main__":
    main()