from unsloth import FastLanguageModel
import torch
import json
import os

# 强制离线，防止它去联网查更新
os.environ["HF_HUB_OFFLINE"] = "1"

# 1. 加载我们已经下载好的基座模型
model_path = "/root/autodl-tmp/ZhangWuji_Project/models/base/Qwen/Qwen2.5-7B-Instruct"
print(f"🚀 Loading Qwen2.5 from: {model_path} ...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# 2. 加载试题
with open("data/test_set/exam.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

print("\n" + "="*50)
print("🤖 张无忌 (Base Model) 开始答题...")
print("="*50 + "\n")

# 3. 循环做题
for i, item in enumerate(questions):
    q = item['question']
    
    # 构造 Prompt
    messages = [
        {"role": "system", "content": "你现在是张无忌。"},
        {"role": "user", "content": q}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    # 生成
    outputs = model.generate(input_ids=inputs, max_new_tokens=256, temperature=0.7)
    response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

    # 打印结果
    print(f"📝 题目 [{i+1}]: {q}")
    print(f"🗣️ 回答: {response}")
    print("-" * 50)

print("✅ 所有题目回答完毕！")
