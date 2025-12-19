import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os

# 1. 路径配置
model_path = "/root/autodl-tmp/ZhangWuji_Project/models/base/Qwen/Qwen2.5-7B-Instruct"
data_path = "data/test_set/exam.json"

print(f"🚀 正在使用原生 Transformers 加载模型: {model_path} ...")

# 2. 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 3. 加载模型 (使用 bfloat16 原生精度，显存占用约 15GB，4090 轻松拿下)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

print("✅ 模型加载成功！开始答题...\n")

# 4. 读取试题
with open(data_path, "r", encoding="utf-8") as f:
    questions = json.load(f)

# 5. 循环答题
print("="*60)
for i, item in enumerate(questions):
    q = item['question']
    
    # 构造 Prompt
    messages = [
        {"role": "system", "content": "你现在是张无忌。"},
        {"role": "user", "content": q}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 生成回答
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.7,   # 稍微有点创造力
        top_p=0.9
    )
    
    # 解码
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 打印
    print(f"📝 问题 [{i+1}]: {q}")
    print(f"🗣️ 回答: {response}")
    print("-" * 60)

print("🎉 全部测试完成！")
