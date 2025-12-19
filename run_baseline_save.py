import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import pandas as pd
import os
from datetime import datetime

# 1. 配置
model_path = "/root/autodl-tmp/ZhangWuji_Project/models/base/Qwen/Qwen2.5-7B-Instruct"
data_path = "data/test_set/exam.json"
output_dir = "logs/eval_reports"
os.makedirs(output_dir, exist_ok=True)

print(f"🚀 Loading Model from: {model_path} ...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="auto", 
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True
)

# 2. 读取题目
with open(data_path, "r", encoding="utf-8") as f:
    questions = json.load(f)

results = []
print("="*60)

# 3. 答题并记录
for item in questions:
    q = item['question']
    ref = item['ref_answer']
    
    messages = [
        {"role": "system", "content": "你现在是张无忌。"},
        {"role": "user", "content": q}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9
    )
    
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # 存入列表
    results.append({
        "question": q,
        "ref_answer": ref,
        "base_model_answer": response
    })
    print(f"✅ 已记录: {q[:10]}...")

# 4. 保存为 CSV
df = pd.DataFrame(results)
timestamp = datetime.now().strftime("%Y%m%d")
filename = f"{output_dir}/exam_result_baseline_{timestamp}.csv"
df.to_csv(filename, index=False, encoding="utf-8-sig")

print("="*60)
print(f"🎉 存档完成！文件已保存至: {filename}")
print("这就是我们的‘一号选手’答卷，请务必妥善保管！")
