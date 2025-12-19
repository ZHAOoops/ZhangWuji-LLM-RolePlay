import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

# 解析版本号
parser = argparse.ArgumentParser()
parser.add_argument("--version", type=str, required=True, help="版本号，如 v3_final")
args = parser.parse_args()

MODEL_PATH = "/root/autodl-tmp/ZhangWuji_Project/models/base/Qwen/Qwen2.5-7B-Instruct"
DATA_PATH = "data/processed/train_dataset_mixed.json"
OUTPUT_DIR = f"models/lora/zhangwuji_{args.version}"

print(f"🎯 训练目标: {args.version}")
print(f"📂 输出路径: {OUTPUT_DIR}")

def main():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
    model = prepare_model_for_kbit_training(model)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(r=16, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")

    def format_prompts(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            messages = [
                {"role": "system", "content": example['system'][i]},
                {"role": "user", "content": example['instruction'][i]},
                {"role": "assistant", "content": example['output'][i]}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            if not text.endswith(tokenizer.eos_token): text += tokenizer.eos_token     
            output_texts.append(text)
        return output_texts

    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    # 稍微增加步数到 150，因为数据越来越多了
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR, per_device_train_batch_size=2, gradient_accumulation_steps=4, learning_rate=2e-4, 
        max_steps=150, logging_steps=1, save_strategy="steps", save_steps=50, fp16=False, bf16=True, optim="paged_adamw_32bit", report_to="none"
    )

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=dataset, peft_config=peft_config, formatting_func=format_prompts, max_seq_length=2048, args=training_args, packing=False
    )
    
    print("🔥 Starting Training...")
    trainer.train()
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ 完成！模型已保存到 {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
