import os
import sys

# 🔥 强力修复：强制安装最稳定的 trl 版本
# (这步会自动运行，不用手动敲)
print("🔧正在锁定黄金环境 (trl==0.8.6)...")
os.system("pip install trl==0.8.6 -q")
os.system("pip install peft==0.10.0 -q") 

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer # 现在这是稳定的 0.8.6 版本

# ================= 配置区 =================
MODEL_PATH = "/root/autodl-tmp/ZhangWuji_Project/models/base/Qwen/Qwen2.5-7B-Instruct"
DATA_PATH = "data/processed/train_dataset_final_quality.json"
OUTPUT_DIR = "models/lora/zhangwuji_v1_native"
os.environ["HF_HUB_OFFLINE"] = "1"
# =========================================

def main():
    print(f"🚀 [Stable] Loading Model: {MODEL_PATH} ...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("🔧 Configuring LoRA...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # 格式化函数
    def format_prompts(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            text = f"<|im_start|>system\n{example['system'][i]}<|im_end|>\n<|im_start|>user\n{example['instruction'][i]}<|im_end|>\n<|im_start|>assistant\n{example['output'][i]}<|im_end|>"
            output_texts.append(text)
        return output_texts

    print(f"📚 Loading Data: {DATA_PATH}")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    print("⚙️ Setting up Trainer (Standard Mode)...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_steps=100,
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,
        fp16=False,
        bf16=True,
        optim="paged_adamw_32bit",
        report_to="none",
    )

    # 🔥 0.8.6 版本的经典写法，绝对不报错
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,       # 老版本就叫 tokenizer，稳！
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=format_prompts,
        max_seq_length=2048,       # 老版本这里支持 max_seq_length，稳！
        args=training_args,
        packing=False,
    )

    print("\n🔥 [Stable] Starting Training...")
    trainer.train()

    print(f"\n💾 Saving Model to: {OUTPUT_DIR}")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("="*50)
    print("✅ 训练大成功！")
    print("="*50)

if __name__ == "__main__":
    main()
