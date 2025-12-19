import os
import sys

# ==========================================
# 🔥 自动修复环境区
# ==========================================
print("🔧 正在补全缺失的依赖 (rich)...")
os.system("pip install rich -q") 
# 再次确认 trl 版本
os.system("pip install trl==0.8.6 peft==0.10.0 -q")

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer 

# ================= 配置区 =================
MODEL_PATH = "/root/autodl-tmp/ZhangWuji_Project/models/base/Qwen/Qwen2.5-7B-Instruct"
DATA_PATH = "data/processed/train_dataset_final_quality.json"
OUTPUT_DIR = "models/lora/zhangwuji_v1_native"
os.environ["HF_HUB_OFFLINE"] = "1"
# =========================================

def main():
    print(f"🚀 [Final] Loading Model: {MODEL_PATH} ...")
    
    # 1. 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 2. 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)

    # 3. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 4. 配置 LoRA
    print("🔧 Configuring LoRA...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # 5. 格式化提示词
    def format_prompts(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            text = f"<|im_start|>system\n{example['system'][i]}<|im_end|>\n<|im_start|>user\n{example['instruction'][i]}<|im_end|>\n<|im_start|>assistant\n{example['output'][i]}<|im_end|>"
            output_texts.append(text)
        return output_texts

    print(f"📚 Loading Data: {DATA_PATH}")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    # 6. 训练参数 (使用最兼容的写法)
    print("⚙️ Setting up Trainer...")
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

    # 7. 启动训练 (v0.8.6 风格)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=format_prompts,
        max_seq_length=2048,
        args=training_args,
        packing=False,
    )

    print("\n🔥 [Final] Starting Training...")
    trainer.train()

    # 8. 保存
    print(f"\n💾 Saving Model to: {OUTPUT_DIR}")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("="*50)
    print("✅ 训练完成！")
    print("="*50)

if __name__ == "__main__":
    main()
