import os
import sys

# 锁定稳定版本
os.system("pip install rich -q") 
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

# 🔥 数据源换成：原著精华 + 合成数据的【混合版】
DATA_PATH = "data/processed/train_dataset_mixed.json" 

# 🔥 输出目录改名：v2_mixed，防止覆盖老模型
OUTPUT_DIR = "models/lora/zhangwuji_v2_mixed"

os.environ["HF_HUB_OFFLINE"] = "1"
# =========================================

def main():
    print(f"🚀 [V2 Mixed] Loading Model: {MODEL_PATH} ...")
    
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
    
    # 依然保留 EOS 修复逻辑，这个不能丢
    def format_prompts(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            messages = [
                {"role": "system", "content": example['system'][i]},
                {"role": "user", "content": example['instruction'][i]},
                {"role": "assistant", "content": example['output'][i]}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            if not text.endswith(tokenizer.eos_token):
                text += tokenizer.eos_token     
            output_texts.append(text)
        return output_texts

    print(f"📚 Loading Mixed Data: {DATA_PATH}")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    print("⚙️ Setting up Trainer...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        # 因为混合后数据量变多了(约250条)，我们稍微增加一点步数，保证训练充分
        max_steps=120, 
        logging_steps=1,
        save_strategy="steps",
        save_steps=60,
        fp16=False,
        bf16=True,
        optim="paged_adamw_32bit",
        report_to="none",
    )

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

    print("\n🔥 [V2 Mixed] Starting Training...")
    trainer.train()

    print(f"\n💾 Saving V2 Model to: {OUTPUT_DIR}")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("="*50)
    print("✅ V2 混合版训练完成！")
    print("="*50)

if __name__ == "__main__":
    main()
