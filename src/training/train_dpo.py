import os
import sys

# =====================================================
# 🔥 保持环境锁定
# =====================================================
print("🔧 环境检查 (transformers==4.41.2, trl==0.8.6)...")
# 这里假设你刚才已经运行过安装命令了，不需要重复跑
# os.system("pip install transformers==4.41.2 trl==0.8.6 accelerate==0.30.1 -q")

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel
from trl import DPOTrainer

# ================= 配置区 =================
MODEL_PATH = "/root/autodl-tmp/ZhangWuji_Project/models/base/Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "models/lora/zhangwuji_v4_family_fix" 
DATA_PATH = "data/dpo/train_dpo.json"
OUTPUT_DIR = "models/lora/zhangwuji_v5_dpo"
os.environ["HF_HUB_OFFLINE"] = "1"
# =========================================

def main():
    print("🚀 [DPO] Loading Base Model...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)
    
    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    # 🔥🔥🔥 核心修复区：强行焊死 Token ID，防止 NoneType 报错 🔥🔥🔥
    # Qwen 的 <|im_end|> ID 是 151645
    print("🔧 强制设置 Token IDs...")
    tokenizer.pad_token_id = 151645
    tokenizer.eos_token_id = 151645
    # DPO 经常需要 BOS token，Qwen 没有，我们强行指向 EOS，防止报错
    tokenizer.bos_token_id = 151645 
    
    # 这一步非常关键：同步给模型配置，否则训练器可能读取错误的 config
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # 2. 加载 SFT LoRA
    print(f"🔗 Loading Adapter: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH, is_trainable=True)

    # 3. 数据处理
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    def process_dpo_data(example):
        messages = [
            {"role": "system", "content": example['system']},
            {"role": "user", "content": example['instruction']}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {
            "prompt": prompt,
            "chosen": example['chosen'],
            "rejected": example['rejected']
        }

    print("🔄 Formatting Data...")
    dataset = dataset.map(process_dpo_data)

    print("⚔️ 配置 DPO Trainer...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-6,
        max_steps=50,
        logging_steps=1,
        save_strategy="steps",
        save_steps=25,
        fp16=False,
        bf16=True,
        optim="paged_adamw_32bit",
        report_to="none",
        remove_unused_columns=False,
        # 🔥 显式告诉 Trainer 忽略某些不匹配，防止它自作聪明去检查
        label_names=["labels"] 
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        beta=0.1, 
        max_length=1024,
        max_prompt_length=512,
    )

    print("🔥 Starting DPO Training...")
    trainer.train()

    print(f"💾 Saving DPO Model to: {OUTPUT_DIR}")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ DPO 训练大成功！")

if __name__ == "__main__":
    main()
