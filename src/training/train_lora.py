from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import os

# ================= 配置区 =================
MODEL_PATH = "/root/autodl-tmp/ZhangWuji_Project/models/base/Qwen/Qwen2.5-7B-Instruct"

# 🔥 核心：指向 DeepSeek 提取的高质量数据
DATA_PATH = "data/processed/train_dataset_final_quality.json"

OUTPUT_DIR = "models/lora/zhangwuji_v1"
MAX_SEQ_LENGTH = 2048
# =========================================

def main():
    print(f"🚀 Loading Model: {MODEL_PATH} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_PATH,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,
        load_in_4bit = True,
    )

    print("🔧 Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, 
        bias = "none", 
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    alpaca_prompt = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        outputs      = examples["output"]
        systems      = examples["system"]
        texts = []
        for input, output, system in zip(instructions, outputs, systems):
            text = alpaca_prompt.format(
                system = system,
                instruction = input,
                output = output,
            ) + tokenizer.eos_token
            texts.append(text)
        return { "text" : texts, }

    print(f"📚 Loading Dataset: {DATA_PATH}")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    dataset = dataset.map(formatting_prompts_func, batched = True,)

    print("⚙️ Setting up Trainer...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = 2, 
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            # 3.7 个 Epoch (221条数据 / 8 batch size * 3.7 ≈ 100步)
            max_steps = 100, 
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = OUTPUT_DIR,
        ),
    )

    print("\n🔥 Starting Training...")
    trainer_stats = trainer.train()

    print(f"\n💾 Saving LoRA Model to: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("="*50)
    print("✅ 训练完成！")
    print(f"模型已保存至: {OUTPUT_DIR}")
    print("="*50)

if __name__ == "__main__":
    main()
