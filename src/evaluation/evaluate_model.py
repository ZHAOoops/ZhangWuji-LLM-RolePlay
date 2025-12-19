import argparse
import torch
import pandas as pd
import json
import jieba
import os
from datetime import datetime
from unsloth import FastLanguageModel
from rouge_chinese import Rouge
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_metrics(generated, reference, embedding_model):
    rouge = Rouge()
    gen_tokens = ' '.join(jieba.cut(generated))
    ref_tokens = ' '.join(jieba.cut(reference))
    scores = rouge.get_scores(gen_tokens, ref_tokens)
    rouge_l = scores[0]['rouge-l']['f']
    
    embeddings = embedding_model.encode([generated, reference], convert_to_tensor=True)
    cosine_sim = util.cos_sim(embeddings[0], embeddings[1]).item()
    return rouge_l, cosine_sim

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/ZhangWuji_Project/models/base/Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/ZhangWuji_Project/logs/eval_reports")
    args = parser.parse_args()

    print(f"🚀 Loading model: {args.model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_path,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    print("📥 Loading embedding model...")
    sim_model = SentenceTransformer('sentence-transformers/text2vec-base-chinese')

    test_data = load_data(args.data_path)
    results = []
    
    print("🔥 Starting Inference...")
    for item in tqdm(test_data):
        question = item['question']
        inputs = tokenizer.apply_chat_template(
            [{"role": "system", "content": "你现在是张无忌。"}, {"role": "user", "content": question}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")

        outputs = model.generate(input_ids=inputs, max_new_tokens=256, temperature=0.7)
        generated = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        
        r_l, sim = calculate_metrics(generated, item['ref_answer'], sim_model)
        results.append({"q": question, "gen": generated, "rouge": r_l, "sim": sim})
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(os.path.join(args.output_dir, f"report_{timestamp}.csv"), index=False, encoding='utf-8-sig')
    print(f"✅ Average ROUGE-L: {df['rouge'].mean():.4f}")

if __name__ == "__main__":
    main()
