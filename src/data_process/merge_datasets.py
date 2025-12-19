import json
import random

# 1. 原著提取的精华数据 (假设你刚才清洗完叫 train_dataset_pure.json，如果没有就用 train_dataset_final_quality.json)
NOVEL_DATA = "data/processed/train_dataset_final_quality.json"

# 2. 刚才生成的合成数据
SYNTHETIC_DATA = "data/processed/train_dataset_synthetic.json"

# 3. 最终混合数据
OUTPUT_FILE = "data/processed/train_dataset_mixed.json"

def main():
    print("🥣 正在混合数据...")
    
    with open(NOVEL_DATA, 'r') as f:
        novel = json.load(f)
        
    with open(SYNTHETIC_DATA, 'r') as f:
        synthetic = json.load(f)
        
    # 可以在这里调整比例，比如合成数据复制一遍增加权重
    # synthetic = synthetic * 2 
    
    combined = novel + synthetic
    random.shuffle(combined) # 打乱顺序
    
    print(f"   📖 原著数据: {len(novel)} 条")
    print(f"   🧪 合成数据: {len(synthetic)} 条")
    print(f"   📦 总计: {len(combined)} 条")
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
        
    print(f"✅ 混合完毕: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
