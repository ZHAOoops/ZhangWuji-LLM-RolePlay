import json
import os
from openai import OpenAI
from tqdm import tqdm
import concurrent.futures
import time

# ================= 配置区 =================
# 你提供的火山引擎配置
API_KEY = "718e7455-5e90-4d7b-8c47-7a2ac5c89611"
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
MODEL_NAME = "deepseek-v3-2-251201"

# 文件路径
INPUT_FILE = "data/processed/wuji_chunks.jsonl"
OUTPUT_FILE = "data/processed/train_dataset_dialogue.json"

# 核心人物过滤（只处理这些人的片段，保证含金量）
TARGET_ROLES = ["赵敏", "周芷若", "小昭", "殷离", "蛛儿", "谢逊", "义父", "张三丰", "杨逍", "范遥", "灭绝", "金花婆婆"]

# 最大处理片段数（设为 500 足够凑齐几百条高质量对话了，想跑全本可以改大）
MAX_CHUNKS = 500
# =========================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def extract_dialogue(chunk_text):
    """调用 API 提取对话"""
    prompt = f"""
    你是《倚天屠龙记》剧本专家。请阅读片段，提取【他人】与【张无忌】的精彩对话。
    
    要求：
    1. 输出标准的 JSON List。
    2. 包含两个字段："instruction" (对方说的话，带上人名，如"赵敏笑道：...") 和 "output" (张无忌的回答)。
    3. 只提取直接对话，去除无关旁白。
    4. 如果片段中没有有效对话，返回空列表 []。
    
    片段：
    {chunk_text}
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个严格遵循JSON格式的数据提取助手。"},
                {"role": "user", "content": prompt}
            ],
            # DeepSeek V3 听得懂这个指令，会强制返回 JSON
            response_format={"type": "json_object"}, 
            temperature=0.7
        )
        content = response.choices[0].message.content
        
        # 解析 JSON
        data = json.loads(content)
        
        # 兼容处理：有时候模型会把 list 包在 key 里，有时候直接返回 list
        if isinstance(data, dict):
            # 尝试找可能的 key
            for key in ["dialogues", "conversations", "data", "pairs"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
            # 如果没找到常见 key，看看能不能强行拿 values
            return []
        elif isinstance(data, list):
            return data
        return []
        
    except Exception as e:
        # print(f"API Error: {e}") # 调试时可以打开
        return []

def process_chunk_wrapper(chunk):
    """包装函数，用于线程池"""
    text = chunk['text']
    
    # === 本地预过滤 (省钱大法) ===
    # 1. 必须有引号
    if "“" not in text: return []
    # 2. 必须有核心人物
    if not any(role in text for role in TARGET_ROLES): return []
    # 3. 必须有无忌
    if "无忌" not in text and "教主" not in text: return []
    
    # 满足条件，调用 API
    raw_pairs = extract_dialogue(text)
    
    # 格式化为训练数据
    formatted_data = []
    for pair in raw_pairs:
        if "instruction" in pair and "output" in pair:
            # 简单清洗
            instr = pair['instruction'].strip()
            out = pair['output'].strip()
            if len(instr) > 2 and len(out) > 1:
                formatted_data.append({
                    "instruction": instr,
                    "input": "",
                    "output": out,
                    "system": "你现在是张无忌，请以明教教主的身份，用武侠风格回答。"
                })
    return formatted_data

def main():
    print(f"🚀 启动 API 极速提取模式 (Model: {MODEL_NAME})")
    
    # 1. 读取片段
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        all_chunks = [json.loads(line) for line in f]
    
    # 限制数量
    target_chunks = all_chunks[:MAX_CHUNKS]
    print(f"📂 待扫描片段: {len(target_chunks)} 个 (已开启核心人物过滤)")
    
    final_dataset = []
    
    # 2. 并发执行
    # 建议设为 10-20，取决于你的 API 限流策略
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # 提交任务
        futures = [executor.submit(process_chunk_wrapper, chunk) for chunk in target_chunks]
        
        # 进度条监控
        pbar = tqdm(total=len(futures), desc="⚡️ API 提取中")
        for future in concurrent.futures.as_completed(futures):
            results = future.result()
            if results:
                final_dataset.extend(results)
                pbar.set_postfix({"已获数据": len(final_dataset)})
            pbar.update(1)
            
    # 3. 保存结果
    print(f"\n💾 正在保存 {len(final_dataset)} 条数据...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)
        
    print("="*50)
    print(f"🎉 大功告成！")
    print(f"   最终数据集: {OUTPUT_FILE}")
    print(f"   数据量: {len(final_dataset)} 条")
    print("="*50)
    
    if final_dataset:
        print("\n👀 质量抽查:")
        print(json.dumps(final_dataset[:2], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
