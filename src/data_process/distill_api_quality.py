import json
import os
from openai import OpenAI
from tqdm import tqdm
import concurrent.futures

# ================= 配置区 =================
API_KEY = ""
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
MODEL_NAME = "deepseek-v3-2-251201"

INPUT_FILE = "data/processed/wuji_chunks.jsonl"
OUTPUT_FILE = "data/processed/train_dataset_final_quality.json"

# 仅保留成年后的核心人物，排除父辈人物（如谢逊早期、张三丰早期）
TARGET_ROLES = ["赵敏", "周芷若", "小昭", "殷离", "杨逍", "韦一笑", "范遥", "灭绝", "成昆", "鹿杖客", "鹤笔翁", "朱元璋"]

# 硬性黑名单：如果 Output 里出现这些词，直接视为脏数据丢弃
BLACKLIST_WORDS = ["张翠山", "翠山", "五弟", "素素", "殷素素", "五哥", "恩师", "郭襄"]
# =========================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def extract_quality_dialogue(chunk_text):
    """
    使用极严苛的 Prompt 提取数据
    """
    prompt = f"""
    任务：从《倚天屠龙记》片段中提取【成年张无忌】（明教教主时期）的对话。
    
    ❌ 严禁提取以下内容（负面约束）：
    1. 严禁提取张翠山（父亲）、殷素素（母亲）的对话。
    2. 严禁提取童年/少年时期的对话（如冰火岛时期、蝴蝶谷时期）。
    3. 严禁提取旁白、心理活动、动作描写，只提取“口语”。
    4. 严禁角色互换（Instruction必须是他人，Output必须是张无忌）。

    ✅ 提取格式要求：
    1. 返回 JSON List。
    2. "instruction": 对方的名字 + 冒号 + 对方说的话 (例如 "赵敏：...")
    3. "output": 张无忌说的话 (不要带 "张无忌道：", 直接写内容)

    小说片段：
    {chunk_text}
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个严格的数据清洗专家。绝不提取错误的父辈剧情。"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3 # 降低温度，让模型更保守、更听话
        )
        data = json.loads(response.choices[0].message.content)
        
        # 兼容性处理
        results = []
        if isinstance(data, dict):
            for key, val in data.items():
                if isinstance(val, list): results = val
        elif isinstance(data, list):
            results = data
            
        return results
    except:
        return []

def process_chunk(chunk):
    text = chunk['text']
    
    # 1. 预过滤：如果没有核心人物，大概率不是我们要的教主剧情
    # (这能帮我们过滤掉大量张翠山时期的剧情，因为那时候赵敏周芷若还没出生)
    if not any(role in text for role in TARGET_ROLES):
        return []
    
    raw_items = extract_quality_dialogue(text)
    
    clean_items = []
    for item in raw_items:
        instr = item.get("instruction", "").strip()
        out = item.get("output", "").strip()
        
        # 2. Python 硬规则清洗
        if len(instr) < 3 or len(out) < 2: continue
        
        # 检查 Output 是否包含父辈黑名单词汇 (如 "我是翠山")
        if any(bad_word in out for bad_word in BLACKLIST_WORDS):
            continue
        
        # 检查 Instruction 是否包含张无忌 (防止无忌自言自语被录入)
        if "无忌" in instr or "教主" in instr:
             # 如果 Instruction 是“无忌道：...”，这说明提取反了，丢弃
             if "道" in instr or "说" in instr:
                 continue

        # 检查 Output 是否包含他人名字 (防止角色互换)
        # 例如 Output: "赵敏笑道..." -> 错
        if any(role in out for role in ["赵敏", "芷若", "小昭", "杨逍"]):
             if "道" in out or "说" in out:
                 continue

        clean_items.append({
            "instruction": instr,
            "input": "",
            "output": out,
            "system": "你现在是张无忌，请以明教教主的身份，用武侠风格回答。"
        })
        
    return clean_items

def main():
    print(f"🚀 启动终极质量提取 (DeepSeek V3)...")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        all_chunks = [json.loads(line) for line in f]
    
    # 全量扫描 (既然你有API，我们就跑全一点，保证数量)
    # 这里的 filter 会过滤掉大概 60% 的非核心剧情片段
    print(f"📂 待扫描片段池: {len(all_chunks)} 个")
    
    final_data = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in all_chunks]
        
        pbar = tqdm(total=len(futures), desc="⚡️ 提取中")
        for future in concurrent.futures.as_completed(futures):
            results = future.result()
            if results:
                final_data.extend(results)
                pbar.set_postfix({"✅ 高质量条目": len(final_data)})
            pbar.update(1)
            
    # 保存
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
        
    print("="*50)
    print(f"🎉 提取结束！")
    print(f"   最终数据集: {OUTPUT_FILE}")
    print(f"   条数: {len(final_data)}")
    print("="*50)
    
    if final_data:
        print("\n👀 抽查第一条 (必须是教主):")
        print(json.dumps(final_data[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
