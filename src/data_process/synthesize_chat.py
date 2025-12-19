import json
import os
from openai import OpenAI
from tqdm import tqdm
import time

# ================= 配置区 =================
API_KEY = "718e7455-5e90-4d7b-8c47-7a2ac5c89611"
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
MODEL_NAME = "deepseek-v3-2-251201"

OUTPUT_FILE = "data/processed/train_dataset_synthetic.json"

# 🔥 四大核心场景，解决当前所有的痛点
SCENARIOS = {
    "identity": {
        "count": 30,
        "prompt": "生成用户关于询问张无忌身份的对话。包括：'你是谁'、'你是教主吗'、'你会什么武功'、'你义父是谁'。要求回答必须确认自己是明教教主，性格谦逊但自信。"
    },
    "modern_tech": {
        "count": 30,
        "prompt": "生成现代人向张无忌询问现代科技的对话。包括：手机、电脑、AI、飞机、DeepSeek、微信。要求张无忌表现出好奇、不懂，或者用武侠世界的概念去理解（比如把手机当成千里传音），切勿惊恐或喊打喊杀。"
    },
    "romance": {
        "count": 30,
        "prompt": "生成用户向张无忌表达爱意或询问情感的对话。包括：'我喜欢你'、'选赵敏还是周芷若'、'想和你结婚'。要求张无忌表现出优柔寡断、害羞、心里惦记着赵敏但又不想伤人的‘渣男/暖男’特质。"
    },
    "daily_chat": {
        "count": 30,
        "prompt": "生成日常闲聊对话。包括：'吃了吗'、'在干嘛'、'累不累'、'心情好吗'。要求回答充满江湖气息，比如在练功、在处理教务、在想念义父等。"
    }
}
# =========================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def generate_batch(scenario_name, config):
    print(f"⚡️ 正在生成场景: {scenario_name} ...")
    
    prompt = f"""
    你是一个数据生成助手。请帮我生成 {config['count']} 组关于《倚天屠龙记》张无忌的角色扮演对话数据。
    
    【场景要求】：{config['prompt']}
    
    【格式要求】：
    1. 返回一个 JSON List。
    2. 每个元素包含 "instruction" (用户的话) 和 "output" (张无忌的回答)。
    3. 张无忌的回答必须简短（50字以内），口语化，符合原著人设。
    4. 不要带翻译，只要中文。
    
    【示例】：
    [
        {{"instruction": "你有手机吗？", "output": "手机？那是何物？可是西域传来的新奇暗器？我从未听闻。"}},
        {{"instruction": "你是明教教主吗？", "output": "承蒙各位兄弟错爱，推举在下暂代教主之位。在下才疏学浅，只盼能为驱除鞑子尽一份力。"}}
    ]
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.8
        )
        data = json.loads(response.choices[0].message.content)
        
        # 兼容处理
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, list): return v
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"Error generating {scenario_name}: {e}")
        return []

def main():
    all_data = []
    
    for name, config in SCENARIOS.items():
        batch_data = generate_batch(name, config)
        print(f"   ✅ {name}: 生成了 {len(batch_data)} 条")
        
        # 格式化
        for item in batch_data:
            all_data.append({
                "instruction": item['instruction'],
                "input": "",
                "output": item['output'],
                "system": "你现在是张无忌，请以明教教主的身份，用武侠风格回答。"
            })
            
        time.sleep(1) # 防止API限流

    # 保存
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
        
    print("="*50)
    print(f"🎉 合成数据生成完毕！")
    print(f"   文件路径: {OUTPUT_FILE}")
    print(f"   总条数: {len(all_data)} 条")
    print("="*50)
    
    # 预览
    print(json.dumps(all_data[:3], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
