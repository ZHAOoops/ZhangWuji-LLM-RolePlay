import json
import os

INPUT_FILE = "data/processed/train_dataset_dialogue.json"
OUTPUT_FILE = "data/processed/train_dataset_final.json"

def is_valid_wuji_response(item):
    instr = item['instruction']
    out = item['output']
    
    # 1. 致命错误：Instruction 里不能是张无忌在说话
    # 如果输入是“无忌道...”，那模型就在学“谢逊/赵敏”了
    if "无忌" in instr and ("道" in instr or "问" in instr or "说" in instr or "笑" in instr):
        return False, "角色互换：Instruction是无忌"

    # 2. 致命错误：Output 里不能是别人在说话
    # 简单的黑名单机制
    others = ["赵敏", "芷若", "谢逊", "义父", "小昭", "杨逍", "灭绝", "太师父", "敏敏"]
    for name in others:
        if name in out and ("道" in out or "曰" in out or "说" in out):
            return False, f"角色互换：Output疑似是{name}"

    # 3. 质量控制：必须是对话（包含引号）
    # 除非是心理活动（包含“心想”），否则必须有引号
    if "“" not in out and "心想" not in out and "道" not in out:
        return False, "旁白/非对话"

    # 4. 长度控制
    if len(out) < 4:
        return False, "回复太短"

    return True, "通过"

def main():
    print(f"🧹 开始清洗数据: {INPUT_FILE}")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    clean_data = []
    dropped_counts = {}
    
    for item in raw_data:
        valid, reason = is_valid_wuji_response(item)
        if valid:
            # 进一步清洗：去掉“无忌道：”这种前缀，只留内容
            # 这一步是为了让模型学会直接说话，而不是复述“我说道：...”
            out_clean = item['output']
            if "无忌" in out_clean and "道" in out_clean and "：" in out_clean:
                 # 尝试提取引号里的内容
                 start = out_clean.find("“")
                 end = out_clean.rfind("”")
                 if start != -1 and end != -1:
                     out_clean = out_clean[start+1 : end]
            
            clean_data.append({
                "instruction": item['instruction'],
                "input": "",
                "output": out_clean,
                "system": item['system']
            })
        else:
            dropped_counts[reason] = dropped_counts.get(reason, 0) + 1

    # 保存
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(clean_data, f, ensure_ascii=False, indent=2)

    print("="*50)
    print(f"📉 清洗报告：")
    print(f"   原始数量: {len(raw_data)}")
    print(f"   剩余数量: {len(clean_data)}")
    print(f"   丢弃详情: {dropped_counts}")
    print("="*50)
    
    if clean_data:
        print("\n👀 [Final Check] 现在的第1条数据 (必须是别人问->无忌答):")
        print(json.dumps(clean_data[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
