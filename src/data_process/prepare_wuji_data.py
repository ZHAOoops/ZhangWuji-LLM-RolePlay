import re
import json
import os
from tqdm import tqdm

# ================= 配置区 =================
INPUT_FILE = "data/v0_raw/yitian_full.txt"
OUTPUT_FILE = "data/processed/wuji_chunks.jsonl"
KEYWORDS = ["张无忌", "无忌", "张教主", "曾阿牛"]
CHUNK_SIZE = 1000  # 稍微加大一点
OVERLAP = 200      # 重叠区域
# =========================================

def clean_text(text):
    """深度清洗文本"""
    print("   [1/3] 正在去除网页标记...")
    text = re.sub(r'={3,}.*?={3,}', '', text)
    text = re.sub(r'第.+?回\s+.+', '', text)
    
    print("   [2/3] 正在压缩空白字符...")
    # 将连续的换行和空格压缩成单一换行，保持段落结构
    text = re.sub(r'\n\s*\n', '\n\n', text) 
    return text.strip()

def create_chunks(text, chunk_size, overlap):
    chunks = []
    start = 0
    text_len = len(text)
    
    # 进度条
    pbar = tqdm(total=text_len, desc="🔪 切片进度", unit="char")
    
    while start < text_len:
        # 1. 确定粗略结束点
        end = min(start + chunk_size, text_len)
        
        # 2. 优化截断点（尽量在句号或换行处截断，不要切断句子）
        # 只在不是文件末尾时寻找截断点
        if end < text_len:
            # 在最后100个字符里找句号
            search_buffer = text[max(start, end-150):end]
            last_break = max(search_buffer.rfind('。'), search_buffer.rfind('\n'))
            
            if last_break != -1:
                # 调整 end 到句号后面
                end = max(start, end - 150) + last_break + 1
        
        chunk = text[start:end]
        
        if chunk.strip():
            chunks.append(chunk)
        
        # 3. 计算步长 (关键修复：防止死循环)
        # 正常步长是 (当前片段长度 - 重叠量)
        # 如果到了末尾片段很短，可能会导致步长为负，这里强制最小步长为 1
        step = max(1, len(chunk) - overlap)
        
        # 如果已经到了文件末尾，强制结束
        if end == text_len:
            pbar.update(text_len - start)
            break
            
        start += step
        pbar.update(step)
        
    pbar.close()
    return chunks

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误：找不到文件 {INPUT_FILE}")
        return

    print(f"📖 读取文件: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_content = f.read()
    
    print(f"   原始大小: {len(raw_content)} 字符")

    print("🧹 开始清洗...")
    cleaned_text = clean_text(raw_content)

    print("🔪 开始切片...")
    all_chunks = create_chunks(cleaned_text, CHUNK_SIZE, OVERLAP)
    print(f"   共切出 {len(all_chunks)} 个片段")

    print("🔍 筛选含关键词片段...")
    selected_chunks = []
    for chunk in tqdm(all_chunks, desc="筛选进度"):
        if any(keyword in chunk for keyword in KEYWORDS):
            selected_chunks.append({"text": chunk, "source": "yitian_novel"})

    # 保存
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in selected_chunks:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("="*40)
    print(f"🎉 处理完成！")
    print(f"   原始片段: {len(all_chunks)}")
    print(f"   张无忌相关片段: {len(selected_chunks)} (保留率: {len(selected_chunks)/len(all_chunks):.1%})")
    print(f"   结果已保存至: {OUTPUT_FILE}")
    print("="*40)

if __name__ == "__main__":
    main()
