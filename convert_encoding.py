import os
import shutil

# 配置路径
source_file = "data/raw/yitian_full.txt"  # 假设你已经重命名了
temp_file = "data/raw/yitian_utf8.txt"

def convert_to_utf8(filename):
    if not os.path.exists(filename):
        print(f"❌ 错误：找不到文件 {filename}")
        print("请确认你是否已经上传文件，并将其重命名为 yitian_full.txt")
        return

    # 尝试常见的中文编码
    encodings = ['utf-8', 'gb18030', 'gbk', 'big5']
    
    content = ""
    success = False
    
    # 1. 读取
    with open(filename, 'rb') as f:
        raw_data = f.read()
    
    for enc in encodings:
        try:
            content = raw_data.decode(enc)
            print(f"✅ 成功检测到编码: {enc}")
            success = True
            break
        except UnicodeDecodeError:
            continue
            
    if not success:
        print("❌ 无法识别文件编码，请检查文件是否损坏。")
        return

    # 2. 写入标准的 UTF-8
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # 3. 覆盖原文件
    shutil.move(temp_file, filename)
    print(f"🎉 文件已转换为标准 UTF-8 格式：{filename}")
    print(f"📖 字数统计：{len(content)} 字")

if __name__ == "__main__":
    convert_to_utf8(source_file)
