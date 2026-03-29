import os
import pandas as pd
from datasets import Dataset
from PIL import Image
import io
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 在这里填入您的 parquet 文件路径
PARQUET_FILE = "train-00000-of-00001-1028f23e353fbe3e.parquet"

# 2. 输出设置
OUTPUT_CSV = "scienceqa_data_fixed.csv"       # 新的 CSV 文件名
OUTPUT_IMG_DIR = "scienceqa_images_fixed"     # 新的图片文件夹名
IMG_FORMAT = "jpg"
# ===========================================

def get_image_from_item(item_image):
    """
    辅助函数：处理不同格式的图片数据 (PIL Image 或 Dict)
    返回 PIL Image 对象，如果失败则返回 None
    """
    if item_image is None:
        return None
    
    # 情况 1: 已经是 PIL Image 对象
    if isinstance(item_image, Image.Image):
        return item_image
    
    # 情况 2: 是字典 (HuggingFace 常见格式 {'path': ..., 'bytes': ...})
    if isinstance(item_image, dict):
        # 优先尝试从 'bytes' 读取
        if 'bytes' in item_image and item_image['bytes'] is not None:
            try:
                return Image.open(io.BytesIO(item_image['bytes']))
            except Exception:
                pass
        # 如果 bytes 不行，尝试从 'path' 读取 (如果是本地路径)
        if 'path' in item_image and item_image['path']:
            try:
                return Image.open(item_image['path'])
            except Exception:
                pass
    
    # 其他未知格式
    return None

def main():
    if not os.path.exists(PARQUET_FILE):
        print(f"❌ 错误：找不到文件 {PARQUET_FILE}")
        return

    print(f"⏳ 正在加载数据文件：{PARQUET_FILE} ...")
    dataset = Dataset.from_parquet(PARQUET_FILE)
    
    if not os.path.exists(OUTPUT_IMG_DIR):
        os.makedirs(OUTPUT_IMG_DIR)

    print("🔄 开始处理数据并保存图片 (已修复字典格式图片)...")
    
    csv_rows = []
    success_count = 0
    fail_count = 0
    
    for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc="处理中"):
        row_id = item.get('id', str(idx))
        
        # --- 1. 处理图片 (使用修复后的函数) ---
        img_filename = ""
        raw_img = item.get('image')
        img_obj = get_image_from_item(raw_img)
        
        if img_obj is not None:
            try:
                safe_name = str(row_id).replace("/", "_").replace("\\", "_").replace(":", "_")
                img_filename = f"{safe_name}.{IMG_FORMAT}"
                img_path = os.path.join(OUTPUT_IMG_DIR, img_filename)
                
                # 转换模式以防保存 JPG 报错
                if img_obj.mode == "RGBA" and IMG_FORMAT.lower() == "jpg":
                    img_obj = img_obj.convert("RGB")
                elif img_obj.mode == "P": # 处理调色板模式
                    img_obj = img_obj.convert("RGB")
                
                img_obj.save(img_path)
                success_count += 1
            except Exception as e:
                # print(f"⚠️ 第 {idx} 条数据图片保存失败: {e}") # 太多报错就不打印了
                fail_count += 1
                img_filename = "ERROR"
        else:
            img_filename = "" 

        # --- 2. 准备 CSV 行数据 ---
        choices = item.get('choices', [])
        if isinstance(choices, list):
            choices_str = " | ".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
        else:
            choices_str = str(choices)

        csv_row = {
            'id': row_id,
            'question': item.get('question', ''),
            'choices': choices_str,
            'answer_index': item.get('answer', -1),
            'answer_text': choices[item.get('answer')] if isinstance(choices, list) and 0 <= item.get('answer', -1) < len(choices) else "",
            'lecture': item.get('lecture', ''),
            'explanation': item.get('explanation', ''),
            'category': item.get('category', ''),
            'grade': item.get('grade', ''),
            'image_filename': img_filename
        }
        csv_rows.append(csv_row)

    # --- 3. 保存 CSV ---
    print(f"\n💾 正在保存 CSV 文件：{OUTPUT_CSV} ...")
    df = pd.DataFrame(csv_rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*40)
    print("✅ 转换完成！")
    print(f"📊 表格文件：{os.path.abspath(OUTPUT_CSV)}")
    print(f"🖼️ 图片统计：成功保存 {success_count} 张，失败/无图 {fail_count + (len(dataset)-success_count-fail_count)} 张")
    print(f"📁 图片文件夹：{os.path.abspath(OUTPUT_IMG_DIR)}")
    print("="*40)

if __name__ == "__main__":
    main()