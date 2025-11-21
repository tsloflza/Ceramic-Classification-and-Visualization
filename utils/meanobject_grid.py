import argparse
import os
import math
from PIL import Image

# =====================
# 1. 資料集定義 (Dataset Definition)
# =====================
datasets = {
    "dynasty": {
        "classes": ["漢", "北宋", "南宋", "金", "元", "明 永樂", "明 宣德", "明 成化", "明 弘治", "明 正德", "明 嘉靖", "明 萬曆", "清 康熙", "清 雍正", "清 乾隆", "清 嘉慶", "清 道光", "清 光緒"],
    },
    "shape": {
        "classes": ["碗", "碟", "洗", "觚", "管", "盤", "壺", "指", "爐", "插", "瓶", "筒", "尊", "托", "盛", "杯", "盆", "盒", "斗", "板", "罐", "片", "鈎", "鍾"],
    },
    "glaze": {
        "classes": ["茄皮紫釉", "孔雀綠釉", "松石綠釉", "寶石紅釉", "豇豆紅釉", "茶葉末釉", "天青釉", "仿官釉", "仿哥釉", "青花釉", "紫金釉", "天藍釉", "仿鈞釉", "白瓷釉", "嬌黃釉", "爐鈞釉", "霽紅釉", "冬青釉", "霽青釉", "甜白釉"],
    },
    "decoration": {
        "classes": ["花卉紋", "雲龍紋", "番蓮紋", "團鳳紋", "蓮花紋", "花果紋", "雙龍戲珠紋", "八寶紋", "魚紋", "牡丹紋", "花鳥紋", "壽字紋", "弦紋", "蓮瓣紋", "福壽紋", "海獸紋",
                    "雲紋", "蝶紋", "蓮塘紋", "鶴紋", "鴛鴦紋", "螭紋", "鳳凰紋", "魚藻紋", "八卦紋", "靈芝紋", "幾何紋", "雙龍紋", "團龍紋", "夔龍紋", "波濤龍紋", "龍鳳紋", "雲鳳紋", "團花紋", "菊花紋", "梅花紋"],
    },
    "kiln": {
        "classes": ["定窯", "官窯", "鈞窯", "哥窯", "彭窯", "廣窯", "汝窯", "龍泉窯", "有田窯", "石灣窯", "德化窯", "吉州窯", "臨川窯", "景德鎮窯"],
    },
}

def format_text_block(text, length=4):
    """
    強制將文字格式化為固定長度：
    1. 超過 length 則截斷
    2. 不足 length 則使用全形空格 (\u3000) 補齊，以確保中文對齊
    """
    # 截斷
    text = text[:length]
    # 計算填充量
    padding = length - len(text)
    # 回傳：文字 + 全形空格填充
    return text + '\u3000' * padding

# =====================
# 2. 網格生成函數
# =====================
def create_image_grid(png_paths, class_labels, output_filename, N=None):
    """
    將多個 PNG 圖片合成一個 N x N 網格，並在 console 印出排版。
    新增參數: class_labels (對應 png_paths 的類別名稱列表)
    """
    valid_images = []
    valid_labels = []

    # --- 1. 同步過濾圖片與標籤 ---
    for path, label in zip(png_paths, class_labels):
        if os.path.exists(path):
            try:
                img = Image.open(path).convert("RGBA")
                valid_images.append(img)
                valid_labels.append(label)
            except Exception as e:
                print(f"❌ 錯誤：無法讀取 {path} ({e})")
        else:
            # 找不到檔案時，安靜跳過或印出警告皆可
            # print(f"⚠️ 警告：找不到檔案 {path} (將在網格中跳過)")
            pass

    if not valid_images:
        print("❌ 錯誤：沒有有效的圖片可以合成，程序終止。")
        return

    num_images = len(valid_images)
    
    # --- 2. 確定網格尺寸 N ---
    if N is None:
        N = math.ceil(math.sqrt(num_images))
    
    if N * N < num_images:
        N = math.ceil(math.sqrt(num_images))
        print(f"⚠️ 提示：指定的 N 太小，已自動調整為 {N}")

    # --- 3. 印出 Grid 排版預覽 (核心修改) ---
    print(f"\n=== Grid Layout Preview ({N}x{N}) ===")
    print("-" * (N * 10 + 1)) # 分隔線長度粗估

    for i in range(num_images):
        # 取得格式化後的名稱 (4個全形字寬)
        fmt_name = format_text_block(valid_labels[i], length=4)
        
        # 印出名稱，不換行
        print(f"| {fmt_name} ", end="")
        
        # 如果到達行尾 (是 N 的倍數)，則換行
        if (i + 1) % N == 0:
            print("|") # 該行結束

    # 如果最後一行沒有填滿，補上結尾換行
    if num_images % N != 0:
        print("|")
    
    print("-" * (N * 10 + 1))
    # print(f"-> 圖片總數：{num_images}\n")

    # --- 4. 圖片合成處理 ---
    base_width, base_height = valid_images[0].size
    grid_width = base_width * N
    grid_height = base_height * N
    grid_image = Image.new('RGBA', (grid_width, grid_height), (255, 255, 255, 0))

    for i, img in enumerate(valid_images):
        row = i // N
        col = i % N
        
        if img.size != (base_width, base_height):
            img = img.resize((base_width, base_height))
            
        x_offset = col * base_width
        y_offset = row * base_height
        
        grid_image.paste(img, (x_offset, y_offset))

    # --- 5. 儲存 ---
    try:
        output_dir = os.path.dirname(output_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        grid_image.save(output_filename)
        print(f"✅ 成功合成網格圖！ 儲存在：{output_filename}")
    except Exception as e:
        print(f"❌ 儲存失敗：{e}")

# =====================
# 3. 主程式與參數解析
# =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="shape",
                        choices=datasets.keys(),
                        help="分類方法名稱，decoration / dynasty / glaze / kiln / shape")
    
    args = parser.parse_args()
    method = args.method

    # 取得該方法的類別列表
    target_classes = datasets[method]["classes"]
    
    # 建構檔案路徑
    base_dir = os.path.join("visualize", method, "mean_object")
    
    png_paths = []
    # 這裡直接傳入 target_classes 作為標籤
    labels = target_classes 

    # print(f"正在為分類 '{method}' 準備資料...")
    
    for class_name in target_classes:
        file_path = os.path.join(base_dir, f"{class_name}.png")
        png_paths.append(file_path)

    output_filename = os.path.join(base_dir, "mean_objects.png")

    # 呼叫函數，傳入 labels
    create_image_grid(png_paths, labels, output_filename)