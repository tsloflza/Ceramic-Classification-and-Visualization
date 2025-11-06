# build_dataset.py
import json
import os

RAW_PATH = "./raw_data/ceramics.json"
OUT_DIR = "./data"

# 五個分類與對應規則
datasets = {
    "dynasty": {
        "output": "dynasty.json",
        "classes": ["漢", "北宋", "南宋", "金", "元", "明 永樂", "明 宣德", "明 成化", "明 弘治", "明 正德", "明 嘉靖", "明 萬曆", "清 康熙", "清 雍正", "清 乾隆", "清 嘉慶", "清 道光", "清 光緒"],
        "rule": lambda item, c: item.get("era") == c
    },
    "shape": {
        "output": "shape.json",
        "classes": ["碗", "碟", "洗", "觚", "管", "盤", "壺", "指", "爐", "插", "瓶", "筒", "尊", "托", "盛", "杯", "盆", "盒", "斗", "板", "罐", "片", "鈎", "鍾"],
        "rule": lambda item, c: item.get("name", "").endswith(c)
    },
    "glaze": {
        "output": "glaze.json",
        "classes": ["茄皮紫釉", "孔雀綠釉", "松石綠釉", "寶石紅釉", "豇豆紅釉", "茶葉末釉", "天青釉", "仿官釉", "仿哥釉", "青花釉", "紫金釉", "天藍釉", "仿鈞釉", "白瓷釉", "嬌黃釉", "爐鈞釉", "霽紅釉", "冬青釉", "霽青釉", "甜白釉"],
        "rule": lambda item, c: c in item.get("name", "")
    },
    "decoration": {
        "output": "decoration.json",
        "classes": ["花卉紋", "雲龍紋", "番蓮紋", "團鳳紋", "蓮花紋", "花果紋", "雙龍戲珠紋", "八寶紋", "魚紋", "牡丹紋", "花鳥紋", "壽字紋", "弦紋", "蓮瓣紋", "福壽紋", "海獸紋",
                    "雲紋", "蝶紋", "蓮塘紋", "鶴紋", "鴛鴦紋", "螭紋", "鳳凰紋", "魚藻紋", "八卦紋", "靈芝紋", "幾何紋", "雙龍紋", "團龍紋", "夔龍紋", "波濤龍紋", "龍鳳紋", "雲鳳紋", "團花紋", "菊花紋", "梅花紋"],
        "rule": lambda item, c: c in item.get("name", "")
    },
    "kiln": {
        "output": "kiln.json",
        "classes": ["定窯", "官窯", "鈞窯", "哥窯", "彭窯", "廣窯", "汝窯", "龍泉窯", "有田窯", "石灣窯", "德化窯", "吉州窯", "臨川窯", "景德鎮窯"],
        "rule": lambda item, c: c in item.get("name", "")
    },
}

# 固定間隔抽樣（非隨機）
def sample_fixed_interval(data, limit=100):
    if len(data) <= limit:
        return data
    step = len(data) // limit
    return [data[i] for i in range(0, len(data), step)][:limit]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(RAW_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # 移除「沒有圖片」的樣本
    NO_IMAGE_URL = "https://digitalarchive.npm.gov.tw/Image/GetImage?ImageId=0&randomCode=0"
    raw_data = [item for item in raw_data if item.get("imageUrl_m") != NO_IMAGE_URL]
    print(f"🧹 過濾後剩餘 {len(raw_data)} 筆資料（已排除無圖片項目）")

    for ds_name, cfg in datasets.items():
        output_path = os.path.join(OUT_DIR, cfg["output"])
        all_selected = []

        for c in cfg["classes"]:
            # 依規則挑選該類別的資料
            selected = [dict(item, **{"class": c}) for item in raw_data if cfg["rule"](item, c)]

            # 固定間隔抽樣（先過濾後抽樣）
            selected = sample_fixed_interval(selected, 100)
            all_selected.extend(selected)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_selected, f, ensure_ascii=False, indent=2)

        print(f"✅ {ds_name}: {len(all_selected)} items saved to {output_path}")


if __name__ == "__main__":
    main()