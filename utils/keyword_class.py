import json

INPUT_PATH = "data/decoration.json"
OUTPUT_PATH = "tmp.txt"
KEYWORD = "白"

def count_keyword_by_class(json_path=INPUT_PATH, keyword=KEYWORD, output_path=OUTPUT_PATH):
    # 讀取 JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stats = {}  # {class: {"total": x, "match": y}}

    # 統計各 class 的資料
    for item in data:
        cls = item.get("class", "Unknown")
        name = item.get("name", "")

        if cls not in stats:
            stats[cls] = {"total": 0, "match": 0}

        stats[cls]["total"] += 1
        if keyword in name:
            stats[cls]["match"] += 1

    # 依照 match 數量排序
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]["match"], reverse=True)

    # 輸出到檔案
    with open(output_path, "w", encoding="utf-8") as f:
        print(f"關鍵字 '{keyword}'  數量/總數", file=f)
        for cls, counts in sorted_stats:
            if counts["match"] == 0:
                break
            f.write(f"{cls}: {counts['match']}/{counts['total']}\n")

    print(f"結果已輸出至 {output_path}")


# Example usage
if __name__ == "__main__":
    count_keyword_by_class()
