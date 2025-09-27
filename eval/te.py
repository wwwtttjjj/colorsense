import json
from collections import defaultdict

def compute_accuracy(json_path, output_md="accuracy_stats.md"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for item in data:
        # 1. 获取图表类型
        img_type = item["image_name"].split("_")[0]
        description = item["description"]

        # 2. 提取答案
        gold = item["extract_answer"].strip()
        ans  = item["answer"].strip()

        # 3. 统计
        key = (img_type, description)
        stats[key]["total"] += 1
        if gold == ans:
            stats[key]["correct"] += 1

    # 4. 写入 Markdown 文件
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("| Type | Description | Accuracy | Correct | Total |\n")
        f.write("|------|-------------|----------|---------|-------|\n")
        for key, val in stats.items():
            img_type, description = key
            acc = val["correct"] / val["total"] if val["total"] > 0 else 0
            f.write(f"| {img_type} | {description} | {acc:.2%} | {val['correct']} | {val['total']} |\n")

    print(f"统计结果已写入 {output_md}")

# 用法
compute_accuracy("./save_chart_output/Qwen2.5-VL-72B-Instruct.json", "accuracy_stats_2.md")