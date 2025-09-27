import os
import json
import uuid

# 目录路径
metadata_dir = "./easy/metadata"
output_file  = "./merged_metadata.json"

# 结果列表
merged_data = []

# 遍历 metadata 目录下所有 .json 文件
for filename in os.listdir(metadata_dir):
    if not filename.endswith(".json"):
        continue

    file_path = os.path.join(metadata_dir, filename)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 生成随机 id
    new_id = str(uuid.uuid4())

    # image 路径（如果要改相对路径可自行调整）
    image_path = data.get("image_file", "")

    # 自定义 query —— 这里你可以直接修改为自己的问题模板
    query = f"Which block contains the odd color? (A) row {data['odd_position']['row']} col {data['odd_position']['col']})"

    # answer 转成文本（例如 'row 2, col 5'）
    answer_text = f"Row {data['odd_position']['row']}, Column {data['odd_position']['col']}"

    # 构造目标格式
    merged_data.append({
        "id": new_id,
        "image": image_path,
        "query": query,
        "answer": answer_text,
        "delta_e": data.get("delta_e", None),
        "answer_position": data.get("odd_position", None),
    })

# 保存为一个合并的 json
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(merged_data, f, indent=4, ensure_ascii=False)

print(f"✅ 合并完成，共 {len(merged_data)} 条记录，保存到：{output_file}")
