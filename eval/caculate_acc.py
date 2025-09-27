import os
import json
import pandas as pd

def compute_total_accuracy(data):
    """
    Compute overall accuracy across all samples.
    """
    correct = 0
    total = 0
    for item in data:
        pred = item.get("extract_answer", "").strip()
        predict_answer = item.get("predict_answer", "").strip()
        gold = item.get("answer", "").strip()
        if pred == gold or predict_answer == gold:
            correct += 1
        total += 1
    return correct / total * 100 if total > 0 else 0.0


def main(json_dir, md_path):
    """
    Process all JSON files in a directory to compute only total accuracy,
    then save results as a markdown table.
    """
    results = []

    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(json_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            total_acc = compute_total_accuracy(data)
            row = {"file_name": filename.replace(".json", ""), "total_acc": total_acc}
            results.append(row)

    if not results:
        print(f"No JSON files found in {json_dir}.")
        return

    # Create DataFrame
    df = pd.DataFrame(results)
    df = df[["file_name", "total_acc"]]

    # Save as markdown
    md_table = df.round(1).to_markdown(index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_table)

    print(f"✅ Markdown results saved to {md_path}")


if __name__ == "__main__":
    dir_md_pairs = [
        ("output", "final_result_table/accuracy.md"),
    ]

    for json_dir, md_path in dir_md_pairs:
        main(json_dir, md_path)
