from pathlib import Path
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import signal, sys

def build_prompt(data: dict) -> str:
    rows, cols = data.get("grid_size", [0, 0])
    shape = data.get("shape", "object")

    prompt = f"""
    You are given an image containing many identical {shape}s 
    arranged in an {rows}×{cols} grid. Each grid cell contains one {shape}. 
    All {shape}s have the same shape and size, but exactly one {shape} has a 
    different color compared to the others.

    Your task:
    - Carefully examine the grid.
    - Identify the grid position (row and column) of the {shape} with the different color.
    - Counting starts from the top-left corner, which is Row 1, Column 1.

    Output format:
    - Answer strictly in the form: "Row X, Column Y"
    - Do not provide any additional explanation or text.
    """
    return prompt



def Extract_answer(predict_answer: str) -> str:
    """
    Extract a single-letter answer from LaTeX-style \boxed{(A–F)} 
    even if there is trailing content like \boxed{(D) 120}.

    Args:
        predict_answer (str): The raw model output.

    Returns:
        str: The extracted answer in the form (A–F),
             "Many answers found" if multiple matches are detected,
             or "No answer found" if none is found.
    """
    # 匹配 \boxed{(A–F)} 后可有空格或其他字符直到 '}'，忽略大小写
    matches = re.findall(r"boxed\{\(\s*([A-Fa-f])\s*\)[^}]*\}", predict_answer)

    if len(matches) == 1:
        return f"({matches[0].upper()})"
    elif len(matches) > 1:
        return "Many answers found"
    else:
        return "No answer found"

def remove_existing_file(save_path):
    """
    Remove the file at the specified path if it exists.

    Args:
        save_path (str): Path of the file to remove.
    """
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Existing file removed: {save_path}")

def write_json(save_json_path, save_json_data):
    if os.path.exists(save_json_path):
        with open(save_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            data.append(save_json_data)
        else:
            data = [data, save_json_data]
    else:
        data = [save_json_data]

    with open(save_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def run_model_parallel(root_dir, save_json_path, max_workers, init_worker, worker_inference, args):
    """
    Run model inference in parallel and save results incrementally.

    Args:
        root_dir (str): Root directory containing images.
        save_json_path (str): Path to save aggregated JSON results.
        max_workers (int): Number of parallel worker processes.
        init_worker (callable): Initialization function for each worker (loads model).
        worker_inference (callable): Function to run inference on a single task.
        model_name (str): Model name passed to init_worker for loading.
    """
    # Prepare all tasks and corresponding raw data for result writing
    tasks = []
    raw_datas = []
            
    
    # 读取已有结果文件，避免重复处理
    processed_image_ids = set()
    if os.path.exists(save_json_path) and os.path.getsize(save_json_path) > 0:
        with open(save_json_path, 'r', encoding='utf-8') as f:
            try:
                existing_results = json.load(f)
                for item in existing_results:
                    image_id = item.get("image_id")
                    if image_id:
                        processed_image_ids.add(image_id)
            except json.JSONDecodeError:
                print(f"[WARN] JSON 文件损坏或为空: {save_json_path}，已忽略。")
                existing_results = []
    else:
        existing_results = []

    for json_file in [args.json_path]:
        with open(json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        for data in json_data:
            image_id = data.get("id")
            if image_id in processed_image_ids:
                continue  # 已处理，跳过

            # query = data.get("query")
            prompt = build_prompt(data)
            data["prompt"] = prompt
            image_name = data.get("image")
            image_path = os.path.join(root_dir, image_name)

            tasks.append((prompt, image_path))
            raw_datas.append(data)
    if not raw_datas:
        print("No new data to process. Exiting.")
        return
    try:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=init_worker,
            initargs=(args.model_name,)
        ) as executor:

            futures = {executor.submit(worker_inference, task): idx for idx, task in enumerate(tasks)}

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    predict_answer = future.result()
                except Exception as e:
                    # print(f"[ERROR] Task {idx} failed: {e}")
                    continue

                data = raw_datas[idx]
                extract_answer = Extract_answer(predict_answer)

                save_json_data = {
                    "image_id": data.get('id'),
                    "image_name": data.get("image"),
                    "prompt": data.get('prompt'),
                    "predict_answer": predict_answer,
                    "extract_answer": extract_answer,
                    "answer": data.get('answer'),
                    "delta_e": data.get('delta_e')
                }
                write_json(save_json_path, save_json_data)
                print(f"[INFO] Written result for {data.get('id')}")

    except KeyboardInterrupt:
        print("⚠️ 收到 Ctrl+C，准备终止所有进程...")
        os.killpg(os.getpgid(os.getpid()), signal.SIGTERM)
        sys.exit(1)
