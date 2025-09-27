import os
import time
import argparse
import pathlib
import textwrap
import PIL.Image
import google.generativeai as genai
import os
import argparse
from configs import get_configs, max_new_tokens
from utils import run_model_parallel
from PIL import Image
from multiprocessing import current_process

model = None
keys = [
    "AIzaSyD1-MdBBo8QWuYg8iJ60WMJBIjAH9SRLig",
    "AIzaSyCWWothJ_jkjN9eVoi0gKLxEsdLTLD_D0o",
    "AIzaSyCxSAxHJAjU-HPiiGamvv3YrCmAfSpk5ko",
    "AIzaSyDCaokTY0ApxbjhAbdAe7jvadmydd3m4zA",
    "AIzaSyBeEatZVn-n7TOFexXSuQNk-pEQmoJ8Wac"
]


def init_worker(model_name="gemini-2.0-flash"):
    global model
    # 获取 worker 在池中的编号（从1开始）
    wid = current_process()._identity[0] - 1
    api_key = keys[wid % len(keys)]
    print(f"worker {wid} 使用 key {api_key}")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

def worker_inference(task, max_tries=6):
    global model
    prompt, image_path = task
    img = Image.open(image_path)

    for attempt in range(max_tries):
        try:
            response = model.generate_content(
                [prompt, img],
                generation_config={"max_output_tokens": max_new_tokens}
            )
            time.sleep(10)
            return response.text

        except Exception as e:
            print(f"[WARN] 第{attempt+1}次调用失败: {e}")
            if attempt < max_tries - 1:
                # 指数回退等待：1, 2, 4, 8, 16...
                sleep_time = 2 ** attempt
                time.sleep(sleep_time)
            else:
                print("[ERROR] 连续失败，放弃此任务")
                return None


def main():
    # global Result_root  # 告诉 Python 使用全局变量
    """
    Parse command-line arguments and run parallel inference.
    """
    parser = argparse.ArgumentParser(description="Run multimodal inference using gemini")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemini-2.5-flash",
        help="Model name. Example: gemini-2.0-flash, gemini-2.5-flash"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=5,
        help="Number of parallel workers for inference"
    )
    
    parser.add_argument(
        "--red_box",
        action="store_true",
        help="Whether to use red box for inference (default: False)"
    )
    
    parser.add_argument(
        "--data_type",
        type=str,
        default="geo",
        help="geo or chart"
    )

    args = parser.parse_args()
    configs_para = get_configs(args.data_type)
    Result_root = configs_para["Result_root"]
    args.json_path = configs_para["json_path"]
    if args.red_box:
        Result_root = Result_root.replace("save_geo_output", "save_red_box_geo_output")
        
    save_json_path = os.path.join(Result_root, f"{args.model_name}.json")

    run_model_parallel(
        root_dir=configs_para["root_dir"],
        save_json_path=save_json_path,
        max_workers=args.max_workers,
        init_worker=init_worker,
        worker_inference=worker_inference,
        args=args
    )

if __name__ == "__main__":
    main()
