# import os

# index = 1
# image_type = "geo" if index == 0 else "chart"
# IMAGE_HEIGHT = 448 if index == 0 else 1000
# IMAGE_WIDTH = 448 if index == 0 else 1000

# Geo_root = "save_geo_output/"
# Chart_root = "save_chart_output/"
# models_dir = "/data/wengtengjin/models/"

# image_dir = ["/data/wengtengjin/Geometry-Chart-Perception/create_images/geo_images/test_images","/data/wengtengjin/Geometry-Chart-Perception/create_images/chart_images/images"][index]
# json_path = ["/data/wengtengjin/Geometry-Chart-Perception/create_images/geo_images/Total_QA/test_data.json", "/data/wengtengjin/Geometry-Chart-Perception/create_images/chart_images/Total_QA/test_data.json"][index]
# root_dir = image_dir

# if index == 0:
#     Result_root = Geo_root
# else:
#     Result_root = Chart_root

# if not os.path.exists(Result_root):
#     os.mkdir(Result_root)

import os
models_dir = "/data/wengtengjin/models/"
max_new_tokens = 1024

def get_configs(data_type: str):
    # 目录定义

    image_dir = "/data/wengtengjin/colorsense/create_data/test_data/image"
    json_path = "/data/wengtengjin/colorsense/create_data/test_data.json"
    root_dir = image_dir

    # 输出路径
    Result_root = "output/"
    if not os.path.exists(Result_root):
        os.mkdir(Result_root)

    return {
        "image_type": "colorsense",
        "image_dir": image_dir,
        "json_path": json_path,
        "root_dir": root_dir,
        "Result_root": Result_root,
        "models_dir": models_dir,
    }

# model_list = [
#     "Random.json",
#     "phi3_5.json",
#     "llava-v1.5-7b.json",
#     "llava-v1.5-13b.json",
#     "llava-v1.6-34b.json",
#     "llava-onevision-qwen2-7b-ov-hf.json",
#     "llava-onevision-qwen2-72b-si-hf.json",
#     "Internvl2_5-8B.json",
#     "Internvl2_5-38B.json",
#     "Internvl2_5-78B.json",
#     "Janus-Pro-7B.json",
#     "Qwen2-VL-2B-Instruct.json",
#     "Qwen2-VL-7B-Instruct.json",
#     "Qwen2-VL-72B-Instruct.json",
#     "Qwen2.5-VL-3B-Instruct.json",
#     "Qwen2.5-VL-7B-Instruct.json",
#     "Qwen2.5-VL-72B-Instruct.json",
#     "GPT-4o.json",
#     "gemini-1.5-flash.json",
#     "gemini-2.0-flash.json",
#     "gemini-1.5-pro.json",
#     "Human.json",
#     "Internvl-8B.json",
#     "Internvl-40B.json",
#     "InternVL2-8B-MPO.json",
#     "llava-v1.5-13b.json",
#     "Math-llava-13b.json",
#     "Llama-VL-3_2-11B.json",
#     "Llama-3.2V-11B-cot.json",
#     "Qwen2.5-VL-7B-Instruct.json",
#     "R1-Onevision-7B.json"
# ]