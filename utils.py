import numpy as np
import matplotlib.pyplot as plt
from skimage import color
import os
import json
import numpy
import shutil

def generate_lab_color(l_range=(20, 70), a_range=(-40, 40), b_range=(-40, 40)):
    """随机生成一个更深的 LAB 颜色（避免白色或过亮）"""
    L = np.random.uniform(*l_range)
    a = np.random.uniform(*a_range)
    b = np.random.uniform(*b_range)
    return np.array([L, a, b])

def perturb_color(base_lab, target_delta_e, step=1.0, max_iter=5000, tol=0.5):
    """
    生成与 base_lab 相差约 target_delta_e 的颜色 (ΔE2000)
    """
    best_candidate = base_lab
    best_diff = 1e9
    for _ in range(max_iter):
        candidate = base_lab + np.random.uniform(-step, step, 3) * target_delta_e
        dE = color.deltaE_ciede2000(
            base_lab[np.newaxis, :], candidate[np.newaxis, :]
        )[0]
        diff = abs(dE - target_delta_e)
        if diff < best_diff:
            best_diff = diff
            best_candidate = candidate
        if diff < tol:
            break
    return best_candidate

def lab_to_rgb(lab):
    """LAB 转 RGB，并裁剪到 [0,1]"""
    rgb = color.lab2rgb(lab[np.newaxis, np.newaxis, :])
    return np.clip(rgb[0, 0, :], 0, 1)

def ensure_dirs(difficulty_name: str):
    """
    创建 难度/image 与 难度/metadata 目录
    - 若已存在，则先清空再重新创建
    """
    img_dir = os.path.join(difficulty_name, "image")
    meta_dir = os.path.join(difficulty_name, "metadata")

    # 如果存在旧目录则先删除
    for d in [img_dir, meta_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
    print(f"Creating directories '{img_dir}' and '{meta_dir}'...")

    # 重新创建空目录
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    return img_dir, meta_dir

def save_image_as_png(image: np.ndarray, path: str):
    """保存为 PNG 文件"""
    plt.imsave(path, image)


def save_pair(image, meta, img_dir, meta_dir, index):
    img_name = f"image_{index}.png"
    meta_name = f"metadata_{index}.json"

    img_path = os.path.join(img_dir, img_name)
    meta_path = os.path.join(meta_dir, meta_name)

    meta = dict(meta)  # 复制一份
    meta["image_file"] = os.path.join("image", img_name)
    meta["metadata_file"] = os.path.join("metadata", meta_name)
    save_image_as_png(image, img_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)