import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from utils import *
from configs import configs, randomize_config
import argparse
def generate_odd_one_out_image(
    grid_size,
    block_size,
    gap,
    delta_e,
    max_image_size,
    margin,
    background_rgb = (1.0, 1.0, 1.0)
):
    h, w = grid_size
    core_h = h * block_size + (h - 1) * gap
    core_w = w * block_size + (w - 1) * gap

    # 保证包含 margin 后不超过 max_image_size，必要时缩放 block/gap
    avail_h = max_image_size[0] - 2 * margin
    avail_w = max_image_size[1] - 2 * margin
    
    scale = min(avail_h / core_h, avail_w / core_w, 1.0)
    block_size = max(1, int(round(block_size * scale)))
    gap = max(0, int(round(gap * scale)))
    core_h = h * block_size + (h - 1) * gap
    core_w = w * block_size + (w - 1) * gap

    img_h = core_h + 2 * margin
    img_w = core_w + 2 * margin

    # 颜色
    base_lab = generate_lab_color()
    odd_lab = perturb_color(base_lab, delta_e)
    base_rgb = lab_to_rgb(base_lab)
    odd_rgb = lab_to_rgb(odd_lab)

    # 位置
    odd_pos = np.random.randint(0, h * w)
    odd_row = odd_pos // w + 1  # 1-indexed
    odd_col = odd_pos % w + 1

    # 绘制
    img = np.ones((img_h, img_w, 3), dtype=np.float32)
    img[:] = np.array(background_rgb, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            idx = i * w + j
            color_rgb = odd_rgb if idx == odd_pos else base_rgb
            y0 = margin + i * (block_size + gap)
            x0 = margin + j * (block_size + gap)
            img[y0:y0 + block_size, x0:x0 + block_size, :] = color_rgb

    # 元数据
    meta = {
        "grid_size": [int(h), int(w)],
        "block_size": int(block_size),
        "gap": int(gap),
        "delta_e": float(delta_e),
        "margin": int(margin),
        "odd_position": {"row": int(odd_row), "col": int(odd_col)},  # 左上为 (1,1)
        "base_color_lab": [float(x) for x in base_lab],
        "odd_color_lab": [float(x) for x in odd_lab],
    }
    return img, meta

def build_dataset(args):
    img_dir, meta_dir = ensure_dirs(args.difficulty)
    # 2. 生成数据
    for idx in range(1, args.number + 1):
        img, meta = generate_odd_one_out_image(
            grid_size=(args.grid_y, args.grid_x),
            block_size=args.block_size,
            gap=args.gap,                               # 可以改成参数
            delta_e=args.de,
            max_image_size=(args.image_size, args.image_size)
                      if isinstance(args.image_size, int)
                      else tuple(args.image_size),
            margin=args.margin
        )

        meta["difficulty"] = args.difficulty
        meta["index"] = idx

        save_pair(img, meta, img_dir, meta_dir, idx)
    print(f"Generated {args.number} images for difficulty '{args.difficulty}' in folder '{args.difficulty}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", type=str, default="easy", help="which difficulty to generate (easy, medium, hard)")
    parser.add_argument("--number", type=int, default=2, help="the number of generate images")
    
    args = parser.parse_args()
    cfg = randomize_config(configs)

    # 把 cfg 的键值直接添加到 args
    for k, v in cfg.items():
        setattr(args, k, v)
    args.de = cfg[f"{args.difficulty}_de"]
    
    build_dataset(args)
    