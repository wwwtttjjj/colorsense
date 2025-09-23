import numpy as np
import matplotlib.pyplot as plt
from skimage import color

def generate_lab_color():
    """随机生成一个 LAB 颜色"""
    L = np.random.uniform(60, 90)   # 偏亮
    a = np.random.uniform(-40, 40)
    b = np.random.uniform(-40, 40)
    return np.array([L, a, b])

def perturb_color(base_lab, target_delta_e, step=1.0):
    """
    生成与 base_lab 相差约 target_delta_e 的颜色 (ΔE2000)
    """
    best_candidate = base_lab
    best_diff = 1e9

    for _ in range(5000):
        candidate = base_lab + np.random.uniform(-step, step, 3) * target_delta_e
        dE = color.deltaE_ciede2000(
            base_lab[np.newaxis, :], candidate[np.newaxis, :]
        )[0]
        diff = abs(dE - target_delta_e)
        if diff < best_diff:
            best_diff = diff
            best_candidate = candidate
        if diff < 0.5:
            break
    return best_candidate

def lab_to_rgb(lab):
    """LAB 转 RGB"""
    rgb = color.lab2rgb(lab[np.newaxis, np.newaxis, :])
    return np.clip(rgb[0, 0, :], 0, 1)

def generate_odd_one_out(delta_e=10, grid_size=(8, 8), block_size=50, gap=5):
    """生成带间隔的 Odd-One-Out 色块图"""
    h, w = grid_size
    base_lab = generate_lab_color()
    odd_lab = perturb_color(base_lab, delta_e)

    base_rgb = lab_to_rgb(base_lab)
    odd_rgb = lab_to_rgb(odd_lab)

    odd_pos = np.random.randint(0, h * w)

    # 图像画布大小（包含间隔）
    img_h = h * block_size + (h - 1) * gap
    img_w = w * block_size + (w - 1) * gap
    img = np.ones((img_h, img_w, 3))  # 默认白色背景

    for i in range(h):
        for j in range(w):
            idx = i * w + j
            color_rgb = odd_rgb if idx == odd_pos else base_rgb
            y0 = i * (block_size + gap)
            x0 = j * (block_size + gap)
            img[y0:y0+block_size, x0:x0+block_size, :] = color_rgb

    return img, odd_pos

if __name__ == "__main__":
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, dE, title in zip(axes, [20, 10, 3], ["Easy", "Medium", "Hard"]):
        img, pos = generate_odd_one_out(delta_e=dE, grid_size=(8, 8), block_size=40, gap=6)
        ax.imshow(img)
        ax.set_title(f"{title} (odd={pos})")
        ax.axis("off")
    plt.show()
