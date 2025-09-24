# shapes/base_shape.py
import numpy as np
from .registry import register_shape
from PIL import Image, ImageDraw
@register_shape("square")
def draw_square(block_size, color=(1,0,0), bgcolor=(1,1,1)):
    img = np.ones((block_size, block_size, 3), dtype=np.float32)
    img[:] = color
    return img

@register_shape("circle")
def draw_circle(block_size, color=(0,1,0), bgcolor=(1,1,1)):
    img = np.ones((block_size, block_size, 3), dtype=np.float32)
    yy, xx = np.ogrid[:block_size, :block_size]
    cx, cy = block_size/2, block_size/2
    r = block_size*0.45
    mask = (xx-cx)**2 + (yy-cy)**2 <= r**2
    img[:] = bgcolor
    img[mask] = color
    return img

@register_shape("triangle")
def draw_triangle(block_size, color=(0,0,1), bgcolor=(1,1,1)):
    img = np.ones((block_size, block_size, 3), dtype=np.float32)
    img[:] = bgcolor
    for y in range(block_size):
        x_min = int(block_size/2 - (y/block_size)*block_size/2)
        x_max = int(block_size/2 + (y/block_size)*block_size/2)
        img[y, x_min:x_max] = color
    return img


@register_shape("shoe")
def draw_shoe(block_size, color=(0.2, 0.4, 0.8), bgcolor=(1, 1, 1)):
    """
    在 block_size×block_size 的方块中绘制一只卡通运动鞋
    返回 numpy 数组 (block_size, block_size, 3)
    """
    # 创建底图
    img = Image.new("RGB", (block_size, block_size),
                    tuple(int(c * 255) for c in bgcolor))
    draw = ImageDraw.Draw(img)

    # 相对尺寸比例，方便适应不同 block_size
    b = block_size
    sole_h = b * 0.15         # 鞋底厚度
    upper_h = b * 0.5         # 鞋面高度
    lace_h = b * 0.1          # 鞋带区域高度

    # 1️⃣ 绘制鞋底：一个长方形
    draw.rectangle(
        [(b * 0.1, b * 0.8), (b * 0.9, b * 0.8 + sole_h)],
        fill=tuple(int(c * 255) for c in (0.2, 0.2, 0.2))
    )

    # 2️⃣ 绘制鞋面：一个带弧度的多边形
    draw.polygon(
        [
            (b * 0.1, b * 0.8),
            (b * 0.2, b * 0.4),
            (b * 0.6, b * 0.3),
            (b * 0.9, b * 0.4),
            (b * 0.9, b * 0.8)
        ],
        fill=tuple(int(c * 255) for c in color)
    )

    # 3️⃣ 鞋带区域：用几条细线表示
    for i in range(3):
        y = b * 0.45 + i * lace_h * 0.8
        draw.line(
            [(b * 0.25, y), (b * 0.7, y)],
            fill=(255, 255, 255),
            width=max(1, int(b * 0.03))
        )

    # 4️⃣ 鞋头高光：给鞋前端一点装饰
    draw.ellipse(
        [(b * 0.1, b * 0.7), (b * 0.3, b * 0.9)],
        outline=(200, 200, 200),
        width=max(1, int(b * 0.02))
    )

    return np.asarray(img, dtype=np.float32) / 255.0