# shapes/text_shapes.py
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from .registry import register_shape

_current_char = "A"

def set_current_char(ch):
    global _current_char
    _current_char = ch

# @register_shape("char")
def draw_char(block_size, color=(0,0,0), bgcolor=(1,1,1)):
    img = Image.new("RGB", (block_size, block_size),
                    tuple(int(c*255) for c in bgcolor))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", size=int(block_size * 0.8))
    bbox = draw.textbbox((0, 0), _current_char, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((block_size - w)/2, (block_size - h)/2),
              _current_char,
              fill=tuple(int(c*255) for c in color),
              font=font)
    return np.asarray(img, dtype=np.float32)/255.0
