# shapes/registry.py
import random

shape_registry = {}

def register_shape(name):
    def decorator(func):
        shape_registry[name] = func
        return func
    return decorator

def draw_shape_by_name(name, block_size, color, bgcolor):
    func = shape_registry[name]
    return func(block_size, color=color, bgcolor=bgcolor), name

def draw_random_shape(block_size, color, bgcolor, allow=None):
    # 延迟导入 set_current_char 避免循环
    from .text_shapes import set_current_char

    candidates = allow if allow else list(shape_registry.keys())
    name = random.choice(candidates)

    if name == "char":
        import string
        ch = random.choice(string.ascii_uppercase + string.digits)
        set_current_char(ch)

    img = shape_registry[name](block_size, color=color, bgcolor=bgcolor)
    return img, name
