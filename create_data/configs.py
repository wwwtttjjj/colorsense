import random
configs = {
    "image_size": (600, 600),
    "grid_x" : [5, 11],
    "grid_y" : [5, 11],
    "de": [5, 30],
    "medium_de": [10, 20],
    "hard_de": [5, 10],
    "margin": 30,
    "block_size":[40, 50],
    "gap": [4, 7],
}
def randomize_config(cfg):
    out = {}
    for k, v in cfg.items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            # 判断元素是否是整数或浮点数范围
            if all(isinstance(x, int) for x in v):
                out[k] = random.randint(v[0], v[1])
            elif all(isinstance(x, (int, float)) for x in v):
                out[k] = random.uniform(v[0], v[1])
            else:
                # 不是数值范围就直接复制
                out[k] = v
        else:
            out[k] = v
    return out


