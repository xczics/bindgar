from typing import List

class CyclicList:
    def __init__(self, data: List):
        self.data = data
        self.length = len(data)
    def __getitem__(self, index: int):
        return self.data[index % self.length]
    def __len__(self):
        return self.length
    def __iter__(self):
        import itertools
        return itertools.cycle(self.data)
    
default_colors = CyclicList([
    "blue",
    "orange",
    "green",
    "red",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
])# avoid out-of-index error when accessing colors

def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(
        int(r * 255),
        int(g * 255),
        int(b * 255)
    )

def lab_to_rgb(L, a, b):
    # 首先将LAB转换到XYZ
    # 参考标准转换公式
    # 使用D65白点
    L = max(0, min(100, L))
    a = max(-128, min(128, a))
    b = max(-128, min(128, b))
    
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    
    def f_inverse(t):
        if t > 6/29:
            return t**3
        else:
            return 3*(6/29)**2*(t - 4/29)
    
    X = Xn * f_inverse(fx)
    Y = Yn * f_inverse(fy)
    Z = Zn * f_inverse(fz)
    
    # 然后将XYZ转换为RGB（使用sRGB转换矩阵）
    # XYZ to linear RGB
    R_linear = 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
    G_linear = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
    B_linear = 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z
    
    # 线性RGB到sRGB（gamma校正）
    def gamma_correct(channel):
        if channel <= 0.0031308:
            return 12.92 * channel
        else:
            return 1.055 * (channel ** (1/2.4)) - 0.055
    
    R = gamma_correct(R_linear)
    G = gamma_correct(G_linear)
    B = gamma_correct(B_linear)
    
    # 裁剪到[0, 1]范围
    R = max(0, min(1, R))
    G = max(0, min(1, G))
    B = max(0, min(1, B))
    
    return R, G, B
