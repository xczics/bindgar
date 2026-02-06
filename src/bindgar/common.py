from typing import List
import numpy as np
from typing import Union, Optional, Tuple, Dict
from matplotlib.axes import Axes

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

def bind_color(L_array: float | List[float] | np.ndarray,
        theta_array: float | List[float] | np.ndarray,
        r_array: float | List[float] | np.ndarray,
        L_color_range: tuple = (0, 100),
        theta_color_range: tuple = (0, 360),
        r_color_range: tuple = (0, 1),
        L_value_range: tuple|None = None,
        theta_value_range: tuple|None = None,
        r_value_range: tuple|None = None,
        log_scale: Tuple[bool, bool, bool]  = (False, False, False)) -> List[str]:
    """
    Bind L, theta, r arrays to colors by converting them to LAB color space and then to RGB hex strings.
    Args:
        L_array: values binds for L component.
        theta_array: values binds for angle component.
        r_array: values binds for saturation component.
        L_color_range: Tuple specifying the range of L for color mapping
        theta_color_range: Tuple specifying the range of theta for color mapping
        r_color_range: Tuple specifying the range of r for color mapping
        L_value_range: Optional tuple specifying the actual value range of L for normalization
        theta_value_range: Optional tuple specifying the actual value range of theta for normalization
        r_value_range: Optional tuple specifying the actual value range of r for normalization
        log_scale: Tuple indicating whether to apply logarithmic scaling to each binding values (L, theta, r)
    """
    # Normalize and optionally log-scale the input arrays
    def normalize_and_log_scale(array, value_range, color_range, log):
        # if value_range is None and the array is a single value, it will be seen as a single color value.
        if value_range is None:
            if isinstance(array, (int, float)):
                return np.array([array])
            if isinstance(array, list) and len(array) == 1:
                return np.array([array])
            if isinstance(array, np.ndarray) and array.size == 1:
                return array
        # normal case: convert to numpy array and normalize to color range
        if isinstance(array, (int, float)):
            array = np.array([array])
        else:
            array = np.array(array)
        if log:
            array = np.log10(array)
        if value_range is not None:
            min_val, max_val = value_range
        else:
            min_val, max_val = np.min(array), np.max(array)
        color_min, color_max = color_range
        normalized = (array - min_val) / (max_val - min_val) * (color_max - color_min) + color_min
        return normalized
    # Normalize each array
    L_normalized = normalize_and_log_scale(L_array, L_value_range, L_color_range, log_scale[0])
    theta_normalized = normalize_and_log_scale(theta_array, theta_value_range, theta_color_range, log_scale[1])
    r_normalized = normalize_and_log_scale(r_array, r_value_range, r_color_range, log_scale[2])
    # convert to the same shape. If one of them is a single value, broadcast it to the shape of the others.
    max_length = max(L_normalized.size, theta_normalized.size, r_normalized.size)
    # even the shape is not the same, just repeat the single or shorter arrays to match the longest one.
    if L_normalized.size < max_length:
        L_normalized = np.resize(L_normalized, max_length)
    if theta_normalized.size < max_length:
        theta_normalized = np.resize(theta_normalized, max_length)
    if r_normalized.size < max_length:
        r_normalized = np.resize(r_normalized, max_length)
    # Convert to RGB hex colors
    colors = []
    for L, theta, r in zip(L_normalized, theta_normalized, r_normalized):
        a = r * np.cos(np.radians(theta)) * 128
        b = r * np.sin(np.radians(theta)) * 128
        R, G, B = lab_to_rgb(L, a, b)
        colors.append(rgb_to_hex(R, G, B))
    return colors

def bg_colorbar(ax: Axes,
                L_lim: tuple | float = 60,
                theta_lim: tuple | float = 90,
                r_lim: tuple | float = 0.5,
                color_bar_height: float = 0.35,
                color_bar_width: float = 0.95,
                color_bar_h_pad: float = 0.05,
                color_presision: int = 100,
                color_bar_title: str = "None",
                color_bar_labels: Dict[str, float]|None = None,
                label_shift: float = 0.02,
                title_shift: float = 0.02,
                title_size: float = 8.0,
                ):
    """
    Add a colorbar to the given Axes that illustrates the color mapping used in bind_color function.
    Args:
        ax: The Matplotlib Axes to which the colorbar will be added.
        L_lim: The limit for the L component (either a single value or a tuple specifying the range).
        theta_lim: The limit for the theta component (either a single value or a tuple specifying the range).
        r_lim: The limit for the r component (either a single value or a tuple specifying the range).
        color_presision: The number of discrete colors to generate for each component in the colorbar.
        NOTE: only one of the L_lim, theta_lim, r_lim should be a tuple specifying the range. The others should be single values to fix those components in the colorbar.
    """
    # Check the validity of the limits
    lim_list = [L_lim, theta_lim, r_lim]
    tuple_check = [isinstance(lim, tuple) for lim in lim_list]
    if sum(tuple_check) != 1:
        raise ValueError("Exactly one of L_lim, theta_lim, r_lim should be a tuple specifying the range, and the others should be single values.")
    # Generate the colorbar data
    prefix_list = ["L", "theta", "r"]
    array_base = np.array(range(color_presision),dtype=float) / (color_presision - 1)
    array_kwargs = {
        f"{prefix}_array": array_base * (lim[1] - lim[0]) + lim[0] if tuple_check else lim for prefix, lim, tuple_check in zip(prefix_list, lim_list, tuple_check) #type: ignore
    }
    tuple_index = tuple_check.index(True)
    range_kwrages = {
        f"{prefix_list[tuple_index]}_color_range": lim_list[tuple_index],
        f"{prefix_list[tuple_index]}_value_range": lim_list[tuple_index],
    }
    colors = bind_color(**array_kwargs, **range_kwrages) # type: ignore
    # Create a colorbar image
    from matplotlib.patches import Rectangle
    start_x = (1 - color_bar_width) / 2
    cell_width = color_bar_width / color_presision
    for i in range(color_presision):
        ax.add_patch(Rectangle(
            xy = (start_x + i * cell_width, color_bar_h_pad), 
            width= cell_width,
            height=color_bar_height, 
            facecolor=colors[i], transform=ax.transAxes, edgecolor='none'))
    # Add labels and title
    lim = lim_list[tuple_index]
    if color_bar_labels is not None:
        for label, value in color_bar_labels.items():
            ax.text(
                x = start_x + value * color_bar_width, #type: ignore
                y = color_bar_h_pad -  label_shift,
                s = label,
                transform=ax.transAxes,
                ha='center',
                va='top',
                fontsize=title_size * 0.8,
            )
    else:
        label_shift = 0
    ax.text(
        x = start_x + color_bar_width / 2,
        y = color_bar_h_pad - title_shift - label_shift,
        s = color_bar_title,
        transform=ax.transAxes,
        ha='center',
        va='top',
        fontsize=title_size,
    )
    
    

    
