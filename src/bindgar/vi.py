from typing import List, Tuple, Optional
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np  # type: ignore
from numpy import ndarray
from .physics import M_SUN, M_EARTH
from .input import InputLoader, InputAcceptable
from .cli import register_command, _dynamic_import_analyze_modules
import sys

subplot_drawers = {}
subplot_inputers = {}

def scatter_positions(ax: Axes, xym_arrays: ndarray, **kwargs) -> None:
    """Scatter plot of x-y positions sizes by mass.

    Args:
        ax (Axes): Matplotlib Axes object to plot on.
        xym_arrays (ndarray): Numpy array of shape (N, 3) where each row is (x, y, mass).
    """
    x = xym_arrays[:, 0]
    y = xym_arrays[:, 1]
    masses = xym_arrays[:, 2]
    ref_mass = kwargs.pop('ref_mass', min(masses))
    set_equal = kwargs.pop('set_equal', True)
    size_scale = kwargs.pop('size_scale', 2)

    sizes = (masses / ref_mass) ** (2/3) * size_scale  # Scale sizes by mass^(2/3)

    max_r = max((x**2 + y**2)**0.5)
    print(f"Max radius in scatter_positions: {max_r}")
    if set_equal:
        ax.set_aspect('equal')
    ax.set_xlim(-max_r*1.1, max_r*1.1)
    ax.set_ylim(-max_r*1.1, max_r*1.1)
    split_colors_by_mass(masses, kwargs)
    ax.scatter(x, y, s=sizes, **kwargs)

    ax.set_xlabel('x (a.u.)')
    ax.set_ylabel('y (a.u.)')

def split_colors_by_mass(m_array: ndarray,
                         kwargs: dict) -> None:
    """Helper function to split colors by mass threshold.
    """
    if 'm_split' in kwargs and m_array is not None and ('c' not in kwargs and 'color' not in kwargs):
        m_split = kwargs.pop('m_split')
        if 'color_small' in kwargs and 'color_large' in kwargs:
            color_small = kwargs.pop('color_small')
            color_large = kwargs.pop('color_large')
            colors = [color_small if m <= m_split else color_large for m in m_array] 
        else:
            colors = ['blue' if m <= m_split else 'orange' for m in m_array]
        kwargs['c'] = colors
    

def scatter_ae(ax: Axes, 
               a_array: ndarray, e_array: ndarray, 
               m_array: Optional[ndarray] = None,
               **kwargs) -> None:
    """Scatter plot of semi-major axis vs eccentricity.
    Args:
        ax (Axes): Matplotlib Axes object to plot on.
        a_array (ndarray): Numpy array of semi-major axes.
        e_array (ndarray): Numpy array of eccentricities.
        m_array (ndarray, optional): Numpy array of masses for size scaling. Defaults to None.
    """
    if m_array is not None:
        ref_mass = kwargs.pop('ref_mass', min(m_array))
        size_scale = kwargs.pop('size_scale', 2)
        sizes = (m_array / ref_mass) ** (2/3) * size_scale
    else:
        sizes = np.full_like(a_array, 8)
    if m_array is not None:
        split_colors_by_mass(m_array, kwargs)
    ax.scatter(a_array, e_array, s=sizes, **kwargs)
    ax.set_xlabel('Semi-major axis (a.u.)')
    ax.set_ylabel('Eccentricity')
    ax.set_xlim(0, max(a_array)*1.1)
    ax.set_ylim(0, max(e_array)*1.1)

def scatter_am(ax: Axes, 
               a_array: ndarray, m_array: ndarray, 
               **kwargs) -> None:
    """Scatter plot of semi-major axis vs mass.
    Args:
        ax (Axes): Matplotlib Axes object to plot on.
        a_array (ndarray): Numpy array of semi-major axes.
        m_array (ndarray): Numpy array of masses.
    """
    m_array = m_array.copy()
    ref_mass = kwargs.pop('ref_mass', min(m_array))
    size_scale = kwargs.pop('size_scale', 2)
    sizes = (m_array / ref_mass) ** (2/3) * size_scale
    max_mas = kwargs.pop('max_mas', 200.0)
    split_colors_by_mass(m_array, kwargs)
    m_array /= M_EARTH / M_SUN  # Convert mass to M_earth
    if max_mas == -1:
        max_mas = max(m_array)
    ax.scatter(a_array, m_array, s=sizes, **kwargs)
    ax.set_xlabel('Semi-major axis (a.u.)')
    ax.set_ylabel(r'Mass ($M_{earth}$)')
    ax.set_xlim(0, max(a_array)*1.1)
    ax.set_ylim(min(m_array)*0.85, min(max_mas, max(m_array)*1.1))
    ax.set_yscale('log')

@register_command("multi-draw", help_msg="Run several sub-commands, and output a combined figure.")
def draw_multi_plots():
    """
    An user-friendly command to draw multiple subplots.
    """
    DEFAULT_PARAMS: InputAcceptable = {
        "figure_file" : {
            "default": "multi_collision_relative_speeds.png",
            "help": "output figure file name",
            "short": "o",
            "type": str,
        },
        "n_cols": {
            "default": 1,
            "help": "number of columns in the figure",
            "type": int,
        },
        "n_rows": {
            "default": 0,
            "help": "number of rows in the figure",
            "type": int,
        },
        "sub_params" : {
            "default": None,
            "help": "list of yaml files for each subplot. Each yaml file should be a valid input for draw-collision-speeds command.",
            "type": list,
            "short": "i",
        },
        "sub_commands" :{
            "default": None,
            "help": "list of commands for each subplot. Supported sub commands are " + ",".join(subplot_drawers.keys()),
            "type": list,
            "short": "t",
        }
    }
    input_params = InputLoader(DEFAULT_PARAMS).load()
    #subparam_loader = InputLoader(COLLISION_DRAWER_PARAMS)
    figure_file = input_params["figure_file"]
    sub_params_files = input_params["sub_params"]
    sub_commands = input_params["sub_commands"]
    # 断言sub_params_files和sub_commands的长度相同
    assert len(sub_params_files) == len(sub_commands), "The length of sub_params_files and sub_commands should be the same."
    n_subplots = len(sub_params_files)
    if input_params["n_rows"] > 0:
        n_rows = input_params["n_rows"]
        n_cols = (n_subplots + n_rows - 1) // n_rows
    elif input_params["n_cols"] > 0:
        n_cols = input_params["n_cols"]
        n_rows = (n_subplots + n_cols - 1) // n_cols
    else:
        n_cols = 1
        n_rows = n_subplots
    plt.figure(figsize=(8*n_cols,4*n_rows))
    index_chapter = 'a'
    # iter over subplots. each subplot has a sub_params_files and a sub_commd.
    for i, (sub_param_file, sub_command) in enumerate(zip(sub_params_files, sub_commands)):
        loader = InputLoader(subplot_inputers[sub_command])
        sub_params = loader.load_yaml(sub_param_file)
        ax = plt.subplot(n_rows, n_cols, i+1)
        factory = subplot_drawers[sub_command]
        factory(ax, sub_params)
        ax.text(0.02, 0.95, f"{index_chapter})", transform=ax.transAxes, fontsize=16, verticalalignment='top')
        index_chapter = chr(ord(index_chapter) + 1)
    plt.tight_layout()
    plt.savefig(figure_file)


def register_subplot_drawer(name: str, params: InputAcceptable):
    """
    an wrapper function to register subplot drawing functions. 
    the wrapper should take 2 parameters: one is the name of subplot, another is an InputAcceptable dict.
    the function should also take 2 parameters: one is the Axes to draw on, another is the input parameters loaded from the InputAcceptable dict.
    In the meanwhile, the wrapper function will export a function to global namespace of this module, which can be called directly with the Axes and the input parameters as kwargs. The name of the exported function will be "call_" + the original function name.
     """
    def decorator(func):
        def wrapper(ax: Axes, input_params: dict):
            func(ax, input_params)
        def export_function(ax: Axes, **kwargs):
            loader = InputLoader(params)
            input_params = loader.load_by_kwargs(kwargs)
            func(ax, input_params)
        subplot_drawers[name] = wrapper
        subplot_inputers[name] = params
        # 将函数导出到当前模块的全局命名空间
        module = sys.modules[__name__]
        export_function.__name__ = f"call_{name.replace('-', '_')}"
        export_function.__module__ = __name__
        export_function.__doc__ = InputLoader(params).document()
        
        # 将函数设置为模块属性
        setattr(module, f"call_{name.replace('-', '_')}", export_function)
        return wrapper
    return decorator

_dynamic_import_analyze_modules()
