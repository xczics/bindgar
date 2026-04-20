
from typing import List, Tuple, Optional, Dict, Any
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np  # type: ignore
from numpy import ndarray
from .physics import M_SUN, M_EARTH
from .input import InputLoader, InputAcceptable
from .cli import register_command, _dynamic_import_analyze_modules
import sys
import os

def scatter_positions(ax: Axes, xym_arrays: ndarray, **kwargs) -> None:
    ...

def scatter_ae(ax: Axes, 
               a_array: ndarray, e_array: ndarray, 
               m_array: Optional[ndarray] = None,
               **kwargs) -> None:
    ...

def scatter_am(ax: Axes,
                a_array: ndarray, m_array: ndarray, 
                **kwargs) -> None:
     ...

def register_subplot_drawer(name: str, params: InputAcceptable):
    ...

# This file is auto-generated during development. Do not edit manually.
from matplotlib.axes import Axes
from typing import Any, Optional


def call_melt_mass(ax: Axes, simulations_lists = None, color_map = None, m_start = None, m_end = None, m_log = None, f_start = None, f_end = None, group_mode = None, label = None, gass_giant_indexes = None) -> None:
    """ 
- **simulations_lists** (default: `None`, type: `list`): list of simulation paths
- **color_map** (default: `None`, type: `list`): color map, none or the same shape as simulations_lists
- **m_start** (default: `0.0`, type: `float`): starting mass for the plot (in M_earth)
- **m_end** (default: `1.5`, type: `float`): ending mass for the plot (in M_earth)
- **m_log** (default: `True`, type: `bool`): whether to use log scale for mass
- **f_start** (default: `0.0`, type: `float`): starting melt fraction for the plot
- **f_end** (default: `10.0`, type: `float`): ending melt fraction for the plot
- **group_mode** (default: `False`, type: `bool`): whether to group by simulation groups
- **label** (default: `None`, type: `list`): label for the plot, if group_mode is True, it will label the group, else it will label each simulation.
- **gass_giant_indexes** (default: `[-1]`, type: `list`): list of indexes of gas giant in simulations, default is [-1]. Set a list with one vary large element to set all particles as non-gas-giant.
    """
    ... 

def call_collision_speeds(ax: Axes, simulations_lists = None, m_start = None, m_end = None, x_lim = None, label = None, group_mode = None, color_map = None) -> None:
    """ 
- **simulations_lists** (default: `None`, type: `list`): list of simulation paths
- **m_start** (default: `0.5`, type: `float`): starting final mass to consider (in M_earth)
- **m_end** (default: `1.4`, type: `float`): ending final mass to consider (in M_earth)
- **x_lim** (default: `[0.0, 40.0]`, type: `list`): x axis limit, [xmin,xmax] in km/s
- **label** (default: `None`, type: `list`): label for the plot, list of str if group_mode is True, else str
- **group_mode** (default: `False`, type: `bool`): whether to group by simulation groups
- **color_map** (default: `None`, type: `list`): color map for different simulation groups
    """
    ... 

def call_speed_and_result(ax: List[Axes], simulations_lists = None, m_range = None, m_log_scale = None, speed_range = None, speed_log = None, label = None, group_mode = None, shape_map = None, color_map = None, filter_list = None, background_list = None) -> None:
    """ 
- **simulations_lists** (default: `None`, type: `list`): list of simulation paths
- **m_range** (default: `[0.01, 5.0]`, type: `list`): range of merged mass during plot (in M_earth), [m_start, m_end]
- **m_log_scale** (default: `True`, type: `bool`): whether to use log scale for mass axis
- **speed_range** (default: `[0.9, 10.0]`, type: `list`): range of collision relative speed during plot, [v_start, v_end]. The unit is v_esc
- **speed_log** (default: `True`, type: `bool`): whether to use log scale for speed axis
- **label** (default: `None`, type: `list`): label for the plot, list of str if group_mode is True, else str
- **group_mode** (default: `False`, type: `bool`): whether to group by simulation groups
- **shape_map** (default: `['d', 'o', '^', 's', 'v', '<', '>', 'p', '*']`, type: `list`): shape map for different simulation groups
- **color_map** (default: `None`, type: `list`): color map for different simulation groups, if not provided, use default colors from matplotlib
- **filter_list** (default: `None`, type: `list`): list of filters to apply on the collisions in each subplots. Each element should be a tuple with four element: (min_gamma, max_gamma, min_angle, max_angle)
- **background_list** (default: `None`, type: `list`): list of parameters to chose which background to plot. Each element should be a tuple: (background_contour_name, gamma, angle)
    """
    ... 

def call_speed_mass_distribution(ax: Axes, simulations_lists = None, m_range = None, m_log_scale = None, speed_range = None, unit = None, label = None, group_mode = None, shape_map = None, color_map = None, need_nb = None, nb_label = None, figure_title = None) -> None:
    """ 
- **simulations_lists** (default: `None`, type: `list`): list of simulation paths
- **m_range** (default: `[0.01, 5.0]`, type: `list`): range of merged mass during plot (in M_earth), [m_start, m_end]
- **m_log_scale** (default: `True`, type: `bool`): whether to use log scale for mass axis
- **speed_range** (default: `[0.0, 80.0]`, type: `list`): range of collision relative speed during plot, [v_start, v_end]. The unit would be the same as 'unit' parameters
- **unit** (default: `km/s`, type: `str`): unit for collision relative speed, can be 'km/s', 'auptu', or 'v_esc' (escape velocity of merged body)
- **label** (default: `None`, type: `list`): label for the plot, list of str if group_mode is True, else str
- **group_mode** (default: `False`, type: `bool`): whether to group by simulation groups
- **shape_map** (default: `['d', 'o', '^', 's', 'v', '<', '>', 'p', '*']`, type: `list`): shape map for different simulation groups
- **color_map** (default: `None`, type: `list`): color map for different simulation groups, if not provided, default colors will be generate through the bind_color function based on the yita and impact angle of each collision
- **need_nb** (default: `False`, type: `bool`): whether to add a background to show the parameter area where the nakajima's scaling law is intropolated or extropolated.
- **nb_label** (default: `False`, type: `bool`): whether to add label during plotting the nakajima's scaling law background.
- **figure_title** (default: `Collision Speed-Mass Distribution`, type: `str`): title for the figure
    """
    ... 
