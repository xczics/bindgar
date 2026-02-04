from ..input import InputLoader,InputAcceptable
from os import path
import numpy as np  # type: ignore

from ..common import default_colors,CyclicList
from ..input import InputLoader
from ..output import SimulationOutput
from ..physics import M_SUN, M_EARTH, convert_factor_from_auptu_to_kmps, AnalogCriteria, stastic_kde
from ..cli import register_command
from matplotlib import pyplot as plt  # type: ignore
from matplotlib.axes import Axes

from typing import List, Union, Dict, Any

def tell_m_in_range(m_start, m_end, skip_gass_giant: bool = False, gas_giant_indexes: List[int] = [-1]):
    def inner(part):
        if skip_gass_giant and part["i"] in gas_giant_indexes:
            return False
        else:
            m = part["m"] * M_SUN / M_EARTH
            return (m >= m_start) and (m <= m_end)
    return inner

COLLISION_DRAWER_PARAMS: InputAcceptable = {
        "simulations_lists" : {
            "default": None,
            "help": "list of simulation paths",
            "short": "l",
            "type": list,
        },
        "m_start" : {
            "default": 0.5,
            "help": "starting final mass to consider (in M_earth)",
            "type": float,
        },
        "m_end" : {
            "default": 1.4,
            "help": "ending final mass to consider (in M_earth)",
            "type": float,
        },
        "x_lim" : {
            "default": [0.0, 40.0],
            "help": "x axis limit, [xmin,xmax] in km/s",
            "type": list,
        },
        "label" : {
            "default": None,
            "help": "label for the plot, list of str if group_mode is True, else str",
            "type": list,
        },
        "group_mode" : {
            "default": False,
            "help": "whether to group by simulation groups",
            "type": bool,
        },
        "color_map": {
            "default": None,
            "help": "color map for different simulation groups",
            "type": list,
        }
    }

def draw_collision_speeds(ax: Axes,
                          param: Dict[str, Any]) -> None:
    simulations_lists = param["simulations_lists"]
    m_start = param["m_start"]
    m_end = param["m_end"]
    mass_filter = tell_m_in_range(m_start, m_end)
    if param["color_map"] is not None:
        color_map = CyclicList(param["color_map"])
    else:
        color_map = default_colors
    if not param["group_mode"]:
        simulations_groups = [simulations_lists]
    else:
        simulations_groups = simulations_lists
    label = param["label"] if param["label"] is not None else []
    for index, simpath_lists in enumerate(simulations_groups):
        relvs = []
        for simpath in simpath_lists:
            simobj = SimulationOutput(simpath)
            concern_indexes = simobj.filter_final_indexes(mass_filter)
            for collision in simobj.collisions():
                if collision['indexi'] in concern_indexes or collision['indexj'] in concern_indexes:
                    dvx = collision['vxi'] - collision['vxj']
                    dvy = collision['vyi'] - collision['vyj']
                    dvz = collision['vzi'] - collision['vzj']
                    relv = np.sqrt(dvx**2 + dvy**2 + dvz**2)
                    relvs.append(relv)
        if relvs == []:
            continue
        relvs = np.array(relvs)
        relvs *= convert_factor_from_auptu_to_kmps
        ax.scatter(relvs, np.zeros_like(relvs), color=color_map[index], s=5, alpha=0.6)
        x_max = np.max(relvs) * 1.1
        x_min = np.min(relvs) * 0.9
        kde_x = np.linspace(x_min, x_max, 200)
        kde = stastic_kde(relvs, kde_x)
        ax.plot(kde_x, kde, color=color_map[index], lw=2, label=label[index])
    if param["x_lim"] is not None:
        ax.set_xlim(param["x_lim"][0], param["x_lim"][1])
    ax.set_title(f"Collision Relative Speeds for Final Mass in {m_start} - {m_end} " + r"$M_{earth}$")
    ax.set_xlabel("Collision Relative Speed (km/s)")
    ax.set_ylabel("KDE")
    ax.legend()

@register_command("draw-multi-collision-speeds", help_msg="Run several draw-collision-speeds, and output a combined figure.")
def draw_multi_collision_speeds():
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
        }
    }
    input_params = InputLoader(DEFAULT_PARAMS).load()
    subparam_loader = InputLoader(COLLISION_DRAWER_PARAMS)
    figure_file = input_params["figure_file"]
    sub_params_files = input_params["sub_params"]
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
    #print(f"Drawing {n_subplots} subplots with {n_rows} rows and {n_cols} columns.")
    #print(f"Output figure to {figure_file}.")
    #print("Loading sub-parameter files:")
    #print(sub_params_files)
    plt.figure(figsize=(8*n_cols,4*n_rows))
    index_chapter = 'a'
    for i, sub_param_file in enumerate(sub_params_files):
        sub_params = subparam_loader.load_yaml(sub_param_file)
        ax = plt.subplot(n_rows, n_cols, i+1)
        draw_collision_speeds(ax, sub_params)
        ax.text(0.02, 0.95, f"{index_chapter})", transform=ax.transAxes, fontsize=16, verticalalignment='top')
        index_chapter = chr(ord(index_chapter) + 1)
    plt.tight_layout()
    plt.savefig(figure_file)

@register_command("draw-collision-speeds")
def main():
    DEFAULT_PARAMS: InputAcceptable = {
        "figure_file" : {
            "default": "collision_relative_speeds.png",
            "help": "output figure file name",
            "type": str,
        },
        **COLLISION_DRAWER_PARAMS
    }
    input_params = InputLoader(DEFAULT_PARAMS).load()
    figure_file = input_params["figure_file"]
    plt.figure(figsize=(12,6))
    ax = plt.gca()
    draw_collision_speeds(ax, input_params)
    plt.tight_layout()
    plt.savefig(figure_file)
                 
if __name__ == "__main__":
    main()