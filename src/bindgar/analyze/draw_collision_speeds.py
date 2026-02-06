from ..input import InputLoader,InputAcceptable
from os import path
import numpy as np  # type: ignore

from ..common import default_colors,CyclicList,bind_color,bg_colorbar
from ..input import InputLoader
from ..output import SimulationOutput
from ..physics import M_SUN, M_EARTH, convert_factor_from_auptu_to_kmps, AnalogCriteria, stastic_kde
from ..cli import register_command
from ..vi import register_subplot_drawer
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

COLLISION_MASS_DISTRIBUTION_PARAMS: InputAcceptable = {
    "simulations_lists" : {
        "default": None,
        "help": "list of simulation paths",
        "short": "l",
        "type": list,
    },
    "m_range" : {
        "default": [0.01, 5.0],
        "help": "range of merged mass during plot (in M_earth), [m_start, m_end]",
        "type": list,
    },
    "m_log_scale" : {
        "default": True,
        "help": "whether to use log scale for mass axis",
        "type": bool,
    },
    "speed_range" : {
        "default": [0.0, 80.0],
        "help": "range of collision relative speed during plot, [v_start, v_end]. The unit would be the same as 'unit' parameters",
        "type": list,
    },
    "unit" : {
        "default": "km/s",
        "help": "unit for collision relative speed, can be 'km/s', 'auptu', or 'v_esc' (escape velocity of merged body)",
        "type": str,
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
    "shape_map": {
            "default": ['d', 'o', '^', 's', 'v', '<', '>', 'p', '*'],
            "help": "shape map for different simulation groups",
            "type": list,
    },
    "color_map": {
            "default": None,
            "help": "color map for different simulation groups, if not provided, default colors will be generate through the bind_color function based on the yita and impact angle of each collision",
            "type": list,
    },
    "figure_title":{
        "default": "Collision Speed-Mass Distribution",
        "help": "title for the figure",
        "type": str,
    }
}

@register_subplot_drawer("collision-speeds", params=COLLISION_DRAWER_PARAMS)
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

@register_subplot_drawer("speed-mass-distribution", params=COLLISION_MASS_DISTRIBUTION_PARAMS)
def draw_speed_mass_distribution(ax: Axes,
                                 param: Dict[str, Any]) -> None:
    simulations_lists = param["simulations_lists"]
    m_range = param["m_range"]
    speed_range = param["speed_range"]
    unit = param["unit"]
    if param["shape_map"] is not None:
        shape_map = CyclicList(param["shape_map"])
    else:
        shape_map = default_colors
    if not param["group_mode"]:
        simulations_groups = [simulations_lists]
    else:
        simulations_groups = simulations_lists
    label = param["label"] if param["label"] is not None else []
    use_auto_color = param["color_map"] is None
    color_map = CyclicList(['C0'])
    ref_total_mass = None
    if not use_auto_color:
        color_map = CyclicList(param["color_map"])
    for index, simpath_lists in enumerate(simulations_groups):
        masses = []
        relvs = []
        new_rs = []
        colors = []
        total_masses = []
        for simpath in simpath_lists:
            simobj = SimulationOutput(simpath)
            simobj.set_gass_giant_indexes([-1]) # NOTE: IT WAS PROVIDED AS-IS. IT SHOULD BE EDIT TO SUPPORT OTHER GASS GIANT INDEXES FROM INPUT PARAMS. 
            total_init_mass = simobj.get_init_total_mass(skip_gass_gaint=True)
            masses_sim = []
            relvs_sim = []
            new_rs_sim = []
            angles_sim = []
            yita_sim = []
            total_masses.append(total_init_mass)
            for collision in simobj.collisions():
                dvx = collision['vxi'] - collision['vxj']
                dvy = collision['vyi'] - collision['vyj']
                dvz = collision['vzi'] - collision['vzj']
                x = collision['xi'] - collision['xj']
                y = collision['yi'] - collision['yj']
                z = collision['zi'] - collision['zj']
                # calculate the impact angle in [0, 90] degree
                angle = np.arccos(np.abs(x*dvx + y*dvy + z*dvz) / (np.sqrt(x**2 + y**2 + z**2) * np.sqrt(dvx**2 + dvy**2 + dvz**2))) * 180 / np.pi
                relv = np.sqrt(dvx**2 + dvy**2 + dvz**2)
                m_merged = collision['mi'] + collision['mj']
                yita = collision['mi'] / m_merged
                yita = 1 - yita if yita > 0.5 else yita
                yita_sim.append(yita)
                masses_sim.append(m_merged)
                relvs_sim.append(relv)
                new_rs_sim.append((collision['ri']**3 + collision['rj']**3)**(1/3))
                angles_sim.append(angle)
            if use_auto_color:
                colors_sim = bind_color(
                L_array = yita_sim,
                theta_array = angles_sim,
                r_array = 0.8,
                theta_color_range=(0,180),
                L_color_range=(80,20),
                L_value_range=(0.0, 0.5),
                )
                colors += colors_sim
            masses += masses_sim
            relvs += relvs_sim
            new_rs += new_rs_sim
        masses = np.array(masses)
        relvs = np.array(relvs)
        average_total_mass = np.mean(total_masses)
        ref_total_mass = average_total_mass if ref_total_mass is None else ref_total_mass
        if unit == "km/s":
            relvs *= convert_factor_from_auptu_to_kmps
        elif unit == "v_esc":
            v_esc = np.sqrt(2 * masses/new_rs )
            relvs /= v_esc
        m_earth = masses * M_SUN / M_EARTH
        if use_auto_color:
            ax.scatter(m_earth, relvs, marker=shape_map[index], color=colors, s=10 * ( average_total_mass / ref_total_mass ),
                   alpha=0.9, label=label[index])
        else:
            ax.scatter(m_earth, relvs, marker=shape_map[index], color=color_map[index],s=10 * ( average_total_mass / ref_total_mass ),
                   alpha=0.9, label=label[index])
    if param["m_log_scale"]:
        ax.set_xscale("log")
    ax.set_xlim(m_range[0], m_range[1])
    ax.set_ylim(speed_range[0], speed_range[1])
    ax.set_title(param["figure_title"])
    ax.set_xlabel("Merged Mass (" + r"$M_{earth}$" + ")")
    if unit != "v_esc":
        ax.set_ylabel(f"Collision Relative Speed ({unit})")
    else:
        ax.set_ylabel(r"Collision Relative Speed (v/$v_{esc}$)")
    ax.legend()
    if use_auto_color:
        # set the points in legend to be grey
        legend = ax.get_legend()
        for handle in legend.legend_handles: #type: ignore
            handle.set_color('grey')  #type:ignore
        # 在图的右下角添加两个color bar, 说明数据点颜色的意义：明度代表初始总质量，颜色代表碰撞角度
        color_bar_ax = ax.inset_axes((0.35, 0.8, 0.25, 0.2), transform=ax.transAxes)
        color_bar_ax.set_axis_off()
        # 明度 color bar
        bg_colorbar(color_bar_ax,
                L_lim=(80, 20),
                theta_lim = 90,
                r_lim = 0.0,
                color_bar_height=0.2,
                color_bar_h_pad=0.75,
                color_bar_width=0.95,
                color_bar_title=r"$M_{impactor}/M_{merged}$",
                color_bar_labels={r"0.0": 0, r"0.5": 1.0 },
                )
        # 颜色 color bar
        bg_colorbar(color_bar_ax,
                L_lim = 50,
                theta_lim=(0,180),
                r_lim=0.8,
                color_bar_height=0.20,
                color_bar_h_pad=0.28,
                color_bar_width=0.95,
                color_bar_title="Impact Angle",
                color_bar_labels={"0°": 0, "30°": 0.33, "45°": 0.5, "60°":0.67,"90°": 1.0},
                title_shift=0.12,
                )

""" This function is replaced by the draw_multi_plots in ..vi. It is kept here for reference.
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
"""

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

@register_command("speed-mass-distribution", help_msg="Draw the distribution of collision speeds .vs. merged mass.")
def speed_mass_distribution():
    DEFAULT_PARAMS: InputAcceptable = {
        "figure_file" : {
            "default": "collision_speed_mass_distribution.png",
            "help": "output figure file name",
            "type": str,
        },
        **COLLISION_MASS_DISTRIBUTION_PARAMS
    }
    input_params = InputLoader(DEFAULT_PARAMS).load()
    figure_file = input_params["figure_file"]
    plt.figure(figsize=(8,6))
    ax = plt.gca()
    draw_speed_mass_distribution(ax, input_params)
    plt.tight_layout()
    plt.savefig(figure_file)

                 
if __name__ == "__main__":
    main()