from ..input import InputLoader,InputAcceptable
from os import path
import numpy as np  # type: ignore

from ..common import default_colors,CyclicList
from ..input import InputLoader
from ..output import SimulationOutput
from ..physics import M_SUN, M_EARTH, convert_factor_from_auptu_to_kmps, AnalogCriteria, stastic_kde
from ..cli import register_command

from typing import List, Union, Dict, Any

def tell_m_in_range(m_start, m_end, skip_gass_giant: bool = False, gas_giant_indexes: List[int] = [-1]):
    def inner(part):
        if skip_gass_giant and part["i"] in gas_giant_indexes:
            return False
        else:
            m = part["m"] * M_SUN / M_EARTH
            return (m >= m_start) and (m <= m_end)
    return inner

@register_command("draw-collision-speeds")
def main():
    DEFAULT_PARAMS: InputAcceptable = {
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
        "figure_file" : {
            "default": "collision_relative_speeds.png",
            "help": "output figure file name",
            "type": str,
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
    input_params = InputLoader(DEFAULT_PARAMS).load()
    simulations_lists = input_params["simulations_lists"]
    m_start = input_params["m_start"]
    m_end = input_params["m_end"]
    figure_file = input_params["figure_file"]
    mass_filter = tell_m_in_range(m_start, m_end)
    if input_params["color_map"] is not None:
        color_map = CyclicList(input_params["color_map"])
    else:
        color_map = default_colors
    from matplotlib import pyplot as plt  # type: ignore
    plt.figure(figsize=(12,6))
    ax = plt.gca()
    if not input_params["group_mode"]:
        simulations_groups = [simulations_lists]
    else:
        simulations_groups = simulations_lists
    label = input_params["label"] if input_params["label"] is not None else []
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
    if input_params["x_lim"] is not None:
        ax.set_xlim(input_params["x_lim"][0], input_params["x_lim"][1])
    ax.set_title(f"Collision Relative Speeds for Final Mass in {m_start} ~ {m_end} M_earth")
    ax.set_xlabel("Collision Relative Speed (km/s)")
    ax.set_ylabel("KDE")
    ax.legend()
    plt.tight_layout()
    plt.savefig(figure_file)
                 
if __name__ == "__main__":
    main()