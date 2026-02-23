from ..input import InputLoader,InputAcceptable
from ..cli import register_command
from ..output import SimulationOutput
from ..devol.collision_event import CollisionEvent, SimulationMeltsEvolution
from ..common import CyclicList,default_colors
from ..vi import register_subplot_drawer
from ..physics import M_EARTH, M_SUN
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Dict, Any
import numpy as np


@register_command("devol-calc")
def main():
    DEFAULT_PARAMS: InputAcceptable = {
        "simulations_lists" : {
            "default": None,
            "help": "list of simulation paths",
            "short": "l",
            "type": list,
        },
    }
    input_params = InputLoader(DEFAULT_PARAMS).load()
    simulations_lists = input_params["simulations_lists"]
    for simulation in simulations_lists:
        print(f"Processing simulation: {simulation}")
        simobj = SimulationOutput(simulation)
        devol_out = simobj.open_write_pip("devol.out",fmt="<< m_melt:.6e c:9.6f t:.2f m_loss:.6e T_increase:9.2f >>")
        collisions = simobj.collisions
        with collisions:
            for index, collision in enumerate(collisions):
                print(f"Processing collision {index}")
                col_event = CollisionEvent(collision)
                m_melt = col_event.melt_mass
                _, t, m_loss, C = col_event.devoltilize()
                devol_out.write_data({
                    "m_melt": m_melt,
                    "c": C,
                    "t": t/(24*3600*365.25), 
                    "m_loss": m_loss,
                    "T_increase": col_event.melt_T_increase(),
                })
                print(f"  Melted mass: {m_melt:.6e}, Peak Temperature: {(col_event.melt_T_increase()+1200.0):9.2f} Devolatilization loss: {m_loss:.6e}, C: {C:.6f}, Time: {t/(24*3600*365.25):.2f} years")
        devol_out.close()

MELT_DRAW_PARAMETERS: InputAcceptable = {
        "simulations_lists" : {
            "default": None,
            "help": "list of simulation paths",
            "short": "l",
            "type": list,
        },
        "color_map" :{
            "default": None,
            "help": "color map, none or the same shape as simulations_lists",
            "type": list,
        },
        "m_start" : {
            "default": 0.0,
            "help": "starting mass for the plot (in M_earth)",
            "type": float,
        },
        "m_end" : {
            "default": 1.5,
            "help": "ending mass for the plot (in M_earth)",
            "type": float,
        },
        "m_log" : {
            "default": True,
            "help": "whether to use log scale for mass",
            "type": bool,
        },
        "f_start" :{
            "default": 0.0,
            "help": "starting melt fraction for the plot",
            "type": float,
        },
        "f_end" : {
            "default": 10.0,
            "help": "ending melt fraction for the plot",
            "type": float,
        },
        "group_mode" : {
            "default": False,
            "help": "whether to group by simulation groups",
            "type": bool,
        },
        "label" : {
            "default": None,
            "help": "label for the plot, if group_mode is True, it will label the group, else it will label each simulation.",
            "type": list,
        },
        "gass_giant_indexes" : {
            "default": [-1],
            "help": "list of indexes of gas giant in simulations, default is [-1]. Set a list with one vary large element to set all particles as non-gas-giant.",
            "type": list,
        }
    }
@register_subplot_drawer("melt-mass",params=MELT_DRAW_PARAMETERS)
def draw_melt_mass(
                    ax: Axes,
                    param: Dict[str, Any]) -> None:
    simulations_lists = param["simulations_lists"]
    color_map = param["color_map"]
    m_start = param["m_start"]
    m_end = param["m_end"]
    m_log = param["m_log"]
    f_start = param["f_start"]
    f_end = param["f_end"]
    group_mode = param["group_mode"]
    label = param["label"]
    gass_giant_indexes = param["gass_giant_indexes"]
    if color_map is not None:
        color_map = CyclicList(color_map)
    else:
        color_map = default_colors
    if group_mode:
        simulations_groups = simulations_lists
    else:
        simulations_groups = [[sim] for sim in simulations_lists]
    for group_index, simulations in enumerate(simulations_groups):
        group_label = label[group_index] if label is not None else None
        m_list = []
        f_list = []
        for sim_index, simulation in enumerate(simulations):
            print("="*20+simulation+"="*20)
            simobj = SimulationOutput(simulation,gass_giants_indexes=gass_giant_indexes)
            sim_label = simobj.get_input_params("Output name")
            if sim_index == 0 and group_label is None:
                group_label = sim_label
            evolutionobj = SimulationMeltsEvolution(simobj)
            final_particles = simobj.survivals_without_gas_giants_data
            for particle in final_particles:
                m_list.append(particle["m"])
                print(f"Particle {particle['i']}: mass = {particle['m']*M_SUN/M_EARTH:.6e} M_earth, melt fraction = {evolutionobj.get_particle_melt_fractions(particle['i']):.6e}")
                f_list.append(evolutionobj.get_particle_melt_fractions(particle["i"]))
        print(f_list)
        m_array = np.array(m_list) * M_SUN / M_EARTH
        f_array = np.array(f_list)
        ax.scatter(m_array, f_array, label=group_label, alpha=0.7, color=color_map[group_index])
    ax.set_xlim(m_start, m_end)
    ax.set_ylim(f_start, f_end)
    ax.set_xlabel(r"Mass ($M_{earth}$)")
    ax.set_ylabel("Total Melt fraction (× 100 %)")
    if m_log:
        ax.set_xscale("log")
    if label is not None:
        ax.legend()

@register_command("melt-analyze")
def melt_analyze():
    DEFAULT_PARAMS: InputAcceptable = {
        **MELT_DRAW_PARAMETERS,
        "figure_file" : {
            "default": "analogs_plot.png",
            "help": "output figure file name",
            "type": str,
            "short": "o",
        },
    }
    input_params = InputLoader(DEFAULT_PARAMS).load()
    figure_file = input_params["figure_file"]
    plt.figure(figsize=(12,6))
    ax = plt.gca()
    draw_melt_mass(ax, input_params)
    plt.tight_layout()
    plt.savefig(figure_file)


                 
if __name__ == "__main__":
    main()