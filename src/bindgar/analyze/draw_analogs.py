from ..input import InputLoader  # type: ignore
from os import path
import numpy as np  # type: ignore

from ..common import default_colors, CyclicList
from ..physics import AnalogCriteria, calculate_orbital, M_SUN, M_EARTH
from ..output import SimulationOutput
from typing import List, Union
from ..cli import register_command

def get_final_particles(SimulationOutputObject: SimulationOutput,
                        gas_giant_indexes: List[int] = [-1],
                        skip_gass_giant: bool = True):
    last_output = SimulationOutputObject.load_last_output()
    final_particles = []
    for data in last_output:
        if skip_gass_giant and data["i"] in gas_giant_indexes:
            continue
        final_particles.append(data)
    return final_particles

def draw_analogs(simulations_lists: List[str], color_map: List[str], 
                  planet_like_critical: List[AnalogCriteria], a_start: float, 
                  a_end: float, m_start: float, m_end: float, color_analogs: List[str],
                  figure_file: str = "analogs_plot.png"):
    import matplotlib.pyplot as plt  # type: ignore
    plt.figure(figsize=(10, 8))
    ax=plt.gca()
    for sim_index, simulation in enumerate(simulations_lists):
        SimOutObj = SimulationOutput(simulation)
        final_particles = get_final_particles(SimOutObj)
        a_values = []
        for part in final_particles:
            a, e, i = calculate_orbital(
                part['x'], part["y"], part["z"],
                part["vx"], part["vy"], part["vz"],
            )
            a_values.append(a)
        m_values = [part["m"] * M_SUN / M_EARTH for part in final_particles]
        ax.scatter(a_values, m_values, color=color_map[sim_index], label=SimOutObj.get_input_params("Output name"), alpha=0.6, s=10)
    # draw range of planet-like analogs
    for crit_index, criteria in enumerate(planet_like_critical):
        ax.fill_betweenx(
            [criteria.m_lower, criteria.m_upper],
            criteria.a_lower, criteria.a_upper,
            color=color_analogs[crit_index],
            alpha=0.3,
            label=f"{criteria.name} - like"
        )
    ax.set_xlim(a_start, a_end)
    ax.set_ylim(m_start, m_end)
    ax.set_xlabel("Semi-major axis (a.u.)")
    ax.set_ylabel("Mass (M_earth)")
    ax.set_title("Final Particles and Planet-like Analogs")
    ax.legend()
    # automatically adjust layout
    plt.tight_layout()
    plt.savefig(figure_file)

@register_command("draw-analogs")
def main():
    DEFAULT_PARAMS = {
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
        "planet_like_critical" : {
            "default": None,
            "help": "critical infos limit for planet-like analogs",
            "type": list,
        },
        "a_start" : {
            "default": 0.3,
            "help": "starting semi-major axis for the plot (in a.u.)",
            "type": float,
        },
        "a_end" : {
            "default": 3.0,
            "help": "ending semi-major axis for the plot (in a.u.)",
            "type": float,
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
        "color_analogs" : {
            "default": ["yellow","green","red"],
            "help": "color for analog regions",
            "type": list,
        },
    }
    input_params = InputLoader(DEFAULT_PARAMS).load()
    simulations_lists = input_params["simulations_lists"]
    if simulations_lists is None or len(simulations_lists) == 0:
        raise ValueError("simulations_lists is required and cannot be empty.") 
    color_map = CyclicList(input_params["color_map"]) if input_params["color_map"] is not None else default_colors
    planet_like_critical = input_params["planet_like_critical"]
    if planet_like_critical is None:
        from ..physics import DEFAULT_VENUS_ANALOG, DEFAULT_EARTH_ANALOG, DEFAULT_MARS_ANALOG
        planet_like_critical = [DEFAULT_VENUS_ANALOG, DEFAULT_EARTH_ANALOG, DEFAULT_MARS_ANALOG]
    else:
        planet_like_critical = [AnalogCriteria(**crit) for crit in planet_like_critical]
    a_start = input_params["a_start"]
    a_end = input_params["a_end"]
    m_start = input_params["m_start"]
    m_end = input_params["m_end"]
    color_analogs = input_params["color_analogs"]
    draw_analogs(simulations_lists, color_map, planet_like_critical, a_start, a_end, m_start, m_end, color_analogs)

if __name__ == "__main__":
    main()