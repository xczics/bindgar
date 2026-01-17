from ..input import InputLoader  # type: ignore
from os import path
import numpy as np  # type: ignore

from ..common import default_colors, CyclicList
from ..physics import AnalogCriteria, calculate_orbital, M_SUN, M_EARTH, M_Mars, M_Moon
from ..output import SimulationOutput
from typing import List, Union
from ..cli import register_command

import matplotlib.pyplot as plt  # type: ignore
import matplotlib.gridspec as gridspec  # type: ignore
from matplotlib.gridspec import SubplotSpec  # type: ignore

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

def partical_orbital_array(partical_lists: List[dict]):
    a_array = []
    e_array = []
    i_array = []
    m_array = []
    for part in partical_lists:
        a, e, i = calculate_orbital(
            part['x'], part["y"], part["z"],
            part["vx"], part["vy"], part["vz"],
        )
        a_array.append(a)
        e_array.append(e)
        i_array.append(i)
        m_array.append(part["m"] * M_SUN / M_EARTH)
    return np.array(a_array), np.array(e_array), np.array(i_array), np.array(m_array)

def draw_analogs(simulations_lists: List[str], color_map: List[str], 
                  planet_like_critical: List[AnalogCriteria], a_start: float, 
                  a_end: float, m_start: float, m_end: float, color_analogs: List[str],
                  figure_file: str = "analogs_plot.png",
                  auto_color: bool = False,
                  color_params: List[List] = None):
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    for sim_index, simulation in enumerate(simulations_lists):
        SimOutObj = SimulationOutput(simulation)
        if auto_color:
            if color_params is None:
                color = SimOutObj.magic_color()
            else:
                color = SimOutObj.magic_color(properties=color_params[sim_index])
        else:
            color = color_map[sim_index]
        final_particles = get_final_particles(SimOutObj)
        a_values, _, _, m_values = partical_orbital_array(final_particles)
        ax.scatter(a_values, m_values, color=color, label=SimOutObj.get_input_params("Output name"), alpha=0.6, s=20)
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

def draw_horizontal_planets_diagram(a_values: Union[List[float],np.ndarray],
                                    m_values: Union[List[float],np.ndarray],
                                    size_factor: float,
                                    ax: plt.Axes,
                                    x_lim: tuple = (0.3, 3.0),
                                    analog_criteria: List[AnalogCriteria] = None,
                                    analog_color: list = None,
                                    vi_x_tick_labels: bool = False,
                                    ):
    """
    For a given planets list, draw a horizontal diagram showing their semi-major axes, and the size of the circles representing their masses.
    """
    ax.set_xlim(x_lim)
    ax.set_ylim(-1, 1)
    y_zeros = np.zeros_like(a_values)
    ax.scatter(a_values, y_zeros, s=np.array(m_values) ** (2/3) * size_factor, alpha=0.6)
    if analog_criteria is not None:
        if analog_color is None:
            analog_color = ['gray']
        analog_color = CyclicList(analog_color)
        for crit_index, criteria in enumerate(analog_criteria):
            ax.fill_betweenx(
                [-1, 1],
                criteria.a_lower, criteria.a_upper,
                color=analog_color[crit_index],
                alpha=0.3,
                label=f"{criteria.name} - like"
            )
    if not vi_x_tick_labels:    
        ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks(np.linspace(x_lim[0], x_lim[1], num=5))
    ax.set_yticks([-1, 0, 1])
    ax.grid(True)

def draw_horizontal_sim_group(simulations_list: List[str],
                              planet_like_critical: List[AnalogCriteria],
                              plot_area: SubplotSpec,
                              size_factor: float,
                              mass_legend: List[tuple],
                              analog_color: CyclicList ,
                              x_lim: tuple = (0.3, 3.0),
                              analog_criteria: List[AnalogCriteria] = None,
                              group_title: str = None,
                            ):
    """
    For a given group of simulation lists, draw horizontal diagrams for each group in the given plot areas.
    It deals with:
        1. subdivide the plot area to several sub-axes for each simulation group. The simulations are vertical stacked.
        2. read the final particles from each simulation, and call draw_horizontal_planets_diagram to draw each simulation group.
        3. add legends and labels.
    """
    n_simulations = len(simulations_list)
    # 左边留出一点空白放文字
    gs = gridspec.GridSpecFromSubplotSpec(n_simulations + 1, 1, subplot_spec=plot_area, hspace=0, left=0.25, right=0.95, top=0.85, bottom=0.05)
    axes = [plt.subplot(gs[i, 0]) for i in range(n_simulations)]
    for sim_index, simulation in enumerate(simulations_list):
        SimOutObj = SimulationOutput(simulation)
        final_particles = get_final_particles(SimOutObj)
        a_values, _, _, m_values = partical_orbital_array(final_particles)
        ax = axes[sim_index]
        vi_x_tick_labels = (sim_index == n_simulations - 2)
        draw_horizontal_planets_diagram(
            a_values, m_values, size_factor, ax,
            x_lim=x_lim,
            analog_criteria=planet_like_critical,
            analog_color=analog_color,
            vi_x_tick_labels=vi_x_tick_labels,
        )
        # 文字放到左边，框线外侧
        ax.text(x_lim[0] - (x_lim[1] - x_lim[0])*0.01, 0.0, 
                SimOutObj.get_input_params("Output name"), 
                verticalalignment='center', horizontalalignment='right',
                )
    # 在顶部标注analog区域名称，不额外画图例，只在顶部轴上标注名称
    if analog_criteria is not None:
        ax_top = axes[0]
        for crit_index, criteria in enumerate(analog_criteria):
            ax_top.text(
                (criteria.a_lower + criteria.a_upper) / 2,
                1.05,
                f"{criteria.name} - like",
                color=analog_color[crit_index] if analog_color is not None else 'gray',
                horizontalalignment='center',
                verticalalignment='bottom',
            )
    # 添加质量图例
    if mass_legend is not None:
        ax_legend = axes[-1]
        # 用最下方的一个轴来画质量图例，将区域均分为几部分，每个部分画一个质量图例
        n_legend = len(mass_legend)
        legend_x_positions = np.linspace(x_lim[0], x_lim[1], n_legend + 2)[1:-1]
        for legend_index, (mass_value, mass_label) in enumerate(mass_legend):
            ax_legend.scatter(
                legend_x_positions[legend_index], 0.3,
                s=mass_value ** (2/3) * size_factor,
                alpha=0.6,
                label=mass_label,
            )
            ax_legend.text(
                legend_x_positions[legend_index], -0.3,
                mass_label,
                horizontalalignment='center',
                verticalalignment='top',
            )
    ax_legend.set_ylim(-1, 1)
    ax_legend.set_xlim(x_lim)
    #删去坐标轴、刻度和框线
    ax_legend.spines['top'].set_visible(False)
    ax_legend.spines['right'].set_visible(False)
    ax_legend.spines['left'].set_visible(False)
    ax_legend.spines['bottom'].set_visible(False)
    ax_legend.set_xticklabels([])
    ax_legend.set_yticklabels([])
    # 顶部轴添加组标题， 注意，不要盖住analog名称
    if group_title is not None:
        ax_title = axes[0]
        ax_title.text(
            (x_lim[0] + x_lim[1]) / 2,
            1.15,
            group_title,
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=14,
            fontweight='bold',
        )


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
        "figure_file" : {
            "default": "analogs_plot.png",
            "help": "output figure file name",
            "type": str,
            "short": "o",
        },
        "auto_color" : {
            "default": True,
            "help": "auto set the colors by the simulation set",
            "type": bool,
        },
        "color_params" : {
            "default": None,
            "help": "auto set the colors by the simulation set, you can specify the properties for color mapping. It should be the same size as simulations_lists",
            "type": List,
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
    draw_analogs(simulations_lists, color_map, planet_like_critical,
     a_start, a_end, m_start, m_end, color_analogs,
    input_params["figure_file"],input_params["auto_color"], input_params["color_params"])

@register_command("final-planets",help_msg="draw final terrestrial planets.")
def final_planets():
    print("Not implemented yet.")
    exit()
    DEFAULT_PARAMS = {
        "simulations_lists" : {
            "default": None,
            "help": "list of simulation paths, if group_mode is True, it should be list of list",
            "short": "l",
            "type": list,
        },
        "group_mode" : {
            "default": False,
            "help": "whether to group by simulation groups",
            "type": bool,
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
        "planet_like_critical" : {
            "default": None,
            "help": "critical infos limit for planet-like analogs",
            "type": list,
        },
        "color_analogs" : {
            "default": ["yellow","green","red"],
            "help": "color for analog regions",
            "type": list,
        },
        "label" : {
            "default": None,
            "help": "label for the plot, list of str if group_mode is True, else str",
            "type": list,
        },
        "figure_file" : {
            "default": "analogs_plot.png",
            "help": "output figure file name",
            "type": str,
            "short": "o",
        },
        "size_of_earth" : {
            "default": 12.0,
            "help": "the size of a circle representing an Earth-mass planet, the size scales with mass^(2/3)",
            "type": float,
            "short": "s",
        },
        "label_mass_scale" : {
            "default": None,
            "help": "The list of an [mass,label] pair, to set the legend of mass.",
            "type": list,
        },
    }
    input_params = InputLoader(DEFAULT_PARAMS).load()
    simulations_lists = input_params["simulations_lists"]
    a_start = input_params["a_start"]
    a_end = input_params["a_end"]
    planet_like_critical = input_params["planet_like_critical"]
    color_analogs = input_params["color_analogs"]
    if planet_like_critical is None:
        from ..physics import DEFAULT_VENUS_ANALOG, DEFAULT_EARTH_ANALOG, DEFAULT_MARS_ANALOG
        planet_like_critical = [DEFAULT_VENUS_ANALOG, DEFAULT_EARTH_ANALOG, DEFAULT_MARS_ANALOG]
    else:
        planet_like_critical = [AnalogCriteria(**crit) for crit in planet_like_critical]
    label = input_params["label"]
    if label_mass_scale is None:
        label_mass_scale = [(M_Moon / M_EARTH, "Moon"), (M_Mars / M_EARTH, "Mars"), (1.0, "Earth"), (10.0, "10 M_earth")]
    size_of_earth = input_params["size_of_earth"]
    figure_file = input_params["figure_file"]
    if input_params["group_mode"]:
        n_groups = len(simulations_lists)
        plt.figure(figsize=(10, 4 * n_groups))
        gs = gridspec.GridSpec(n_groups, 1, hspace=0.4)
        for group_index, sim_group in enumerate(simulations_lists):
            plot_area = gs[group_index, 0]
            group_title = label[group_index] if label is not None else None
            draw_horizontal_sim_group(
                sim_group,
                planet_like_critical,
                plot_area,
                size_of_earth,
                label_mass_scale,
                CyclicList(color_analogs),
                x_lim=(a_start, a_end),
                analog_criteria=planet_like_critical,
                group_title=group_title,
            )
        plt.savefig(figure_file)
    else:
        draw_horizontal_sim_group(
            simulations_lists,
            planet_like_critical,
            plot_area=SubplotSpec(plt.gcf().add_gridspec(1,1),0,0),
            size_factor=size_of_earth,
            mass_legend=label_mass_scale,
            analog_color=CyclicList(color_analogs),
            x_lim=(a_start, a_end),
            analog_criteria=planet_like_critical,
            group_title=None,
        )
        plt.savefig(figure_file)
   

if __name__ == "__main__":
    main()