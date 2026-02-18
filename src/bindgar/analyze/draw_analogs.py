from ..input import InputLoader,InputAcceptable
from os import path
import numpy as np  # type: ignore

from ..common import default_colors, CyclicList
from ..physics import AnalogCriteria, calculate_orbital, M_SUN, M_EARTH, M_Mars, M_Moon
from ..output import SimulationOutput
from typing import List, Union, Dict, Any
from ..cli import register_command

import matplotlib.pyplot as plt  # type: ignore
import matplotlib.gridspec as gridspec  # type: ignore
from matplotlib.gridspec import SubplotSpec  # type: ignore
from matplotlib.axes import Axes

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

def draw_analogs(simulations_lists: List[str], color_map: List[str]|CyclicList, 
                  planet_like_critical: List[AnalogCriteria], a_start: float, 
                  a_end: float, m_start: float, m_end: float, color_analogs: List[str],
                  figure_file: str = "analogs_plot.png",
                  auto_color: bool = False,
                  color_params: List[List]|None = None,
                  group_mode: bool = False,
                  labels: List[str]|None = None,):
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    if not group_mode:
        group_list = [[sim] for sim in simulations_lists]
    else:
        group_list = simulations_lists
    for group_index, sim_list in enumerate(group_list):
        a_values = np.array([])
        m_values = np.array([])
        color = None
        label = None
        for sim_index, simulation in enumerate(sim_list):
            SimOutObj = SimulationOutput(simulation)
            final_particles = get_final_particles(SimOutObj)
            if sim_index == 0:
                if auto_color:
                    if color_params is None:
                        color = SimOutObj.magic_color()
                    else:
                        color = SimOutObj.magic_color(properties=color_params[group_index])
                else:
                    color = color_map[group_index]
                if labels is not None:
                    label = labels[group_index] if group_index < len(labels) else None
                else:
                    label = None
                if label is None:
                    label = SimOutObj.get_input_params("Output name")
                a_values, _, _, m_values = partical_orbital_array(final_particles)
            else:
                a_values_append, _, _, m_values_append = partical_orbital_array(final_particles)
                a_values = np.concatenate((a_values, a_values_append))
                m_values = np.concatenate((m_values, m_values_append))
            if sim_index == len(sim_list) - 1:
                ax.scatter(a_values, m_values, color=color, label=label, alpha=0.8, s=20)
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
    ax.set_ylabel(r"Mass ($M_{earth}$)")
    ax.set_title("Final Particles and Planet-like Analogs")
    ax.legend()
    # automatically adjust layout
    plt.tight_layout()
    plt.savefig(figure_file)

def draw_horizontal_planets_diagram(a_values: Union[List[float],np.ndarray],
                                    m_values: Union[List[float],np.ndarray],
                                    size_factor: float,
                                    ax: Axes,
                                    x_lim: tuple = (0.3, 3.0),
                                    analog_criteria: List[AnalogCriteria]|None = None,
                                    analog_color: list|None|CyclicList = None,
                                    vi_x_tick_labels: bool = False,
                                    ):
    """
    For a given planets list, draw a horizontal diagram showing their semi-major axes, and the size of the circles representing their masses.
    """
    a_values = np.array(a_values)
    m_values = np.array(m_values)
    ax.set_xlim(x_lim)
    ax.set_ylim(-1, 1)
    y_zeros = np.zeros_like(a_values)
    if analog_criteria is not None:
        if analog_color is None:
            analog_color = ['gray']
        if isinstance(analog_color,list):
            analog_color = CyclicList(analog_color)
        if not isinstance(analog_color,CyclicList):
            raise ValueError("analog_color should be tranformable to CyclicList type")
        for crit_index, criteria in enumerate(analog_criteria):
            ax.fill_betweenx(
                [-1, 1],
                criteria.a_lower, criteria.a_upper,
                color=analog_color[crit_index],
                alpha=0.3,
                label=f"{criteria.name} - like"
            )
    ax.scatter(a_values, y_zeros, s=np.array(m_values) ** (2/3) * size_factor, color='grey')
    # check if any planets have their masses and a fits the analog criteria, if so, color them differently
    if analog_criteria is not None:
        for crit_index, criteria in enumerate(analog_criteria):
            mask = (a_values >= criteria.a_lower) & (a_values <= criteria.a_upper) & \
                   (m_values >= criteria.m_lower) & (m_values <= criteria.m_upper)
            if np.any(mask):
                ax.scatter(
                    a_values[mask], y_zeros[mask],
                    s=np.array(m_values[mask]) ** (2/3) * size_factor,
                    color=analog_color[crit_index], #type: ignore
                    edgecolors='black',
                    linewidths=0.5,
                    zorder=5,
                )
    if not vi_x_tick_labels:    
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("Semi-major axis (a.u.)")
    ax.set_yticklabels([])
    ax.set_yticks([-1, 0, 1])
    # 添加中心水平线
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.7)
    #设置tick位置在图内
    ax.tick_params(axis='y', which='both', direction='in')
    #设置垂直网格线为analog中心区域，与tick独立
    if analog_criteria is not None:
        for criteria in analog_criteria:
            ax.axvline(
                criteria.a_mean if criteria.a_mean is not None else (criteria.a_lower + criteria.a_upper) / 2,
                color='gray',
                linestyle='--',
                linewidth=0.5,
                alpha=0.7,
            )

def draw_horizontal_sim_group(simulations_list: List[str],
                              planet_like_critical: List[AnalogCriteria],
                              plot_area: SubplotSpec,
                              size_factor: float,
                              mass_legend: List[tuple],
                              analog_color: CyclicList,
                              x_lim: tuple = (0.3, 3.0),
                              analog_criteria: List[AnalogCriteria]|None = None,
                              group_title: str|None = None,
                              legend_fixed_height: float = 0.15,  # 固定图例区域高度比例
                              title_fixed_height: float = 0.1,  # 固定标题区域高度比例
                              ):
    """
    For a given group of simulation lists, draw horizontal diagrams for each group in the given plot areas.
    It deals with:
        1. subdivide the plot area to several sub-axes for each simulation group. The simulations are vertical stacked.
        2. read the final particles from each simulation, and call draw_horizontal_planets_diagram to draw each simulation group.
        3. add legends and labels.
    
    Args:
        simulations_list: 模拟列表
        planet_like_critical: 类行星标准
        plot_area: 子图区域
        size_factor: 大小因子
        mass_legend: 质量图例
        analog_color: 颜色循环列表
        x_lim: x轴限制
        analog_criteria: 类似行星标准
        group_title: 组标题
        group_idx: 当前组索引（用于对齐）
        n_groups: 总组数
        legend_fixed_height: 固定图例区域高度比例
        title_fixed_height: 固定标题区域高度比例
    """
    n_simulations = len(simulations_list)
    
    # 计算高度比例
    # 标题区域 + 模拟区域 + 图例区域
    # 使用固定比例确保不同组对齐
    if group_title is not None:
        title_height = title_fixed_height
    else:
        title_height = 0.0
    
    legend_height = legend_fixed_height
    
    # 计算模拟区域的高度（剩余空间）
    sim_height_ratio = 1.0 - title_height - legend_height
    
    # 创建高度比例列表
    height_ratios = []
    if title_height > 0:
        height_ratios.append(title_height)
    
    # 模拟行平均分配模拟区域高度
    for _ in range(n_simulations):
        height_ratios.append(sim_height_ratio / n_simulations)
    
    if legend_height > 0 and mass_legend is not None:
        height_ratios.append(legend_height)
    
    # 创建网格
    n_rows = len(height_ratios)
    gs = gridspec.GridSpecFromSubplotSpec(
        n_rows, 1, 
        subplot_spec=plot_area, 
        hspace=0,
        height_ratios=height_ratios
    )
    
    # 创建子图
    axes = []
    current_row = 0
    
    # 1. 标题轴（如果存在）
    if title_height > 0 and group_title is not None:
        ax_title = plt.subplot(gs[current_row, 0])
        axes.append(ax_title)
        
        # 设置标题轴
        ax_title.set_xlim(x_lim)
        ax_title.set_ylim(0, 1)
        ax_title.axis('off')
        
        # 添加组标题
        ax_title.text(
            (x_lim[0] + x_lim[1]) / 2,
            0.5,
            group_title,
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=12,
            fontweight='bold',
        )
        
        current_row += 1
    
    # 2. 模拟轴
    sim_axes = []
    for i in range(n_simulations):
        ax = plt.subplot(gs[current_row + i, 0])
        sim_axes.append(ax)
        axes.append(ax)
    
    # 绘制模拟
    for sim_index, simulation in enumerate(simulations_list):
        SimOutObj = SimulationOutput(simulation)
        final_particles = get_final_particles(SimOutObj)
        a_values, _, _, m_values = partical_orbital_array(final_particles)
        ax = sim_axes[sim_index]
        
        # 只在最后一个模拟显示x轴刻度标签
        vi_x_tick_labels = (sim_index == n_simulations - 1)
        
        draw_horizontal_planets_diagram(
            a_values, m_values, size_factor, ax,
            x_lim=x_lim,
            analog_criteria=planet_like_critical,
            analog_color=analog_color,
            vi_x_tick_labels=vi_x_tick_labels,
        )
        
        # 在左边添加模拟名称
        ax.text(x_lim[0] - (x_lim[1] - x_lim[0]) * 0.01, 0.0, 
                SimOutObj.get_input_params("Output name"), 
                verticalalignment='center', 
                horizontalalignment='right',
                fontsize=8)
    
    # 3. 在第一个模拟轴上标注analog区域名称
    if analog_criteria is not None and len(sim_axes) > 0:
        ax_top = sim_axes[0]
        for crit_index, criteria in enumerate(analog_criteria):
            ax_top.text(
                (criteria.a_lower + criteria.a_upper) / 2,
                1.1,  # 略微上移，避免重叠
                f"{criteria.name}",
                horizontalalignment='center',
                verticalalignment='bottom',
                fontsize=9,
            )
    
    # 4. 图例轴（如果存在）
    if legend_height > 0 and mass_legend is not None:
        ax_legend = plt.subplot(gs[-1, 0])
        axes.append(ax_legend)
        
        # 设置图例轴范围
        ax_legend.set_xlim(x_lim)
        ax_legend.set_ylim(-0.5, 0.5)
        
        # 隐藏轴线和刻度
        ax_legend.spines['top'].set_visible(False)
        ax_legend.spines['right'].set_visible(False)
        ax_legend.spines['left'].set_visible(False)
        ax_legend.spines['bottom'].set_visible(False)
        ax_legend.set_xticks([])
        ax_legend.set_yticks([])
        ax_legend.set_xticklabels([])
        ax_legend.set_yticklabels([])
        
        # 设置背景透明
        ax_legend.set_facecolor('none')
        
        # 添加质量图例
        n_legend = len(mass_legend)
        legend_x_positions = np.linspace(x_lim[0], x_lim[1], n_legend + 2)[1:-1]
        
        for legend_index, (mass_value, mass_label) in enumerate(mass_legend):
            # 绘制质量点
            ax_legend.scatter(
                legend_x_positions[legend_index], -0.15,
                s=mass_value ** (2/3) * size_factor,  
                color='blue',
                zorder=5
            )
            
            # 添加标签
            ax_legend.text(
                legend_x_positions[legend_index], -0.3,
                mass_label,
                horizontalalignment='center',
                verticalalignment='top',
                fontsize=8
            )
    
def draw_horizontal_sim_group_old(simulations_list: List[str],
                              planet_like_critical: List[AnalogCriteria],
                              plot_area: SubplotSpec,
                              size_factor: float,
                              mass_legend: List[tuple],
                              analog_color: CyclicList ,
                              x_lim: tuple = (0.3, 3.0),
                              analog_criteria: List[AnalogCriteria]|None = None,
                              group_title: str|None = None,
                            ):
    """
    It is the draft verion of `draw_horizontal_sim_group` before AI edition, kept for reference.
    """
    n_simulations = len(simulations_list)
    gs = gridspec.GridSpecFromSubplotSpec(n_simulations + 1, 1, subplot_spec=plot_area, hspace=0)
    axes = [plt.subplot(gs[i, 0]) for i in range(n_simulations + 1)]
    for sim_index, simulation in enumerate(simulations_list):
        SimOutObj = SimulationOutput(simulation)
        final_particles = get_final_particles(SimOutObj)
        a_values, _, _, m_values = partical_orbital_array(final_particles)
        ax = axes[sim_index]
        vi_x_tick_labels = (sim_index == n_simulations - 1)
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
                f"{criteria.name}",
                #color=analog_color[crit_index] if analog_color is not None else 'gray',
                horizontalalignment='center',
                verticalalignment='bottom',
                fontsize=6,
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
                label=mass_label,
                # set color the same as the points, that is, the default color in matplotlib
                color='blue',
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
    ax_legend.spines['right'].set_visible(False)
    ax_legend.spines['left'].set_visible(False)
    ax_legend.spines['bottom'].set_visible(False)
    ax_legend.set_xticklabels([])
    ax_legend.set_yticklabels([])
    # 不要刻度
    ax_legend.set_xticks([])
    ax_legend.set_yticks([])
    # 设置背景为透明，不要盖住上面的刻度
    ax_legend.set_facecolor('none')
    # 顶部轴添加组标题， 注意，不要盖住analog名称
    if group_title is not None:
        ax_title = axes[0]
        ax_title.text(
            (x_lim[0] + x_lim[1]) / 2,
            2.0,
            group_title,
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=14,
            fontweight='bold',
        )


@register_command("draw-analogs")
def main():
    DEFAULT_PARAMS: InputAcceptable = {
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
        "group_mode" : {
            "default": False,
            "help": "whether to group by simulation groups",
            "type": bool,
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
        "label" : {
            "default": None,
            "help": "label for the plot, if group_mode is True, it will label the group, else it will label each simulation.",
            "type": list,
        },
    }
    input_params = InputLoader(DEFAULT_PARAMS).load()
    simulations_lists = input_params["simulations_lists"]
    group_mode = input_params["group_mode"]
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
    input_params["figure_file"],input_params["auto_color"],
    input_params["color_params"],group_mode=group_mode,
    labels=input_params["label"])

@register_command("final-planets",help_msg="draw final terrestrial planets.")
def final_planets():
    #print("Not implemented yet.")
    #exit()
    DEFAULT_PARAMS:InputAcceptable = {
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
            "default": 0.1,
            "help": "starting semi-major axis for the plot (in a.u.)",
            "type": float,
        },
        "a_end" : {
            "default": 2.5,
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
    label_mass_scale = input_params["label_mass_scale"]
    if label_mass_scale is None:
        label_mass_scale = [(M_Moon / M_EARTH, "Moon"), (M_Mars / M_EARTH, "Mars"), (1.0, "Earth"), (5.0, "5 Earth")]
    size_of_earth = input_params["size_of_earth"]
    figure_file = input_params["figure_file"]
    if input_params["group_mode"]:
        n_groups = len(simulations_lists)
        plt.figure(figsize=(4 * n_groups,10))
        gs = gridspec.GridSpec(1,n_groups, wspace=0.4)
        for group_index, sim_group in enumerate(simulations_lists):
            plot_area = gs[0, group_index]
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