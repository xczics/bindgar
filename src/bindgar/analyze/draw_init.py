from os import path
import numpy as np  # type: ignore


from ..vi import scatter_positions, scatter_ae, scatter_am
from ..datahandle import SimulationOutputData
from ..input import InputLoader,InputAcceptable
from ..cli import register_command
from ..physics import calculate_orbital

from matplotlib.axes import Axes
from matplotlib import pyplot as plt
from typing import List, Union, Dict, Any, Optional, Literal, Tuple

def distribution_line(a_range: Tuple[float, float],
                      a_type: Literal["uniform", "powerlaw", "gaussian"] = "uniform",
                      a_factor: float = -1.5,
                        ) -> Tuple[np.ndarray, np.ndarray,str]:
    """
    Generate a line representing the distribution of the semi-major axis of planetesimals or embryos.
    Args:
        a_range (Tuple[float, float]): Range of semi-major axis (min, max).
        a_type (str): Type of distribution. Options: "uniform", "powerlaw", "gaussian". Default is "uniform".
        a_factor (float): Power-law factor if a_type is "powerlaw". Default is -1.5. It should be negative.
    Returns:
        Tuple[np.ndarray, np.ndarray]: Two numpy arrays representing the x and y coordinates of the distribution line.
    """
    if a_type == "uniform":
        x = np.linspace(a_range[0], a_range[1], 100)
        y = np.ones_like(x)
        text = "Uniform ({a_range[0]:.2f} - {a_range[1]:.2f} au)"
    elif a_type == "powerlaw":
        x = np.linspace(0, 1, 100)
        x = a_range[0] + x * (a_range[1] - a_range[0])
        y = x ** ( a_factor + 1.0 )
        if a_factor != -2.0:
            y = y * ( a_factor + 2.0 ) / ( a_range[1] ** ( a_factor + 2.0 ) - a_range[0] ** ( a_factor + 2.0 ) )
        else:
            y = y / np.log(a_range[1] / a_range[0])
        text = r"$\Sigma \propto r^{" + f"{a_factor}" + r"}$"
    elif a_type == "gaussian":
        sigma = 0.5 * (a_range[1] - a_range[0])
        mean = 0.5 * (a_range[1] + a_range[0])
        x = np.linspace(a_range[0] - 2.5 * sigma, a_range[1] + 2.5 * sigma, 100)
        y = np.exp(-0.5 * ((x - mean) / sigma) ** 2)
        text = r"Gaussian ($\sigma$ = " + f"{sigma:.2f}" + ")"
    else:
        raise ValueError(f"Unknown a_type {a_type}.")
    return x, y, text


def distribution_sketch(
                        ax: Axes,
                        a_range_pl: Tuple[float, float], 
                        a_range_em: Tuple[float, float],
                        a_type_pl: Literal["uniform", "powerlaw", "gaussian"] = "uniform", 
                        a_type_em: Literal["uniform", "powerlaw", "gaussian"] = "uniform",
                        a_factor_pl: float = -1.5,
                        a_factor_em: float = -1.5,
                        pl_em_ratio: float = 9.0,
                        scale_figure: bool = True,):
    """
    Draw a sketch of the distribution of the semi-major axis of planetesimals and embryos. 
        X is the semi-major axis, Y is the number density of planetesimals/embryos. 
        The distribution can be uniform, power-law, or gaussian. 
        The figure will be scaled to fit the ranges if scale_figure is True.
    Args:
        ax (Axes): Matplotlib Axes object to plot on.
        a_range_pl (Tuple[float, float]): Range of semi-major axis for planetesimals (min, max).
        a_range_em (Tuple[float, float]): Range of semi-major axis for embryos (min, max).
        a_type_pl (str): Type of distribution for planetesimals. Options: "uniform", "powerlaw", "gaussian". Default is "uniform".
        a_type_em (str): Type of distribution for embryos. Options: "uniform", "powerlaw", "gaussian". Default is "uniform".
        a_factor_pl (float): Power-law factor for planetesimals if a_type_pl is "powerlaw". Default is -1.5. It should be negative.
        a_factor_em (float): Power-law factor for embryos if a_type_em is "powerlaw". Default is -1.5. It should be negative.
        pl_em_ratio (float): Ratio of the number density of planetesimals to embryos. Default is 10.0.
        scale_figure (bool): Whether to scale the figure to fit the ranges. Default is True.
    """
    if scale_figure:
        max_a = max(a_range_pl[1], a_range_em[1])
        ax.set_xlim(0, max_a*1.2)
    x_pl, y_pl, text_pl = distribution_line(a_range_pl, a_type_pl, a_factor_pl)
    x_em, y_em, text_em = distribution_line(a_range_em, a_type_em, a_factor_em)
    y_pl = y_pl * pl_em_ratio
    ax.plot(x_pl, y_pl, label=f"Planetesimals: {text_pl}", color = "blue")
    # 添加截断虚线
    if a_type_pl == "uniform" or a_type_pl == "powerlaw":
        ax.axvline(a_range_pl[0], color='blue', linestyle='--')
        ax.axvline(a_range_pl[1], color='blue', linestyle='--')
    elif a_type_pl == "gaussian":
        mean_pl = 0.5 * (a_range_pl[1] + a_range_pl[0])
        ax.axvline(a_range_pl[0], color='blue', linestyle='--')
        ax.axvline(a_range_pl[1], color='blue', linestyle='--')
        ax.axvline(mean_pl, color='blue', linestyle=':')
    ax.plot(x_em, y_em, label=f"Embryos: {text_em}",color="orange")
    # 添加截断虚线
    if a_type_em == "uniform" or a_type_em == "powerlaw":
        ax.axvline(a_range_em[0], color='orange', linestyle='--')
        ax.axvline(a_range_em[1], color='orange', linestyle='--')
    elif a_type_em == "gaussian":
        mean_em = 0.5 * (a_range_em[1] + a_range_em[0])
        ax.axvline(a_range_em[0], color='orange', linestyle='--')
        ax.axvline(a_range_em[1], color='orange', linestyle='--')
        ax.axvline(mean_em, color='orange', linestyle=':')
    #put text at proper position
    ax.text(0.05, 0.95, f"Pl: {text_pl}", transform=ax.transAxes, fontsize=8, verticalalignment='top', color='blue')
    ax.text(0.05, 0.85, f"Emb: {text_em}", transform=ax.transAxes, fontsize=8, verticalalignment='top', color='orange')

@register_command("draw-setup-sketch", help_msg="Draw a sketch of the simulation setup.")
def draw_setup_sketch():
    """
    Draw a sketch of the simulation setup based on the initial configuration and data.
    It would be a N * (1 + X) figure, where the first column is a sketch of the distribution of the semi-major axis of planet-esimals and embryos.
    The other columns are real a-e or a-m plots if specified.
    """
    DEFAULT_PARAMS: InputAcceptable = {
        "init_config" : {
            "default": None,
            "help": "List of yaml files used as the input of the `bindgar setup-generate` input.",
            "short": "i",
            "type": list,
        },
        "figure_file" : {
            "default": "setup_sketch.png",
            "help": "output figure file name",
            "short": "o",
            "type": str,
        },
        "draw_ae" : {
            "default": True,
            "help": "Whether to draw the a-e plot.",
            "type": bool,
        },
        "draw_am" : {
            "default": False,
            "help": "Whether to draw the a-m plot.",
            "type": bool,
        },
        "init_data_file" : {
            "default": None,
            "help": "If set either draw_ae or draw_am is True, the initial data file is required to get the data for the plots. It should be the same size as the init_config.",
            "type": list,
            "short": "a",
        },
        "format" : {
            "default": "<< x y z m vx vy vz r >>",
            "help": "format of the init_data_file if draw_ae or draw_am is True.",
            "type": str,
        },
        "x_ranges" : {
            "default": None,
            "help": "List of x_ranges for each row in the sketch-up, a-e, a-m plots. Each item should be a tuple of (min, max). If None, it will be determined automatically.",
            "type": list,
        },
        "setup_names" : {
            "default": None,
            "help": "List of names for each setup to be displayed on the sketch-up plot. If None, no names will be displayed.",
            "type": list,
        },
        "simulation_names" : {
            "default": None,
            "help": "List of names for each simulation to be displayed on the a-e and a-m plots. If None, default name will be generated through the init_data_file names.",
            "type": list,
        },
    }
    from ..setupgener import SETUP_PARAMETERS,handle_default_setups
    input_params = InputLoader(DEFAULT_PARAMS).load()
    init_config = input_params["init_config"]
    figure_file = input_params["figure_file"]
    draw_ae = input_params["draw_ae"]
    draw_am = input_params["draw_am"]
    init_data_file = input_params["init_data_file"]
    data_format = input_params["format"]
    x_ranges = input_params["x_ranges"]
    if (draw_ae or draw_am) and (init_data_file is None or len(init_data_file) != len(init_config)):
        raise ValueError("If either draw_ae or draw_am is True, init_data_file must be provided.")
    if init_config is None:
        raise ValueError("init_config must be provided.")
    # calculate the size of the figure  
    n_rows = len(init_config)
    n_cols = 1 + int(draw_ae) + int(draw_am)
    fig = plt.figure(figsize=(5*n_cols, 3*n_rows))
    gs = fig.add_gridspec(n_rows, n_cols)
    mass_ref = 0.0
    for i, config_file in enumerate(init_config):
        init_param = InputLoader(SETUP_PARAMETERS).load_yaml(config_file)
        N_emb, N_pl, _, m_pl, _, _ = handle_default_setups(init_param)
        if mass_ref == 0.0:
            mass_ref = m_pl
        a_range_pl = tuple(init_param["pl_a_range"])
        a_range_em = tuple(init_param["emb_a_range"])
        a_type_pl = init_param["a_type"]
        a_type_em = init_param["a_type"]
        pl_em_ratio = N_pl / N_emb if N_emb > 0 else 1.0
        ax_sketch = fig.add_subplot(gs[i, 0])
        distribution_sketch(ax_sketch, a_range_pl, a_range_em, a_type_pl, a_type_em, pl_em_ratio=pl_em_ratio)
        x_range = x_ranges[i] if (x_ranges is not None and i < len(x_ranges)) else None
        if x_range is not None:
            ax_sketch.set_xlim(x_range[0], x_range[1])
        # 因为是示意图，sketch-up 不需要纵坐标标签
        ax_sketch.set_yticks([])
        ax_sketch.set_ylabel("Number Density")
        ax_sketch.set_xlabel("Semi-major axis (a.u.)")
        if input_params["setup_names"] is not None and i < len(input_params["setup_names"]):
            ax_sketch.text(0.95, 0.95, input_params["setup_names"][i], transform=ax_sketch.transAxes, fontsize=8, verticalalignment='top', horizontalalignment='right')
        if draw_ae or draw_am:
            init_data = SimulationOutputData(init_data_file[i], mode="r", format_spec=data_format, skip_header=False)
            if input_params["simulation_names"] is not None and i < len(input_params["simulation_names"]):
                sim_name = input_params["simulation_names"][i]
            else:
                sim_name = path.basename(init_data_file[i])
                # remove file extension ".dat" if exists
                if sim_name.endswith(".dat"):
                    sim_name = sim_name[:-4]
                # remove "_in" suffix if exists
                if sim_name.endswith("_in"):
                    sim_name = sim_name[:-3]
            a_array = []
            e_array = []
            m_array = []
            index = 0
            with init_data:
                for data in init_data:
                    index += 1
                    if index > N_emb + N_pl:
                        break
                    a, e, _ = calculate_orbital(
                        data['x'], data["y"], data["z"],
                        data["vx"], data["vy"], data["vz"],
                    )
                    a_array.append(a)
                    e_array.append(e)
                    m_array.append(data["m"])
            a_array = np.array(a_array)
            e_array = np.array(e_array)
            m_array = np.array(m_array)
            m_split = np.median(m_array)
            embryo_color = "orange"
            planetesimal_color = "blue"
            if draw_ae:
                ax_ae = fig.add_subplot(gs[i, 1])
                scatter_ae(ax_ae, a_array, e_array, m_array,
                            m_split=m_split,
                            color_small=planetesimal_color,
                            color_large=embryo_color,
                            ref_mass=mass_ref,
                           )
                if x_range is not None:
                    ax_ae.set_xlim(x_range[0], x_range[1])
                ax_ae.text(0.95, 0.95, sim_name, transform=ax_ae.transAxes, fontsize=8, verticalalignment='top', horizontalalignment='right')
            if draw_am:
                ax_am = fig.add_subplot(gs[i, 1 + int(draw_ae)])
                scatter_am(ax_am, a_array, m_array,
                            m_split=m_split,
                            color_small=planetesimal_color,
                            color_large=embryo_color,
                            ref_mass=mass_ref,  
                           )
                if x_range is not None:
                    ax_am.set_xlim(x_range[0], x_range[1])
                ax_am.text(0.95, 0.95, sim_name, transform=ax_am.transAxes, fontsize=8, verticalalignment='top', horizontalalignment='right')
    plt.tight_layout()
    plt.savefig(figure_file)

@register_command("draw-init")
def main():
    DEFAULT_PARAMS: InputAcceptable = {
        "init_dat" : {
            "default": None,
            "help": "File name of the initial data",
            "short": "i",
            "type": str,
        },
        "figure_file" : {
            "default": "initial_positions.png",
            "help": "output figure file name",
            "short": "o",
            "type": str,
        },
        "format" : {
            "default": "<< x y z m vx vy vz r >>",
            "help": "format of the init_dat file",
            "type": str,
        },
        "skip_gass_giants" : {
            "default": True,
            "help": "Whether to skip gas giants in the plot",
            "type": bool,
        },
        "gass_giants_indexes" : {
            "default": [-1],
            "help": "Indexes of gas giants to skip if skip_gass_giants is True",
            "type": List[int],
        }
    }
    input_params = InputLoader(DEFAULT_PARAMS).load()
    init_data_file = input_params["init_dat"]
    figure_file = input_params["figure_file"]
    format_str = input_params["format"]
    sim_data = SimulationOutputData(init_data_file, format_str, mode="r", skip_header=False)
    xym = []
    a_array = []
    e_array = []
    index = 0
    len_sim_data = len(sim_data)
    skip_gass_giants = input_params["skip_gass_giants"]
    gass_giants_indexes = input_params["gass_giants_indexes"]
    # if < 0, count from the last, e.g., -1 means the last one. Note that the index starts from 1.
    if skip_gass_giants:
        gass_giants_indexes = set([idx if idx > 0 else len_sim_data + idx + 1 for idx in gass_giants_indexes])
    with sim_data:
        for data in sim_data:
            index += 1
            if skip_gass_giants and index in gass_giants_indexes:
                continue
            x = data["x"]
            y = data["y"]
            m = data["m"]
            a, e, i = calculate_orbital(
                data['x'], data["y"], data["z"],
                data["vx"], data["vy"], data["vz"],
            )
            a_array.append(a)
            e_array.append(e)
            xym.append((x, y, m))
    xym = np.array(xym)
    a_array = np.array(a_array)
    e_array = np.array(e_array)
    
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2,2)
    ax_position = fig.add_subplot(gs[:,0])
    ax_ae = fig.add_subplot(gs[0,1])
    ax_am = fig.add_subplot(gs[1,1])
    scatter_positions(ax_position, xym)
    scatter_ae(ax_ae, a_array, e_array,xym[:,2])
    scatter_am(ax_am, a_array, xym[:,2])
    plt.tight_layout()
    plt.savefig(figure_file)
    


