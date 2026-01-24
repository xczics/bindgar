from os import path
import numpy as np  # type: ignore


from ..vi import scatter_positions, scatter_ae, scatter_am
from ..datahandle import SimulationOutputData
from ..input import InputLoader,InputAcceptable
from ..cli import register_command
from ..physics import calculate_orbital

from typing import List, Union, Dict, Any

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
        }

    }

    import matplotlib.pyplot as plt
    input_params = InputLoader(DEFAULT_PARAMS).load()
    init_data_file = input_params["init_dat"]
    figure_file = input_params["figure_file"]
    format_str = input_params["format"]
    sim_data = SimulationOutputData(init_data_file, format_str, mode="r", skip_header=False)
    xym = []
    a_array = []
    e_array = []
    with sim_data:
        for data in sim_data:
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
    


