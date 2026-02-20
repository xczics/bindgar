"""
This module provides functionality for handling simulation output data, including loading, 
filtering, and analyzing simulation results. It includes a `SimulationOutput` class that 
provides methods to retrieve simulation parameters, load output data, calculate total mass 
and particle counts, and generate color mappings based on simulation properties.
Classes:
    - SimulationOutput: A class to manage and process simulation output data.
Constants:
    - DEFAULT_Ejction_Format: Default format for ejection data.
    - DEFAULT_Collisions_Format: Default format for collision data.
Functions:
    - SimulationOutput.__init__(path: str):
        Initializes the SimulationOutput object with the given simulation directory path.
    - SimulationOutput.get_input_params(params: Union[str, List[str]]) -> Union[str, List[str]]:
        Retrieves input parameters from the simulation's `param.dat` file.
    - SimulationOutput.load_last_output() -> SimulationOutputData:
        Loads the last output file from the simulation directory, considering error files if present.
    - SimulationOutput.get_init_data() -> SimulationOutputData:
        Retrieves the initial simulation data from the input file.
    - SimulationOutput.collisions(collision_format: str = DEFAULT_Collisions_Format) -> SimulationOutputData:
        Loads collision data from the simulation directory.
    - SimulationOutput.filter_final_indexes(filter_func: callable) -> List[int]:
        Filters and returns the indexes of particles in the final output based on a given filter function.
    - SimulationOutput._get_total_mass(data: SimulationOutputData, skip_indexes: List[int] = None) -> float:
        Calculates the total mass of particles in the given data, optionally skipping specified indexes.
    - SimulationOutput._get_num_particles(data: SimulationOutputData, skip_indexes: List[int] = None) -> int:
        Counts the number of particles in the given data, optionally skipping specified indexes.
    - SimulationOutput.get_init_total_mass(skip_indexes: List[int] = None) -> float:
        Retrieves the total initial mass of particles, with optional caching.
    - SimulationOutput.get_final_total_mass(skip_indexes: List[int] = None) -> float:
        Retrieves the total final mass of particles, with optional caching.
    - SimulationOutput.get_init_num_particles(skip_indexes: List[int] = None) -> int:
        Retrieves the initial number of particles, with optional caching.
    - SimulationOutput.get_final_num_particles(skip_indexes: List[int] = None) -> int:
        Retrieves the final number of particles, with optional caching.
    - SimulationOutput.magic_color(properties: List = None, value_range: List = None) -> str:
        Generates a color string based on simulation properties, using the LAB color space.
Dependencies:
    - typing: For type annotations.
    - glob: For file pattern matching.
    - os.path: For file path manipulations.
    - functools: For caching functions.
    - math: For mathematical operations.
    - .datahandle: For handling simulation data.
    - .physics: For physical constants.
    - .common: For color conversion utilities.
"""
from typing import List, Union, Any, Callable, Optional, overload, Dict
import glob
from os import path
from .datahandle import SimulationOutputData, pharse_format, string2data
from .physics import M_EARTH, M_SUN
from functools import lru_cache, cached_property
from .common import statstic_time
import math

DEFAULT_Ejction_Format = "<< time index m r x y z vx vy vz Sx Sy Sz case >>"
DEFAULT_Collisions_Format = "<< time indexi mi ri xi yi zi vxi vyi vzi Sxi Syi Szi indexj mj rj xj yj zj vxj vyj vzj Sxj Syj Szj >>"


class SimulationOutput:
    def __init__(self, path: str,gass_giants_indexes: List[int]|None = None):
        self.path = path
        self.input_params = {}
        if gass_giants_indexes is not None:
            self.set_gass_giant_indexes(gass_giants_indexes)
        self._history_built = False
    
    @overload
    def get_input_params(self, params: str) -> str: ...
    @overload
    def get_input_params(self, params: List[str]) -> List[str]: ...
    
    def get_input_params(self, params: Union[str,List[str]]) -> Union[str, List[str]]:
        required_params = [params] if isinstance(params, str) else params
        lack_params = [p for p in required_params if p not in self.input_params]
        if len(lack_params) > 0:
            param_file = path.join(self.path, "param.dat")
            with open(param_file, "r") as f:
                for line in f:
                    if "Input file Format" in line or "Output file Format" in line:
                        # I don't know why genga use ":" for these two parameters, so special handle them
                        tokens = line.strip().split(":")
                    else:
                        tokens = line.strip().split("=")
                    if len(tokens) == 2:
                        key = tokens[0].strip()
                        value = tokens[1].strip()
                        if key in lack_params:
                            self.input_params[key] = value
                            lack_params.remove(key)
                    if len(lack_params) == 0:
                        break
            lack_params = [p for p in required_params if p not in self.input_params]
            if len(lack_params) > 0:
                raise ValueError(f"Parameters {lack_params} not found in param.dat")
        if isinstance(params, str):
            return self.input_params[params]
        else:
            return [self.input_params[p] for p in params]
    
    @statstic_time
    def load_last_output(self) -> SimulationOutputData:
        # Output file is like Out{Output name}_xxxx.dat, we need to find the last xxxx, xxxx is the steps
        output_name = self.get_input_params("Output name")
        output_files = glob.glob(path.join(self.path, f"Out{output_name}_*.dat"))
        steps = [int(f.split("_")[-1].split(".")[0]) for f in output_files]
        max_step_index = steps.index(max(steps))        
        output_file = output_files[max_step_index]
        # check if "OutError.dat" is also exists
        error_file = path.join(self.path, "OutError.dat")
        if not path.exists(error_file):
            return SimulationOutputData(output_file, self.get_input_params("Output file Format"), mode="r", skip_header=False)
        else:
            # need to compare the time in both file, and return the one with larger time
            output_format = self.get_input_params("Output file Format")
            last_output = SimulationOutputData(output_file, output_format, mode="r", skip_header=False)
            error_output = SimulationOutputData(error_file, output_format, mode="r", skip_header=False)
            with last_output, error_output:
                last_time = next(iter(last_output))["t"]
                error_time = next(iter(error_output))["t"]
            if error_time > last_time:
                return SimulationOutputData(error_file, output_format, mode="r", skip_header=False)
            else:
                return SimulationOutputData(output_file, output_format, mode="r", skip_header=False)
    
    def get_init_data(self) -> SimulationOutputData:
        init_file = self.get_input_params("Input file")
        init_file =  path.join(self.path, init_file)
        init_format = self.get_input_params("Input file Format")
        return SimulationOutputData(init_file, init_format, mode="r", skip_header=False)
    
    @cached_property
    def collisions(self, collision_format: str = DEFAULT_Collisions_Format) -> SimulationOutputData:
        collision_file = path.join(self.path, f"Collisions{self.get_input_params('Output name')}.dat")
        return SimulationOutputData(collision_file, collision_format, mode="r", skip_header=False)
    
    @cached_property
    def ejections(self, ejection_format: str = DEFAULT_Ejction_Format) -> SimulationOutputData:
        ejection_file = path.join(self.path, f"Ejections{self.get_input_params('Output name')}.dat")
        return SimulationOutputData(ejection_file, ejection_format, mode="r", skip_header=False)
    
    def filter_final_indexes(self, filter_func: Callable) -> List[int]:
        last_output = self.load_last_output()
        final_indexes = []
        for data in last_output:
            if filter_func(data):
                final_indexes.append(data["i"])
        return final_indexes
    
    def _get_total_mass(self, data: SimulationOutputData, skip_indexes: List[int]|None = None) -> float:
        if skip_indexes is None:
            skip_indexes = [-1]
        total_mass = 0.0
        if any(value < 0 for value in skip_indexes):
            len_data = len(data)
            skip_indexes = [i if i >= 0 else len_data + i for i in skip_indexes]
        idx = 0
        for item in data:
            pidx = item["i"] if "i" in item else idx
            if pidx not in skip_indexes:
                total_mass += item["m"]
            idx += 1
        return total_mass
    
    def _get_num_particles(self, data: SimulationOutputData, skip_indexes: List[int]|None = None) -> int:
        if skip_indexes is None:
            skip_indexes = [-1]
        total_num = 0
        if any(value < 0 for value in skip_indexes):
            len_data = len(data)
            skip_indexes = [i if i >= 0 else len_data + i for i in skip_indexes]
        idx = 0
        for item in data:
            pidx = item["i"] if "i" in item else idx
            if pidx not in skip_indexes:
                total_num += 1
            idx += 1
        return total_num 
    
    def open_write_pip(self, file_name: str, fmt: str, **kwargs) -> SimulationOutputData:
        file_path = path.join(self.path, file_name)
        if "mode" not in kwargs:
            kwargs["mode"] = "w"
        return SimulationOutputData(file_path, format_spec=fmt, **kwargs)
    
    def set_gass_giant_indexes(self, indexes: List[int]) -> None:
        self.gass_giant_indexes = indexes
    
    @statstic_time
    def built_history(self) -> None:
        """
        read collisions and ejections, build the history of each particles.
        """
        self._cached_history = {}
        collisions, ejections = self.collisions, self.ejections
        with collisions, ejections:
            ci=0
            ei=0
            for collision in collisions:
                for idx in [collision["indexi"], collision["indexj"]]:
                    if idx not in self._cached_history:
                        self._cached_history[idx] = []
                    self._cached_history[idx].append((0,collision,ci))
                ci+=1
            for ejection in ejections:
                idx = ejection["index"]
                if idx not in self._cached_history:
                    self._cached_history[idx] = []
                self._cached_history[idx].append((1,ejection,ei))
                ei+=1
        self._history_built = True
    
    def _history_from_cache(self, particle_index: int) -> Dict[str, Any]:
        if particle_index not in self._cached_history:
            return {"terminal_type": "survival", "terminal_event": None, "history": []}
        history = []
        terminal_type = None
        terminal_event = None
        last_event = self._cached_history[particle_index][-1]
        if last_event[0] == 0:
            terminal_type = "collision"
            terminal_event = (last_event[1], last_event[2])
            if particle_index in self._survivals:
                terminal_type = "survival"
                terminal_event = None
                history = [(event[1], event[2]) for event in self._cached_history[particle_index]]
            else:
                history = [(event[1], event[2]) for event in self._cached_history[particle_index][:-1]]
        elif last_event[0] == 1:
            terminal_type = "ejection"
            terminal_event = (last_event[1], last_event[2])
            history = [(event[1], event[2]) for event in self._cached_history[particle_index][:-1]]
        return {"terminal_type": terminal_type,"terminal_event":terminal_event,"history": history}

    @lru_cache(maxsize=None)
    @statstic_time
    def particle_history(self, particle_index: int) -> Dict[str, Any]:
        if self._history_built:
            return self._history_from_cache(particle_index)
        history = []
        terminal_type = None
        terminal_event = None
        collisions = self.collisions
        ejections = self.ejections
        ci = 0
        ei = 0
        with collisions, ejections:
            for collision in collisions:
                if collision["indexi"] == particle_index or collision["indexj"] == particle_index:
                    history.append((collision, ci))
                ci+=1
            for ejection in ejections:
                if ejection["index"] == particle_index:
                    terminal_type = "ejection"
                    terminal_event = (ejection, ei)
                    break
                ei+=1
            else:
                # check if the particle is still in the final output
                if particle_index in self._survivals:
                    terminal_type = "survival"
                    terminal_event = None
                else:
                    terminal_type = "collision"
                    terminal_event = history[-1] if len(history) > 0 else None
                    history.pop()  # remove the last collision, because it is not the terminal event
        return {"terminal_type": terminal_type,"terminal_event":terminal_event,"history": history}
    
    @cached_property
    @statstic_time
    def _survivals(self) -> set:
        last_output = self.load_last_output()
        survivals = set()
        with last_output:
            for item in last_output:
                survivals.add(item["i"])
        return survivals
    
    def survivals(self) -> List[int]:
        return list(self._survivals)
    
    @property
    def survivals_without_gas_giants_sets(self) -> set:
        if hasattr(self, "gass_giant_indexes"):
            return self._survivals - set(self.gass_giant_indexes)
        else:
            raise ValueError("Gas giant indexes not set. Please call set_gass_giant_indexes() first.")
    @property
    def survivals_without_gas_giants(self) -> List[int]:
        return list(self.survivals_without_gas_giants_sets)
    
    @lru_cache(maxsize=2)
    def get_init_total_mass(self, skip_gass_gaint: bool = False) -> float:
        if not skip_gass_gaint:
            return self._get_total_mass(self.get_init_data())
        elif hasattr(self, "gass_giant_indexes"):
            return self._get_total_mass(self.get_init_data(), self.gass_giant_indexes)
        else:
            raise ValueError("Gas giant indexes not set. Please call set_gass_giant_indexes() first.")
    @lru_cache(maxsize=2)
    def get_final_total_mass(self, skip_gass_gaint: bool = False) -> float:
        if not skip_gass_gaint:
            return self._get_total_mass(self.load_last_output())
        elif hasattr(self, "gass_giant_indexes"):
            return self._get_total_mass(self.load_last_output(), self.gass_giant_indexes)
        else:
            raise ValueError("Gas giant indexes not set. Please call set_gass_giant_indexes() first.")
    @lru_cache(maxsize=2)
    def get_init_num_particles(self, skip_gass_gaint: bool = False) -> int:
        if not skip_gass_gaint:
            return self._get_num_particles(self.get_init_data())
        elif hasattr(self, "gass_giant_indexes"):
            return self._get_num_particles(self.get_init_data(), self.gass_giant_indexes)
        else:
            raise ValueError("Gas giant indexes not set. Please call set_gass_giant_indexes() first.")
    @lru_cache(maxsize=2)
    def get_final_num_particles(self, skip_gass_gaint: bool = False) -> int:
        if not skip_gass_gaint:
            return self._get_num_particles(self.load_last_output())
        elif hasattr(self, "gass_giant_indexes"):
            return self._get_num_particles(self.load_last_output(), self.gass_giant_indexes)
        else:
            raise ValueError("Gas giant indexes not set. Please call set_gass_giant_indexes() first.")

    def magic_color(self, properties: List|None = None, value_range: List|None = None) -> str:
        """
        It is a magic function, it can auto-select color based on the properties of the simulation.
        The function select color in an LAB color space, and set color with L, r and theta parameters,
        each property will match a parameter.
        Args:
            properties (List): List of properties to determine color.
                It should be a three element list, each should be a name of property, or a fixed value.
            value_range (List): List, set the range of the property, and the color parameter.
        Return:
            A string, can be used in matplotlib.
        """
        from .common import lab_to_rgb, rgb_to_hex
        supported_properties = ["total_init_mass","num_particles","lg_num_particles"]
        default_properties_range = {
            "total_init_mass" : [M_EARTH/M_SUN * 0.1, M_EARTH/M_SUN * 10],
            "num_particles" : [50, 4000],
            "lg_num_particles" : [2, 3],
        }
        get_prop_funcs = {
            "total_init_mass" : self.get_init_total_mass,
            "num_particles" : self.get_init_num_particles,
            "lg_num_particles" : lambda: math.log10(self.get_init_num_particles()),
        }
        default_color_range = [
            [20, 100],
            [1, 128],
            [0, 360]
        ]

        if properties == None:
            properties = ["lg_num_particles", "total_init_mass", 90]
        for prop in properties:
            if type(prop) is str and prop not in supported_properties:
                raise ValueError(f"Unsupported property: {prop}. Supported properties are: {supported_properties}")
        
        if value_range is None:
            value_range = [
                [
                    default_properties_range[prop][0],  # 属性最小值
                    default_properties_range[prop][1],  # 属性最大值
                    default_color_range[i][0],          # 颜色参数最小值
                    default_color_range[i][1]           # 颜色参数最大值
                ] if isinstance(prop, str) else None
                for i, prop in enumerate(properties)
            ]
        
        # 计算三个颜色参数
        color_params = []
        for i, prop in enumerate(properties):
            if isinstance(prop, str):
                # 获取属性值
                prop_func = get_prop_funcs[prop]
                prop_value = prop_func()
                
                # 从value_range中获取映射范围
                prop_min, prop_max, color_min, color_max = value_range[i]
                
                # 线性映射属性值到颜色参数
                if prop_max == prop_min:
                    # 避免除零错误
                    color_param = (color_min + color_max) / 2
                else:
                    # 归一化属性值
                    normalized = (prop_value - prop_min) / (prop_max - prop_min)
                    # 映射到颜色参数范围
                    color_param = color_min + normalized * (color_max - color_min)
            else:
                # 直接使用固定值
                color_param = prop
            
            color_params.append(color_param)

        L, r, theta = color_params
        theta = theta % 360
        
        theta_rad = math.radians(theta)
        a = r * math.cos(theta_rad)
        b = r * math.sin(theta_rad)

        R, B, G = lab_to_rgb(L, a, b)
        return rgb_to_hex(R, G, B)
