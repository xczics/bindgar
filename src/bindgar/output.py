from typing import List, Union
import glob
from os import path
from .datahandle import SimulationOutputData, pharse_format, string2data

DEFAULT_Ejction_Format = "<< time index m r x y z vx vy vz Sx Sy Sz case >>"
DEFAULT_Collisions_Format = "<< time indexi mi ri xi yi zi vxi vyi vzi Sxi Syi Szi indexj mj rj xj yj zj vxj vyj vzj Sxj Syj Szj >>"


class SimulationOutput:
    def __init__(self, path: str):
        self.path = path
        self.input_params = {}
    
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
    
    def collisions(self, collision_format: str = DEFAULT_Collisions_Format) -> SimulationOutputData:
        collision_file = path.join(self.path, f"Collisions{self.get_input_params('Output name')}.dat")
        return SimulationOutputData(collision_file, collision_format, mode="r", skip_header=False)
    def filter_final_indexes(self, filter_func: callable) -> List[int]:
        last_output = self.load_last_output()
        final_indexes = []
        for data in last_output:
            if filter_func(data):
                final_indexes.append(data["i"])
        return final_indexes
        