from dataclasses import dataclass
from typing import List, Union, Sequence
import os

@dataclass
class OutFormat:
    name: str
    format: str
    type: str

def pharse_format(format_str: str, split: str = " ") -> List[OutFormat]:
    # an example: "<< x y:.8f z m vx vy vz >>"
    # another example: "<< index a e inc m:.3e >>""
    FLOAT_END_CHARS = ['f', 'e', 'g', 'E', 'G', 'F']
    INT_END_CHARS = ['d', 'i', 'u', 'x', 'X', 'D', 'I', 'U']
    INT_NAMES = ['i','indexi','indexj','index', 'id', 'ID', 'idx']
    STR_NAMES = ['name', 'type', 'label']
    format_list = []
    tokens = format_str.strip().split(split)
    # Remove the leading and trailing '<<' and '>>'
    if tokens[0] == "<<" :
        tokens = tokens[1:]
    if tokens[-1] == ">>" :
        tokens = tokens[:-1]
    for token in tokens:
        if ":" in token:
            name, fmt = token.split(":")
            if fmt.strip()[-1] in FLOAT_END_CHARS:
                type_str = "float"
            elif fmt.strip()[-1] in INT_END_CHARS:
                type_str = "int"
            else:
                type_str = "str"
        else:
            name = token
            if name in INT_NAMES:
                fmt = "d"
                type_str = "int"
            elif name in STR_NAMES:
                fmt = "s"
                type_str = "str"
            else:
                type_str = "float"
                if name in ['x', 'y', 'z']:
                    fmt = ".8f"
                elif name in ['vx', 'vy', 'vz', 'Sx', 'Sy', 'Sz', 'vxi', 'vyi', 'vzi', 'Sxi', 'Syi', 'Szi', 'vxj', 'vyj', 'vzj', 'Sxj', 'Syj', 'Szj']:
                    fmt = ".8g"
                else:
                    fmt = ".8e"
        format_list.append(OutFormat(name=name, format=fmt, type=type_str))
    return format_list

def string2data(format: List[OutFormat], data_str: str, split: str = " ") -> dict:
    tokens = data_str.strip().split(split)
    if len(tokens) != len(format):
        raise ValueError("Number of tokens does not match format specification")
    data_dict = {}
    for i, out_fmt in enumerate(format):
        if out_fmt.type == "float":
            data_dict[out_fmt.name] = float(tokens[i])
        elif out_fmt.type == "int":
            data_dict[out_fmt.name] = int(tokens[i])
        else:
            data_dict[out_fmt.name] = tokens[i]
    return data_dict

def data2string(format: List[OutFormat], data: Union[dict, Sequence], split: str = " ") -> str:
    if isinstance(data, dict):
        values = [data[out_fmt.name] for out_fmt in format]
    else:
        if len(data) != len(format):
            raise ValueError("Number of data items does not match format specification")
        values = data
    return split.join([f"{values[i]:{format[i].format}}" for i in range(len(format))])

class SimulationOutputData:
    # recevice filename, format string or OutFormat list, and mode of "r" "w" or "a". It also provide a iterator to read data line by line, or from end to start.
    def __init__(self, filename: str, format_spec: Union[List[OutFormat], str], mode: str = "r", split: str = " ", skip_header: bool = False, chunk_size: int=8196):
        self.filename = filename
        if mode != "r" and os.path.exists(filename):
            raise ValueError("Currently only read mode 'r' is supported, for safety reasons. Because writing or appending may overwrite existing data. Getting such data is expensive.")
        self.mode = mode
        self.split = split
        self.skip_header = skip_header
        self.chunk_size = chunk_size
        if isinstance(format_spec, str):
            self.format_list = pharse_format(format_spec, split)
        else:
            self.format_list = format_spec
        self.file = open(self.filename, mode)

    def __iter__(self):
        if self.mode != "r":
            raise ValueError("File not opened in read mode")
        # check if skip_header is True, skip the first line
        try:
            if self.skip_header:
                next(self.file)    
            for line in self.file:
                tokens = line.strip().split(self.split)
                if len(tokens) != len(self.format_list):
                    raise ValueError("Number of tokens does not match format specification")
                data_dict = string2data(self.format_list, line, self.split)
                yield data_dict
        finally:
            self.file.close()
    def __reversed__(self):
        return self.loop_from_end()
    def loop_from_end(self):
        if self.mode != "r":
            raise ValueError("File not opened in read mode")
        try:
            self.file.seek(0, 2)
            file_size = self.file.tell()
            buffer = ""
            position = file_size
            # check if it is not a head line when skip_header is True
            while position > 0:
                read_size = min(self.chunk_size, position)
                position -= read_size
                self.file.seek(position)
                chunk = self.file.read(read_size)
                try:
                    chunk.decode('utf-8')  
                except UnicodeDecodeError:
                    # 如果解码失败，调整读取位置避免截断多字节字符
                    position += 1
                    read_size -= 1
                    self.file.seek(position)
                    chunk = self.file.read(read_size)
                buffer = chunk + buffer
                lines = buffer.splitlines()
                if position > 0:
                    buffer = lines[0] if lines else ""
                    lines = lines[1:]
                else:
                    buffer = ""
                for line in reversed(lines):
                    # 检查是否为首行并跳过
                    if self.skip_header and position == 0 and line == lines[-1]:
                        continue
                    if line.strip():  # 跳过空行
                        data_dict = string2data(self.format_list, line, self.split)
                        yield data_dict
        finally:
            self.file.seek(0)  # Reset file pointer
            self.file.close()  # Ensure the file is closed properly
    def get_names(self) -> List[str]:
        return [out_fmt.name for out_fmt in self.format_list]
    
    def write_data(self, data: Union[dict, Sequence]):
        if self.mode not in ["w", "a"]:
            raise ValueError("File not opened in write or append mode")
        line = data2string(self.format_list, data, self.split)
        self.file.write(line + "\n")
    def close(self):
        self.file.close()
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        # I don't know how python handle the __exit__ parameters, I just use the version suggested by copilot
        self.close()
