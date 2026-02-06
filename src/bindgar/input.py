"""
版权所有 (c) [2025] [Zicong Xiao (Nick name: Zircon)]
Copyright (c) [2025] [Zicong Xiao (Nick name: Zircon)]
GPLv3 许可证。禁止商业用途, 尤其是禁止营销号打包行为。
Licensed under GPLv3. Commercial use is prohibited, especially repackaging by unauthorized marketing accounts on social media.

【开发声明/Development Notice】
本代码在AI辅助下开发 | Code developed with AI assistance
作者免责 | Author not liable for use

"""

import argparse
import yaml
import os
from typing import Dict, Any, Optional, List, Union, Tuple

def deep_update(original: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """递归更新字典"""
    for key, value in updates.items():
        if isinstance(value, dict) and key in original and isinstance(original[key], dict):
            original[key] = deep_update(original[key], value)
        else:
            original[key] = value
    return original

InputAcceptable = Dict[str, Union[Dict[str, Any], 'InputLoader']]

class InputLoader:
    """统一配置加载器（支持嵌套字典定义参数）
    功能：
    - 从命令行/YAML文件加载配置
    - 自动生成短参数和帮助信息
    - 支持类型检查和可选值限制
    """

    def __init__(
        self,
        param_defs: InputAcceptable,
        description: str = "Application Configuration"
    ):
        """
        :param param_defs: 参数定义字典 {
            "参数名": {
                "default": 默认值,           # 必选
                "help": "帮助信息",          # 可选
                "short": "短参数字母",        # 可选（如 "a" 生成 -a）
                "type": 类型,               # 可选（默认自动推断）
                "choices": [可选值列表]       # 可选
                "only_yaml": True/False    # 可选（是否仅从YAML加载，默认False）
                "link": list of str or int   # option, default is empty list. If provided, this parameter will be a alias to a parameter, whose path is given by the list of str.
                "alias": str                # option, default is None. If provided, it will automately generate a "link paramter" in the upper most dict.
            }
        }
        :param description: 命令行描述
        """
        self.param_defs = param_defs
        self._validate_param_defs()
        self.description = description
        self.is_flattened = False

    def load(self) -> Dict[str, Any]:
        """加载配置（优先级：命令行 > YAML > 默认值）"""
        self._add_link_parameters()
        self._flatten_param_defs()
        config_path = self._parse_config_path()
        config = self._load_yaml_config(config_path)
        return self._parse_args_and_merge(config)
    
    def load_yaml(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """仅从YAML加载配置,不解析命令行参数, 并替换默认值"""
        if config_path is None:
            config_path = self._parse_config_path()
        self._add_link_parameters()
        self._flatten_param_defs()
        return self._load_yaml_config(config_path)
    
    def load_by_kwargs(self, kwargs) -> Dict[str, Any]:
        """从给定的kwargs加载配置, 不解析命令行参数, 并替换默认值"""
        self._add_link_parameters()
        self._flatten_param_defs()
        config = {
            name: defs["default"] if not isinstance(defs, InputLoader) else defs.generate_default_dict()
            for name, defs in self.param_defs.items()
        }
        deep_update(config, kwargs)
        return self._update_link_parameters(config)

    def generate_default_dict(self) -> Dict[str, Any] :
        # 生成默认配置字典
        default_config = {}
        for name, defs in self.param_defs.items():
            if isinstance(defs,InputLoader):
                default_config[name] = defs.generate_default_dict()
            else:
                default_config[name] = defs["default"]
        return default_config

    def _flatten_param_defs(self) -> None:
        # 扁平化嵌套的参数定义
        flat_defs = {}
        for name, defs in self.param_defs.items():
            if isinstance(defs, InputLoader):
                default_dict = defs.generate_default_dict()
                flat_defs[name] = {
                    "default": default_dict,
                    "help": defs.description,
                    "only_yaml": True
                } 
            else:
                flat_defs[name] = defs
        self.param_defs = flat_defs
        self.is_flattened = True
    
    def _add_link_parameters(self) -> None:
        # 为别名参数添加链接参数
        alias_params = self.export_alias_parameters()
        for alias_name, link_path, otherinfo in alias_params:
            if len(link_path) <=1:
                continue
            self.param_defs[alias_name] = {
                "default": otherinfo.get("default", None),
                "help": f"Alias parameter for {'.'.join(map(str, link_path))}. {otherinfo.get('help','')}",
                "short": otherinfo.get("short", None),
                "link": link_path
            }

    def export_alias_parameters(self) -> List[Tuple[str, List[Union[str,int]], Dict[str,Any]]]:
        # 导出所有的别名参数
        alias_params = []
        for name, defs in self.param_defs.items():
            if isinstance(defs, InputLoader):
                for exported_alias in defs.export_alias_parameters():
                    alias_name, link_path, otherinfo = exported_alias
                    alias_params.append((alias_name, [name] + link_path, otherinfo))
            else:
                if "alias" in defs:
                    alias_params.append((defs["alias"], [name], {"short": defs.get("short", None), "default": defs.get("default", None), "help": defs.get("help", "")}))
        return alias_params

    def _parse_config_path(self) -> Optional[str]:
        """解析 --config 参数"""
        pre_parser = argparse.ArgumentParser(add_help=False)
        pre_parser.add_argument(
            "-c", "--config",
            help="Path to input YAML file, if not provided, only command line arguments are used",
            metavar="FILE"
        )
        args, _ = pre_parser.parse_known_args()
        return args.config

    def _load_yaml_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """从YAML加载配置"""
        if not self.is_flattened:
            self._flatten_param_defs()
        config = {
            name: defs["default"] if not isinstance(defs, InputLoader) else defs.generate_default_dict()
            for name, defs in self.param_defs.items()
        }
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                yaml_config = yaml.safe_load(f) or {}
            deep_update(config, yaml_config)
        return config

    def _parse_args_and_merge(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """解析命令行参数并合并"""
        if not self.is_flattened:
            self._flatten_param_defs()
        parser = argparse.ArgumentParser(
            description=self.description,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        # 添加 --config 参数
        parser.add_argument("-c", "--config", help="Path to input YAML file, if not provided, only command line arguments are used")
        # 添加特殊参数
        parser.add_argument("-e", "--example", action="store_true", help="print example YAML configuration and exit")
        parser.add_argument("-d", "--document", action="store_true", help="print parameter documentation and exit")

        # 为每个参数添加命令行选项
        for name, defs in self.param_defs.items():
            if isinstance(defs, InputLoader):
                raise ValueError("Nested InputLoader instances should be flattened before parsing arguments.")
            if defs.get("only_yaml", False):
                continue
            self._add_argument(parser, name, defs, config[name])

        # 合并参数
        args = parser.parse_args()

        if args.example:    
            print(self.example_yaml())
            exit(0)
        if args.document:
            print(self.document())
            exit(0)
        deep_update(config,vars(args))
        config = self._update_link_parameters(config)
        return config

    def _update_link_parameters(self,config):
        # check link parameters
        for name, defs in self.param_defs.items():
            if isinstance(defs, InputLoader):
                raise ValueError("Nested InputLoader instances should be flattened before updating link parameters.")
            if "link" in defs:
                link_path = defs["link"]
                if len(link_path) == 0:
                    continue
                if config.get(name) is None:
                    config.pop(name, None)
                    continue
                # change the linked value
                linked_para = config
                for i in range(len(link_path)-1):
                    key = link_path[i]
                    try:
                        linked_para = linked_para[key]
                    except KeyError:
                        raise ValueError(f"Linked parameter '{name}' has an invalid path: {'.'.join(link_path)}")
                linked_para[link_path[-1]] = config[name]
                config.pop(name, None)
        return config
    
    def _add_argument(self, parser: argparse.ArgumentParser, name: str, 
                     defs: Dict[str, Any], default: Any):
        """添加单个命令行参数"""
        arg_names = []
        if "short" in defs and defs["short"] is not None:
            # check short should not conflict with predefined args, "-e", "-d", "-c"
            if defs["short"] in ['e','d','c']:
                raise ValueError(f"Short parameter '-{defs['short']}' for '{name}' conflicts with predefined arguments.")
            arg_names.append(f"-{defs['short']}")
        arg_names.append(f"--{name}")

        kwargs = {
            "default": default,
            "help": defs.get("help", ""),
            "dest": name
        }

        # 类型处理
        param_type = defs.get("type", type(defs["default"]))
        if isinstance(defs["default"], bool):
            kwargs["action"] = "store_false" if defs["default"] else "store_true"
        else:
            kwargs["type"] = param_type

        # 可选值限制
        if "choices" in defs:
            kwargs["choices"] = defs["choices"]
        
        # check if the parameter is a list
        if defs.get("type", None) == list or isinstance(defs["default"], list):
            kwargs["nargs"] = '+'
            if "type" in defs and defs["type"] != list:
                kwargs["type"] = defs["type"]
            elif not isinstance(defs["default"], list) or not defs["default"]:
                # 如果默认值是空列表或无法推断类型，使用字符串类型
                kwargs["type"] = str
            else:
                # 从默认值的第一个元素推断类型
                kwargs["type"] = type(defs["default"][0])

        parser.add_argument(*arg_names, **kwargs)

    def _validate_param_defs(self):
        """验证参数定义"""
        for name, defs in self.param_defs.items():
            if isinstance(defs, InputLoader):
                defs._validate_param_defs()
            elif "default" not in defs:
                raise ValueError(f"参数 '{name}' 必须包含 'default' 字段")
        return self.param_defs
    def document(self):
        """
        生成参数文档, 返回字符串, 用markdown的嵌套列表表示
        """
        doc_lines = []
        for name, defs in self.param_defs.items():
            if isinstance(defs, InputLoader):
                doc_lines.append(f"- **{name}**: {defs.description}")
                nested_doc = defs.document()
                nested_lines = nested_doc.split("\n")
                for line in nested_lines:
                    doc_lines.append(f"  {line}")
            else:
                line = f"- **{name}** (default: `{defs['default']}`"
                if "type" in defs:
                    line += f", type: `{defs['type'].__name__}`"
                if "choices" in defs:
                    line += f", choices: `{defs['choices']}`"
                line += f"): {defs.get('help', '')}"
                doc_lines.append(line)
        return "\n".join(doc_lines)
    def example_yaml(self) -> str:
        """
        生成示例YAML配置字符串
        """
        default_config = self.generate_default_dict()
        return yaml.dump(default_config, sort_keys=False)


# ----------------------------
# 示例代码（可直接运行测试）
# ----------------------------
if __name__ == "__main__":
# 测试嵌套参数定义
    nested_loader = InputLoader(
        param_defs={
            "subparam1": {
                "default": 10,
                "help": "Sub parameter 1",
                "short": "s"
            },
            "subparam2": {
                "default": False,
                "help": "Sub parameter 2",
                "short": "t",
                "alias": "sp2"
            }
        },
        description="Nested Configuration"
    )

    loader = InputLoader(
        param_defs={
            "param1": {
                "default": 42,
                "help": "An integer parameter",
                "short": "p",
                "type": int,
                "alias": "ap1"
            },
            "param2": {
                "default": "default_value",
                "help": "A string parameter",
                "short": "q",
                "type": str,
                "choices": ["default_value", "option1", "option2"]
            },
            "nested": nested_loader,
            "alias_param": {
                "default": None,
                "help": "An alias parameter for param1",
                "short": "a",
                "link": ["param1"]
            }
        },
        description="Application Configuration Loader"
    )
    print(loader.example_yaml())
    config = loader.load()
    print("Loaded Configuration:")
    print(config)