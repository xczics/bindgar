import sys
import os
import inspect
import importlib
import pkgutil

class SubCommandScriptBinder:
    def __init__(self):
        self.binding_dict = {}
        self._subparsers_added = False
    
    def bind(self, command_name: str, script_name: str, running_obj: callable, help_msg: str = ""):
        if help_msg == "":
            help_msg = f"Equals to run the script `python {script_name}`"
        self.binding_dict[command_name] = [script_name, running_obj, help_msg]

    
    def run(self, command_name: str):
        # 修改sys.argv，移除命令名参数
        if len(sys.argv) > 1 and sys.argv[1] == command_name:
            sys.argv.pop(1)
        
        _, running_obj, _ = self.binding_dict[command_name]
        # 修改程序名，使其看起来像是直接运行脚本
        sys.argv[0] = f"bindgar-{command_name}"
        running_obj()
    
    def check_or_help(self):
        if len(sys.argv) < 2:
            self._print_help_with_commands()
            sys.exit(1)
        
        command_name = sys.argv[1]
        if command_name not in self.binding_dict:
            if command_name in ['-h', '--help','help']:
                self._print_help_with_commands()
                sys.exit(0)
            print(f"Unknown command: {command_name}\n")
            self._print_help_with_commands()
            sys.exit(1)
        
        return command_name
    def _print_help_with_commands(self):
        """自定义帮助信息，包含命令列表"""    
        print("Bindgar is Interface for N-body Data, especially for Genga And Rebound.")
        print("Usage: bindgar <command> [options]\n")
        if self.binding_dict:
            print("Available commands:")
            max_cmd_len = max(len(cmd) for cmd in self.binding_dict.keys())
            for cmd, (script_name, _, help_msg) in sorted(self.binding_dict.items()):
                print(f"  {cmd:<{max_cmd_len}}  {help_msg}")
            print(f"\nUse 'bindgar <command> --help' for command-specific help")

# 全局注册表实例
_registry = None

def get_registry():
    """获取全局注册表实例"""
    global _registry
    if _registry is None:
        _registry = SubCommandScriptBinder()
    return _registry

def _get_caller_filename():
    """获取调用者的文件名（不带路径和扩展名）"""
    # 获取调用栈，找到装饰器被调用的位置
    
    stack = inspect.stack()
    for frame_info in stack:
        # 跳过装饰器自身的帧和内部函数帧
        if frame_info.function in ['register_command', '_get_caller_filename', 'decorator']:
            continue
        
        # 检查文件名，跳过当前文件（cli.py）
        caller_file = frame_info.filename
        current_file = os.path.abspath(__file__)
        
        # 如果找到的不是当前文件，说明是外部调用者
        if os.path.abspath(caller_file) != current_file:
            # 提取文件名（不含路径和扩展名）
            filename = os.path.splitext(os.path.basename(caller_file))[0]
            #print(f"Debug: Caller filename determined as {filename}")
            return filename
    return "unknown_script"

def register_command(command_name=None, script_name=None, help_msg=""):
    """
    注册子命令的装饰器
    
    Args:
        command_name: 子命令名称（可选，默认为文件名）
        script_name: 脚本名称（可选，默认为文件名）
    """
    def decorator(func):
        # 自动获取文件名
        caller_filename = _get_caller_filename()
        
        # 设置命令名和脚本名（使用条件判断而不是or）
        actual_command_name = command_name if command_name is not None else caller_filename.replace('_', '-')
        actual_script_name = script_name if script_name is not None else f"{caller_filename}.py"

        # 获取注册表并绑定命令
        registry = get_registry()
        if help_msg == "":
            registry.bind(actual_command_name, actual_script_name, func)
        else:
            registry.bind(actual_command_name, actual_script_name, func, help_msg)
        
        return func
    return decorator

def _dynamic_import_analyze_modules():
    """动态导入.analyze包中的所有模块"""
    try:
        # 导入analyze包
        analyze_package = importlib.import_module('.analyze', package=__package__)
        
        # 获取analyze包中的所有模块
        package_path = analyze_package.__path__
        
        # 遍历包中的所有模块
        for importer, modname, ispkg in pkgutil.iter_modules(package_path):
            if not ispkg:  # 只导入模块，不导入子包
                try:
                    # 动态导入模块
                    full_module_name = f".analyze.{modname}"
                    importlib.import_module(full_module_name, package=__package__)
                    #print(f"Debug: Successfully imported {full_module_name}")
                except ImportError as e:
                    print(f"Warning: Failed to import {modname}: {e}")
                except Exception as e:
                    print(f"Warning: Error importing {modname}: {e}")
                    
    except ImportError as e:
        print(f"Warning: analyze package not found: {e}")
    except Exception as e:
        print(f"Warning: Error loading analyze modules: {e}")

def main():
    registry = get_registry()
    # All sub-commands are registered in .analyze modules
    # dynamically import them to ensure registration, not manually import
    _dynamic_import_analyze_modules()
    from . import setupgener
    command_name = registry.check_or_help()
    registry.run(command_name)

if __name__ == "__main__":
    main()