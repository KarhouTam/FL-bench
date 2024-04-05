import importlib
import sys
import inspect
from pathlib import Path

FLBENCH_ROOT = Path(__file__).parent.absolute()
if FLBENCH_ROOT not in sys.path:
    sys.path.append(FLBENCH_ROOT.as_posix())

SERVER_DIR = Path(__file__).parent.joinpath("src/server").absolute()
if SERVER_DIR not in sys.path:
    sys.path.append(SERVER_DIR.as_posix())

from src.utils.tools import parse_args


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise RuntimeError(
            "No method is specified. Run like `python main.py <method> [config_file_relative_path] [cli_method_args ...]`,\n e.g., python main.py fedavg config/template.yml`"
        )

    method_name = sys.argv[1]

    config_file_path = None
    cli_method_args = []
    if len(sys.argv) > 2:
        config_file_path = sys.argv[2]
        cli_method_args = sys.argv[3:]

    try:
        method_module = importlib.import_module(method_name)
    except:
        raise FileNotFoundError(f"unrecongnized method: {method_name}.")

    try:
        get_method_args_func = getattr(method_module, f"get_{method_name}_args")
    except:
        get_method_args_func = None

    module_attributes = inspect.getmembers(method_module, inspect.isclass)
    server_class = [
        attribute
        for attribute in module_attributes
        if attribute[0].lower() == method_name + "server"
    ][0][1]

    server = server_class(
        args=parse_args(
            config_file_path, method_name, get_method_args_func, cli_method_args
        )
    )

    server.run()
