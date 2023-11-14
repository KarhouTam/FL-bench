import importlib
import sys
import inspect
from pathlib import Path

sys.path.append(Path(__file__).parent.joinpath("src/server").absolute().as_posix())

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError(
            "Need to assign a method. Run like `python main.py <method> [args ...]`, e.g., python main.py fedavg -d cifar10 -m lenet5`"
        )

    method = sys.argv[1]
    args_list = sys.argv[2:]

    try:
        module = importlib.import_module(method)
        argparser_function = getattr(module, f"get_{method}_argparser")
        parser = argparser_function()
        module_attributes = inspect.getmembers(module, inspect.isclass)
        server_class = [
            attribute
            for attribute in module_attributes
            if attribute[0].lower() == method + "server"
        ][0][1]
    except:
        raise ValueError(f"Cannot find method: {method} or its argparser funciton.")

    try:
        server = server_class(args=parser.parse_args(args_list))
    except:
        raise ValueError(f"Undefined argument in the list: {args_list}")

    server.run()
