import importlib
import os
import sys
import inspect
from pathlib import Path

import yaml
import pynvml

FLBENCH_ROOT = Path(__file__).parent.absolute()
if FLBENCH_ROOT not in sys.path:
    sys.path.append(FLBENCH_ROOT.as_posix())


from src.utils.tools import parse_args

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise RuntimeError(
            "No method is specified. Run like `python main.py <method> [config_file_relative_path] [cli_method_args ...]`,",
            "e.g., python main.py fedavg config/template.yml`",
        )

    method_name = sys.argv[1]

    config_file_path = None
    cli_method_args = []
    if len(sys.argv) > 2:
        if ".yaml" in sys.argv[2] or ".yml" in sys.argv[2]:  # ***.yml or ***.yaml
            config_file_path = sys.argv[2]
            cli_method_args = sys.argv[3:]
        else:
            cli_method_args = sys.argv[2:]
    try:
        fl_method_server_module = importlib.import_module(f"src.server.{method_name}")
    except:
        raise ImportError(f"Can't import `src.server.{method_name}`.")

    get_method_args_func = getattr(fl_method_server_module, f"get_{method_name}_args", None)

    module_attributes = inspect.getmembers(fl_method_server_module)
    server_class = [
        attribute
        for attribute in module_attributes
        if attribute[0].lower() == method_name + "server"
    ][0][1]

    config_file_args = None
    if config_file_path is not None and os.path.isfile(
        Path(config_file_path).absolute()
    ):
        with open(Path(config_file_path).absolute(), "r") as f:
            try:
                config_file_args = yaml.safe_load(f)
            except:
                raise TypeError(
                    f"Config file's type should be yaml, now is {Path(config_file_path).absolute()}"
                )

    ARGS = parse_args(
        config_file_args, method_name, get_method_args_func, cli_method_args
    )
    if ARGS.mode == "parallel":
        import ray

        num_available_gpus = ARGS.parallel.num_gpus
        num_available_cpus = ARGS.parallel.num_cpus
        if num_available_gpus is None:
            pynvml.nvmlInit()
            num_total_gpus = pynvml.nvmlDeviceGetCount()
            if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
                num_available_gpus = min(
                    len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")), num_total_gpus
                )
            else:
                num_available_gpus = num_total_gpus
        if num_available_cpus is None:
            num_available_cpus = os.cpu_count()
        try:
            ray.init(
                address=ARGS.parallel.ray_cluster_addr,
                namespace=method_name,
                num_cpus=num_available_cpus,
                num_gpus=num_available_gpus,
                ignore_reinit_error=True,
            )
        except ValueError:
            ray.init(namespace=method_name, ignore_reinit_error=True)

        cluster_resources = ray.cluster_resources()
        ARGS.parallel.num_cpus = cluster_resources["CPU"]
        ARGS.parallel.num_gpus = cluster_resources["GPU"]

    server = server_class(args=ARGS)
    server.run()
