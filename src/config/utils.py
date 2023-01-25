from collections import OrderedDict
from typing import List, Optional, OrderedDict, Tuple, Union

import torch
import random
import numpy as np
from path import Path

_PROJECT_DIR = Path(__file__).parent.parent.parent.abspath()
LOG_DIR = _PROJECT_DIR / "logs"
TEMP_DIR = _PROJECT_DIR / "temp"


def fix_random_seed(seed: int) -> None:
    torch.cuda.empty_cache()
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clone_params(
    src: Union[OrderedDict[str, torch.Tensor], torch.nn.Module]
) -> OrderedDict[str, torch.Tensor]:
    if isinstance(src, OrderedDict):
        return OrderedDict(
            {
                name: param.clone().detach().requires_grad_(param.requires_grad)
                for name, param in src.items()
            }
        )
    if isinstance(src, torch.nn.Module):
        return OrderedDict(
            {
                name: param.clone().detach().requires_grad_(param.requires_grad)
                for name, param in src.state_dict(keep_vars=True).items()
            }
        )


def trainable_params(
    src: Union[OrderedDict[str, torch.Tensor], torch.nn.Module], requires_name=False
) -> Tuple[Optional[List[str]], List[torch.Tensor]]:
    parameters = []
    keys = []
    if isinstance(src, OrderedDict):
        for name, param in src.items():
            if param.requires_grad:
                parameters.append(param)
                keys.append(name)
    elif isinstance(src, torch.nn.Module):
        for name, param in src.state_dict(keep_vars=True).items():
            if param.requires_grad:
                parameters.append(param)
                keys.append(name)

    if requires_name:
        return keys, parameters
    else:
        return parameters
