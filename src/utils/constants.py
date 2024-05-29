import json
import os
from pathlib import Path

from torch import optim

FLBENCH_ROOT = Path(__file__).parent.parent.parent.absolute()
OUT_DIR = FLBENCH_ROOT / "out"
TEMP_DIR = FLBENCH_ROOT / "temp"

DEFAULT_COMMON_ARGS = {
    "dataset": "mnist",
    "seed": 42,
    "model": "lenet5",
    "join_ratio": 0.1,
    "global_epoch": 100,
    "local_epoch": 5,
    "finetune_epoch": 0,
    "batch_size": 32,
    "test_interval": 100,
    "straggler_ratio": 0,
    "straggler_min_local_epoch": 0,
    "external_model_params_file": "",
    "buffers": "local",
    "optimizer": {
        "name": "sgd",  # [sgd, adam, adamw, rmsprop, adagrad]
        "lr": 0.01,
        "dampening": 0,  # SGD,
        "weight_decay": 0,
        "momentum": 0,  # SGD, RMSprop,
        "alpha": 0.99,  # RMSprop,
        "nesterov": False,  # SGD,
        "betas": [0.9, 0.999],  # Adam, AdamW,
        "amsgrad": False,  # Adam, AdamW
    },
    "eval_test": True,
    "eval_val": False,
    "eval_train": False,
    "verbose_gap": 10,
    "visible": False,
    "use_cuda": True,
    "save_log": True,
    "save_model": False,
    "save_fig": True,
    "save_metrics": True,
    "check_convergence": True,
}

DEFAULT_PARALLEL_ARGS = {
    "ray_cluster_addr": None,
    "num_gpus": None,
    "num_cpus": None,
    "num_workers": 2,
}

INPUT_CHANNELS = {
    "mnist": 1,
    "medmnistS": 1,
    "medmnistC": 1,
    "medmnistA": 1,
    "covid19": 3,
    "fmnist": 1,
    "emnist": 1,
    "femnist": 1,
    "cifar10": 3,
    "cinic10": 3,
    "svhn": 3,
    "cifar100": 3,
    "celeba": 3,
    "usps": 1,
    "tiny_imagenet": 3,
    "domain": 3,
}


def _get_domainnet_args():
    if os.path.isfile(FLBENCH_ROOT / "data" / "domain" / "metadata.json"):
        with open(FLBENCH_ROOT / "data" / "domain" / "metadata.json", "r") as f:
            metadata = json.load(f)
        return metadata
    else:
        return {}


def _get_synthetic_args():
    if os.path.isfile(FLBENCH_ROOT / "data" / "synthetic" / "args.json"):
        with open(FLBENCH_ROOT / "data" / "synthetic" / "args.json", "r") as f:
            metadata = json.load(f)
        return metadata
    else:
        return {}


# (C, H, W)
DATA_SHAPE = {
    "mnist": (1, 28, 28),
    "medmnistS": (1, 28, 28),
    "medmnistC": (1, 28, 28),
    "medmnistA": (1, 28, 28),
    "fmnist": (1, 28, 28),
    "svhn": (3, 32, 32),
    "emnist": 62,
    "femnist": 62,
    "cifar10": (3, 32, 32),
    "cinic10": (3, 32, 32),
    "cifar100": (3, 32, 32),
    "covid19": (3, 244, 224),
    "usps": (1, 16, 16),
    "celeba": (3, 218, 178),
    "tiny_imagenet": (3, 64, 64),
    "synthetic": _get_synthetic_args().get("dimension", 0),
    "domain": (3, *(_get_domainnet_args().get("image_size", (0, 0)))),
}

NUM_CLASSES = {
    "mnist": 10,
    "medmnistS": 11,
    "medmnistC": 11,
    "medmnistA": 11,
    "fmnist": 10,
    "svhn": 10,
    "emnist": 62,
    "femnist": 62,
    "cifar10": 10,
    "cinic10": 10,
    "cifar100": 100,
    "covid19": 4,
    "usps": 10,
    "celeba": 2,
    "tiny_imagenet": 200,
    "synthetic": _get_synthetic_args().get("class_num", 0),
    "domain": _get_domainnet_args().get("class_num", 0),
}


DATA_MEAN = {
    "mnist": [0.1307],
    "cifar10": [0.4914, 0.4822, 0.4465],
    "cifar100": [0.5071, 0.4865, 0.4409],
    "emnist": [0.1736],
    "fmnist": [0.286],
    "femnist": [0.9637],
    "medmnist": [124.9587],
    "medmnistA": [118.7546],
    "medmnistC": [124.424],
    "covid19": [125.0866, 125.1043, 125.1088],
    "celeba": [128.7247, 108.0617, 97.2517],
    "synthetic": [0.0],
    "svhn": [0.4377, 0.4438, 0.4728],
    "tiny_imagenet": [122.5119, 114.2915, 101.388],
    "cinic10": [0.47889522, 0.47227842, 0.43047404],
    "domain": [0.485, 0.456, 0.406],
}


DATA_STD = {
    "mnist": [0.3015],
    "cifar10": [0.2023, 0.1994, 0.201],
    "cifar100": [0.2009, 0.1984, 0.2023],
    "emnist": [0.3248],
    "fmnist": [0.3205],
    "femnist": [0.155],
    "medmnist": [57.5856],
    "medmnistA": [62.3489],
    "medmnistC": [58.8092],
    "covid19": [56.6888, 56.6933, 56.6979],
    "celeba": [67.6496, 62.2519, 61.163],
    "synthetic": [1.0],
    "svhn": [0.1201, 0.1231, 0.1052],
    "tiny_imagenet": [58.7048, 57.7551, 57.6717],
    "cinic10": [0.24205776, 0.23828046, 0.25874835],
    "domain": [0.229, 0.224, 0.225],
}

OPTIMIZERS = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "rmsprop": optim.RMSprop,
    "adagrad": optim.Adagrad,
}

LR_SCHEDULERS = {
    "step": optim.lr_scheduler.StepLR,
    "cosine": optim.lr_scheduler.CosineAnnealingLR,
    "constant": optim.lr_scheduler.ConstantLR,
    "plateau": optim.lr_scheduler.ReduceLROnPlateau,
}
