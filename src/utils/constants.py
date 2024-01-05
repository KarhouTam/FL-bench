import json

from .tools import PROJECT_DIR

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


def _get_domain_classes_num():
    try:
        with open(PROJECT_DIR / "data" / "domain" / "metadata.json", "r") as f:
            metadata = json.load(f)
        return metadata["class_num"]
    except:
        return 0


def _get_synthetic_classes_num():
    try:
        with open(PROJECT_DIR / "data" / "synthetic" / "args.json", "r") as f:
            metadata = json.load(f)
        return metadata["class_num"]
    except:
        return 0


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
    "synthetic": _get_synthetic_classes_num(),
    "domain": _get_domain_classes_num(),
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
