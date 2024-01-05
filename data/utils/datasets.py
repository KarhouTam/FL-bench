import json
import os
import pickle
from argparse import Namespace
from pathlib import Path
from typing import List

import torch
import numpy as np
import torchvision
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import pil_to_tensor
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self) -> None:
        self.classes: List = None
        self.data: torch.Tensor = None
        self.targets: torch.Tensor = None
        self.train_data_transform = None
        self.train_target_transform = None
        self.general_data_transform = None
        self.general_target_transform = None
        self.enable_train_transform = True

    def __getitem__(self, index):
        data, targets = self.data[index], self.targets[index]
        if self.enable_train_transform and self.train_data_transform is not None:
            data = self.train_data_transform(data)
        if self.enable_train_transform and self.train_target_transform is not None:
            targets = self.train_target_transform(targets)
        if self.general_data_transform is not None:
            data = self.general_data_transform(data)
        if self.general_target_transform is not None:
            targets = self.general_target_transform(targets)
        return data, targets

    def __len__(self):
        return len(self.targets)


class FEMNIST(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ) -> None:
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        if not os.path.isfile(root / "data.npy") or not os.path.isfile(
            root / "targets.npy"
        ):
            raise RuntimeError(
                "run data/utils/run.py -d femnist for generating the data.npy and targets.npy first."
            )

        data = np.load(root / "data.npy")
        targets = np.load(root / "targets.npy")

        self.data = torch.from_numpy(data).float().reshape(-1, 1, 28, 28)
        self.targets = torch.from_numpy(targets).long()
        self.classes = list(range(62))
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform


class Synthetic(BaseDataset):
    def __init__(self, root, *args, **kwargs) -> None:
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        if not os.path.isfile(root / "data.npy") or not os.path.isfile(
            root / "targets.npy"
        ):
            raise RuntimeError(
                "run data/utils/run.py -d femnist for generating the data.npy and targets.npy first."
            )

        data = np.load(root / "data.npy")
        targets = np.load(root / "targets.npy")

        self.data = torch.from_numpy(data).float()
        self.targets = torch.from_numpy(targets).long()
        self.classes = list(range(len(self.targets.unique())))


class CelebA(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ) -> None:
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        if not os.path.isfile(root / "data.npy") or not os.path.isfile(
            root / "targets.npy"
        ):
            raise RuntimeError(
                "run data/utils/run.py -d femnist for generating the data.npy and targets.npy first."
            )

        data = np.load(root / "data.npy")
        targets = np.load(root / "targets.npy")

        self.data = torch.from_numpy(data).permute([0, -1, 1, 2]).float()
        self.targets = torch.from_numpy(targets).long()
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform
        self.classes = [0, 1]


class MedMNIST(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        self.classes = list(range(11))
        self.data = (
            torch.Tensor(np.load(root / "raw" / "xdata.npy")).float().unsqueeze(1)
        )
        self.targets = (
            torch.Tensor(np.load(root / "raw" / "ydata.npy")).long().squeeze()
        )
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform


class COVID19(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        self.data = (
            torch.Tensor(np.load(root / "raw" / "xdata.npy"))
            .permute([0, -1, 1, 2])
            .float()
        )
        self.targets = (
            torch.Tensor(np.load(root / "raw" / "ydata.npy")).long().squeeze()
        )
        self.classes = [0, 1, 2, 3]
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform


class USPS(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        train_part = torchvision.datasets.USPS(root / "raw", True, download=True)
        test_part = torchvision.datasets.USPS(root / "raw", False, download=True)
        train_data = torch.Tensor(train_part.data).float().unsqueeze(1)
        test_data = torch.Tensor(test_part.data).float().unsqueeze(1)
        train_targets = torch.Tensor(train_part.targets).long()
        test_targets = torch.Tensor(test_part.targets).long()

        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = list(range(10))
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform


class SVHN(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        train_part = torchvision.datasets.SVHN(root / "raw", "train", download=True)
        test_part = torchvision.datasets.SVHN(root / "raw", "test", download=True)
        train_data = torch.Tensor(train_part.data).float()
        test_data = torch.Tensor(test_part.data).float()
        train_targets = torch.Tensor(train_part.labels).long()
        test_targets = torch.Tensor(test_part.labels).long()

        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = list(range(10))
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform


class MNIST(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        super().__init__()
        train_part = torchvision.datasets.MNIST(root, True, download=True)
        test_part = torchvision.datasets.MNIST(root, False)
        train_data = torch.Tensor(train_part.data).float().unsqueeze(1)
        test_data = torch.Tensor(test_part.data).float().unsqueeze(1)
        train_targets = torch.Tensor(train_part.targets).long().squeeze()
        test_targets = torch.Tensor(test_part.targets).long().squeeze()
        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = train_part.classes
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform


class FashionMNIST(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        super().__init__()
        train_part = torchvision.datasets.FashionMNIST(root, True, download=True)
        test_part = torchvision.datasets.FashionMNIST(root, False, download=True)
        train_data = torch.Tensor(train_part.data).float().unsqueeze(1)
        test_data = torch.Tensor(test_part.data).float().unsqueeze(1)
        train_targets = torch.Tensor(train_part.targets).long().squeeze()
        test_targets = torch.Tensor(test_part.targets).long().squeeze()
        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = train_part.classes
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform


class EMNIST(BaseDataset):
    def __init__(
        self,
        root,
        args,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        super().__init__()
        split = None
        if isinstance(args, Namespace):
            split = args.emnist_split
        elif isinstance(args, dict):
            split = args["emnist_split"]
        train_part = torchvision.datasets.EMNIST(
            root, split=split, train=True, download=True
        )
        test_part = torchvision.datasets.EMNIST(
            root, split=split, train=False, download=True
        )
        train_data = torch.Tensor(train_part.data).float().unsqueeze(1)
        test_data = torch.Tensor(test_part.data).float().unsqueeze(1)
        train_targets = torch.Tensor(train_part.targets).long().squeeze()
        test_targets = torch.Tensor(test_part.targets).long().squeeze()
        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = train_part.classes
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform


class CIFAR10(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        super().__init__()
        train_part = torchvision.datasets.CIFAR10(root, True, download=True)
        test_part = torchvision.datasets.CIFAR10(root, False, download=True)
        train_data = torch.Tensor(train_part.data).permute([0, -1, 1, 2]).float()
        test_data = torch.Tensor(test_part.data).permute([0, -1, 1, 2]).float()
        train_targets = torch.Tensor(train_part.targets).long().squeeze()
        test_targets = torch.Tensor(test_part.targets).long().squeeze()
        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = train_part.classes
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform


class CIFAR100(BaseDataset):
    def __init__(
        self,
        root,
        args,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        super().__init__()
        train_part = torchvision.datasets.CIFAR100(root, True, download=True)
        test_part = torchvision.datasets.CIFAR100(root, False, download=True)
        train_data = torch.Tensor(train_part.data).permute([0, -1, 1, 2]).float()
        test_data = torch.Tensor(test_part.data).permute([0, -1, 1, 2]).float()
        train_targets = torch.Tensor(train_part.targets).long().squeeze()
        test_targets = torch.Tensor(test_part.targets).long().squeeze()
        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = train_part.classes
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform
        super_class = None
        if isinstance(args, Namespace):
            super_class = args.super_class
        elif isinstance(args, dict):
            super_class = args["super_class"]

        if super_class:
            # super_class: [sub_classes]
            CIFAR100_SUPER_CLASS = {
                0: ["beaver", "dolphin", "otter", "seal", "whale"],
                1: ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
                2: ["orchid", "poppy", "rose", "sunflower", "tulip"],
                3: ["bottle", "bowl", "can", "cup", "plate"],
                4: ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
                5: ["clock", "keyboard", "lamp", "telephone", "television"],
                6: ["bed", "chair", "couch", "table", "wardrobe"],
                7: ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
                8: ["bear", "leopard", "lion", "tiger", "wolf"],
                9: ["cloud", "forest", "mountain", "plain", "sea"],
                10: ["bridge", "castle", "house", "road", "skyscraper"],
                11: ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
                12: ["fox", "porcupine", "possum", "raccoon", "skunk"],
                13: ["crab", "lobster", "snail", "spider", "worm"],
                14: ["baby", "boy", "girl", "man", "woman"],
                15: ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
                16: ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
                17: ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
                18: ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
                19: ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
            }
            mapping = {}
            for super_cls, sub_cls in CIFAR100_SUPER_CLASS.items():
                for cls in sub_cls:
                    mapping[cls] = super_cls
            new_targets = []
            for cls in self.targets:
                new_targets.append(mapping[self.classes[cls]])
            self.targets = torch.tensor(new_targets, dtype=torch.long)


class TinyImagenet(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        if not os.path.isdir(root / "raw"):
            raise RuntimeError(
                "Using `data/download/tiny_imagenet.sh` to download the dataset first."
            )
        self.classes = pd.read_table(
            root / "raw/wnids.txt", sep="\t", engine="python", header=None
        )[0].tolist()

        if not os.path.isfile(root / "data.pt") or not os.path.isfile(
            root / "targets.pt"
        ):
            mapping = dict(zip(self.classes, list(range(len(self.classes)))))
            data = []
            targets = []
            for cls in os.listdir(root / "raw" / "train"):
                for img_name in os.listdir(root / "raw" / "train" / cls / "images"):
                    img = pil_to_tensor(
                        Image.open(root / "raw" / "train" / cls / "images" / img_name)
                    ).float()
                    if img.shape[0] == 1:
                        img = torch.expand_copy(img, [3, 64, 64])
                    data.append(img)
                    targets.append(mapping[cls])

            table = pd.read_table(
                root / "raw/val/val_annotations.txt",
                sep="\t",
                engine="python",
                header=None,
            )
            test_classes = dict(zip(table[0].tolist(), table[1].tolist()))
            for img_name in os.listdir(root / "raw" / "val" / "images"):
                img = pil_to_tensor(
                    Image.open(root / "raw" / "val" / "images" / img_name)
                ).float()
                if img.shape[0] == 1:
                    img = torch.expand_copy(img, [3, 64, 64])
                data.append(img)
                targets.append(mapping[test_classes[img_name]])
            torch.save(torch.stack(data), root / "data.pt")
            torch.save(torch.tensor(targets, dtype=torch.long), root / "targets.pt")

        self.data = torch.load(root / "data.pt")
        self.targets = torch.load(root / "targets.pt")
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform


class CINIC10(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        if not os.path.isdir(root / "raw"):
            raise RuntimeError(
                "Using `data/download/tiny_imagenet.sh` to download the dataset first."
            )
        self.classes = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        if not os.path.isfile(root / "data.pt") or not os.path.isfile(
            root / "targets.pt"
        ):
            data = []
            targets = []
            mapping = dict(zip(self.classes, range(10)))
            for folder in ["test", "train", "valid"]:
                for cls in os.listdir(Path(root) / "raw" / folder):
                    for img_name in os.listdir(root / "raw" / folder / cls):
                        img = pil_to_tensor(
                            Image.open(root / "raw" / folder / cls / img_name)
                        ).float()
                        if img.shape[0] == 1:
                            img = torch.expand_copy(img, [3, 32, 32])
                        data.append(img)
                        targets.append(mapping[cls])
            torch.save(torch.stack(data), root / "data.pt")
            torch.save(torch.tensor(targets, dtype=torch.long), root / "targets.pt")

        self.data = torch.load(root / "data.pt")
        self.targets = torch.load(root / "targets.pt")
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform


class DomainNet(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        general_data_transform=None,
        general_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ) -> None:
        super().__init__()
        if not isinstance(root, Path):
            root = Path(root)
        if not os.path.isdir(root / "raw"):
            raise RuntimeError(
                "Using `data/download/domain.sh` to download the dataset first."
            )
        targets_path = root / "targets.pt"
        metadata_path = root / "metadata.json"
        filename_list_path = root / "filename_list.pkl"
        if not (
            os.path.isfile(targets_path)
            and os.path.isfile(metadata_path)
            and os.path.isfile(filename_list_path)
        ):
            raise RuntimeError(
                "Run data/domain/preprocess.py to preprocess DomainNet first."
            )

        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        with open(filename_list_path, "rb") as f:
            self.filename_list = pickle.load(f)

        self.classes = list(metadata["classes"].keys())
        self.targets = torch.load(targets_path)
        self.pre_transform = transforms.Compose(
            [
                transforms.Resize([metadata["image_size"], metadata["image_size"]]),
                transforms.ToTensor(),
            ]
        )
        self.general_data_transform = general_data_transform
        self.general_target_transform = general_target_transform
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform

    def __getitem__(self, index):
        data = self.pre_transform(Image.open(self.filename_list[index]).convert("RGB"))
        targets = self.targets[index]
        if self.enable_train_transform and self.train_data_transform is not None:
            data = self.train_data_transform(data)
        if self.enable_train_transform and self.train_target_transform is not None:
            targets = self.train_target_transform(targets)
        if self.general_data_transform is not None:
            data = self.general_data_transform(data)
        if self.general_target_transform is not None:
            targets = self.general_target_transform(targets)
        return data, targets


DATASETS = {
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    "mnist": MNIST,
    "emnist": EMNIST,
    "fmnist": FashionMNIST,
    "femnist": FEMNIST,
    "medmnistS": MedMNIST,
    "medmnistC": MedMNIST,
    "medmnistA": MedMNIST,
    "covid19": COVID19,
    "celeba": CelebA,
    "synthetic": Synthetic,
    "svhn": SVHN,
    "usps": USPS,
    "tiny_imagenet": TinyImagenet,
    "cinic10": CINIC10,
    "domain": DomainNet,
}
