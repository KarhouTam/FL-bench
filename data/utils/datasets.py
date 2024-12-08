import json
import os
import pickle
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Optional, Type

import numpy as np
import pandas as pd
import torch
import torchvision
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io.image import ImageReadMode, read_image


class BaseDataset(Dataset):
    def __init__(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        classes: List[int],
        train_data_transform: Optional[transforms.Compose] = None,
        train_target_transform: Optional[transforms.Compose] = None,
        test_data_transform: Optional[transforms.Compose] = None,
        test_target_transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.data = data
        self.targets = targets
        self.classes = classes
        self.train_data_transform = train_data_transform
        self.train_target_transform = train_target_transform
        self.test_data_transform = test_data_transform
        self.test_target_transform = test_target_transform
        self.data_transform = self.train_data_transform
        self.target_transform = self.train_target_transform

        # rescale data to fit in [0, 1.0] if needed
        self._rescale_data()

    def _rescale_data(self):
        max_val = self.data.max()
        if max_val > 1.0:
            self.data /= 255.0

    def __getitem__(self, index):
        data, targets = self.data[index], self.targets[index]
        if self.data_transform is not None:
            data = self.data_transform(data)
        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return data, targets

    def train(self):
        self.data_transform = self.train_data_transform
        self.target_transform = self.train_target_transform

    def eval(self):
        self.data_transform = self.test_data_transform
        self.target_transform = self.test_target_transform

    def __len__(self):
        return len(self.targets)


class FEMNIST(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        test_data_transform=None,
        test_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ) -> None:
        if not isinstance(root, Path):
            root = Path(root)
        if not os.path.isfile(root / "data.npy") or not os.path.isfile(
            root / "targets.npy"
        ):
            raise RuntimeError(
                "Please run generate_data.py -d synthetic for generating the data.npy and targets.npy first."
            )

        data = np.load(root / "data.npy")
        targets = np.load(root / "targets.npy")

        super().__init__(
            data=torch.from_numpy(data).float().reshape(-1, 1, 28, 28),
            targets=torch.from_numpy(targets).long(),
            classes=list(range(62)),
            test_data_transform=test_data_transform,
            test_target_transform=test_target_transform,
            train_data_transform=train_data_transform,
            train_target_transform=train_target_transform,
        )


class Synthetic(BaseDataset):
    def __init__(self, root, *args, **kwargs) -> None:
        if not isinstance(root, Path):
            root = Path(root)
        if not os.path.isfile(root / "data.npy") or not os.path.isfile(
            root / "targets.npy"
        ):
            raise RuntimeError(
                "Please run generate_data.py -d synthetic for generating the data.npy and targets.npy first."
            )

        data = np.load(root / "data.npy")
        targets = np.load(root / "targets.npy")

        super().__init__(
            data=torch.from_numpy(data).float(),
            targets=torch.from_numpy(targets).long(),
            classes=sorted(np.unique(targets).tolist()),
        )


class CelebA(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        test_data_transform=None,
        test_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ) -> None:
        if not isinstance(root, Path):
            root = Path(root)
        if not os.path.isfile(root / "data.npy") or not os.path.isfile(
            root / "targets.npy"
        ):
            raise RuntimeError(
                "Please run generate_data.py -d synthetic for generating the data.npy and targets.npy first."
            )

        data = np.load(root / "data.npy")
        targets = np.load(root / "targets.npy")

        super().__init__(
            data=torch.from_numpy(data).permute([0, -1, 1, 2]).float(),
            targets=torch.from_numpy(targets).long(),
            classes=[0, 1],
            test_data_transform=test_data_transform,
            test_target_transform=test_target_transform,
            train_data_transform=train_data_transform,
            train_target_transform=train_target_transform,
        )


class MedMNIST(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        test_data_transform=None,
        test_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        if not isinstance(root, Path):
            root = Path(root)

        super().__init__(
            data=torch.Tensor(np.load(root / "raw" / "xdata.npy")).float().unsqueeze(1),
            targets=torch.Tensor(np.load(root / "raw" / "ydata.npy")).long().squeeze(),
            classes=list(range(11)),
            test_data_transform=test_data_transform,
            test_target_transform=test_target_transform,
            train_data_transform=train_data_transform,
            train_target_transform=train_target_transform,
        )


class COVID19(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        test_data_transform=None,
        test_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        if not isinstance(root, Path):
            root = Path(root)
        super().__init__(
            data=torch.Tensor(np.load(root / "raw" / "xdata.npy"))
            .permute([0, -1, 1, 2])
            .float(),
            targets=torch.Tensor(np.load(root / "raw" / "ydata.npy")).long().squeeze(),
            classes=[0, 1, 2, 3],
            test_data_transform=test_data_transform,
            test_target_transform=test_target_transform,
            train_data_transform=train_data_transform,
            train_target_transform=train_target_transform,
        )


class USPS(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        test_data_transform=None,
        test_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        if not isinstance(root, Path):
            root = Path(root)
        train_part = torchvision.datasets.USPS(root / "raw", True, download=True)
        test_part = torchvision.datasets.USPS(root / "raw", False, download=True)
        train_data = torch.Tensor(train_part.data).float().unsqueeze(1)
        test_data = torch.Tensor(test_part.data).float().unsqueeze(1)
        train_targets = torch.Tensor(train_part.targets).long()
        test_targets = torch.Tensor(test_part.targets).long()

        super().__init__(
            data=torch.cat([train_data, test_data]),
            targets=torch.cat([train_targets, test_targets]),
            classes=list(range(10)),
            test_data_transform=test_data_transform,
            test_target_transform=test_target_transform,
            train_data_transform=train_data_transform,
            train_target_transform=train_target_transform,
        )


class SVHN(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        test_data_transform=None,
        test_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        if not isinstance(root, Path):
            root = Path(root)
        train_part = torchvision.datasets.SVHN(root / "raw", "train", download=True)
        test_part = torchvision.datasets.SVHN(root / "raw", "test", download=True)
        train_data = torch.Tensor(train_part.data).float()
        test_data = torch.Tensor(test_part.data).float()
        train_targets = torch.Tensor(train_part.labels).long()
        test_targets = torch.Tensor(test_part.labels).long()

        super().__init__(
            data=torch.cat([train_data, test_data]),
            targets=torch.cat([train_targets, test_targets]),
            classes=list(range(10)),
            test_data_transform=test_data_transform,
            test_target_transform=test_target_transform,
            train_data_transform=train_data_transform,
            train_target_transform=train_target_transform,
        )


class MNIST(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        test_data_transform=None,
        test_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        train_part = torchvision.datasets.MNIST(root, True, download=True)
        test_part = torchvision.datasets.MNIST(root, False)
        train_data = torch.Tensor(train_part.data).float().unsqueeze(1)
        test_data = torch.Tensor(test_part.data).float().unsqueeze(1)
        train_targets = torch.Tensor(train_part.targets).long().squeeze()
        test_targets = torch.Tensor(test_part.targets).long().squeeze()

        super().__init__(
            data=torch.cat([train_data, test_data]),
            targets=torch.cat([train_targets, test_targets]),
            classes=list(range(10)),
            test_data_transform=test_data_transform,
            test_target_transform=test_target_transform,
            train_data_transform=train_data_transform,
            train_target_transform=train_target_transform,
        )


class FashionMNIST(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        test_data_transform=None,
        test_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        train_part = torchvision.datasets.FashionMNIST(root, True, download=True)
        test_part = torchvision.datasets.FashionMNIST(root, False, download=True)
        train_data = torch.Tensor(train_part.data).float().unsqueeze(1)
        test_data = torch.Tensor(test_part.data).float().unsqueeze(1)
        train_targets = torch.Tensor(train_part.targets).long().squeeze()
        test_targets = torch.Tensor(test_part.targets).long().squeeze()

        super().__init__(
            data=torch.cat([train_data, test_data]),
            targets=torch.cat([train_targets, test_targets]),
            classes=list(range(10)),
            test_data_transform=test_data_transform,
            test_target_transform=test_target_transform,
            train_data_transform=train_data_transform,
            train_target_transform=train_target_transform,
        )


class EMNIST(BaseDataset):
    def __init__(
        self,
        root,
        args,
        test_data_transform=None,
        test_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        split = None
        if isinstance(args, Namespace):
            split = args.emnist_split
        elif isinstance(args, dict):
            split = args["emnist_split"]
        elif isinstance(args, DictConfig):
            split = args.emnist_split
        train_part = torchvision.datasets.EMNIST(
            root, split=split, train=True, download=True
        )
        test_part = torchvision.datasets.EMNIST(
            root, split=split, train=False, download=False
        )
        train_data = torch.Tensor(train_part.data).float().unsqueeze(1)
        test_data = torch.Tensor(test_part.data).float().unsqueeze(1)
        train_targets = torch.Tensor(train_part.targets).long().squeeze()
        test_targets = torch.Tensor(test_part.targets).long().squeeze()

        super().__init__(
            data=torch.cat([train_data, test_data]),
            targets=torch.cat([train_targets, test_targets]),
            classes=list(range(len(train_part.classes))),
            test_data_transform=test_data_transform,
            test_target_transform=test_target_transform,
            train_data_transform=train_data_transform,
            train_target_transform=train_target_transform,
        )


class CIFAR10(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        test_data_transform=None,
        test_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        train_part = torchvision.datasets.CIFAR10(root, True, download=True)
        test_part = torchvision.datasets.CIFAR10(root, False, download=True)
        train_data = torch.Tensor(train_part.data).permute([0, -1, 1, 2]).float()
        test_data = torch.Tensor(test_part.data).permute([0, -1, 1, 2]).float()
        train_targets = torch.Tensor(train_part.targets).long().squeeze()
        test_targets = torch.Tensor(test_part.targets).long().squeeze()

        super().__init__(
            data=torch.cat([train_data, test_data]),
            targets=torch.cat([train_targets, test_targets]),
            classes=list(range(10)),
            test_data_transform=test_data_transform,
            test_target_transform=test_target_transform,
            train_data_transform=train_data_transform,
            train_target_transform=train_target_transform,
        )


class CIFAR100(BaseDataset):
    def __init__(
        self,
        root,
        args,
        test_data_transform=None,
        test_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        train_part = torchvision.datasets.CIFAR100(root, True, download=True)
        test_part = torchvision.datasets.CIFAR100(root, False, download=True)
        train_data = torch.Tensor(train_part.data).permute([0, -1, 1, 2]).float()
        test_data = torch.Tensor(test_part.data).permute([0, -1, 1, 2]).float()
        train_targets = torch.Tensor(train_part.targets).long().squeeze()
        test_targets = torch.Tensor(test_part.targets).long().squeeze()
        data = torch.cat([train_data, test_data])
        targets = torch.cat([train_targets, test_targets])
        classes = list(range(100))

        super_class = False
        if isinstance(args, (Namespace, DictConfig)):
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
            for cls in targets:
                new_targets.append(mapping[train_part.classes[cls]])
            targets = torch.tensor(new_targets, dtype=torch.long)
            classes = list(range(20))

        super().__init__(
            data=data,
            targets=targets,
            classes=classes,
            test_data_transform=test_data_transform,
            test_target_transform=test_target_transform,
            train_data_transform=train_data_transform,
            train_target_transform=train_target_transform,
        )


class TinyImagenet(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        test_data_transform=None,
        test_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        if not isinstance(root, Path):
            root = Path(root)
        if not os.path.isdir(root / "raw"):
            raise RuntimeError(
                "Using `data/download/tiny_imagenet.sh` to download the dataset first."
            )
        classes = pd.read_table(
            root / "raw/wnids.txt", sep="\t", engine="python", header=None
        )[0].tolist()

        if not os.path.isfile(root / "data.pt") or not os.path.isfile(
            root / "targets.pt"
        ):
            mapping = dict(zip(classes, list(range(len(classes)))))
            data = []
            targets = []
            for cls in os.listdir(root / "raw" / "train"):
                for img_name in os.listdir(root / "raw" / "train" / cls / "images"):
                    img = read_image(
                        str(root / "raw" / "train" / cls / "images" / img_name),
                        mode=ImageReadMode.RGB,
                    ).float()
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
                img = read_image(
                    str(root / "raw" / "val" / "images" / img_name),
                    mode=ImageReadMode.RGB,
                ).float()
                data.append(img)
                targets.append(mapping[test_classes[img_name]])
            torch.save(torch.stack(data), root / "data.pt")
            torch.save(torch.tensor(targets, dtype=torch.long), root / "targets.pt")

        super().__init__(
            data=torch.load(root / "data.pt"),
            targets=torch.load(root / "targets.pt"),
            classes=list(range(len(classes))),
            test_data_transform=test_data_transform,
            test_target_transform=test_target_transform,
            train_data_transform=train_data_transform,
            train_target_transform=train_target_transform,
        )


class CINIC10(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        test_data_transform=None,
        test_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ):
        if not isinstance(root, Path):
            root = Path(root)
        if not os.path.isdir(root / "raw"):
            raise RuntimeError(
                "Using `data/download/tiny_imagenet.sh` to download the dataset first."
            )
        classes = [
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
            mapping = dict(zip(classes, range(10)))
            for folder in ["test", "train", "valid"]:
                for cls in os.listdir(Path(root) / "raw" / folder):
                    for img_name in os.listdir(root / "raw" / folder / cls):
                        img = read_image(
                            str(root / "raw" / folder / cls / img_name),
                            mode=ImageReadMode.RGB,
                        ).float()
                        data.append(img)
                        targets.append(mapping[cls])
            torch.save(torch.stack(data), root / "data.pt")
            torch.save(torch.tensor(targets, dtype=torch.long), root / "targets.pt")

        super().__init__(
            data=torch.load(root / "data.pt"),
            targets=torch.load(root / "targets.pt"),
            classes=list(range(len(classes))),
            test_data_transform=test_data_transform,
            test_target_transform=test_target_transform,
            train_data_transform=train_data_transform,
            train_target_transform=train_target_transform,
        )


class DomainNet(BaseDataset):
    def __init__(
        self,
        root,
        args=None,
        test_data_transform=None,
        test_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    ) -> None:
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

        self.pre_transform = transforms.Compose(
            [transforms.Resize(metadata["image_size"]), transforms.ToTensor()]
        )
        super().__init__(
            data=torch.empty(1, 1, 1, 1),  # dummy data
            targets=torch.load(targets_path),
            classes=list(range(len(metadata["classes"]))),
            test_data_transform=test_data_transform,
            test_target_transform=test_target_transform,
            train_data_transform=train_data_transform,
            train_target_transform=train_target_transform,
        )

    def __getitem__(self, index):
        data = self.pre_transform(Image.open(self.filename_list[index]).convert("RGB"))
        targets = self.targets[index]
        if self.data_transform is not None:
            data = self.data_transform(data)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
        return data, targets


DATASETS: Dict[str, Type[BaseDataset]] = {
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
