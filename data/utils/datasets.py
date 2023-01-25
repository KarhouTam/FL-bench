from argparse import Namespace

import torch
import numpy as np
import torchvision
import os
import pandas as pd
from path import Path
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from torch.utils.data import Dataset


class FEMNIST(Dataset):
    def __init__(self, data, targets) -> None:
        self.data = torch.tensor(data, dtype=torch.float).reshape(-1, 1, 28, 28)
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.classes = list(range(62))

    def __getitem__(self, index):
        data, target = self.data[index], self.targets[index]
        return data, target

    def __len__(self):
        return len(self.targets)


class Synthetic(Dataset):
    def __init__(self, X, Y):
        self.data = X
        self.targets = Y

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.targets)


class CelebA(Dataset):
    def __init__(self, data, targets, transform=None, target_transform=None):
        self.data = torch.stack(data)  # [3, 218, 178]
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.transform = transform
        self.target_transform = target_transform
        self.classes = [0, 1]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        data, targets = self.data[idx], self.targets[idx]
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
        return data, targets


class MedMNIST(Dataset):
    def __init__(self, root, args=None, transform=None, target_transform=None):
        if not isinstance(root, Path):
            root = Path(root)
        self.data = (
            torch.Tensor(np.load(root / "raw" / "xdata.npy")).float().unsqueeze(1)
        )
        self.targets = (
            torch.Tensor(np.load(root / "raw" / "ydata.npy")).long().squeeze()
        )
        self.transform = transform
        self.target_transform = target_transform
        self.classes = list(range(11))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        data, targets = self.data[idx], self.targets[idx]
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
        return data, targets


class COVID19(Dataset):
    # (3,244,224)
    def __init__(self, root, args=None, transform=None, target_transform=None):
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
        self.transform = transform
        self.target_transform = target_transform
        self.classes = list(range(4))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        data, targets = self.data[idx], self.targets[idx]
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
        return data, targets


class USPS(Dataset):
    # 3 x 32 x 32
    def __init__(self, root, args=None, transform=None, target_transform=None):
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
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        data, targets = self.data[idx], self.targets[idx]
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
        return data, targets


class SVHN(Dataset):
    # 3 x 32 x 32
    def __init__(self, root, args=None, transform=None, target_transform=None):
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
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        data, targets = self.data[idx], self.targets[idx]
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
        return data, targets


class MNIST(Dataset):
    def __init__(self, root, args=None, transform=None, target_transform=None):
        train_part = torchvision.datasets.MNIST(
            root, True, transform, target_transform, download=True
        )
        test_part = torchvision.datasets.MNIST(root, False, transform, target_transform)
        train_data = torch.Tensor(train_part.data).float().unsqueeze(1)
        test_data = torch.Tensor(test_part.data).float().unsqueeze(1)
        train_targets = torch.Tensor(train_part.targets).long().squeeze()
        test_targets = torch.Tensor(test_part.targets).long().squeeze()
        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = train_part.classes
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        data, targets = self.data[idx], self.targets[idx]
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
        return data, targets


class FashionMNIST(Dataset):
    def __init__(self, root, args=None, transform=None, target_transform=None):
        train_part = torchvision.datasets.FashionMNIST(root, True, download=True)
        test_part = torchvision.datasets.FashionMNIST(root, False, download=True)
        train_data = torch.Tensor(train_part.data).float().unsqueeze(1)
        test_data = torch.Tensor(test_part.data).float().unsqueeze(1)
        train_targets = torch.Tensor(train_part.targets).long().squeeze()
        test_targets = torch.Tensor(test_part.targets).long().squeeze()
        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = train_part.classes
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        data, targets = self.data[idx], self.targets[idx]
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
        return data, targets


class EMNIST(Dataset):
    def __init__(self, root, args, transform=None, target_transform=None):
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
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        data, targets = self.data[idx], self.targets[idx]
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
        return data, targets


class CIFAR10(Dataset):
    def __init__(self, root, args=None, transform=None, target_transform=None):
        train_part = torchvision.datasets.CIFAR10(root, True, download=True)
        test_part = torchvision.datasets.CIFAR10(root, False, download=True)
        train_data = torch.Tensor(train_part.data).permute([0, -1, 1, 2]).float()
        test_data = torch.Tensor(test_part.data).permute([0, -1, 1, 2]).float()
        train_targets = torch.Tensor(train_part.targets).long().squeeze()
        test_targets = torch.Tensor(test_part.targets).long().squeeze()
        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = train_part.classes
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        data, targets = self.data[idx], self.targets[idx]
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
        return data, targets


class CIFAR100(Dataset):
    def __init__(self, root, args, transform=None, target_transform=None):
        train_part = torchvision.datasets.CIFAR100(root, True, download=True)
        test_part = torchvision.datasets.CIFAR100(root, False, download=True)
        train_data = torch.Tensor(train_part.data).permute([0, -1, 1, 2]).float()
        test_data = torch.Tensor(test_part.data).permute([0, -1, 1, 2]).float()
        train_targets = torch.Tensor(train_part.targets).long().squeeze()
        test_targets = torch.Tensor(test_part.targets).long().squeeze()
        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = train_part.classes
        self.transform = transform
        self.target_transform = target_transform
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

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        data, targets = self.data[idx], self.targets[idx]
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
        return data, targets


class TinyImagenet(Dataset):
    def __init__(self, root, args=None, transform=None, target_transform=None):
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
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        data, target = self.data[idx], self.targets[idx]
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target


DATASETS = {
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    "mnist": MNIST,
    "emnist": EMNIST,
    "fmnist": FashionMNIST,
    "medmnistS": MedMNIST,
    "medmnistC": MedMNIST,
    "medmnistA": MedMNIST,
    "covid19": COVID19,
    "svhn": SVHN,
    "usps": USPS,
    "tiny_imagenet": TinyImagenet,
}
