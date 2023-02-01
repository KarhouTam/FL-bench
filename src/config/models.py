from collections import OrderedDict
from typing import Dict, OrderedDict, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# CNN used in FedAvg
class FedAvgCNN(nn.Module):
    def __init__(self, dataset: str, need_all_features=False):
        super(FedAvgCNN, self).__init__()
        config = {
            "mnist": (1, 1024, 10),
            "medmnistS": (1, 1024, 11),
            "medmnistC": (1, 1024, 11),
            "medmnistA": (1, 1024, 11),
            "covid19": (3, 196736, 4),
            "fmnist": (1, 1024, 10),
            "emnist": (1, 1024, 62),
            "femnist": (1, 1, 62),
            "cifar10": (3, 1600, 10),
            "cifar100": (3, 1600, 100),
            "tiny_imagenet": (3, 3200, 200),
            "celeba": (3, 133824, 2),
            "svhn": (3, 1600, 10),
            "usps": (1, 800, 10),
        }
        self.base = nn.Sequential(
            OrderedDict(
                conv1=nn.Conv2d(config[dataset][0], 32, 5),
                activation1=nn.ReLU(),
                pool1=nn.MaxPool2d(2),
                conv2=nn.Conv2d(32, 64, 5),
                activation2=nn.ReLU(),
                pool2=nn.MaxPool2d(2),
                flatten=nn.Flatten(),
                fc1=nn.Linear(config[dataset][1], 512),
            )
        )
        self.classifier = nn.Linear(512, config[dataset][2])
        self.all_features = []
        self.need_all_features_flag = False

    def need_all_features(self):
        self.need_all_features_flag = True
        target_modules = [
            module
            for module in self.base.modules()
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)
        ]

        def get_feature_hook_fn(model, input, output):
            self.all_features.append(output)

        for module in target_modules:
            module.register_forward_hook(get_feature_hook_fn)

    def forward(self, x):
        out = self.classifier(F.relu(self.base(x)))
        if self.need_all_features_flag:
            self.all_features = []
        return out

    def get_final_features(self, x, detach=True):
        func = (lambda x: x.clone().detach()) if detach else lambda x: x
        out = self.base(x)
        return func(out)

    def get_all_features(self, x, detach=True):
        if self.need_all_features_flag:
            func = (lambda x: x.clone().detach()) if detach else (lambda x: x)
            _ = self.base(x)
            feature_list = [func(feature) for feature in self.all_features]
            self.all_features = []
            return feature_list


class LeNet5(nn.Module):
    def __init__(self, dataset: str, need_all_features=False) -> None:
        super(LeNet5, self).__init__()
        config = {
            "mnist": (1, 256, 10),
            "medmnistS": (1, 256, 11),
            "medmnistC": (1, 256, 11),
            "medmnistA": (1, 256, 11),
            "covid19": (3, 49184, 4),
            "fmnist": (1, 256, 10),
            "emnist": (1, 256, 62),
            "femnist": (1, 256, 62),
            "cifar10": (3, 400, 10),
            "svhn": (3, 400, 10),
            "cifar100": (3, 400, 100),
            "celeba": (3, 33456, 2),
            "usps": (1, 200, 10),
            "tiny_imagenet": (3, 2704, 200),
        }
        self.base = nn.Sequential(
            OrderedDict(
                conv1=nn.Conv2d(config[dataset][0], 6, 5),
                bn1=nn.BatchNorm2d(6),
                activation1=nn.ReLU(),
                pool1=nn.MaxPool2d(2),
                conv2=nn.Conv2d(6, 16, 5),
                bn2=nn.BatchNorm2d(16),
                activation2=nn.ReLU(),
                pool2=nn.MaxPool2d(2),
                flatten=nn.Flatten(),
                fc1=nn.Linear(config[dataset][1], 120),
                activation3=nn.ReLU(),
                fc2=nn.Linear(120, 84),
            )
        )

        self.classifier = nn.Linear(84, config[dataset][2])
        self.all_features = []
        self.need_all_features_flag = False

    def need_all_features(self):
        self.need_all_features_flag = True
        target_modules = [
            module
            for module in self.base.modules()
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)
        ]

        def get_feature_hook_fn(model, input, output):
            self.all_features.append(output)

        for module in target_modules:
            module.register_forward_hook(get_feature_hook_fn)

    def forward(self, x):
        out = self.classifier(F.relu(self.base(x)))
        if self.need_all_features_flag:
            self.all_features = []
        return out

    def get_final_features(self, x, detach=True):
        func = (lambda x: x.clone().detach()) if detach else (lambda x: x)
        out = self.base(x)
        return func(out)

    def get_all_features(self, x, detach=True):
        if self.need_all_features_flag:
            func = (lambda x: x.clone().detach()) if detach else (lambda x: x)
            _ = self.base(x)
            feature_list = [func(feature) for feature in self.all_features]
            self.all_features = []
            return feature_list


class TwoNN(nn.Module):
    def __init__(self, dataset, need_all_features=False):
        super(TwoNN, self).__init__()
        config = {
            "mnist": (784, 10),
            "medmnistS": (784, 11),
            "medmnistC": (784, 11),
            "medmnistA": (784, 11),
            "fmnist": (784, 10),
            "emnist": (784, 62),
            "femnist": (784, 62),
            "cifar10": (3072, 10),
            "svhn": (3072, 10),
            "cifar100": (3072, 100),
            "usps": (1536, 10),
            "synthetic": (60, 10),  # default dimension and classes
        }

        self.fc1 = nn.Linear(config[dataset][0], 200)
        self.classifier = nn.Linear(200, config[dataset][1])
        self.activation = nn.ReLU()

    def need_all_features(self):
        return

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.activation(self.fc1(x))
        x = self.classifier(x)
        return x

    def get_final_features(self, x, detach=True):
        func = (lambda x: x.clone().detach()) if detach else (lambda x: x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return func(x)

    def get_all_features(self, x, detach=True):
        raise RuntimeError("2NN has 0 Conv layer, so is disable to get all features.")


class MobileNetV2(nn.Module):
    def __init__(self, dataset, need_all_features=False):
        super(MobileNetV2, self).__init__()
        config = {
            "mnist": 10,
            "medmnistS": 11,
            "medmnistC": 11,
            "medmnistA": 11,
            "fmnist": 10,
            "svhn": 10,
            "emnist": 62,
            "femnist": 62,
            "cifar10": 10,
            "cifar100": 100,
            "covid19": 4,
            "usps": 10,
            "tiny_imagenet": 200,
        }
        # NOTE: If you don't want parameters pretrained, uncomment the lines below
        # self.base = models.MobileNetV2(config[dataset])
        # self.classifier = deepcopy(self.base.classifier[1])
        self.base = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V2
        )
        self.classifier = nn.Linear(
            self.base.classifier[1].in_features, config[dataset]
        )

        self.base.classifier[1] = nn.Identity()

        self.dropouts = [
            module for module in self.base.modules() if isinstance(module, nn.Dropout)
        ]

        self.all_features = []
        self.need_all_features_flag = False

    def need_all_features(self):
        self.need_all_features_flag = True
        target_modules = [
            module
            for module in self.base.features.modules()
            if isinstance(module, nn.Conv2d)
        ]

        def get_feature_hook_fn(model, input, output):
            self.all_features.append(output)

        for module in target_modules:
            module.register_forward_hook(get_feature_hook_fn)

    def forward(self, x):
        out = self.classifier(F.relu(self.base(x)))
        if self.need_all_features_flag:
            self.all_features = []
        return out

    def get_final_features(self, x, detach=True):
        for dropout in self.dropouts:
            dropout.eval()

        func = (lambda x: x.clone().detach()) if detach else (lambda x: x)
        out = self.base(x)

        for dropout in self.dropouts:
            dropout.train()
        return func(out)

    def get_all_features(self, x, detach=True):
        if self.need_all_features_flag:
            func = (lambda x: x.clone().detach()) if detach else (lambda x: x)
            _ = self.base(x)
            feature_list = [func(feature) for feature in self.all_features]
            self.all_features = []
            return feature_list


MODEL_DICT: Dict[str, Type[nn.Module]] = {
    "lenet5": LeNet5,
    "avgcnn": FedAvgCNN,
    "2nn": TwoNN,
    "mobile": MobileNetV2,
}
