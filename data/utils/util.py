import json
import os
from argparse import Namespace
from collections import Counter
from typing import Dict

import torch
import numpy as np
from path import Path
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from datasets import FEMNIST, CelebA, Synthetic

_DATA_ROOT = Path(__file__).parent.parent.abspath()


def prune_args(args: Namespace) -> Dict:
    args_dict = {}
    # general settings
    args_dict["dataset"] = args.dataset
    args_dict["client_num_in_total"] = args.client_num_in_total
    args_dict["testset_ratio"] = args.testset_ratio
    args_dict["seed"] = args.seed

    args_dict["split"] = args.split
    if args.split == "user":
        args_dict["train_user_ratio"] = args.fraction

    if args.dataset == "emnist":
        args_dict["emnist_split"] = args.emnist_split
    elif args.dataset == "cifar100":
        args_dict["super_class"] = bool(args.super_class)
    elif args.dataset == "synthetic":
        args_dict["beta"] = args.beta
        args_dict["gamma"] = args.gamma
        args_dict["dimension"] = args.dimension

    if args.iid == 1:
        args_dict["iid"] = True
    else:
        # Dirchlet
        if args.alpha > 0:
            args_dict["alpha"] = args.alpha
            args_dict["least_samples"] = args.least_samples
        # randomly assign classes
        elif args.classes > 0:
            args_dict["classes_per_client"] = args.classes
        # allocate shards
        elif args.shards > 0:
            args_dict["num_shards_per_client"] = args.shards

    return args_dict


def process_femnist(split):
    train_dir = _DATA_ROOT / "femnist" / "data" / "train"
    test_dir = _DATA_ROOT / "femnist" / "data" / "test"
    stats = {}
    i = 0
    datasets = []
    # load data of train clients
    if split == "sample":
        train_filename_list = sorted(os.listdir(train_dir))
        test_filename_list = sorted(os.listdir(test_dir))
        for train_js_file, test_js_file in zip(train_filename_list, test_filename_list):
            with open(train_dir / train_js_file, "r") as f:
                train = json.load(f)
            with open(test_dir / test_js_file, "r") as f:
                test = json.load(f)
            for writer in train["users"]:
                stats[i] = {}
                train_data = train["user_data"][writer]["x"]
                train_targets = train["user_data"][writer]["y"]
                test_data = test["user_data"][writer]["x"]
                test_targets = test["user_data"][writer]["y"]

                all_data = train_data + test_data
                all_targets = train_targets + test_targets
                stats[i]["x"] = len(all_data)
                stats[i]["y"] = Counter(all_targets)

                datasets.append(FEMNIST(all_data, all_targets))
                i += 1

        clients_4_test = list(range(i))
        clients_4_train = list(range(i))

    else:
        for js_filename in os.listdir(train_dir):
            with open(train_dir / js_filename, "r") as f:
                json_data = json.load(f)
            for writer in json_data["users"]:
                stats[i] = {"x": None, "y": None}
                data = json_data["user_data"][writer]["x"]
                targets = json_data["user_data"][writer]["y"]
                stats["train"][i]["x"] = len(data)
                stats["train"][i]["y"] = Counter(targets)
                datasets.append(FEMNIST(data, targets))
                i += 1

        clients_4_train = list(range(i))

        # load data of test clients
        for js_filename in os.listdir(test_dir):
            with open(test_dir / js_filename, "r") as f:
                json_data = json.load(f)
            for writer in json_data["users"]:
                stats[i] = {"x": None, "y": None}
                data = json_data["user_data"][writer]["x"]
                targets = json_data["user_data"][writer]["y"]
                stats["test"][i]["x"] = len(data)
                stats["test"][i]["y"] = Counter(targets)
                datasets.append(FEMNIST(data, targets))
                i += 1

        clients_4_test = list(range(len(clients_4_train), i))

    num_samples = np.array(list(map(lambda stat_i: stat_i["x"], stats.values())))
    stats["sample per client"] = {
        "std": num_samples.mean(),
        "stddev": num_samples.std(),
    }

    return datasets, stats, len(datasets), clients_4_train, clients_4_test


def process_celeba(split):
    train_dir = _DATA_ROOT / "celeba" / "data" / "train"
    test_dir = _DATA_ROOT / "celeba" / "data" / "test"
    raw_data_dir = _DATA_ROOT / "celeba" / "data" / "raw" / "img_align_celeba"
    train_filename = os.listdir(train_dir)[0]
    test_filename = os.listdir(test_dir)[0]
    with open(train_dir / train_filename, "r") as f:
        train = json.load(f)
    with open(test_dir / test_filename, "r") as f:
        test = json.load(f)

    stats = {}
    datasets = []
    i = 0
    clients_4_test, clients_4_train = None, None

    if split == "sample":
        for i, ori_id in enumerate(train["users"]):
            stats[i] = {"x": None, "y": None}
            data = []
            targets = []
            targets = train["user_data"][ori_id]["y"] + test["user_data"][ori_id]["y"]
            for img_name in (
                train["user_data"][ori_id]["x"] + test["user_data"][ori_id]["x"]
            ):
                data.append(pil_to_tensor(Image.open(raw_data_dir / img_name)))
            stats[i]["x"] = train["num_samples"][i] + test["num_samples"][i]
            stats[i]["y"] = Counter(targets)

            datasets.append(CelebA(data, targets))
            i += 1

        clients_4_train = list(range(i))
        clients_4_test = list(range(i))

    else:  # t == "user"
        # process data of train clients
        stats["train"] = {}
        for i, ori_id in enumerate(train["users"]):
            stats[i] = {"x": None, "y": None}
            data = []
            targets = []
            targets = train["user_data"][ori_id]["y"]
            for img_name in train["user_data"][ori_id]["x"]:
                data.append(pil_to_tensor(Image.open(raw_data_dir / img_name)))

            stats[i]["x"] = train["num_samples"][i]
            stats[i]["y"] = Counter(targets)
            datasets.append(CelebA(data, targets))
            i += 1

        clients_4_train = list(range(i))

        # process data of test clients
        stats["test"] = {}
        for i, ori_id in enumerate(test["users"]):
            stats[i] = {"x": None, "y": None}
            data = []
            targets = []
            targets = test["user_data"][ori_id]["y"]
            for img_name in test["user_data"][ori_id]["x"]:
                data.append(pil_to_tensor(Image.open(raw_data_dir / img_name)))

            stats["test"][i]["x"] = test["num_samples"][i]
            stats["test"][i]["y"] = Counter(targets)
            datasets.append(CelebA(data, targets))
            i += 1

        clients_4_test = list(range(len(clients_4_train), i))

    num_samples = np.array(list(map(lambda stat_i: stat_i["x"], stats.values())))
    stats["sample per client"] = {
        "std": num_samples.mean(),
        "stddev": num_samples.std(),
    }

    return datasets, stats, len(datasets), clients_4_train, clients_4_test


def generate_synthetic_data(args):
    def softmax(x):
        ex = np.exp(x)
        sum_ex = np.sum(np.exp(x))
        return ex / sum_ex

    # All codes below are modified from https://github.com/litian96/FedProx/tree/master/data
    NUM_CLASS = 10 if args.classes <= 0 else args.classes

    samples_per_user = (
        np.random.lognormal(4, 2, args.client_num_in_total).astype(int) + 50
    ).tolist()
    # samples_per_user = [10 for _ in range(args.client_num_in_total)]
    W_global = np.zeros((args.dimension, NUM_CLASS))
    b_global = np.zeros(NUM_CLASS)
    X_split = [[] for _ in range(args.client_num_in_total)]
    y_split = [[] for _ in range(args.client_num_in_total)]

    mean_b = mean_W = np.random.normal(0, args.gamma, args.client_num_in_total)
    B = np.random.normal(0, args.beta, args.client_num_in_total)
    mean_x = np.zeros((args.client_num_in_total, args.dimension))
    diagonal = np.zeros(args.dimension)
    for j in range(args.dimension):
        diagonal[j] = np.power((j + 1), -1.2)
    cov_x = np.diag(diagonal)

    for client_id in range(args.client_num_in_total):
        if args.iid:
            mean_x[client_id] = np.ones(args.dimension) * B[client_id]  # all zeros
        else:
            mean_x[client_id] = np.random.normal(B[client_id], 1, args.dimension)

    if args.iid:
        W_global = np.random.normal(0, 1, (args.dimension, NUM_CLASS))
        b_global = np.random.normal(0, 1, NUM_CLASS)

    all_datasets = []
    stats = {}
    for client_id in range(args.client_num_in_total):

        W = np.random.normal(mean_W[client_id], 1, (args.dimension, NUM_CLASS))
        b = np.random.normal(mean_b[client_id], 1, NUM_CLASS)

        if args.iid != 0:
            W = W_global
            b = b_global

        xx = np.random.multivariate_normal(
            mean_x[client_id], cov_x, samples_per_user[client_id]
        )
        yy = np.zeros(samples_per_user[client_id])

        for j in range(samples_per_user[client_id]):
            tmp = np.dot(xx[j], W) + b
            yy[j] = np.argmax(softmax(tmp))

        X_split[client_id] = torch.tensor(xx, dtype=torch.float)
        y_split[client_id] = torch.tensor(yy, dtype=torch.long)
        all_datasets.append(Synthetic(X_split[client_id], y_split[client_id]))
        stats[client_id] = {}
        stats[client_id]["x"] = samples_per_user[client_id]
        stats[client_id]["y"] = Counter(y_split[client_id].tolist())

    num_samples = np.array(list(map(lambda stat_i: stat_i["x"], stats.values())))
    stats["sample per client"] = {
        "std": num_samples.mean(),
        "stddev": num_samples.std(),
    }

    return all_datasets, stats
