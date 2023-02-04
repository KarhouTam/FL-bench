import json
import os
from argparse import Namespace
from collections import Counter
from typing import Dict

import numpy as np
from path import Path
from PIL import Image

_DATA_ROOT = Path(__file__).parent.parent.abspath()


def prune_args(args: Namespace) -> Dict:
    args_dict = {}
    # general settings
    args_dict["dataset"] = args.dataset
    args_dict["client_num_in_total"] = args.client_num_in_total
    args_dict["fraction"] = args.fraction
    args_dict["seed"] = args.seed
    args_dict["split"] = args.split

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
            args_dict["shards_per_client"] = args.shards

    return args_dict


def process_femnist(args):
    train_dir = _DATA_ROOT / "femnist" / "data" / "train"
    test_dir = _DATA_ROOT / "femnist" / "data" / "test"
    stats = {}
    client_cnt = 0
    data_cnt = 0
    all_data = []
    all_targets = []
    partition = {"separation": None, "data_indices": {}}
    clients_4_train, clients_4_test = None, None

    # load data of train clients
    if args.split == "sample":
        train_filename_list = sorted(os.listdir(train_dir))
        test_filename_list = sorted(os.listdir(test_dir))
        for train_js_file, test_js_file in zip(train_filename_list, test_filename_list):
            with open(train_dir / train_js_file, "r") as f:
                train = json.load(f)
            with open(test_dir / test_js_file, "r") as f:
                test = json.load(f)
            for writer in train["users"]:
                stats[client_cnt] = {}
                train_data = train["user_data"][writer]["x"]
                train_targets = train["user_data"][writer]["y"]
                test_data = test["user_data"][writer]["x"]
                test_targets = test["user_data"][writer]["y"]

                data = train_data + test_data
                targets = train_targets + test_targets
                all_data.append(np.array(data))
                all_targets.append(np.array(targets))
                partition["data_indices"][client_cnt] = {
                    "train": list(range(data_cnt, data_cnt + len(train_data))),
                    "test": list(
                        range(data_cnt + len(train_data), data_cnt + len(data))
                    ),
                }
                stats[client_cnt]["x"] = len(data)
                stats[client_cnt]["y"] = Counter(targets)

                data_cnt += len(data)
                client_cnt += 1

        clients_4_test = list(range(client_cnt))
        clients_4_train = list(range(client_cnt))

        num_samples = np.array(list(map(lambda stat_i: stat_i["x"], stats.values())))
        stats["sample per client"] = {
            "std": num_samples.mean(),
            "stddev": num_samples.std(),
        }

    else:
        stats["train"] = {}
        stats["test"] = {}
        for js_filename in os.listdir(train_dir):
            with open(train_dir / js_filename, "r") as f:
                json_data = json.load(f)
            for writer in json_data["users"]:
                stats["train"][client_cnt] = {"x": None, "y": None}
                data = json_data["user_data"][writer]["x"]
                targets = json_data["user_data"][writer]["y"]

                all_data.append(np.array(data))
                all_targets.append(np.array(targets))
                partition["data_indices"][client_cnt] = {
                    "train": list(range(data_cnt, data_cnt + len(data))),
                    "test": [],
                }
                stats["train"][client_cnt]["x"] = len(data)
                stats["train"][client_cnt]["y"] = Counter(targets)

                data_cnt += len(data)
                client_cnt += 1

        clients_4_train = list(range(client_cnt))

        num_samples = np.array(
            list(map(lambda stat_i: stat_i["x"], stats["train"].values()))
        )
        stats["train"]["sample per client"] = {
            "std": num_samples.mean(),
            "stddev": num_samples.std(),
        }

        # load data of test clients
        for js_filename in os.listdir(test_dir):
            with open(test_dir / js_filename, "r") as f:
                json_data = json.load(f)
            for writer in json_data["users"]:
                stats["test"][client_cnt] = {"x": None, "y": None}
                data = json_data["user_data"][writer]["x"]
                targets = json_data["user_data"][writer]["y"]
                all_data.append(np.array(data))
                all_targets.append(np.array(targets))
                partition["data_indices"][client_cnt] = {
                    "train": [],
                    "test": list(range(data_cnt, data_cnt + len(data))),
                }
                stats["test"][client_cnt]["x"] = len(data)
                stats["test"][client_cnt]["y"] = Counter(targets)

                data_cnt += len(data)
                client_cnt += 1

        clients_4_test = list(range(len(clients_4_train), client_cnt))

        num_samples = np.array(
            list(map(lambda stat_i: stat_i["x"], stats["test"].values()))
        )
        stats["test"]["sample per client"] = {
            "std": num_samples.mean(),
            "stddev": num_samples.std(),
        }

    np.save(_DATA_ROOT / "femnist" / "data", np.concatenate(all_data))
    np.save(_DATA_ROOT / "femnist" / "targets", np.concatenate(all_targets))

    args.client_num_in_total = client_cnt
    partition["separation"] = {
        "train": clients_4_train,
        "test": clients_4_test,
        "total": client_cnt,
    }

    return partition, stats


def process_celeba(args):
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
    data_cnt = 0
    all_data = []
    all_targets = []
    partition = {"separation": None, "data_indices": {}}
    client_cnt = 0
    clients_4_test, clients_4_train = None, None

    if args.split == "sample":
        for client_cnt, ori_id in enumerate(train["users"]):
            stats[client_cnt] = {"x": None, "y": None}
            train_data = np.stack(
                [
                    np.asarray(Image.open(raw_data_dir / img_name))
                    for img_name in train["user_data"][ori_id]["x"]
                ],
                axis=0,
            )
            test_data = np.stack(
                [
                    np.asarray(Image.open(raw_data_dir / img_name))
                    for img_name in test["user_data"][ori_id]["x"]
                ],
                axis=0,
            )
            train_targets = train["user_data"][ori_id]["y"]
            test_targets = test["user_data"][ori_id]["y"]

            data = np.concatenate([train_data, test_data])
            targets = train_targets + test_targets
            if all_data == []:
                all_data = data
            else:
                all_data = np.concatenate([all_data, data])
            if all_targets == []:
                all_targets = targets
            else:
                all_targets = np.concatenate([all_targets, targets])
            partition["data_indices"][client_cnt] = {
                "train": list(range(data_cnt, data_cnt + len(train_data))),
                "test": list(range(data_cnt + len(train_data), data_cnt + len(data))),
            }
            stats[client_cnt]["x"] = (
                train["num_samples"][client_cnt] + test["num_samples"][client_cnt]
            )
            stats[client_cnt]["y"] = Counter(targets)

            data_cnt += len(data)
            client_cnt += 1

        clients_4_train = list(range(client_cnt))
        clients_4_test = list(range(client_cnt))
        num_samples = np.array(list(map(lambda stat_i: stat_i["x"], stats.values())))
        stats["sample per client"] = {
            "std": num_samples.mean(),
            "stddev": num_samples.std(),
        }

    else:  # t == "user"
        # process data of train clients
        stats["train"] = {}
        for client_cnt, ori_id in enumerate(train["users"]):
            stats[client_cnt] = {"x": None, "y": None}
            data = np.stack(
                [
                    np.asarray(Image.open(raw_data_dir / img_name))
                    for img_name in train["user_data"][ori_id]["x"]
                ],
                axis=0,
            )
            targets = train["user_data"][ori_id]["y"]
            if all_data == []:
                all_data = data
            else:
                all_data = np.concatenate([all_data, data])
            if all_targets == []:
                all_targets = targets
            else:
                all_targets = np.concatenate([all_targets, targets])
            partition["data_indices"][client_cnt] = {
                "train": list(range(data_cnt, data_cnt + len(data))),
                "test": [],
            }
            stats[client_cnt]["x"] = train["num_samples"][client_cnt]
            stats[client_cnt]["y"] = Counter(targets)

            data_cnt += len(data)
            client_cnt += 1

        clients_4_train = list(range(client_cnt))

        num_samples = np.array(
            list(map(lambda stat_i: stat_i["x"], stats["train"].values()))
        )
        stats["sample per client"] = {
            "std": num_samples.mean(),
            "stddev": num_samples.std(),
        }

        # process data of test clients
        stats["test"] = {}
        for client_cnt, ori_id in enumerate(test["users"]):
            stats[client_cnt] = {"x": None, "y": None}
            data = np.stack(
                [
                    np.asarray(Image.open(raw_data_dir / img_name))
                    for img_name in test["user_data"][ori_id]["x"]
                ],
                axis=0,
            )
            targets = test["user_data"][ori_id]["y"]
            if all_data == []:
                all_data = data
            else:
                all_data = np.concatenate([all_data, data])
            if all_targets == []:
                all_targets = targets
            else:
                all_targets = np.concatenate([all_targets, targets])
            partition["data_indices"][client_cnt] = {
                "train": [],
                "test": list(range(data_cnt, data_cnt + len(data))),
            }
            stats["test"][client_cnt]["x"] = test["num_samples"][client_cnt]
            stats["test"][client_cnt]["y"] = Counter(targets)

            data_cnt += len(data)
            client_cnt += 1

        clients_4_test = list(range(len(clients_4_train), client_cnt))

        num_samples = np.array(
            list(map(lambda stat_i: stat_i["x"], stats["test"].values()))
        )
        stats["sample per client"] = {
            "std": num_samples.mean(),
            "stddev": num_samples.std(),
        }

    np.save(_DATA_ROOT / "celeba" / "data", all_data)
    np.save(_DATA_ROOT / "celeba" / "targets", all_targets)

    args.client_num_in_total = client_cnt
    partition["separation"] = {
        "train": clients_4_train,
        "test": clients_4_test,
        "total": client_cnt,
    }

    return partition, stats


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

    mean_W = np.random.normal(0, args.gamma, args.client_num_in_total)
    mean_b = mean_W
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

    all_data = []
    all_targets = []
    data_cnt = 0
    partition = {"separation": None, "data_indices": None}
    data_indices = [[] for _ in range(args.client_num_in_total)]
    stats = {}
    if args.split == "user":
        stats["train"] = {}
        stats["test"] = {}
    for client_id in range(args.client_num_in_total):

        W = np.random.normal(mean_W[client_id], 1, (args.dimension, NUM_CLASS))
        b = np.random.normal(mean_b[client_id], 1, NUM_CLASS)

        if args.iid != 0:
            W = W_global
            b = b_global

        data = np.random.multivariate_normal(
            mean_x[client_id], cov_x, samples_per_user[client_id]
        )
        targets = np.zeros(samples_per_user[client_id])

        for j in range(samples_per_user[client_id]):
            true_logit = np.dot(data[j], W) + b
            targets[j] = np.argmax(softmax(true_logit))

        all_data.append(data)
        all_targets.append(targets)

        data_indices[client_id] = list(range(data_cnt, data_cnt + len(data)))

        data_cnt += len(data)

        if args.split == "sample":
            stats[client_id] = {}
            stats[client_id]["x"] = samples_per_user[client_id]
            stats[client_id]["y"] = Counter(targets.tolist())

        else:
            if client_id < int(args.client_num_in_total * args.fraction):
                stats["train"][client_id] = {}
                stats["train"][client_id]["x"] = samples_per_user[client_id]
                stats["train"][client_id]["y"] = Counter(targets.tolist())
            else:
                stats["test"][client_id] = {}
                stats["test"][client_id]["x"] = samples_per_user[client_id]
                stats["test"][client_id]["y"] = Counter(targets.tolist())

    np.save(_DATA_ROOT / "synthetic" / "data", np.concatenate(all_data))
    np.save(_DATA_ROOT / "synthetic" / "targets", np.concatenate(all_targets))

    num_samples = np.array(list(map(lambda stat_i: stat_i["x"], stats.values())))
    stats["sample per client"] = {
        "std": num_samples.mean(),
        "stddev": num_samples.std(),
    }

    partition["data_indices"] = data_indices

    return partition, stats
