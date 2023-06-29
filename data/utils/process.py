import json
import os
from argparse import Namespace
from collections import Counter
from typing import Dict
from pathlib import Path

import numpy as np
from PIL import Image

DATA_ROOT = Path(__file__).parent.parent.absolute()


def prune_args(args: Namespace) -> Dict:
    args_dict = {}
    # general settings
    args_dict["dataset"] = args.dataset
    args_dict["client_num"] = args.client_num
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
    elif args.dataset == "domain":
        with open(DATA_ROOT / "domain" / "metadata.json", "r") as f:
            metadata = json.load(f)
            args_dict["data_amount"] = metadata["data_amount"]
            args_dict["image_size"] = metadata["image_size"]
            args_dict["class_num"] = metadata["class_num"]
            args_dict["preprocess_seed"] = metadata["seed"]
    elif args.dataset in ["femnist", "celeba"]:
        with open(DATA_ROOT / args.dataset / "preprocess_args.json") as f:
            preprocess_args = json.load(f)
        args_dict.pop("seed")
        args_dict["split"] = preprocess_args["t"]
        args_dict["fraction"] = preprocess_args["tf"]
        args_dict["sample_seed"] = preprocess_args["smplseed"]
        args_dict["split_seed"] = preprocess_args["spltseed"]
        args_dict["least_samples"] = preprocess_args["k"]
        if preprocess_args["s"] == "iid":
            args_dict["iid"] = True
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
        elif args.semantic:
            args_dict["pca_components"] = args.pca_components
            args_dict["efficient_net_type"] = args.efficient_net_type
    return args_dict


def process_femnist():
    train_dir = DATA_ROOT / "femnist" / "data" / "train"
    test_dir = DATA_ROOT / "femnist" / "data" / "test"
    stats = {}
    client_cnt = 0
    data_cnt = 0
    all_data = []
    all_targets = []
    partition = {"separation": None, "data_indices": {}}
    clients_4_train, clients_4_test = None, None
    with open(DATA_ROOT / "femnist" / "preprocess_args.json", "r") as f:
        args = json.load(f)

    # load data of train clients
    if args["t"] == "sample":
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

    np.save(DATA_ROOT / "femnist" / "data", np.concatenate(all_data))
    np.save(DATA_ROOT / "femnist" / "targets", np.concatenate(all_targets))

    partition["separation"] = {
        "train": clients_4_train,
        "test": clients_4_test,
        "total": client_cnt,
    }

    return partition, stats, client_cnt


def process_celeba():
    train_dir = DATA_ROOT / "celeba" / "data" / "train"
    test_dir = DATA_ROOT / "celeba" / "data" / "test"
    raw_data_dir = DATA_ROOT / "celeba" / "data" / "raw" / "img_align_celeba"
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

    with open(DATA_ROOT / "celeba" / "preprocess_args.json") as f:
        args = json.load(f)

    if args["t"] == "sample":
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

    np.save(DATA_ROOT / "celeba" / "data", all_data)
    np.save(DATA_ROOT / "celeba" / "targets", all_targets)

    partition["separation"] = {
        "train": clients_4_train,
        "test": clients_4_test,
        "total": client_cnt,
    }

    return partition, stats, client_cnt


def generate_synthetic_data(args):
    def softmax(x):
        ex = np.exp(x)
        sum_ex = np.sum(np.exp(x))
        return ex / sum_ex

    # All codes below are modified from https://github.com/litian96/FedProx/tree/master/data
    class_num = 10 if args.classes <= 0 else args.classes

    samples_per_user = (
        np.random.lognormal(4, 2, args.client_num).astype(int) + 50
    ).tolist()
    # samples_per_user = [10 for _ in range(args.client_num)]
    w_global = np.zeros((args.dimension, class_num))
    b_global = np.zeros(class_num)

    mean_w = np.random.normal(0, args.gamma, args.client_num)
    mean_b = mean_w
    B = np.random.normal(0, args.beta, args.client_num)
    mean_x = np.zeros((args.client_num, args.dimension))

    diagonal = np.zeros(args.dimension)
    for j in range(args.dimension):
        diagonal[j] = np.power((j + 1), -1.2)
    cov_x = np.diag(diagonal)

    for client_id in range(args.client_num):
        if args.iid:
            mean_x[client_id] = np.ones(args.dimension) * B[client_id]  # all zeros
        else:
            mean_x[client_id] = np.random.normal(B[client_id], 1, args.dimension)

    if args.iid:
        w_global = np.random.normal(0, 1, (args.dimension, class_num))
        b_global = np.random.normal(0, 1, class_num)

    all_data = []
    all_targets = []
    data_cnt = 0
    partition = {"separation": None, "data_indices": None}
    data_indices = [[] for _ in range(args.client_num)]
    stats = {}
    if args.split == "user":
        stats["train"] = {}
        stats["test"] = {}
    for client_id in range(args.client_num):
        w = np.random.normal(mean_w[client_id], 1, (args.dimension, class_num))
        b = np.random.normal(mean_b[client_id], 1, class_num)

        if args.iid != 0:
            w = w_global
            b = b_global

        data = np.random.multivariate_normal(
            mean_x[client_id], cov_x, samples_per_user[client_id]
        )
        targets = np.zeros(samples_per_user[client_id])

        for j in range(samples_per_user[client_id]):
            true_logit = np.dot(data[j], w) + b
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
            if client_id < int(args.client_num * args.fraction):
                stats["train"][client_id] = {}
                stats["train"][client_id]["x"] = samples_per_user[client_id]
                stats["train"][client_id]["y"] = Counter(targets.tolist())
            else:
                stats["test"][client_id] = {}
                stats["test"][client_id]["x"] = samples_per_user[client_id]
                stats["test"][client_id]["y"] = Counter(targets.tolist())

    np.save(DATA_ROOT / "synthetic" / "data", np.concatenate(all_data))
    np.save(DATA_ROOT / "synthetic" / "targets", np.concatenate(all_targets))

    num_samples = np.array(list(map(lambda stat_i: stat_i["x"], stats.values())))
    stats["sample per client"] = {
        "std": num_samples.mean(),
        "stddev": num_samples.std(),
    }

    partition["data_indices"] = data_indices

    return partition, stats
