import importlib
import json
import os
from argparse import Namespace
from collections import Counter
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from data.utils.datasets import FEMNIST, CelebA, Synthetic

DATA_ROOT = Path(__file__).parent.parent.absolute()


def prune_args(args: Namespace) -> dict:
    args_dict = {}
    # general settings
    args_dict["client_num"] = args.client_num
    args_dict["test_ratio"] = args.test_ratio
    args_dict["val_ratio"] = args.val_ratio
    args_dict["seed"] = args.seed
    args_dict["split"] = args.split
    args_dict["IID_ratio"] = args.iid
    args_dict["monitor_window_name_suffix"] = (
        f"{args.dataset}-{args.client_num}clients-{int(args.iid * 100)}%IID"
    )

    if args.dataset == "emnist":
        args_dict["emnist_split"] = args.emnist_split
    elif args.dataset == "cifar100":
        args_dict["super_class"] = bool(args.super_class)
        args_dict["monitor_window_name_suffix"] += "-use20superclasses"
    elif args.dataset == "synthetic":
        args_dict["beta"] = args.beta
        args_dict["gamma"] = args.gamma
        args_dict["dimension"] = args.dimension
        args_dict["class_num"] = 10 if args.classes <= 0 else args.classes
        args_dict["monitor_window_name_suffix"] += f"-beta{args.beta}-gamma{args.gamma}"
    elif args.dataset == "domain":
        with open(DATA_ROOT / "domain" / "metadata.json", "r") as f:
            metadata = json.load(f)
            args_dict["data_amount"] = metadata["data_amount"]
            args_dict["image_size"] = metadata["image_size"]
            args_dict["class_num"] = metadata["class_num"]
            args_dict["preprocess_seed"] = metadata["seed"]
            args_dict["monitor_window_name_suffix"] += f"-class{metadata['class_num']}"
    elif args.dataset in ["femnist", "celeba"]:
        with open(DATA_ROOT / args.dataset / "preprocess_args.json") as f:
            preprocess_args = json.load(f)
        args_dict.pop("seed")
        args_dict["split"] = preprocess_args["t"]
        args_dict["sample_seed"] = preprocess_args["smplseed"]
        args_dict["split_seed"] = preprocess_args["spltseed"]
        args_dict["min_samples_per_client"] = preprocess_args["k"]
        args_dict["test_ratio"] = 1.0 - preprocess_args["tf"]
        args_dict["val_ratio"] = 0.0
        args_dict["monitor_window_name_suffix"] = "{}-{}clients-k{}-{}".fotmat(
            args.dataset, args.client_num, preprocess_args["k"], preprocess_args["t"]
        )
        args_dict.pop("seed")
        if preprocess_args["s"] == "iid":
            args_dict["iid"] = True
            args_dict["monitor_window_name_suffix"] += f"-IID"

    if args.ood_domains is not None:
        args_dict["ood_domains"] = args.ood_domains
        args_dict["monitor_window_name_suffix"] += f"-{args.ood_domains}OODdomains"
    else:
        # Dirchlet
        if args.alpha > 0:
            args_dict["alpha"] = args.alpha
            args_dict["min_samples_per_client"] = args.min_samples_per_client
            args_dict["monitor_window_name_suffix"] += f"-Dir({args.alpha})"
        # randomly assign classes
        elif args.classes > 0:
            args_dict["classes_per_client"] = args.classes
            args_dict["monitor_window_name_suffix"] += f"-{args.classes}classes"
        # allocate shards
        elif args.shards > 0:
            args_dict["shards_per_client"] = args.shards
            args_dict["monitor_window_name_suffix"] += f"-{args.shards}shards"
        elif args.semantic:
            args_dict["pca_components"] = args.pca_components
            args_dict["efficient_net_type"] = args.efficient_net_type
            args_dict["monitor_window_name_suffix"] += f"-semantic"

    if args.dataset not in ["femnist", "celeba"]:
        args_dict["monitor_window_name_suffix"] += f"-seed{args.seed}"
    return args_dict


def process_femnist(args, partition: dict, stats: dict):
    train_dir = DATA_ROOT / "femnist" / "data" / "train"
    test_dir = DATA_ROOT / "femnist" / "data" / "test"
    client_cnt = 0
    data_cnt = 0
    all_data = []
    all_targets = []
    data_indices = {}
    clients_4_train, clients_4_test = None, None
    with open(DATA_ROOT / "femnist" / "preprocess_args.json", "r") as f:
        preprocess_args = json.load(f)

    # load data of train clients
    if preprocess_args["t"] == "sample":
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
                data_indices[client_cnt] = {
                    "train": list(range(data_cnt, data_cnt + len(train_data))),
                    "val": [],
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
        stats["samples_per_client"] = {
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
                data_indices[client_cnt] = {
                    "train": list(range(data_cnt, data_cnt + len(data))),
                    "val": [],
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
        stats["train"]["samples_per_client"] = {
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
                data_indices[client_cnt] = {
                    "train": [],
                    "val": [],
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
        stats["test"]["samples_per_client"] = {
            "std": num_samples.mean(),
            "stddev": num_samples.std(),
        }

    np.save(DATA_ROOT / "femnist" / "data", np.concatenate(all_data))
    np.save(DATA_ROOT / "femnist" / "targets", np.concatenate(all_targets))

    partition["separation"] = {
        "train": clients_4_train,
        "val": [],
        "test": clients_4_test,
        "total": client_cnt,
    }
    partition["data_indices"] = [indices for indices in data_indices.values()]
    args.client_num = client_cnt
    return FEMNIST(
        root=DATA_ROOT / "femnist",
        args=None,
        test_data_transform=None,
        test_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    )


def process_celeba(args, partition: dict, stats: dict):
    train_dir = DATA_ROOT / "celeba" / "data" / "train"
    test_dir = DATA_ROOT / "celeba" / "data" / "test"
    raw_data_dir = DATA_ROOT / "celeba" / "data" / "raw" / "img_align_celeba"
    train_filename = os.listdir(train_dir)[0]
    test_filename = os.listdir(test_dir)[0]
    with open(train_dir / train_filename, "r") as f:
        train = json.load(f)
    with open(test_dir / test_filename, "r") as f:
        test = json.load(f)

    data_cnt = 0
    all_data = []
    all_targets = []
    data_indices = {}
    client_cnt = 0
    clients_4_test, clients_4_train = None, None

    with open(DATA_ROOT / "celeba" / "preprocess_args.json") as f:
        preprocess_args = json.load(f)

    if preprocess_args["t"] == "sample":
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
            data_indices[client_cnt] = {
                "train": list(range(data_cnt, data_cnt + len(train_data))),
                "val": [],
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
        stats["samples_per_client"] = {
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
            data_indices[client_cnt] = {
                "train": list(range(data_cnt, data_cnt + len(data))),
                "val": [],
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
        stats["samples_per_client"] = {
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
                "val": [],
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
        stats["samples_per_client"] = {
            "std": num_samples.mean().item(),
            "stddev": num_samples.std().item(),
        }

    np.save(DATA_ROOT / "celeba" / "data", all_data)
    np.save(DATA_ROOT / "celeba" / "targets", all_targets)

    partition["separation"] = {
        "train": clients_4_train,
        "val": [],
        "test": clients_4_test,
        "total": client_cnt,
    }
    partition["data_indices"] = [indices for indices in data_indices.values()]
    args.client_num = client_cnt
    return CelebA(
        root=DATA_ROOT / "celeba",
        args=None,
        test_data_transform=None,
        test_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    )


def generate_synthetic_data(args, partition: dict, stats: dict):
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

    for client_id in range(args.client_num):
        w = np.random.normal(mean_w[client_id], 1, (args.dimension, class_num))
        b = np.random.normal(mean_b[client_id], 1, class_num)

        if args.iid != 0:
            w = w_global
            b = b_global

        data = np.random.multivariate_normal(
            mean_x[client_id], cov_x, samples_per_user[client_id]
        )
        targets = np.zeros(samples_per_user[client_id], dtype=np.int32)

        for j in range(samples_per_user[client_id]):
            true_logit = np.dot(data[j], w) + b
            targets[j] = np.argmax(softmax(true_logit))

        all_data.append(data)
        all_targets.append(targets)

        partition["data_indices"][client_id] = list(
            range(data_cnt, data_cnt + len(data))
        )

        data_cnt += len(data)

        stats[client_id] = {}
        stats[client_id]["x"] = samples_per_user[client_id]
        stats[client_id]["y"] = Counter(targets.tolist())

    np.save(DATA_ROOT / "synthetic" / "data", np.concatenate(all_data))
    np.save(DATA_ROOT / "synthetic" / "targets", np.concatenate(all_targets))

    num_samples = np.array(list(map(lambda stat_i: stat_i["x"], stats.values())))
    stats["samples_per_client"] = {
        "std": num_samples.mean().item(),
        "stddev": num_samples.std().item(),
    }
    return Synthetic(root=DATA_ROOT / "synthetic")


def exclude_domain(
    client_num: int,
    targets: np.ndarray,
    domain_map: dict[str, int],
    domain_indices_bound: dict,
    ood_domains: set[str],
    partition: dict,
    stats: dict,
):
    ood_domain_num = 0
    data_indices = np.arange(len(targets), dtype=np.int64)
    for domain in ood_domains:
        if domain not in domain_map:
            Warning(f"One of `args.ood_domains` {domain} is unrecongnized and ignored.")
        else:
            ood_domain_num += 1

    def _idx_2_domain_label(index):
        # get the domain label of a specific data index
        for domain, bound in domain_indices_bound.items():
            if bound["begin"] <= index < bound["end"]:
                return domain_map[domain]

    domain_targets = np.vectorize(_idx_2_domain_label)(data_indices)

    id_label_set = set(
        label for domain, label in domain_map.items() if domain not in ood_domains
    )

    train_clients = list(range(client_num - ood_domain_num))
    ood_clients = list(range(client_num - ood_domain_num, client_num))
    partition["separation"] = {
        "train": train_clients,
        "val": [],
        "test": ood_clients,
        "total": client_num,
    }
    for ood_domain, client_id in zip(ood_domains, ood_clients):
        indices = np.where(domain_targets == domain_map[ood_domain])[0]
        partition["data_indices"][client_id] = {
            "train": np.array([], dtype=np.int64),
            "val": np.array([], dtype=np.int64),
            "test": indices,
        }
        stats[client_id] = {
            "x": len(indices),
            "y": {domain_map[ood_domain]: len(indices)},
        }

    return id_label_set, domain_targets, len(train_clients)


def plot_distribution(client_num: int, label_counts: np.ndarray, save_path: str):
    plt.figure()
    ax = plt.gca()
    left = np.zeros(client_num)
    client_ids = np.arange(client_num)
    for y, cnts in enumerate(label_counts):
        ax.barh(client_ids, width=cnts, label=y, left=left)
        left += cnts
    ax.set_yticks([])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.legend(bbox_to_anchor=(1.2, 1))
    plt.savefig(save_path, bbox_inches="tight")


def class_from_string(class_string: str) -> type:
    """Dynamically loads a class from a string representation.

    Args:
        class_string (str): The string representation of the class, including the module path.

    Returns:
        type: The loaded class.

    Example:
        class_from_string('path.to.module.ClassName') returns the class 'ClassName' from the module 'path.to.module'.
    """
    module = importlib.import_module('.'.join(class_string.split('.')[:-1]))
    class_ = getattr(module, class_string.split('.')[-1])
    return class_


def partitioner_class_from_flwr_datasets(flower_partitioner_class: str):
    """Dynamically loads a partitioner class from a string representation of a
    flwr.datasets.Dataset.

    Args:
        flower_partitioner_class (str): The string representation of the partitioner class, including the module path.

    Returns:
        type: The loaded partitioner class.
    """
    return class_from_string(f"flwr_datasets.partitioner.{flower_partitioner_class}")
