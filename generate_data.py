import hashlib
import json
import os
import pickle
import random
from argparse import ArgumentParser
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np

from data.utils.datasets import DATASETS, BaseDataset
from data.utils.process import (
    exclude_domain,
    generate_synthetic_data,
    plot_distribution,
    process_celeba,
    process_femnist,
    prune_args,
)
from data.utils.schemes import (
    allocate_shards,
    dirichlet,
    iid_partition,
    randomly_assign_classes,
    semantic_partition,
)
from data.utils.schemes.flower import flower_partition
from src.utils.functional import fix_random_seed

CURRENT_DIR = Path(__file__).parent.absolute()


def main(args):
    dataset_root = CURRENT_DIR / "data" / args.dataset

    fix_random_seed(args.seed, args.use_cuda)

    if not os.path.isdir(dataset_root):
        os.mkdir(dataset_root)

    if os.path.isfile(dataset_root / "partition_md5.txt"):
        with open(dataset_root / "partition_md5.txt", "r") as f:
            md5 = f.read()
            if md5 == hashlib.md5(json.dumps(args.__dict__).encode()).hexdigest():
                print("Partition file already exists. Skip partitioning.")
                return

    client_num = args.client_num
    partition = {"separation": None, "data_indices": [[] for _ in range(client_num)]}
    # x: num of samples,
    # y: label distribution
    stats = {i: {"x": 0, "y": {}} for i in range(args.client_num)}
    dataset: BaseDataset = None

    if args.dataset == "femnist":
        dataset = process_femnist(args, partition, stats)
        partition["val"] = []
    elif args.dataset == "celeba":
        dataset = process_celeba(args, partition, stats)
        partition["val"] = []
    elif args.dataset == "synthetic":
        dataset = generate_synthetic_data(args, partition, stats)
    else:  # MEDMNIST, COVID, MNIST, CIFAR10, ...
        # NOTE: If `args.ood_domains`` is not empty, then FL-bench will map all labels (class space) to the domain space
        # and partition data according to the new `targets` array.
        dataset = DATASETS[args.dataset](dataset_root, args)
        targets = dataset.targets.numpy().astype(np.int32)
        target_indices = np.arange(len(targets), dtype=np.int32)
        valid_label_set = set(range(len(dataset.classes)))
        if args.dataset in ["domain"] and args.ood_domains:
            metadata = json.load(open(dataset_root / "metadata.json", "r"))
            valid_label_set, targets, client_num = exclude_domain(
                client_num=client_num,
                domain_map=metadata["domain_map"],
                targets=targets,
                domain_indices_bound=metadata["domain_indices_bound"],
                ood_domains=set(args.ood_domains),
                partition=partition,
                stats=stats,
            )

        iid_data_partition = deepcopy(partition)
        iid_stats = deepcopy(stats)
        if 0 < args.iid <= 1.0:  # iid partition
            sampled_indices = np.array(
                random.sample(
                    target_indices.tolist(), int(len(target_indices) * args.iid)
                )
            )

            # if args.iid < 1.0, then residual indices will be processed by another partition method
            target_indices = np.array(
                list(set(target_indices) - set(sampled_indices)), dtype=np.int32
            )

            iid_partition(
                targets=targets[sampled_indices],
                target_indices=sampled_indices,
                label_set=valid_label_set,
                client_num=client_num,
                partition=iid_data_partition,
                stats=iid_stats,
            )

        if len(target_indices) > 0:
            if args.alpha > 0:  # Dirichlet(alpha)
                dirichlet(
                    targets=targets[target_indices],
                    target_indices=target_indices,
                    label_set=valid_label_set,
                    client_num=client_num,
                    alpha=args.alpha,
                    min_samples_per_client=args.min_samples_per_client,
                    partition=partition,
                    stats=stats,
                )
            elif args.classes != 0:  # randomly assign classes
                args.classes = max(1, min(args.classes, len(dataset.classes)))
                randomly_assign_classes(
                    targets=targets[target_indices],
                    target_indices=target_indices,
                    label_set=valid_label_set,
                    client_num=client_num,
                    class_num=args.classes,
                    partition=partition,
                    stats=stats,
                )
            elif args.shards > 0:  # allocate shards
                allocate_shards(
                    targets=targets[target_indices],
                    target_indices=target_indices,
                    label_set=valid_label_set,
                    client_num=client_num,
                    shard_num=args.shards,
                    partition=partition,
                    stats=stats,
                )
            elif args.semantic:
                semantic_partition(
                    dataset=dataset,
                    targets=targets[target_indices],
                    target_indices=target_indices,
                    label_set=valid_label_set,
                    efficient_net_type=args.efficient_net_type,
                    client_num=client_num,
                    pca_components=args.pca_components,
                    gmm_max_iter=args.gmm_max_iter,
                    gmm_init_params=args.gmm_init_params,
                    seed=args.seed,
                    use_cuda=args.use_cuda,
                    partition=partition,
                    stats=stats,
                )
            elif args.flower_partitioner_class != "":
                flower_partition(
                    targets=targets[target_indices],
                    target_indices=target_indices,
                    label_set=valid_label_set,
                    client_num=client_num,
                    flower_partitioner_class=args.flower_partitioner_class,
                    flower_partitioner_kwargs=args.flower_partitioner_kwargs,
                    partition=partition,
                    stats=stats,
                )
            elif args.dataset in ["domain"] and args.ood_domains is None:
                with open(dataset_root / "original_partition.pkl", "rb") as f:
                    partition = {}
                    partition["data_indices"] = pickle.load(f)
                    partition["separation"] = None
                    args.client_num = len(partition["data_indices"])
            else:
                raise RuntimeError(
                    "Part of data indices are ignored. Please set arbitrary one arg from"
                    " [--alpha, --classes, --shards, --semantic] for partitioning."
                )

    # merge the iid and niid partition results
    if 0 < args.iid <= 1.0:
        num_samples = []
        for i in range(args.client_num):
            partition["data_indices"][i] = np.concatenate(
                [partition["data_indices"][i], iid_data_partition["data_indices"][i]]
            ).astype(np.int32)
            stats[i]["x"] += iid_stats[i]["x"]
            stats[i]["y"] = {
                cls: stats[i]["y"].get(cls, 0) + iid_stats[i]["y"].get(cls, 0)
                for cls in dataset.classes
            }
            num_samples.append(stats[i]["x"])
        num_samples = np.array(num_samples)
        stats["samples_per_client"] = {
            "mean": num_samples.mean().item(),
            "stddev": num_samples.std().item(),
        }

    if partition["separation"] is None:
        if args.split == "user":
            test_clients_num = int(args.client_num * args.test_ratio)
            val_clients_num = int(args.client_num * args.val_ratio)
            train_clients_num = args.client_num - test_clients_num - val_clients_num
            clients_4_train = list(range(train_clients_num))
            clients_4_val = list(
                range(train_clients_num, train_clients_num + val_clients_num)
            )
            clients_4_test = list(
                range(train_clients_num + val_clients_num, args.client_num)
            )

        elif args.split == "sample":
            clients_4_train = list(range(args.client_num))
            clients_4_val = clients_4_train
            clients_4_test = clients_4_train

        partition["separation"] = {
            "train": clients_4_train,
            "val": clients_4_val,
            "test": clients_4_test,
            "total": args.client_num,
        }

    if args.dataset not in ["femnist", "celeba"]:
        if args.split == "sample":
            for client_id in partition["separation"]["train"]:
                indices = partition["data_indices"][client_id]
                np.random.shuffle(indices)
                testset_size = int(len(indices) * args.test_ratio)
                valset_size = int(len(indices) * args.val_ratio)
                trainset, valset, testset = (
                    indices[testset_size + valset_size :],
                    indices[testset_size : testset_size + valset_size],
                    indices[:testset_size],
                )
                partition["data_indices"][client_id] = {
                    "train": trainset,
                    "val": valset,
                    "test": testset,
                }
        elif args.split == "user":
            for client_id in partition["separation"]["train"]:
                indices = partition["data_indices"][client_id]
                partition["data_indices"][client_id] = {
                    "train": indices,
                    "val": np.array([], dtype=np.int64),
                    "test": np.array([], dtype=np.int64),
                }

            for client_id in partition["separation"]["val"]:
                indices = partition["data_indices"][client_id]
                partition["data_indices"][client_id] = {
                    "train": np.array([], dtype=np.int64),
                    "val": indices,
                    "test": np.array([], dtype=np.int64),
                }

            for client_id in partition["separation"]["test"]:
                indices = partition["data_indices"][client_id]
                partition["data_indices"][client_id] = {
                    "train": np.array([], dtype=np.int64),
                    "val": np.array([], dtype=np.int64),
                    "test": indices,
                }

    if args.dataset in ["domain"]:
        class_targets = np.array(dataset.targets, dtype=np.int32)
        metadata = json.load(open(dataset_root / "metadata.json", "r"))

        def _idx_2_domain_label(index):
            for domain, bound in metadata["domain_indices_bound"].items():
                if bound["begin"] <= index < bound["end"]:
                    return metadata["domain_map"][domain]

        domain_targets = np.vectorize(_idx_2_domain_label)(
            np.arange(len(class_targets), dtype=np.int64)
        )
        for client_id in range(args.client_num):
            indices = np.concatenate(
                [
                    partition["data_indices"][client_id]["train"],
                    partition["data_indices"][client_id]["val"],
                    partition["data_indices"][client_id]["test"],
                ]
            ).astype(np.int64)
            stats[client_id] = {
                "x": len(indices),
                "class space": Counter(class_targets[indices].tolist()),
                "domain space": Counter(domain_targets[indices].tolist()),
            }
        stats["domain_map"] = metadata["domain_map"]

    # plot
    if args.plot_distribution:
        if args.dataset in ["domain"]:
            # class distribution
            counts = np.zeros((len(dataset.classes), args.client_num), dtype=np.int64)
            client_ids = range(args.client_num)
            for i, client_id in enumerate(client_ids):
                for j, cnt in stats[client_id]["class space"].items():
                    counts[j][i] = cnt
            plot_distribution(
                client_num=args.client_num,
                label_counts=counts,
                save_path=f"{dataset_root}/class_distribution.png",
            )
            # domain distribution
            counts = np.zeros(
                (len(metadata["domain_map"]), args.client_num), dtype=np.int64
            )
            client_ids = range(args.client_num)
            for i, client_id in enumerate(client_ids):
                for j, cnt in stats[client_id]["domain space"].items():
                    counts[j][i] = cnt
            plot_distribution(
                client_num=args.client_num,
                label_counts=counts,
                save_path=f"{dataset_root}/domain_distribution.png",
            )

        else:
            counts = np.zeros((len(dataset.classes), args.client_num), dtype=np.int64)
            client_ids = range(args.client_num)
            for i, client_id in enumerate(client_ids):
                for j, cnt in stats[client_id]["y"].items():
                    counts[j][i] = cnt
            plot_distribution(
                client_num=args.client_num,
                label_counts=counts,
                save_path=f"{dataset_root}/class_distribution.png",
            )

    with open(dataset_root / "partition.pkl", "wb") as f:
        pickle.dump(partition, f)

    with open(dataset_root / "all_stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    with open(dataset_root / "args.json", "w") as f:
        json.dump(prune_args(args), f, indent=4)

    with open(dataset_root / "partition_md5.txt", "w") as f:
        f.write(hashlib.md5(json.dumps(args.__dict__).encode()).hexdigest())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-d", "--dataset", type=str, choices=DATASETS.keys(), required=True
    )
    parser.add_argument("--iid", type=float, default=0.0)
    parser.add_argument("-cn", "--client_num", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "-sp", "--split", type=str, choices=["sample", "user"], default="sample"
    )
    parser.add_argument("-vr", "--val_ratio", type=float, default=0.0)
    parser.add_argument("-tr", "--test_ratio", type=float, default=0.25)
    parser.add_argument("-pd", "--plot_distribution", type=int, default=1)

    # Randomly assign classes
    parser.add_argument("-c", "--classes", type=int, default=0)

    # Shards
    parser.add_argument("-s", "--shards", type=int, default=0)

    # Dirichlet
    parser.add_argument("-a", "--alpha", type=float, default=0)
    parser.add_argument("-ms", "--min_samples_per_client", type=int, default=10)

    # Flower partitioner
    parser.add_argument("-fpc", "--flower_partitioner_class", type=str, default="")
    parser.add_argument("-fpk", "--flower_partitioner_kwargs", type=str, default="{}")

    # For synthetic data only
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--dimension", type=int, default=60)

    # For CIFAR-100 only
    parser.add_argument("--super_class", type=int, default=0)

    # For EMNIST only
    parser.add_argument(
        "--emnist_split",
        type=str,
        choices=["byclass", "bymerge", "letters", "balanced", "digits", "mnist"],
        default="byclass",
    )

    # For domain generalization datasets only
    parser.add_argument("--ood_domains", nargs="+", default=None)

    # For semantic partition only
    parser.add_argument("-sm", "--semantic", type=int, default=0)
    parser.add_argument("--efficient_net_type", type=int, default=7)
    parser.add_argument("--gmm_max_iter", type=int, default=100)
    parser.add_argument(
        "--gmm_init_params", type=str, choices=["random", "kmeans"], default="random"
    )
    parser.add_argument("--pca_components", type=Optional[int], default=None)
    parser.add_argument("--use_cuda", type=int, default=1)
    args = parser.parse_args()
    main(args)
