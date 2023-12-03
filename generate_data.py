import json
import os
import pickle

from argparse import ArgumentParser
from pathlib import Path
import numpy as np

from src.utils.tools import fix_random_seed
from data.utils.process import (
    prune_args,
    generate_synthetic_data,
    process_celeba,
    process_femnist,
    dirichlet_4_domainnet,
)
from data.utils.schemes import (
    dirichlet,
    iid_partition,
    randomly_assign_classes,
    allocate_shards,
    semantic_partition,
)
from data.utils.datasets import DATASETS

CURRENT_DIR = Path(__file__).parent.absolute()


def main(args):
    dataset_root = CURRENT_DIR / "data" / args.dataset

    fix_random_seed(args.seed)

    if not os.path.isdir(dataset_root):
        os.mkdir(dataset_root)

    partition = {"separation": None, "data_indices": None}

    if args.dataset == "femnist":
        partition, stats, args.client_num = process_femnist()
    elif args.dataset == "celeba":
        partition, stats, args.client_num = process_celeba()
    elif args.dataset == "synthetic":
        partition, stats = generate_synthetic_data(args)
    else:  # MEDMNIST, COVID, MNIST, CIFAR10, ...
        dataset = DATASETS[args.dataset](dataset_root, args)

        if not args.iid:
            if args.dataset == "domain":
                with open(dataset_root / "original_partition.pkl", "rb") as f:
                    partition = {}
                    partition["data_indices"] = pickle.load(f)
                    partition["separation"] = None
                    args.client_num = len(partition["data_indices"])
                with open(dataset_root / "original_stats.json", "r") as f:
                    stats = json.load(f)
            elif args.alpha > 0:  # Dirichlet(alpha)
                partition, stats = dirichlet(
                    dataset=dataset,
                    client_num=args.client_num,
                    alpha=args.alpha,
                    least_samples=args.least_samples,
                )
            elif args.classes != 0:  # randomly assign classes
                args.classes = max(1, min(args.classes, len(dataset.classes)))
                partition, stats = randomly_assign_classes(
                    dataset=dataset, client_num=args.client_num, class_num=args.classes
                )
            elif args.shards > 0:  # allocate shards
                partition, stats = allocate_shards(
                    ori_dataset=dataset,
                    client_num=args.client_num,
                    shard_num=args.shards,
                )
            elif args.semantic:
                partition, stats = semantic_partition(
                    dataset=dataset,
                    efficient_net_type=args.efficient_net_type,
                    client_num=args.client_num,
                    pca_components=args.pca_components,
                    gmm_max_iter=args.gmm_max_iter,
                    gmm_init_params=args.gmm_init_params,
                    seed=args.seed,
                    use_cuda=args.use_cuda,
                )

            else:
                raise RuntimeError(
                    "Please set arbitrary one arg from [--alpha, --classes, --shards] to split the dataset."
                )

        else:  # iid partition
            partition, stats = iid_partition(
                dataset=dataset, client_num=args.client_num
            )

    if partition["separation"] is None:
        if args.split == "user":
            train_clients_num = int(args.client_num * args.fraction)
            clients_4_train = list(range(train_clients_num))
            clients_4_test = list(range(train_clients_num, args.client_num))
        elif args.split == "domain" and args.dataset == "domain":
            # we choose sketch as test domain by default
            clients_per_domain = int(args.client_num / 6)
            clients_4_train = list(range(0, 5 * clients_per_domain))
            clients_4_test = list(
                set(range(0, clients_per_domain * 6)) - set(clients_4_train)
            )
        else:
            clients_4_train = list(range(args.client_num))
            clients_4_test = list(range(args.client_num))

        partition["separation"] = {
            "train": clients_4_train,
            "test": clients_4_test,
            "total": args.client_num,
        }

    if args.dataset not in ["femnist", "celeba"]:
        for client_id, idx in enumerate(partition["data_indices"]):
            if args.split == "sample":
                num_train_samples = int(len(idx) * args.data_ratio)
                np.random.shuffle(idx)
                idx_train, idx_test = idx[:num_train_samples], idx[num_train_samples:]
                partition["data_indices"][client_id] = {
                    "train": idx_train,
                    "test": idx_test,
                }
            else:
                if client_id in clients_4_train:
                    num_train_samples = int(len(idx) * args.data_ratio)
                    partition["data_indices"][client_id] = {
                        "train": idx[:num_train_samples],
                        "test": idx[num_train_samples:],
                    }
                else:
                    partition["data_indices"][client_id] = {"train": [], "test": idx}
        # generate heterogeneous partition for DomainNet
        if args.dataset == "domain" and args.alpha > 0:
            with open(dataset_root / "domain_indices_bound.pkl", "rb") as f:
                domain_indices_bound = pickle.load(f)
            targets_numpy = np.array(dataset.targets, dtype=np.int64)
            partition, stats = dirichlet_4_domainnet(
                partition, args, clients_4_train, domain_indices_bound, targets_numpy
            )

    with open(dataset_root / "partition.pkl", "wb") as f:
        pickle.dump(partition, f)

    with open(dataset_root / "all_stats.json", "w") as f:
        json.dump(stats, f)

    with open(dataset_root / "args.json", "w") as f:
        json.dump(prune_args(args), f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=[
            "mnist",
            "cifar10",
            "cifar100",
            "synthetic",
            "femnist",
            "emnist",
            "fmnist",
            "celeba",
            "medmnistS",
            "medmnistA",
            "medmnistC",
            "covid19",
            "svhn",
            "usps",
            "tiny_imagenet",
            "cinic10",
            "domain",
        ],
        default="cifar10",
    )
    parser.add_argument("--iid", type=int, default=0)
    parser.add_argument("-cn", "--client_num", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split", type=str, choices=["sample", "user", "domain"], default="sample"
    )
    parser.add_argument("-f", "--fraction", type=float, default=0.5)
    parser.add_argument("-c", "--classes", type=int, default=0)
    parser.add_argument("-s", "--shards", type=int, default=0)
    parser.add_argument("-a", "--alpha", type=float, default=0)
    parser.add_argument("-ls", "--least_samples", type=int, default=40)
    parser.add_argument("-a", "--alpha", type=float, default=0)
    parser.add_argument(
        "-dr",
        "--data_ratio",
        type=float,
        default=0.9,
        help="data ratio for spliting the dataset on training client",
    )
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

    # For semantic partition only
    parser.add_argument("-sm", "--semantic", type=int, default=0)
    parser.add_argument("--efficient_net_type", type=int, default=0)
    parser.add_argument("--gmm_max_iter", type=int, default=100)
    parser.add_argument(
        "--gmm_init_params", type=str, choices=["random", "kmeans"], default="kmeans"
    )
    parser.add_argument("--pca_components", type=int, default=256)
    parser.add_argument("--use_cuda", type=int, default=1)
    args = parser.parse_args()
    main(args)
