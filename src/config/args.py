from argparse import ArgumentParser

# Cause the value setting deficit with bool type arguments, all bool type arguments are expressed as int.
# Non-zero value for True; 0 for False.
def get_fedavg_argparser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="lenet5",
        choices=["lenet5", "2nn", "avgcnn", "mobile", "res18", "alex", "sqz"],
    )
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-jr", "--join_ratio", type=float, default=0.1)
    parser.add_argument("-ge", "--global_epoch", type=int, default=100)
    parser.add_argument("-le", "--local_epoch", type=int, default=5)
    parser.add_argument("-fe", "--finetune_epoch", type=int, default=0)
    parser.add_argument("-tg", "--test_gap", type=int, default=100)
    parser.add_argument("-ee", "--eval_test", type=int, default=1)
    parser.add_argument("-er", "--eval_train", type=int, default=0)
    parser.add_argument("-lr", "--local_lr", type=float, default=1e-2)
    parser.add_argument("-mom", "--momentum", type=float, default=0.0)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-vg", "--verbose_gap", type=int, default=100000)
    parser.add_argument("-bs", "--batch_size", type=int, default=32)
    parser.add_argument("--use_lr_scheduler", type=int, default=0)
    parser.add_argument("--server_cuda", type=int, default=1)
    parser.add_argument("--client_cuda", type=int, default=1)
    parser.add_argument("--visible", type=int, default=0)
    parser.add_argument("--save_log", type=int, default=1)
    parser.add_argument("--save_model", type=int, default=0)
    parser.add_argument("--save_fig", type=int, default=1)
    parser.add_argument("--save_metrics", type=int, default=1)
    return parser


def get_fedavgm_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--server_momentum", type=float, default=0.9)
    return parser


def get_fedprox_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--mu", type=float, default=1.0)
    return parser


def get_fedap_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument(
        "--version", type=str, choices=["original", "f", "d"], default="original"
    )
    parser.add_argument("--pretrain_ratio", type=float, default=0.3)
    parser.add_argument("--warmup_round", type=float, default=0.5)
    parser.add_argument("--model_momentum", type=float, default=0.5)
    return parser


def get_fedfomo_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--M", type=int, default=5)
    parser.add_argument("--valset_ratio", type=float, default=0.2)
    return parser


def get_perfedavg_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--version", choices=["fo", "hf"], default="fo")
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--delta", type=float, default=1e-3)
    return parser


def get_pfedme_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lamda", type=float, default=15)
    parser.add_argument("--pers_lr", type=float, default=0.01)
    parser.add_argument("--mu", type=float, default=1e-3)
    parser.add_argument("--k", type=int, default=5)
    return parser


def get_fedrep_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--train_body_epoch", type=int, default=1)
    return parser


def get_moon_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--mu", type=float, default=5)
    return parser


def get_scaffold_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--global_lr", type=float, default=1.0)
    return parser


def get_pfedla_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--hn_lr", type=float, default=5e-3)
    parser.add_argument("--hn_momentum", type=float, default=0.0)
    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=100)
    return parser


def get_pfedhn_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument(
        "--version", type=str, choices=["pfedhn", "pfedhn_pc"], default="pfedhn"
    )
    parser.add_argument("--embed_dim", type=int, default=-1)
    parser.add_argument("--hn_lr", type=float, default=1e-2)
    parser.add_argument("--embed_lr", type=float, default=None)
    parser.add_argument("--hn_momentum", type=float, default=0.9)
    parser.add_argument("--hn_weight_decay", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--hidden_num", type=int, default=3)
    parser.add_argument("--norm_clip", type=int, default=50)
    return parser


def get_fedlc_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--tau", type=float, default=1.0)
    return parser


def get_cfl_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--eps_1", type=float, default=0.4)
    parser.add_argument("--eps_2", type=float, default=1.6)
    parser.add_argument("--min_cluster_size", type=int, default=2)
    parser.add_argument("--start_clustering_round", type=int, default=20)
    return parser


def get_feddyn_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--alpha", type=float, default=0.01)
    return parser


def get_apfl_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--adaptive_alpha", type=int, default=1)
    return parser


def get_lgfedavg_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--num_global_layers", type=int, default=1)
    return parser


def get_knnper_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--capacity", type=int, default=500)
    parser.add_argument("--weight", type=float, default=0.5)
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--k", type=int, default=5)
    return parser


def get_ditto_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--pers_epoch", type=int, default=1)
    parser.add_argument("--lamda", type=float, default=1)
    return parser


def get_fedmd_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--digest_epoch", type=int, default=1)
    parser.add_argument("--public_dataset", type=str, default="mnist")
    parser.add_argument("--public_batch_size", type=int, default=32)
    parser.add_argument("--public_batch_num", type=int, default=5)
    return parser


def get_metafed_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--valset_ratio", type=float, default=0.2)
    parser.add_argument("--warmup_epoch", type=int, default=30)
    parser.add_argument("--lamda", type=float, default=1.0)
    parser.add_argument("--threshold_1", type=float, default=0.6)
    parser.add_argument("--threshold_2", type=float, default=0.5)
    return parser
