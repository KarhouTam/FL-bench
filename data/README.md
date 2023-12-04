# Generating Federated dataset

## Download Datasets 🎨

This benchmark also integrates [*LEAF*](https://github.com/TalwalkarLab/leaf), and supports *FEMNIST*, *CelebA*. For these datasets, this benchmark does not partition them further.

Most of the datasets supported by this benchmark are integrated into `torchvision.datasets`, expect *Tiny-ImageNet-200*, *Covid-19*, *Organ-S/A/CMNIST*, *DomainNet*. 

For those datasets, I prepare download scripts (at [`data/download`](https://github.com/KarhouTam/FL-bench/blob/master/data/download)) for you. 🤗

e.g.

```shell
cd data/download
sh tiny_imagenet.sh
```

## Generic Arguments 🔧
📢 All arguments have their default value.
| Arguments for general datasets | Description                                                                                                                                                                                                                                                      |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--dataset`, `-d`              | The name of dataset.                                                                                                                                                                                                                                             |
| `--iid`                        | Non-zero value for randomly partitioning data and disabling all other Non-IID partition methods.                                                                                                                                                                 |
| `--client_num`, `-cn`          | The number of clients.                                                                                                                                                                                                                                           |
| `--split`                      | Chooses from `[sample, user, domain]`.  `user`: partition clients into train-test groups; `sample`: partition each client's data samples into train-test groups; `domain`: choose sketch as test domain by default (only available if the dataset is DomainNet). |
| `--fraction`, `-f`             | Propotion of training clients that depends on `--split`. *Note that this argument is unused for FEMNIST and CelebA, which already split clients' dataset when you run their `preprocess.sh` according to its argument `--tf`.*                                   |
| `--data_ratio`, `-dr`          | Proportion of training data on training clients.                                                                                                                                                                                                                 |

⭐ For *CIFAR-100* specifically, this benchmark supports partitioning it into the superclass category (*CIFAR-100*'s 100 classes can also be classified into 20 superclasses) by setting `--super_class` to non-zero.

## Partitioning Schemes 🌌

This benckmark offers 5 partitioning schemes.

- ***Dirichlet***: Refers to [Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification (*FedAvgM*)](https://arxiv.org/abs/1909.06335). Dataset would be splitted according to $Dir(\alpha)$. Smaller $\alpha$ means stronger label heterogeneity.
  - `--alpha` or `-a`: The parameter for controlling intensity of label heterogeneity.
  - `--least_samples` or `-ls`: The parameter for defining the minimum number of samples each client would be distributed. *A small `--least_samples` along with small `--alpha` or big `--client_num` might considerablely prolong the partition.*

- ***Shards***: Refers to [Communication-Efficient Learning of Deep Networks from Decentralized Data (*FedAvg*)](https://arxiv.org/abs/1602.05629). The whole dataset would be evenly splitted into many equal-size shards.
  - `--shards` or `-s`: Number of data shards that each client holds. The same partition method as in *FedAvg*.

- ***Randomly Assigning Classes***: Each client would be allocated data that belongs to `-c` classes. And classes for each client are randomly choosed.
  - `--classes` or `-c`: Number of classes that each client's data belong to.

- ***Synthetic***: Refers to [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127)
. The whole dataset are generated according to $(\alpha, \beta)$. Check the paper for all details.
  - [`--gamma`, `--beta`]: The parameters $(\alpha, \beta)$ in paper respectively. 
  - `--dimension`: The dimension of synthetic data.
  - `--iid`: Non-zero value for generating IID synthetic dataset.
- ***Semantic Partition***: Refers to [What Do We Mean by Generalization in Federated Learning?](https://arxiv.org/abs/2110.14216). Each client's data are correspond to a gaussian distribution that generated by a gaussian mixture model. You can learn the whole process precedure in paper's Appendix D.
  - `--semantic` or `-sm`: Non-zero value for performing semantic partition.
  - `--efficient_net_type`: The type of EfficientNet for computing the embeddings of data.
  - `--pca_components`: The number of dimension for PCA decomposition.
  - `--gmm_max_iter`: The maximum number of fitting iteration of the gaussian mixture model.
  - `--gmm_init_params`: The way for initializing gaussian mixture model (`kmeans` / `random`).
  - `--use_cuda`: Non-zero value for using CUDA to accelerate the computation. 

## Usage 🚀

e.g.

```shell
python generate_data.py -d cifar10 -a 0.1 -cn 100
```

The command above splits the *CIFAR-10* dataset into 100 subsets (for 100 clients) according to $Dir(0.1)$.
### The LEAF 🍂 
You should set all arguments well already when running `preprocess.sh`. 

All generic arguments in `generate_data.py` will be deactivated when processing LEAF datasets, except `-d` that used for specifying dataset. 

When processing LEAF datasets, `generate_data.py` only responsible for translating the output of `preprocess.sh` (data files in .json format) into a single `data.npy` and `targets.npy`.

## Guideline for Processing DomainNet 🧾
See more details in [Processing DomainNet](/data/domain/README.md).


## Acknowledgement 🤗

[`data/femnist`](https://github.com/KarhouTam/FL-bench/tree/master/data/femnist), [`data/celeba`](https://github.com/KarhouTam/FL-bench/tree/master/data/celeba), [`data/leaf_utils`](https://github.com/KarhouTam/FL-bench/tree/master/data/leaf_utils) are copied from [*LEAF*](https://github.com/TalwalkarLab/leaf) with subtle modifications to be integrated into this benchmark. [`data/femnist/README.md`](https://github.com/KarhouTam/FL-bench/tree/master/data/femnist#readme) and [`data/celeba/README.md`](https://github.com/KarhouTam/FL-bench/tree/master/data/celeba#readme) for full details.

We are not use the test set of *Tiny-ImageNet-200* because it is unlabeled.

