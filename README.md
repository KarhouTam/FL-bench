```


                             ______ __               __                        __  
                            / ____// /              / /_   ___   ____   _____ / /_ 
                           / /_   / /     ______   / __ \ / _ \ / __ \ / ___// __ \
                          / __/  / /___  /_____/  / /_/ //  __// / / // /__ / / / /
                         /_/    /_____/          /_____/ \___//_/ /_/ \___//_/ /_/


```

<p align="center">
  <a href="https://github.com/KarhouTam/FL-bench/blob/master/LICENSE">
    <img alt="GitHub License" src="https://img.shields.io/github/license/KarhouTam/FL-bench?style=for-the-badge&logo=github&color=8386e0"/>
  </a>
  <a href="https://github.com/KarhouTam/FL-bench/issues?q=is%3Aissue+is%3Aclosed">
    <img alt="GitHub closed issues" src="https://img.shields.io/github/issues-closed-raw/KarhouTam/FL-bench?style=for-the-badge&logo=github&color=8386e0">
  </a>
  <a href="https://github.com/KarhouTam/FL-bench/stargazers">
    <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/KarhouTam/FL-bench?style=for-the-badge&logo=github&color=8386e0">
  </a>
  <a href="https://github.com/KarhouTam/FL-bench/forks">
    <img alt="GitHub Repo forks" src="https://img.shields.io/github/forks/KarhouTam/FL-bench?style=for-the-badge&logo=github&color=8386e0">
  </a>
</p>
<h4 align="center"><i>This is a benchmark for evaluating well-known traditional, personalized and domain generalization federated learning methods. This benchmark is straightforward and easy to extend.</i></h4>

## Methods 🧬

### Traditional FL Methods

- ***FedAvg*** -- [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) (AISTATS'17)

- ***FedAvgM*** -- [Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification](https://arxiv.org/abs/1909.06335) (ArXiv'19)

- ***FedProx*** -- [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127) (MLSys'20)

- ***SCAFFOLD*** -- [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](https://arxiv.org/abs/1910.06378) (ICML'20)

- ***MOON*** -- [Model-Contrastive Federated Learning](http://arxiv.org/abs/2103.16257) (CVPR'21)
  
- ***FedDyn*** -- [Federated Learning Based on Dynamic Regularization](http://arxiv.org/abs/2111.04263) (ICLR'21)

- ***FedLC*** -- [Federated Learning with Label Distribution Skew via Logits Calibration](http://arxiv.org/abs/2209.00189) (ICML'22)

- ***FedGen*** -- [Data-Free Knowledge Distillation for Heterogeneous Federated Learning](https://arxiv.org/abs/2105.10056) (ICML'21)

- ***CCVR*** -- [No Fear of Heterogeneity: Classifier Calibration for Federated Learning with Non-IID Data](https://arxiv.org/abs/2106.05001) (NIPS'21)

- ***FedOpt*** -- [Adaptive Federated Optimization](https://arxiv.org/abs/2003.00295) (ICLR'21)

- ***Elastic Aggregation*** -- [Elastic Aggregation for Federated Optimization](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Elastic_Aggregation_for_Federated_Optimization_CVPR_2023_paper.html) (CVPR'23)

### Personalized FL Methods

- ***pFedSim (My Work⭐)*** -- [pFedSim: Similarity-Aware Model Aggregation Towards Personalized Federated Learning](https://arxiv.org/abs/2305.15706) (ArXiv'23)

- ***Local-Only*** -- Local training only (without communication).

- ***FedMD*** -- [FedMD: Heterogenous Federated Learning via Model Distillation](http://arxiv.org/abs/1910.03581) (NIPS'19)

- ***APFL*** -- [Adaptive Personalized Federated Learning](http://arxiv.org/abs/2003.13461) (ArXiv'20)

- ***LG-FedAvg*** -- [Think Locally, Act Globally: Federated Learning with Local and Global Representations](https://arxiv.org/abs/2001.01523) (ArXiv'20)

- ***FedBN*** -- [FedBN: Federated Learning On Non-IID Features Via Local Batch Normalization](http://arxiv.org/abs/2102.07623) (ICLR'21)

- ***FedPer*** -- [Federated Learning with Personalization Layers](http://arxiv.org/abs/1912.00818) (AISTATS'20)

- ***FedRep*** -- [Exploiting Shared Representations for Personalized Federated Learning](http://arxiv.org/abs/2102.07078) (ICML'21)

- ***Per-FedAvg*** -- [Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach](https://proceedings.neurips.cc/paper/2020/hash/24389bfe4fe2eba8bf9aa9203a44cdad-Abstract.html) (NIPS'20)

- ***pFedMe*** -- [Personalized Federated Learning with Moreau Envelopes](http://arxiv.org/abs/2006.08848) (NIPS'20)

- ***Ditto*** -- [Ditto: Fair and Robust Federated Learning Through Personalization](http://arxiv.org/abs/2012.04221) (ICML'21)

- ***pFedHN*** -- [Personalized Federated Learning using Hypernetworks](http://arxiv.org/abs/2103.04628) (ICML'21)
  
- ***pFedLA*** -- [Layer-Wised Model Aggregation for Personalized Federated Learning](https://openaccess.thecvf.com/content/CVPR2022/html/Ma_Layer-Wised_Model_Aggregation_for_Personalized_Federated_Learning_CVPR_2022_paper.html) (CVPR'22)

- ***CFL*** -- [Clustered Federated Learning: Model-Agnostic Distributed Multi-Task Optimization under Privacy Constraints](https://arxiv.org/abs/1910.01991) (ArXiv'19)

- ***FedFomo*** -- [Personalized Federated Learning with First Order Model Optimization](http://arxiv.org/abs/2012.08565) (ICLR'21)

- ***FedBabu*** -- [FedBabu: Towards Enhanced Representation for Federated Image Classification](https://arxiv.org/abs/2106.06042) (ICLR'22)

- ***FedAP*** -- [Personalized Federated Learning with Adaptive Batchnorm for Healthcare](https://arxiv.org/abs/2112.00734) (IEEE'22)

- ***kNN-Per*** -- [Personalized Federated Learning through Local Memorization](http://arxiv.org/abs/2111.09360) (ICML'22)

- ***MetaFed*** -- [MetaFed: Federated Learning among Federations with Cyclic Knowledge Distillation for Personalized Healthcare](http://arxiv.org/abs/2206.08516) (IJCAI'22)

- ***FedRoD*** -- [On Bridging Generic and Personalized Federated Learning for Image Classification](https://arxiv.org/abs/2107.00778) (ICLR'22)

### FL Domain Generalization Methods

- ***FedSR*** -- [FedSR: A Simple and Effective Domain Generalization Method for Federated Learning](https://openreview.net/forum?id=mrt90D00aQX) (NIPS'22)
  
- ***ADCOL*** -- [Adversarial Collaborative Learning on Non-IID Features](https://proceedings.mlr.press/v202/li23j.html) (ICML'23)
  
- ***FedIIR*** -- [Out-of-Distribution Generalization of Federated Learning via Implicit Invariant Relationships](https://openreview.net/pdf?id=JC05k0E2EM) (ICML'23)

## Environment Preparation 🧩

### PyPI 🐍
```sh
pip install -r .environment/requirements.txt
```

### Conda 💻
```sh
conda env create -f .environment/environment.yml
```

### Poetry 🎶

```sh
# For those China mainland users
cd .environment && poetry install --no-root

# For those oversea users
cd .environment && sed -i "10,14d" pyproject.toml && poetry lock --no-update && poetry install --no-root
```

### Docker 🐳

```shell
# For those China mainland users
docker pull registry.cn-hangzhou.aliyuncs.com/karhoutam/fl-bench:master

# For those oversea users
docker pull ghcr.io/karhoutam/fl-bench:master
or
docker pull docker.io/karhoutam/fl-bench:master

# An example for building container
docker run -it --name fl-bench --privileged -p 8097:8097 --gpus all ghcr.io/karhoutam/fl-bench:master
```


## Easy Run 🏃‍♂️

ALL classes of methods are inherited from `FedAvgServer` and `FedAvgClient`. If you wanna figure out the entire workflow and detail of variable settings, go check [`src/server/fedavg.py`](src/server/fedavg.py) and [`src/client/fedavg.py`](src/client/fedavg.py).


```shell
# partition the CIFAR-10 according to Dir(0.1) for 100 clients
python generate_data.py -d cifar10 -a 0.1 -cn 100

# run FedAvg on CIFAR-10 with default settings.
# Use main.py like python main.py <method> [args ...]
# ❗ Method name should be identical to the `.py` file name in `src/server`.
python main.py fedavg -d cifar10
```

About methods of generating federated dastaset, go check [`data/README.md`](data/#readme) for full details.


### Monitor 📈 (recommended 👍)
1. Run `python -m visdom.server` on terminal.
2. Run `python main.py <method> --visible 1`
3. Go check `localhost:8097` on your browser.
## Generic Arguments 🔧

📢 All generic arguments have their default value. Go check `get_fedavg_argparser()` in [`FL-bench/src/server/fedavg.py`](src/server/fedavg.py) for full details of generic arguments. 

You can also write your own `.yaml` config file. I offer you a [template](config/template.yaml) in `config` and recommend you to save your config files there also. 

One example: `python main.py fedavg -cfg config/template.yaml`

About the default values and hyperparameters of advanced FL methods, go check corresponding `FL-bench/src/server/<method>.py` for full details.
| Argument                       | Description                                                                                                                                                                                                                                                                                                                               |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--dataset`                    | The name of dataset that experiment run on.                                                                                                                                                                                                                                                                                               |
| `--model`                      | The model backbone experiment used.                                                                                                                                                                                                                                                                                                       |
| `--seed`                       | Random seed for running experiment.                                                                                                                                                                                                                                                                                                       |
| `--join_ratio`                 | Ratio for (client each round) / (client num in total).                                                                                                                                                                                                                                                                                    |
| `--global_epoch`               | Global epoch, also called communication round.                                                                                                                                                                                                                                                                                            |
| `--local_epoch`                | Local epoch for client local training.                                                                                                                                                                                                                                                                                                    |
| `--finetune_epoch`             | Epoch for clients fine-tunning their models before test.                                                                                                                                                                                                                                                                                  |
| `--test_gap`                   | Interval round of performing test on clients.                                                                                                                                                                                                                                                                                             |
| `--eval_test`                  | Non-zero value for performing evaluation on joined clients' testset before and after local training.                                                                                                                                                                                                                                      |
| `--eval_val`                  | Non-zero value for performing evaluation on joined clients' valset before and after local training.                                                                                                                                                                                                                                      |
| `--eval_train`                 | Non-zero value for performing evaluation on joined clients' trainset before and after local training.                                                                                                                                                                                                                                     |
| `-op, --optimizer` | Client local optimizer, selected from `[sgd, adam]` |
| `--local_lr`                   | Learning rate for client local training.                                                                                                                                                                                                                                                                                                  |
| `--momentum`                   | Momentum for client local opitimizer.                                                                                                                                                                                                                                                                                                     |
| `--weight_decay`               | Weight decay for client local optimizer.                                                                                                                                                                                                                                                                                                  |
| `--verbose_gap`                | Interval round of displaying clients training performance on terminal.                                                                                                                                                                                                                                                                    |
| `--batch_size`                 | Data batch size for client local training.                                                                                                                                                                                                                                                                                                |
| `--use_cuda`                   | Non-zero value indicates that tensors are in gpu.                                                                                                                                                                                                                                                                                         |
| `--visible`                    | Non-zero value for using Visdom to monitor algorithm performance on `localhost:8097`.                                                                                                                                                                                                                                                     |
| `--straggler_ratio`            | The ratio of stragglers (set in `[0, 1]`). Stragglers would not perform full-epoch local training as normal clients. Their local epoch would be randomly selected from range `[--straggler_min_local_epoch, --local_epoch)`.                                                                                                              |
| `--straggler_min_local_epoch`  | The minimum value of local epoch for stragglers.                                                                                                                                                                                                                                                                                          |
| `--external_model_params_file` | The relative file path of external model parameters. Please confirm whether the shape of parameters compatible with the model by yourself. ⚠ This feature is enabled only when `unique_model=False`, which is pre-defined by each FL method.                          |
| `--save_log`                   | Non-zero value for saving algorithm running log in `out/<method>/<start_time>`.                                                                                                                                                                                                                                                               |
| `--save_model`                 | Non-zero value for saving output model(s) parameters in `out/<method>/<start_time>`.pt`.                                                                                                                                                                                 |
| `--save_fig`                   | Non-zero value for saving the accuracy curves showed on Visdom into a `.pdf` file at `out/<method>/<start_time>`.                                                                                                                                                                                                                            |
| `--save_metrics`               | Non-zero value for saving metrics stats into a `.csv` file at `out/<method>/<start_time>`.                                                                                                                                                                                                                                                    |
| `--viz_win_name`               | Custom visdom window name (active when setting `--visible` as a non-zero value).                                                                                                                                                                                                                                                          |
| `--config_file`                | Relative file path of custom config `.yaml` file.                                                                                                                                                                                                                                                                                         |
| `--check_convergence`          | Non-zero value for checking convergence after training.                                                                                                                                                                                                                                                                                   |

## Supported Models 🚀

This benchmark supports bunch of models that common and integrated in Torchvision:

- ResNet family
- EfficientNet family
- DenseNet family
- MobileNet family
- LeNet5
...

🤗 You can define your own custom model by filling the `CustomModel` class in [`src/utils/models.py`](src/utils/models.py) and use it by specifying `--model custom` when running.

## Supported Datasets 🎨

This benchmark only supports to solve image classification task for now.


Regular Image Datasets

- *MNIST* (1 x 28 x 28, 10 classes)

- *CIFAR-10/100* (3 x 32 x 32, 10/100 classes)

- *EMNIST* (1 x 28 x 28, 62 classes)

- *FashionMNIST* (1 x 28 x 28, 10 classes)

- [*Syhthetic Dataset*](https://arxiv.org/abs/1812.06127)

- [*FEMNIST*](https://leaf.cmu.edu/) (1 x 28 x 28, 62 classes)

- [*CelebA*](https://leaf.cmu.edu/) (3 x 218 x 178, 2 classes)

- [*SVHN*](http://ufldl.stanford.edu/housenumbers/) (3 x 32 x 32, 10 classes)

- [*USPS*](https://ieeexplore.ieee.org/document/291440) (1 x 16 x 16, 10 classes)

- [*Tiny-ImageNet-200*](https://arxiv.org/pdf/1707.08819.pdf) (3 x 64 x 64, 200 classes)

- [*CINIC-10*](https://datashare.ed.ac.uk/handle/10283/3192) (3 x 32 x 32, 10 classes)

Domain Generalization Image Datasets

- [*DomainNet*](http://ai.bu.edu/DomainNet/) (3 x ? x ?, 345 classes) 
  - Go check [`data/README.md`](data#readme) for the full process guideline 🧾.
  
Medical Image Datasets

- [*COVID-19*](https://www.researchgate.net/publication/344295900_Curated_Dataset_for_COVID-19_Posterior-Anterior_Chest_Radiography_Images_X-Rays) (3 x 244 x 224, 4 classes)

- [*Organ-S/A/CMNIST*](https://medmnist.com/) (1 x 28 x 28, 11 classes)

## Citation 🧐

```bibtex
@software{Tan_FL-bench,
  author = {Tan, Jiahao and Wang, Xinpeng},
  license = {GPL-2.0},
  title = {{FL-bench: A federated learning benchmark for solving image classification tasks}},
  url = {https://github.com/KarhouTam/FL-bench}
}

@misc{tan2023pfedsim,
  title={pFedSim: Similarity-Aware Model Aggregation Towards Personalized Federated Learning}, 
  author={Jiahao Tan and Yipeng Zhou and Gang Liu and Jessie Hui Wang and Shui Yu},
  year={2023},
  eprint={2305.15706},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}

```
