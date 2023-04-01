# Federated Learning Benchmark

## Method

### Regular FL Methods

- ***FedAvg*** -- [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) (AISTATS'17)

- ***FedAvgM*** -- [Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification](https://arxiv.org/abs/1909.06335) (ArXiv'19)

- ***FedProx*** -- [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127) (MLSys'20)

- ***SCAFFOLD*** -- [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](https://arxiv.org/abs/1910.06378) (ICML'20)

- ***MOON*** -- [Model-Contrastive Federated Learning](http://arxiv.org/abs/2103.16257) (CVPR'21)
- ***FedDyn*** -- [Federated Learning Based on Dynamic Regularization](http://arxiv.org/abs/2111.04263) (ICLR'21)

- ***FedLC*** -- [Federated Learning with Label Distribution Skew via Logits Calibration](http://arxiv.org/abs/2209.00189) (ICML'22)
  

### Personalized FL Methods

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


More reproductions/features would come soon or later (depends on my mood 🤣).

## Easy Run
e.g.
```shell
# partition the CIFAR-10 according to Dir(0.1) for 100 clients
cd data/utils
python run.py -d cifar10 -a 0.1 -cn 100
cd ../../

# run FedAvg under default setting.
cd src/server
python fedavg.py
```

About methods of generating federated dastaset, go check [`data/README.md`](https://github.com/KarhouTam/FL-bench/tree/master/data/#readme) for full details.


### Monitor
1. Run `python -m visdom.server` on terminal.
2. Run `src/server/${algo}.py --visible 1`
3. Go check `localhost:8097` on your browser.
## Arguments

📢 All arguments have default value.

About the default values and hyperparameters of advanced FL methods, go check [`src/config/args.py`](https://github.com/KarhouTam/FL-bench/tree/master/src/config/args.py) for full details.
| General Argument          | Description                                                                                                   |
| ------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `--dataset`, `-d`         | The name of dataset that experiment run on.                                                                   |
| `--model`, `-m`           | The model backbone experiment used.                                                                           |
| `--seed`                  | Random seed for running experiment.                                                                           |
| `--join_ratio`, `-jr`     | Ratio for (client each round) / (client num in total).                                                        |
| `--global_epoch`, `-ge`   | Global epoch, also called communication round.                                                                |
| `--local_epoch`, `-le`    | Local epoch for client local training.                                                                        |
| `--finetune_epoch`, `-fe` | Epoch for clients fine-tunning their models before test.                                                      |
| `--test_gap`, `-tg`       | Interval round of performing test on clients.                                                                 |
| `--eval_test`, `-ee`      | Non-zero value for performing evaluation on joined clients' testset before and after local training.          |
| `--eval_train`, `-er`     | Non-zero value for performing evaluation on joined clients' trainset before and after local training.         |
| `--local_lr`, `-lr`       | Learning rate for client local training.                                                                      |
| `--momentum`, `-mom`      | Momentum for client local opitimizer.                                                                         |
| `--weight_decay`, `-wd`   | Weight decay for client local optimizer.                                                                      |
| `--verbose_gap`, `-vg`    | Interval round of displaying clients training performance on terminal.                                        |
| `--batch_size`, `-bs`     | Data batch size for client local training.                                                                    |
| `--server_cuda`           | Non-zero value indicates that tensors at server side are in gpu.                                              |
| `--client_cuda`           | Non-zero value indicates that tensors at client side are in gpu.                                              |
| `--visible`               | Non-zero value for using Visdom to monitor algorithm performance on `localhost:8097`.                         |
| `--save_log`              | Non-zero value for saving algorithm running log in `FL-bench/out/{$algo}`.                                    |
| `--save_model`            | Non-zero value for saving output model(s) parameters in `FL-bench/out/{$algo}`.                               |
| `--save_fig`              | Non-zero value for saving the accuracy curves showed on Visdom into a `.jpeg` file at `FL-bench/out/{$algo}`. |
| `--save_metrics`          | Non-zero value for saving metrics stats into a `.csv` file at `FL-bench/out/{$algo}`.                         |

## Supported Datasets

🤗 This benchmark only support algorithms to solve image classification task for now.


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

Medical Image Datasets

- [*COVID-19*](https://www.researchgate.net/publication/344295900_Curated_Dataset_for_COVID-19_Posterior-Anterior_Chest_Radiography_Images_X-Rays) (3 x 244 x 224, 4 classes)

- [*Organ-S/A/CMNIST*](https://medmnist.com/) (1 x 28 x 28, 11 classes)

## Acknowledgement

Some reproductions in this benchmark are referred to <https://github.com/TsingZ0/PFL-Non-IID>, which is a great FL benchmark. 👍

This benchmark is still young, which means I will update it frequently and unpredictably. Therefore, periodically fetching the latest code is recommended. 🤖

If this benchmark is helpful to your research, it's my pleasure. 😏






