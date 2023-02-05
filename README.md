# Federated Learning Benchmark

## Methods

### Regular FL methods

- ***FedAvg*** -- [Communication-Efficient Learning of Deep Networks from Decentralized Data (AISTATS 2017)](https://arxiv.org/abs/1602.05629)

- ***FedAvgM*** -- [Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification (ArXiv)](https://arxiv.org/abs/1909.06335)

- ***FedProx*** -- [Federated Optimization in Heterogeneous Networks (MLSys 2020)](https://arxiv.org/abs/1812.06127)

- ***SCAFFOLD*** -- [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning (ICML 2020)](https://arxiv.org/abs/1910.06378)

- ***MOON*** -- [Model-Contrastive Federated Learning (CVPR 2021)](http://arxiv.org/abs/2103.16257)
- ***FedDyn*** -- [Federated Learning Based on Dynamic Regularization (ICLR 2021)](http://arxiv.org/abs/2111.04263)

- ***FedLC*** -- [Federated Learning with Label Distribution Skew via Logits Calibration (ICML 2022)](http://arxiv.org/abs/2209.00189)
  

### Personalized FL methods

- ***Local*** -- Local training only (without communication).

- ***APFL*** -- [Adaptive Personalized Federated Learning (ArXiv)](http://arxiv.org/abs/2003.13461)

- ***LG-FedAvg*** -- [Think Locally, Act Globally: Federated Learning with Local and Global Representations (ArXiv)](https://arxiv.org/abs/2001.01523)

- ***FedBN*** -- [FedBN: Federated Learning On Non-IID Features Via Local Batch Normalization (ICLR 2021)](http://arxiv.org/abs/2102.07623)

- ***FedPer*** -- [Federated Learning with Personalization Layers (AISTATS 2020)](http://arxiv.org/abs/1912.00818)

- ***FedRep*** -- [Exploiting Shared Representations for Personalized Federated Learning (ICML 2021)](http://arxiv.org/abs/2102.07078)

- ***Per-FedAvg*** -- [Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach (NIPS 2020)](https://proceedings.neurips.cc/paper/2020/hash/24389bfe4fe2eba8bf9aa9203a44cdad-Abstract.html)

- ***pFedMe*** -- [Personalized Federated Learning with Moreau Envelopes (NIPS 2020)](http://arxiv.org/abs/2006.08848)

- ***pFedHN*** -- [Personalized Federated Learning using Hypernetworks (ICML 2021)](http://arxiv.org/abs/2103.04628)
  
- ***pFedLA*** -- [Layer-Wised Model Aggregation for Personalized Federated Learning (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Ma_Layer-Wised_Model_Aggregation_for_Personalized_Federated_Learning_CVPR_2022_paper.html)

- ***CFL*** -- [Clustered Federated Learning: Model-Agnostic Distributed Multi-Task Optimization under Privacy Constraints (ArXiv)](https://arxiv.org/abs/1910.01991)

- ***FedFomo*** -- [Personalized Federated Learning with First Order Model Optimization (ICLR 2021)](http://arxiv.org/abs/2012.08565)

- ***FedBabu*** -- [FedBabu: Towards Enhanced Representation for Federated Image Classification (ICLR 2022)](https://arxiv.org/abs/2106.06042)

- ***FedAP*** -- [Personalized Federated Learning with Adaptive Batchnorm for Healthcare (IEEE Transactions on Big Data 2022)](https://arxiv.org/abs/2112.00734)


More reproductions/features would come soon or later (depends on my mood 🤣).

## Usage

So easy, right? 😎

```shell
cd data/utils
python run.py -d cifar10 -a 0.1 -cn 100
cd ../../

cd src/server
python ${algo}.py
```

About methods of generating federated dastaset, go check `data/utils/README.md` for full details.


### Monitor

1. Run `python -m visdom.server` on terminal.
2. Go check `localhost:8097` on your browser.

## Supported Datasets

🤗 This benchmark only support algorithms to solve image classification problem for now.


Regular image datasets

- *MNIST* (1 x 28 x 28, 10 classes)

- *CIFAR-10/100* (3 x 32 x 32, 10/100 classes)

- *EMNIST* (1 x 28 x 28, 62 classes)

- *FashionMNIST* (1 x 28 x 28, 10 classes)

- *FEMNIST* (1 x 28 x 28, 62 classes)

- *CelebA* (3 x 218 x 178, 2 classes)

- *SVHN* (3 x 32 x 32, 10 classes)

- *USPS* (1 x 16 x 16, 10 classes)

- *Tiny-Imagenet-200* (3 x 64 x 64, 200 classes)

Medical image datasets

- [*COVID-19*](https://www.researchgate.net/publication/344295900_Curated_Dataset_for_COVID-19_Posterior-Anterior_Chest_Radiography_Images_X-Rays) (3 x 244 x 224, 4 classes)

- [*Organ-S/A/CMNIST*](https://medmnist.com/) (1 x 28 x 28, 11 classes)

## Acknowledgement

Some reproductions in this benchmark are referred to <https://github.com/TsingZ0/PFL-Non-IID>, which is a great FL benchmark.  👍






