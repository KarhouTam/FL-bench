There are three steps to partition DomainNet:
1. Split clients to test clients and training clients.
2. Split samples on training clients into training set and test set
3. Construct heterogeneous data partition 

# Split Clients
We split clients and data accrding to `--split`, more details in [Generating Federated dataset](/data/README.md).\
If `--split='domain'` we choose sketch as test domain by default. Users can easily choose their own test domain by modifying `generate_data.py`

# Split Samples on Training Clients
We split samples on each training client into training set and test set at a ratio of `--fraction`. There will be no test set on training data when `--fraction=1`.
# Construct Heterogeneous Data Partition
We generate heterogeneous paration following the method adopted by [FedAVGM](https://arxiv.org/pdf/1909.06335.pdf).
- `--alpha` or `-a`: The parameter for controlling intensity of label heterogeneity. *Smaller `--alpha` -> more hetergeneous partition.*
- `--least_samples` or `-ls`: The parameter for defining the minimum number of samples each client would be distributed. *A small `--least_samples` along with small `--alpha` or big `--client_num` might considerablely prolong the partition.*
- If `--alpha=0`, each client would have only one domain.

Finally, if we set `--split='domain'` and `--fraction>0`, we have out of domain test set (sketch by default) and in domain test set (extracted from samples on training clients by `--fraction`), which is designed for domain generalization task. Besides, we can generate heterogeneous training data partition by `--alpha`.\
Users can check the size of samples from each domain on each training client in `'domain_info'` from `all_stats.json` when `--alpha>0`.
