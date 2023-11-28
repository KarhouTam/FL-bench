There are three steps to partition DomainNet
1. select test clients
2. split samples on training clients into training set and test set
3. Construct heterogeneous data partition

# Select Test Clients
If `--split='user' or 'sample'`, we split clients and data accrding.\
If `--split='domain'`, we choose Sketch as test domain and other styles as training domain.
