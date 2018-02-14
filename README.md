# MixtureFA (Mixtures of Factor Analyzers)

This is a toolbox written in python for training the (Mixture of) Factor Analyizer.

>(Only in Mandarin) Detailed math of FA/MFA: see [blog](https://bobondemon.github.io/2018/02/11/Mixtures-of-Factor-Analyzers/)

---
## Usage

### FA

```python
# Input:
#   X: (D,N) matrix, where D is obervation dimension and N is observation number
#   K: scalar, indicates the dimension of latent space
# Return:
#   FA follows the following linear-Gaussian Model: p(x|z) ~ N(mu+Wz, diag(psi))
#   mu, (D,1)
#   W, (D,K)
#   psi, (D,)
from mixture_latent_analysis import fa_em
W_fa, mu_fa, psi = fa_em(X,K)
```

### MFA

```python
# Input:
#   X: (D,N) matrix, where D is obervation dimension and N is observation number
#   K: scalar, indicates the dimension of latent space
#   M: scalar, indicates the number of mixture components
# Return:
#   For a given mixture component j, p(x|zj) ~ N(mu[j,:]+W[j,...]zj, diag(psi))
#   pi, (M,): the mixture weights, sum(pi)=1
#   mu, (M,D)
#   W, (M,D,K)
#   psi, (D,)
from mixture_latent_analysis import mfa_em
pi, mu, W, psi = mfa_em(X,K,M)
```

---
## Toy Example

For running the toy examples, type:

```bash
>> # FA toy example
>> python expToyData.py 
```

<img src="fig/FA_toy_example.png" width="70%" height="70%">

```bash
>> # MFA toy example
>> python expToyData_mixture.py 
```

<img src="fig/MFA_toy_example.png" width="70%" height="70%">

---
## TODO

1. 修改 comments
2. MFA: `mu` 初始化使用 k-means
