## Optimizing Polynomial Graph Filters: A Novel Adaptive Krylov Subspace Approach

**AdaptKry** is a polynomial graph filter optimized by adaptive Krylov basis. This repository contains the source codes of UniFilter, data process, and split generation codes. For detailed information about **AdaptKry**, please refer to our paper in **WWW 2024**. If any issues are observed, please contact Keke Huang, thanks!



## Environment Settings    

- pytorch 1.7.0
- torch-geometric 1.6.1
- scipy 1.9.3
- seaborn 0.12.0
- scikit-learn 1.1.3
- ogb 1.3.1
- optuna 3.1.1
- gdown

## Datasets

We follow the folder structure as [ChebNet II](https://github.com/ivam-he/ChebNetII). Please acquire all the data from ChebNet II and put the data in the subfolder './data'. 
The ogb datasets (ogbn-arxiv and ogbn-papers100M) and non-homophilous datasets (from [LINKX](https://arxiv.org/abs/2110.14446) ) can be downloaded automatically.



## Folders

Please create a folder named 'pretrained' before running.



## Citation

Please cite our work if interested

```
@inproceedings{HuangCTXL24,
  author       = {Keke Huang and
                  Wencai Cao and
                  Hoang Ta and
                  Xiaokui Xiao and
                  Pietro Li{\`{o}}},
  title        = {Optimizing Polynomial Graph Filters: {A} Novel Adaptive Krylov Subspace
                  Approach},
  booktitle    = {{WWW}},
  pages        = {1057--1068},
  year         = {2024},
}
```
