# DWP for MRI Sematic Segmentation


Pytorch implementation of the paper [Bayesian Generative Models for Knowledge Transfer in MRI Semantic Segmentation Problems](https://www.frontiersin.org/articles/10.3389/fnins.2019.00844/full)

We use Deep Weight Prior (DWP) [1] to perform knowledge transfer from large source to the smaller target dataset. 
We use a common benchmark - BRATS18 [2] as a target and MS [3] dataset as source dataset. 
   

![model](pics/dwp_im.png)


The method perform better that randomly initialized and fine-tuned models. 

![predict](pics/predictions_pics.png)



[1] Atanov, A., Ashukha, A., Struminsky, K., Vetrov, D., and Welling, M. (2018). The deep weight prior.

[2] Menze, B. H., Jakab, A., Bauer, S., Kalpathy-Cramer, J., Farahani, K., Kirby, J., et al. (2015). The multimodal brain tumor image segmentation benchmark (brats). 

[3] CoBrain analytics, (2018). Multiple Sclerosis Human Brain MR Imaging Dataset


# Experiments
- Load Source and target dataset into `data/dataset_name` folders. Below are examples for MNIST abd notMNIST datasets (they will be loaded automatically)

- Train N models on the source dataset
```bash
python3 train.py --dataset_name notMNIST --trian_size -1 
```

- Train VAE
```bash
python3 train_vae.py --kernel_size 7
```

- Train model on the target dataset with VAE as a prior
```bash
python3 train.py --dataset_name MNIST --prior notMNIST --trian_size 100 
```

# Citation

```text
@article{kuzina2019bayesian,
  title={Bayesian generative models for knowledge transfer in mri semantic segmentation problems},
  author={Kuzina, Anna and Egorov, Evgenii and Burnaev, Evgeny},
  journal={Frontiers in neuroscience},
  volume={13},
  pages={844},
  year={2019},
  publisher={Frontiers}
} 
```


