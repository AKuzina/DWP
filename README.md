# DWP for MRI Sematic Segmentation


Pytorch implementation of the paper [Bayesian Generative Models for Knowledge Transfer in MRI Semantic Segmentation Problems](https://www.frontiersin.org/articles/10.3389/fnins.2019.00844/full)

We use Deep Weight Prior (DWP) [1] to perform knowledge transfer from large source to the smaller target dataset. 
We use a common benchmark - BRATS18 [2] as a target and MS [3] dataset as source dataset. 
The method perform better that randomly initialized and fine-tuned models.   

![res](!pics/BRATS_res.png)


![predict](!pics/predictions_pics.png)



[1] Atanov, A., Ashukha, A., Struminsky, K., Vetrov, D., and Welling, M. (2018). The deep weight prior.

[2] Menze, B. H., Jakab, A., Bauer, S., Kalpathy-Cramer, J., Farahani, K., Kirby, J., et al. (2015). The multimodal brain tumor image segmentation benchmark (brats). 

[3] CoBrain analytics, (2018). Multiple Sclerosis Human Brain MR Imaging Dataset 