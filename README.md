# LA-ViT: A Network with Transformers Constrained by Learned-Parameter-Free Attention for Interpretable Grading in a New Laryngeal Histopathology Image Dataset
Pan Huang, Hualiang Xiao, Chentao Li, Xiaodong Guo, Peng He, Sukun Tian, Peng Feng, Hu Chen, Yuchun Sun, Francesco Mercaldo, Antonella Santone, Jing Qin 

Our manuscript is in the peer review stage (IEEE-JBHI 2024), and we will publish the code and dataset completely when the review is finished. If you interested in this research and dataset, contact us at panhuang@cqu.edu.cn or n2308704h@e.ntu.edu.sg.

***
# Introduction
Grading laryngeal squamous cell carcinoma (LSCC) based on histopathological images is a clinically significant yet challenging task. However, more low-effect background semantic information appeared in the feature maps, feature channels, and class activation maps, which caused a serious impact on the accuracy and interpretability of LSCC grading. While the traditional transformer block makes extensive use of parameter attention, the model overlearns the low-effect background semantic information, resulting in ineffectively reducing the proportion of background semantics. Therefore, we propose an end-to-end network with transformers constrained by learned-parameter-free attention (LA-ViT), which improve the ability to learn high-effect target semantic information and reduce the proportion of background semantics. Firstly, according to generalized linear model and probabilistic, we demonstrate that learned-parameter-free attention (LA) has a stronger ability to learn highly effective target semantic information than parameter attention. Secondly, the first-type LA transformer block of LA-ViT utilizes the feature map position subspace to realize the query. Then, it uses the feature channel subspace to realize the key, and adopts the average convergence to obtain a value. And those construct the LA mechanism. Thus, it reduces the proportion of background semantics in the feature maps and feature channels. Thirdly, the second-type LA transformer block of LA-ViT uses the model probability matrix information and decision level weight information to realize key and query, respectively. And those realize the LA mechanism. So, it reduces the proportion of background semantics in class activation maps. Finally, we build a new complex semantic LSCC pathology image dataset to address the problem, which is less research on LSCC grading models because of lacking clinically meaningful datasets. After extensive experiments, the whole metrics of LA-ViT outperform those of other state-of-the-art methods, and the visualization maps match better with the regions of interest in the pathologists' decision-making. Moreover, the experimental results conducted on a public LSCC pathology image dataset show that LA-ViT has superior generalization performance to that of other state-of-the-art methods.

---
![image](https://github.com/Baron-Huang/LA-ViT/blob/main/Images/Fig_4.png)
Fig.1. Overview of the proposed end-to-end LA-ViT for the pathology grading of LSCC. 



![image](https://github.com/Baron-Huang/LA-ViT/blob/main/Images/Fig_2.png)
Fig.2. There is an overabundance of low-effect background semantics in the feature maps, feature channels, and class activation maps of traditional transformer models (such as the vision transformer (ViT) and Swin transformer (SwinT)). Tumor and bird are target semantics, and tree and lymph are background semantics. 

---
# Citing LA-ViT
We manuscript is in peer review stage, waiting a moment.

---
# Notice
1. We fixed the random seeds, and the effect is consistent on different GPUs on the same server, but there may be subtle differences if the server is changed, and such differences are caused by Pytorch's random method, which we cannot solve. 
2. if you replace other datasets for experiments, the hyperparameters mentioned in the text may not be optimal, please re-optimize them, and most importantly, optimize the learning rate. If you have any doubts, please do not hesitate to contact panhuang@cqu.edu.cn.
3. Use of our datasets requires a request to be made to co-author Hualiang Xiao (dpbl_xhl@163.com) by filling out the relevant request form(https://github.com/Baron-Huang/LA-ViT/blob/main/Application%20Form%20for%20Datasets.docx), which can be downloaded after review.
4. The algorithms, ideas, and datasets in this paper are for scientific study and research only, and commercial activities are prohibited. If you use our methods, ideas, and datasets in your publications and patent, please follow the principles of academic norms and cite this paper.

---
# API
|API|Version|
|--|--|
|d2l|0.17.5|
|numpy|1.21.5|
|scikit-image|0.18.3|
|scikit-learn|0.24.2|
|torch|1.9.0+cu111|
|torchvision|0.10.0+cu111|
|Python|3.9.0|
|Cuda|11.1.105|


