# ViT-AMCNet: A ViT-AMC Network with Adaptive Model Fusion and Multiobjective Optimization for Interpretable Laryngeal Tumor Grading from Histopathological Images (IEEE Transactions on Medical Imaging 2022)

***
# Introduction
The tumor grading of laryngeal cancer pathological images needs to be accurate and interpretable. The deep learning model based on the attention mechanism-integrated convolution (AMC) block has good inductive bias capability but poor interpretability, whereas the deep learning model based on the vision transformer (ViT) block has good interpretability but weak inductive bias ability. Therefore, we propose an end-to-end ViT-AMC network (ViT-AMCNet) with adaptive model fusion and multiobjective optimization that integrates and fuses the ViT and AMC blocks. However, existing model fusion methods often have negative fusion: 1). There is no guarantee that the ViT and AMC blocks will simultaneously have good feature representation capability. 2). The difference in feature representations learning between the ViT and AMC blocks is not obvious, so there is much redundant information in the two feature representations. Accordingly, we first prove the feasibility of fusing the ViT and AMC blocks based on Hoeffding's inequality. Then, we propose a multiobjective optimization method to solve the problem that ViT and AMC blocks cannot simultaneously have good feature representation. Finally, an adaptive model fusion method integrating the metrics block and the fusion block is proposed to increase the differences between feature representations and improve the deredundancy capability. Our methods improve the fusion ability of ViT-AMCNet, and experimental results demonstrate that ViT-AMCNet significantly outperforms state-of-the-art methods. Importantly, the visualized interpretive maps are closer to the region of interest of concern by pathologists, and the generalization ability is also excellent. 

---
![image](https://raw.githubusercontent.com/Baron-Huang/ViT-AMCNet/main/Img/main.jpg)

---
# Citing ViT-AMCNet
Huang, Pan, et al. "A ViT-AMC Network with Adaptive Model Fusion and Multiobjective Optimization for Interpretable Laryngeal Tumor Grading from Histopathological Images." IEEE Transactions on Medical Imaging (2022).DOI: 10.1109/TMI.2022.3202248

---
# Notice
1. We fixed the random seeds, and the effect is consistent on different GPUs on the same server, but there may be subtle differences if the server is changed, and such differences are caused by Pytorch's random method, which we cannot solve. 
2. if you replace other datasets for experiments, the hyperparameters mentioned in the text may not be optimal, please re-optimize them, and most importantly, optimize the learning rate. If you have any doubts, please do not hesitate to contact panhuang@cqu.edu.cn.
3. The dataset in this paper was requested from the Medical Signal Processing Laboratory of the University of Athens, and we do not have the permission to upload its data, so interested parties can request it from the University of Greece (link in the dataset section of the paper).
4. The algorithms and ideas in this paper are for scientific study and research only, and commercial activities are prohibited. If you refer to the improvement methods of the ideas in this paper, please follow the principles of academic norms and cite this paper.

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
|Python|3.7.9|
|Cuda|11.1.105|


# LA-ViT
LA-ViT: A Network with Transformers constrained by Learned-parameters-free Attention for Interpretable Grading in A New Laryngeal Histopathology Images Dataset

Our manuscript is in the peer review stage, and we will publish the code and dataset completely when the review is finished. If interested in this research and dataset, contact us at panhuang@cqu.edu.cn.
