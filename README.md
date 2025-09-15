<img src="https://github.com/QinZhou-work/MicNet/blob/53d36fc3b68835ceb8aa94c17664171aad5e1576/Pictures/qbrc_logo.png?inline=false" width="40%"/>


# MicNet
MicNet: Integrating spatially resolved transcriptomes and pathology images by contrastive deep neural network

# Introduction
Exploring the spatial organization of cells alongside their gene expression is key to understanding how tissues acquire distinct structures and functions. Recent advances in spatial transcriptomics (SRT) technologies have enabled the joint profiling of tissue morphology and mRNA expression, yet integrating these two modalities remains a major challenge. To address this, we developed MicNet, an unsupervised deep learning framework that bridges histology images and transcriptomic data, providing robust, scalable, and biologically meaningful representations for spatial domain identification and downstream analyses.

<div align="center">
  <img src="https://github.com/QinZhou-work/MicNet/blob/8196b806a4f8179e8d4d838c8d23649ad5bc8e09/Pictures/MicNet_figures.png?inline=True" alt="Alt text" width="90%"/>
</div>
# Installation

```
conda create -n "MicNet" python=3.6.10
conda activate MicNet
pip install -r MicNet_requirements.txt
```

# User Guideline
- [ ] [Spatial transcriptomic data pre-processing](https://github.com/QinZhou-work/MicNet/blob/464a7f6974ca83b80a688c73a8075c21bc498664/tutorial/MicNet_1_data_check_and_preprocessing.ipynb)

- [ ] [Training MicNet](https://github.com/QinZhou-work/MicNet/blob/f3324491b3dc150e10969607dfc755ef122239b9/tutorial/MicNet_2_train_MicNet.ipynb)

- [ ] [Inference MicNet](https://github.com/QinZhou-work/MicNet/blob/f3324491b3dc150e10969607dfc755ef122239b9/tutorial/MicNet_3_Inference.ipynb)
      
- [ ] [Clustering the domains](https://github.com/QinZhou-work/MicNet/blob/93805f9c47b349e579ff31d8f86b6a987fe8bfb8/tutorial/MicNet_4_clustering_with_MicNet.ipynb)

