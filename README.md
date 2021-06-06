# Dynamic-Vision-Transformer (Pytorch)


This repo contains the official code and pre-trained models for the Dynamic Vision Transformer (DVT).

- [Not All Images are Worth 16x16 Words: Dynamic Vision Transformers with Adaptive Sequence Length](https://arxiv.org/pdf/2105.15075.pdf)


**Update on 2021/06/01: Release Pre-trained Models and the Inference Code on ImageNet.**

## Introduction

<p align="center">
    <img src="figures/examples.png" width= "400">
</p>

We develop a Dynamic Vision Transformer (DVT) to automatically configure a proper number of tokens for each individual image, leading to a significant improvement in computational efficiency,  both theoretically and empirically.
<p align="center">
    <img src="figures/overview.png" width= "810">
</p>



## Citation

If you find this work valuable or use our code in your own research, please consider citing us with the following bibtex:

```
@article{wang2021not,
        title = {Not All Images are Worth 16x16 Words: Dynamic Vision Transformers with Adaptive Sequence Length},
       author = {Wang, Yulin and Huang, Rui and Song, Shiji and Huang, Zeyi and Huang, Gao},
      journal = {arXiv preprint arXiv:2105.15075},
         year = {2021}
}
```

## Results

- Top-1 accuracy on ImageNet v.s. GFLOPs 
<p align="center">
    <img src="figures/result_main.png" width= "810">
</p>


- Top-1 accuracy on CIFAR v.s. GFLOPs 
<p align="center">
    <img src="figures/cifar.png" width= "500">
</p>


- Top-1 accuracy on ImageNet v.s. Throughput 
<p align="center">
    <img src="figures/result_speed.png" width= "400">
</p>


- Visualization
<p align="center">
    <img src="figures/result_visual.png" width= "700">
</p>


## Pre-trained Models


|Backbone|# of Exits|# of Tokens|Links|
|-----|------|-----|-----|
|T2T-ViT-12| 3| 7x7-10x10-14x14|[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/f7987d559f00435caa3d/?dl=1) / [Google Drive](https://drive.google.com/file/d/1-hu-OMWiwuc8iTd1PV-c8tv6jLZ8EVv9/view?usp=sharing)|

- What are contained in the checkpoints:

```
**.pth.tar
├── model_state_dict: state dictionaries of the model
├── flops: a list containing the GFLOPs corresponding to exiting at each exit
├── anytime_classification: Top-1 accuracy of each exit
├── dynamic_threshold: the confidence thresholds used in budgeted batch classification
├── budgeted_batch_classification: results of budgeted batch classification (a two-item list, [0] and [1] correspond to the two coordinates of a curve)

```

## Requirements
- python 3.7.7
- pytorch 1.3.1
- torchvision 0.4.2


## Evaluate Pre-trained Models

Read the evaluation results saved in pre-trained models
```
CUDA_VISIBLE_DEVICES=0 python inference.py --batch_size 128 --model DVT_T2t_vit_12 --checkpoint_path PATH_TO_CHECKPOINTS  --eval_mode 0
```

Read the confidence thresholds saved in pre-trained models and infer the model on the validation set
```
CUDA_VISIBLE_DEVICES=0 python inference.py --data_url PATH_TO_DATASET --batch_size 128 --model DVT_T2t_vit_12 --checkpoint_path PATH_TO_CHECKPOINTS  --eval_mode 1
```

Determine confidence thresholds on the training set and infer the model on the validation set
```
CUDA_VISIBLE_DEVICES=0 python inference.py --data_url PATH_TO_DATASET --batch_size 128 --model DVT_T2t_vit_12 --checkpoint_path PATH_TO_CHECKPOINTS  --eval_mode 2
```

The dataset is expected to be prepared as follows:
```
ImageNet
├── train
│   ├── folder 1 (class 1)
│   ├── folder 2 (class 1)
│   ├── ...
├── val
│   ├── folder 1 (class 1)
│   ├── folder 2 (class 1)
│   ├── ...

```

## Contact
If you have any question, please feel free to contact the authors. Yulin Wang: wang-yl19@mails.tsinghua.edu.cn.

## Acknowledgment
Our code of T2T-ViT from [here](https://github.com/yitu-opensource/T2T-ViT). 

## To Do
- Update the code for training.
