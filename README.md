# Leaving Footprints for Online Out-of-Distribution Detection by Leveraging Low-Bit Representations
(@NVIDIA Research Team, 2023)


## Requirements

* python3
* pytorch==1.7.1
* cudatoolkit==11.0.221 
* numpy==1.19.2
* tensorboardx==1.4

## Introduction
Out-of-distribution (OOD) detection is developed to identify the inference data with a similar -or- distinct distribution from training data, i.e., in-distribution (ID) -or- out-of-distribution (OOD) data, to prevent the OOD data from generating a distorted prediction result and thus deteriorating the model performance. 

## Motivations
Existing works utilized pruning or
outlier removal to retain more essential prediction information of ID data than OOD data for identification. However,
they focus on the structures, e.g., columns of activation matrices, or long-tailed values, without examining the overall
distribution. In this project, we design a novel algorithm, Leaf, to employ quantization to preserve ID distribution with low-bit representations, whereas OOD features under the same quantization criterion tend to suffer from more distortions, e.g., fewer representations or more information
collision. Accordingly, two groups are more easily to be
distinguished. Moreover, existing research detects images
individually and independently during inference. By contrast, we propose a new mindset, online OOD detection, to
store the detection results of previous data as footprints for
the subsequent detection. Based on the prior knowledge, we
estimate the ID and OOD testing distributions to separate
the two groups further apart to effectively enhance of the detection results.


## Overview

Overview of online OOD detection using low-bit features (see ["fig/overview.pdf"](fig/motivation.pdf))


## Implementation (eg. ResNet-50 on ImageNet-1K [ID] and iNaturalist [OOD] datasets)

#### Training low-bit models

```shell
cd oodq_cnn/resnet-50-imagenet/
bash run_train.sh
```

#### Evaluation (FPR95 and AUROC)

```shell
bash run_eval.sh
```
## Results

See ["fig/performance_plot.pdf"](fig/performance_plot.pdf).

## Paper Reference

* [\[1\]](https://arxiv.org/abs/1610.02136)
* [\[2\]](https://proceedings.neurips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html)
* [\[3\]](https://openreview.net/forum?id=H1VGkIxRZ)
* [\[4\]](https://proceedings.neurips.cc/paper/2021/hash/01894d6f048493d2cacde3c579c315a3-Abstract.html)
* [\[5\]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840680.pdf)
* [\[6\]](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_ViM_Out-of-Distribution_With_Virtual-Logit_Matching_CVPR_2022_paper.pdf)
