# Covariance-Guided and Direction-Aware Scan Attention for Hyperspectral Image Fusion（CGSFN）
![Language](https://img.shields.io/badge/language-python-brightgreen) 
## Abstract
Hyperspectral and multispectral image fusion aims to reconstruct high-resolution hyperspectral image (HR-HSI) by integrating complementary spectral and spatial information. Despite recent advances, existing methods still suffer from two key limitations: (1) predominantly rely on feature-level aggregation or similarity-driven attention mechanisms, which limits their ability to effectively characterize higher-order statistical dependencies between heterogeneous modalities; and (2) mainly focus on scale variation, while lacking explicit mechanisms for modeling direction-aware spatial dependencies inherent in hyperspectral data.  To overcome these challenges, we propose CGSFN—a Covariance-Guided Scan Fusion Network. Specifically, a Covariance-Guided Cross-Modal Attention Module (CG-CAM) is designed to capture covariance-level inter-modal dependencies, enabling more effective cross-modal interaction between complementary spectral and spatial information. A Direction-Aware Scan Attention Module (DA-SAM) is developed to achieve structure-aware spatial representation learning by jointly modeling multi-scale spatial context and direction-aware dependencies through scan-inspired attention mechanisms. Experimental results on multiple benchmark datasets demonstrate that the proposed method consistently outperforms existing approaches across various performance metrics.

## 🌈 Method

![CGSFN](./overall.png)


## 👉 Dataset
We conduct experiments on three publicly available datasets and use a data simulation strategy to generate training and test image pairs.
* [Cave](https://cave.cs.columbia.edu/repository/Multispectral)

* [Harvard](http://vision.seas.harvard.edu/hyperspec/)

* [KAIST](https://vclab.kaist.ac.kr/siggraphasia2017p1/)

Please place the downloaded datasets in the following directory structure:
```text
Dataset/
├── Cave/
│   ├── Train/      # CAVE training images
│   └── Test/       # CAVE testing images
├── Harvard/
│   ├── Train/      # Harvard training images
│   └── Test/       # Harvard testing images
├── KAIST/
     ├── Train/      # KAIST training images
     └── Test/       # KAIST testing images
```
## 🌿 Getting Started

### Environment Setup

To get started, we recommend setting up a conda environment and installing dependencies via pip. Use the following commands to set up your environment.
```
conda create -n env python=3.8.20
conda activate env
pip install -r requirements.txt
```
### Train and Test
python train_cave_M.py

### Citation

If this project helps your research, please cite our paper

## 🌸 Acknowledgment

We are deeply grateful to repositories [CD-UNet](https://github.com/liuofficial/CD-UNet), which served as the foundational basis for our code implementation.

