# MSLEC

This project is the official implementation of 'Restoring Layer-wise Representations for Extreme Image Exposure Correction'.


## Installation

```
conda create -n RLREC python=3.11
conda activate RLREC
conda install pytorch==2.5.0 torchvision==0.20.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install opencv-python
pip install einops
pip install basicsr

```

## Prepare Data

Please refer to the link below to download the dataset
- LCDP https://www.whyy.site/paper/lcdp
- MSEC https://github.com/mahmoudnafifi/Exposure_Correction
- SICE https://github.com/KevinJ-Huang/ExposureNorm-Compensation


### Crop data

```
python generate_patches_lcdp.py
```

## Prepare DINOv3 model

Please download [dinov3_vitb16.pth](https://github.com/facebookresearch/dinov3) and place it in /home/your-username/.cache/torch/hub/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth 

## Train Stage I (*option/train_RLREC_S1_lcdp.yml*)

```
sh trainS1.sh
```

## Train Stage II (*option/train_RLREC_S2_lcdp.yml*)

```
sh trainS2.sh
```

## Test (*option/test_RLREC_lcdp.yml*)

```
sh test.sh
```











