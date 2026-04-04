# MSLEC

This project is the official implementation of 'MSLEC: Multi-Scale Latent Feature Restoration for Exposure Correction'.


## Installation

```
conda create -n mslec python=3.11
conda activate mslec
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

## Prepare Multi-Scale VQ-VAE model

Please download [ms-vqvae.pth](https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth) and place it in your-path/vae_ch160v4096z32.pth. 

## Train Stage I (*option/train_VAEEC_S1_lcdp.yml*)

```
sh trainS1.sh
```

## Train Stage II (*option/train_VAEEC_S2_lcdp.yml*)

```
sh trainS2.sh
```

## Test (*option/test_VAEECS2_lcdp.yml*)

```
sh test.sh
```











