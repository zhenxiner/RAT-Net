# RAT-Net (Accept at MIDL2022)
Region Aware Transformer for Automatic Breast Ultrasound Tumor Segmentation

# Prerequisites

# Train on your dataset

# Test on your dataset
Please modify lines 43~45 of test.py and set the paths of the .pth file, test image and test annotation file. Once done, you can use 
```python  
python test.py  
```
command to test the network.

-There are many verification metrics in the metrics file, and you can choose the verification metric you want to add.

# Predict
Please modify lines 40~41 of predict.py and set the paths of the .pth file, predict images. Once done, you can use 
```python  
python predict.py  
```
command to predict the image.

# Citation 

If you find this repo helps, please kindly cite our paper, thanks!

# References
Links to the methods compared in our experiments:

UNet: https://github.com/milesial/Pytorch-UNet

ResUNet: https://github.com/rishikksh20/ResUnet

Swin-UNet: https://github.com/HuCaoFighting/Swin-Unet

UTNet: https://github.com/yhygao/UTNet

TransUNet: https://github.com/Beckschen/TransUNet

SegFormer: https://github.com/NVlabs/SegFormer

nnUNet: https://github.com/MIC-DKFZ/nnUNet
