# RAT-Net (Accept at MIDL2022)
Region Aware Transformer for Automatic Breast Ultrasound Tumor Segmentation

## Prerequisites

## Train on your dataset

### Dataset format

### Train

Please modify lines 22~27 of trainRATt3.py

-load_pretrain: load your pretrain model. if you need load, set it to True.

-load_path: the pretrain model path.

-img_paths: the path of train data

-label_paths: the path of train label data

-valimg_paths: the path of validation data

-vallabel_paths: the path of validation label data

Please modify line 152 of ./model/RAtrasformer.py

For example, the line is: "out[:, :, (h_x * 200 // 768):(h_x * 500 // 768), (w_x * 100 // 768):(w_x * 600 // 768)] = x"

the number 768 is the original size of the training data, the number of other such as 200, 500 is the region you can set.

Please modify line 53, 54 and 87 of ./model/RASAB_t3.py

The modification method is the same as the RAtrasformer.py.

 Once done, you can use 
```python  
python trainRATt3.py  
```
command to train the network.


## Test on your dataset
Please modify lines 43~45 of test.py and set the paths of the .pth file, test image and test annotation file. Once done, you can use 
```python  
python test.py  
```
command to test the network.

-There are many verification metrics in the metrics file, and you can choose the verification metric you want to add.

## Predict
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
