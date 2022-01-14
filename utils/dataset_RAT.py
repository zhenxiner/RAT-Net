'''
这个代码用来测试是否loader数据成功
'''
import random
from glob import glob
from tqdm import tqdm
import cv2
import os

import numpy as np
import torch
import torch.utils.data
from torchvision import datasets, models, transforms
from collections import Counter



class Dataset(torch.utils.data.Dataset):

    def __init__(self, img_paths, mask_paths, scale=1, train_flag=True):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.train_flag = train_flag
        self.scale = scale

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        npimage = cv2.imread(img_path)
        npmask = cv2.imread(mask_path, -1)
        if self.train_flag:
            if npmask.sum() == 0:
                randseed = random.randint(0, 99)
                if randseed < 75:
                    npimage, npmask = self.imgaug(npimage, npmask)
                else:
                    pass
        npimage = cv2.resize(npimage, (384, 384), interpolation=cv2.INTER_LINEAR)
        npmask = cv2.resize(npmask, (384, 384), interpolation=cv2.INTER_LINEAR)

        npimage = npimage[np.newaxis, :, :, :]
        npmask = npmask[np.newaxis, :, :, np.newaxis]

        if self.train_flag:
            npimage, npmask = self.imgaug_seg(npimage, npmask)
        npimage = npimage.transpose((0, 3, 1, 2))
        npimage = npimage.squeeze()
        npmask = npmask.squeeze()

        npimage = npimage.astype("float32")
        npmask = npmask.astype("float32")

        return npimage, npmask

    def imgaug(self, npimage, npmask):
        localinformation = '/home/Public/localinformation.txt'
        fo = open(localinformation)
        img_targetfile = "/home/train_img/*"
        mask_targetfile = "/home/train_label/*"
        imglist = os.listdir(img_targetfile)
        a = random.randint(0, 3697)
        imgname = imglist[a]
        img_target = cv2.imread(img_targetfile + imgname)
        mask_target = cv2.imread(mask_targetfile + imgname, -1)
        x1 = 0
        x2 = 0
        x3 = 0
        x4 = 0
        lines = fo.readlines()
        for line in lines:
            if imgname == line.split(":")[0]:
                x1 = int(line.split(":")[1].split(",")[0])
                x2 = int(line.split(":")[1].split(",")[1])
                x3 = int(line.split(":")[1].split(",")[2])
                x4 = int(line.split(":")[1].split(",")[3])
        random_localx = random.randint(270, 320)
        random_localy = random.randint(16, (700-x4+x3))
        if x4-x3==70:
            npimage[random_localx:random_localx+x2-x1, random_localy:random_localy+x4-x3-1, :] = img_target[x1:x2, x3:x4, :]
            npmask[random_localx:random_localx+x2-x1, random_localy:random_localy+x4-x3-1] = mask_target[x1:x2, x3:x4]
        else:
            npimage[random_localx:(random_localx+x2-x1), random_localy:(random_localy+x4-x3), :] = img_target[x1:x2, x3:x4, :]
            npmask[random_localx:(random_localx+x2-x1), random_localy:(random_localy+x4-x3)] = mask_target[x1:x2, x3:x4]
        fo.close()
        return npimage, npmask


    def imgaug_seg(self, images, segmaps):
        import imgaug.augmenters as iaa

        seq = iaa.Sequential([
            iaa.GaussianBlur((0, 3.0)),
            iaa.Affine(translate_px={"x": (-40, 40)}),
            iaa.Crop(px=(0, 10)),
            iaa.Fliplr(0.3),
            iaa.Flipud(0.3),
        ])

        images_aug, segmaps_aug = seq(images=images, segmentation_maps=segmaps)

        return images_aug, segmaps_aug

if __name__ == "__main__":
    pass