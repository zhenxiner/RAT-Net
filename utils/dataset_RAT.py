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