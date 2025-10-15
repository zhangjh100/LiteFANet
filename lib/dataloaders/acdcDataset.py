# -*- encoding: utf-8 -*-
import os
import glob

import cv2
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

import lib.utils as utils
import lib.transforms.two as my_transforms


class acdcDataset(Dataset):
    """
    load acdc dataset
    """

    def __init__(self, opt, mode):
        """
        initialize acdc dataset
        :param opt: params dict
        :param mode: train/valid
        """
        super(acdcDataset, self).__init__()
        self.opt = opt
        self.mode = mode
        self.root = opt["dataset_path"]
        self.train_dir = os.path.join(self.root, "train")
        self.valid_dir = os.path.join(self.root, "test")
        self.transforms_dict = {
            "train": my_transforms.Compose([
                my_transforms.RandomResizedCrop(self.opt["resize_shape"], scale=(0.4, 1.0), ratio=(3. / 4., 4. / 3.), interpolation='BILINEAR'),
                my_transforms.ColorJitter(brightness=self.opt["color_jitter"], contrast=self.opt["color_jitter"], saturation=self.opt["color_jitter"], hue=0),
                my_transforms.RandomGaussianNoise(p=self.opt["augmentation_p"]),
                my_transforms.RandomHorizontalFlip(p=self.opt["augmentation_p"]),
                my_transforms.RandomVerticalFlip(p=self.opt["augmentation_p"]),
                my_transforms.RandomRotation(self.opt["random_rotation_angle"]),
                my_transforms.Cutout(p=self.opt["augmentation_p"], value=(0, 0)),
                my_transforms.ToTensor(),
                my_transforms.Normalize(mean=self.opt["normalize_means"], std=self.opt["normalize_stds"])
            ]),
            "test": my_transforms.Compose([
                my_transforms.Resize(self.opt["resize_shape"]),
                my_transforms.ToTensor(),
                my_transforms.Normalize(mean=self.opt["normalize_means"], std=self.opt["normalize_stds"])
            ])
        }

        if mode == "train":
            self.images_list = sorted(glob.glob(os.path.join(self.train_dir, "images", "*.tif")))
            self.labels_list = sorted(glob.glob(os.path.join(self.train_dir, "annotations_0", "*.tif")))
        else:
            self.images_list = sorted(glob.glob(os.path.join(self.valid_dir, "images", "*.tif")))
            self.labels_list = sorted(glob.glob(os.path.join(self.valid_dir, "annotations_0", "*.tif")))

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        image = cv2.imread(self.images_list[index], -1)
        label = cv2.imread(self.labels_list[index], -1)

        if len(image.shape) == 2:  # 若图像是单通道（H, W）
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # 灰度图转 3 通道 BGR
        elif image.shape[2] == 1:  # 若图像是单通道带维度（H, W, 1）
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if len(label.shape) == 2:  # 若图像是单通道（H, W）
            label = cv2.cvtColor(label, cv2.COLOR_GRAY2BGR)  # 灰度图转 3 通道 BGR
        elif image.shape[2] == 1:  # 若图像是单通道带维度（H, W, 1）
            label = cv2.cvtColor(label, cv2.COLOR_GRAY2BGR)

        label[label == 171] = 1
        label[label == 114] = 2
        label[label == 57] = 3
        image, label = self.transforms_dict[self.mode](image, label)
        return image, label
