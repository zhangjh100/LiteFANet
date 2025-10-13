# -*- encoding: utf-8 -*-
import os
import glob

import cv2
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

import lib.utils as utils
import lib.transforms.two as my_transforms


class RevisionDataset(Dataset):
    """
    load MMOTU dataset
    """

    def __init__(self, opt, mode):
        """
        initialize MMOTU dataset
        :param opt: params dict
        :param mode: train/valid
        """
        super(RevisionDataset, self).__init__()
        self.opt = opt
        self.mode = mode
        self.root = opt["dataset_path"]
        self.train_dir = os.path.join(self.root, "train")
        self.valid_dir = os.path.join(self.root, "valid")
        self.transforms_dict = {
            "train": my_transforms.Compose([
                my_transforms.RandomResizedCrop(self.opt["resize_shape"], scale=(0.4, 1.0), ratio=(3. / 4., 4. / 3.),
                                                interpolation='BILINEAR'),
                my_transforms.ColorJitter(brightness=self.opt["color_jitter"], contrast=self.opt["color_jitter"],
                                          saturation=self.opt["color_jitter"], hue=0),
                my_transforms.RandomGaussianNoise(p=self.opt["augmentation_p"]),
                my_transforms.RandomHorizontalFlip(p=self.opt["augmentation_p"]),
                my_transforms.RandomVerticalFlip(p=self.opt["augmentation_p"]),
                my_transforms.RandomRotation(self.opt["random_rotation_angle"]),
                my_transforms.Cutout(p=self.opt["augmentation_p"], value=(0, 0)),
                my_transforms.ToTensor(),
                my_transforms.Normalize(mean=self.opt["normalize_means"], std=self.opt["normalize_stds"])
            ]),
            "valid": my_transforms.Compose([
                my_transforms.Resize(self.opt["resize_shape"]),
                my_transforms.ToTensor(),
                my_transforms.Normalize(mean=self.opt["normalize_means"], std=self.opt["normalize_stds"])
            ])
        }

        # if mode == "train":
        #     self.images_list = sorted(glob.glob(os.path.join(self.train_dir, "images", "*.JPG")))
        #     self.labels_list = sorted(glob.glob(os.path.join(self.train_dir, "labels", "*.PNG")))
        # else:
        #     self.images_list = sorted(glob.glob(os.path.join(self.valid_dir, "images", "*.JPG")))
        #     self.labels_list = sorted(glob.glob(os.path.join(self.valid_dir, "labels", "*.PNG")))
        if mode == "train":
            self.images_list = sorted(glob.glob(os.path.join(self.train_dir, "images", "*.JPG")))
            self.labels_list = sorted(glob.glob(os.path.join(self.train_dir, "labels", "*.JPG")))
        else:
            self.images_list = sorted(glob.glob(os.path.join(self.valid_dir, "images", "*.JPG")))
            self.labels_list = sorted(glob.glob(os.path.join(self.valid_dir, "labels", "*.JPG")))

        # ✅ 打印加载了多少张图片（调试用）
        # print(
        #      f"[DEBUG] Loaded {len(self.images_list)} images for mode = '{mode}' from path: {self.valid_dir if mode == 'valid' else self.train_dir}")
        print(
            f"[DEBUG] Loaded {len(self.images_list)} images for mode = '{mode}' from: {self.train_dir if mode == 'train' else self.valid_dir}")
        print(
            f"[DEBUG] Loaded {len(self.labels_list)} labels for mode = '{mode}' from: {self.train_dir if mode == 'train' else self.valid_dir}")

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        image = cv2.imread(self.images_list[index], -1)
        label = cv2.imread(self.labels_list[index], -1)

        # 2. 把所有非 0 的像素（255）都当作前景 1
        #    你可以用 ==255 或者 >127，都行：
        label = (label > 127).astype(np.uint8)

        image, label = self.transforms_dict[self.mode](image, label)
        return image, label



