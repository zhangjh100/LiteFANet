# -*- encoding: utf-8 -*-
# import os
# import glob
#
# import cv2
# import numpy as np
#
# from torch.utils.data import Dataset
# import lib.transforms.two as my_transforms
#
#
# class KvasirSEGDataset(Dataset):
#     """
#     load Kvasir-SEG dataset
#     """
#
#     def __init__(self, opt, mode):
#         """
#         initialize Kvasir-SEG dataset
#         :param opt: params dict
#         :param mode: train/valid
#         """
#         super(KvasirSEGDataset, self).__init__()
#         self.opt = opt
#         self.mode = mode
#         self.root = opt["dataset_path"]
#         self.train_dir = os.path.join(self.root, "train")
#         self.valid_dir = os.path.join(self.root, "test")
#
#         self.transforms_dict = {
#             "train": my_transforms.Compose([
#                 my_transforms.RandomResizedCrop(self.opt["resize_shape"], scale=(0.4, 1.0), ratio=(3. / 4., 4. / 3.), interpolation='BILINEAR'),
#                 my_transforms.ColorJitter(brightness=self.opt["color_jitter"], contrast=self.opt["color_jitter"], saturation=self.opt["color_jitter"], hue=0),
#                 my_transforms.RandomGaussianNoise(p=self.opt["augmentation_p"]),
#                 my_transforms.RandomHorizontalFlip(p=self.opt["augmentation_p"]),
#                 my_transforms.RandomVerticalFlip(p=self.opt["augmentation_p"]),
#                 my_transforms.RandomRotation(self.opt["random_rotation_angle"]),
#                 my_transforms.Cutout(p=self.opt["augmentation_p"], value=(0, 0)),
#                 my_transforms.ToTensor(),
#                 my_transforms.Normalize(mean=self.opt["normalize_means"], std=self.opt["normalize_stds"])
#             ]),
#             "test": my_transforms.Compose([
#                 my_transforms.Resize(self.opt["resize_shape"]),
#                 my_transforms.ToTensor(),
#                 my_transforms.Normalize(mean=self.opt["normalize_means"], std=self.opt["normalize_stds"])
#             ])
#         }
#
#         if mode == "train":
#             self.images_list = sorted(glob.glob(os.path.join(self.train_dir, "images", "*.jpg")))
#             self.labels_list = sorted(glob.glob(os.path.join(self.train_dir, "annotations", "*.jpg")))
#         else:
#             self.images_list = sorted(glob.glob(os.path.join(self.valid_dir, "images", "*.jpg")))
#             self.labels_list = sorted(glob.glob(os.path.join(self.valid_dir, "annotations", "*.jpg")))
#
#         print(f"[DEBUG] Loaded {len(self.images_list)} images for mode = '{mode}' from: {self.train_dir if mode == 'train' else self.valid_dir}")
#         print(f"[DEBUG] Loaded {len(self.labels_list)} labels for mode = '{mode}' from: {self.train_dir if mode == 'train' else self.valid_dir}")
#
#     def __len__(self):
#         return len(self.images_list)
#
#     def __getitem__(self, index):
#         image = cv2.imread(self.images_list[index], cv2.IMREAD_COLOR)
#         label = cv2.imread(self.labels_list[index], cv2.IMREAD_GRAYSCALE)  # ✅ 读取为灰度图
#
#         if label is None:
#             raise RuntimeError(f"Label not found or unreadable: {self.labels_list[index]}")
#         if image is None:
#             raise RuntimeError(f"Image not found or unreadable: {self.images_list[index]}")
#
#         # 将所有非0像素视为前景
#         label = (label > 127).astype(np.uint8)
#
#         # 确保 label 维度为 (H, W, 1)
#         if label.ndim == 2:
#             label = label[:, :, np.newaxis]
#
#         image, label = self.transforms_dict[self.mode](image, label)
#         return image, label



# -*- encoding: utf-8 -*-
import os
import glob

import cv2
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

import lib.utils as utils
import lib.transforms.two as my_transforms


class KvasirSEGDataset(Dataset):
    """
    load ISIC 2018 dataset
    """

    def __init__(self, opt, mode):
        """
        initialize ISIC 2018 dataset
        :param opt: params dict
        :param mode: train/valid
        """
        super(KvasirSEGDataset, self).__init__()
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
            self.images_list = sorted(glob.glob(os.path.join(self.train_dir, "images", "*.jpg")))
            self.labels_list = sorted(glob.glob(os.path.join(self.train_dir, "annotations", "*.png")))
        else:
            self.images_list = sorted(glob.glob(os.path.join(self.valid_dir, "images", "*.jpg")))
            self.labels_list = sorted(glob.glob(os.path.join(self.valid_dir, "annotations", "*.png")))

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        image = cv2.imread(self.images_list[index], -1)
        label = cv2.imread(self.labels_list[index], -1)
        label[label == 255] = 1
        image, label = self.transforms_dict[self.mode](image, label)
        return image, label
