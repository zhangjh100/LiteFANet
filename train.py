# -*- encoding: utf-8 -*-
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
# Ignore MONAI tf32 warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"monai\.utils\.tf32"
)
# Ignore PyTorch bilinear align_corners warning
warnings.filterwarnings(
    "ignore",
    message=r".*Default upsampling behavior when mode=bilinear.*",
    category=UserWarning
)

import argparse
import os
import time
import torch

import nni

from lib import utils, dataloaders, models, losses, metrics, trainers

params_3D_CBCT_Tooth = {
    # ——————————————————————————————————————————————    Launch Initialization    —————————————————————————————————————————————————
    "CUDA_VISIBLE_DEVICES": "0",
    "seed": 1777777,
    "cuda": True,
    "benchmark": False,
    "deterministic": True,
    # —————————————————————————————————————————————     Preprocessing      ————————————————————————————————————————————————————
    "resample_spacing": [0.5, 0.5, 0.5],
    "clip_lower_bound": -1412,
    "clip_upper_bound": 17943,
    "samples_train": 2048,
    "crop_size": (160, 160, 96),
    "crop_threshold": 0.5,
    # ——————————————————————————————————————————————    Data Augmentation    ——————————————————————————————————————————————————————
    "augmentation_probability": 0.3,
    "augmentation_method": "Choice",
    "open_elastic_transform": True,
    "elastic_transform_sigma": 20,
    "elastic_transform_alpha": 1,
    "open_gaussian_noise": True,
    "gaussian_noise_mean": 0,
    "gaussian_noise_std": 0.01,
    "open_random_flip": True,
    "open_random_rescale": True,
    "random_rescale_min_percentage": 0.5,
    "random_rescale_max_percentage": 1.5,
    "open_random_rotate": True,
    "random_rotate_min_angle": -50,
    "random_rotate_max_angle": 50,
    "open_random_shift": True,
    "random_shift_max_percentage": 0.3,
    "normalize_mean": 0.05029342141696459,
    "normalize_std": 0.028477091559295814,
    # —————————————————————————————————————————————    Data Loading     ——————————————————————————————————————————————————————
    "dataset_name": "3D-CBCT-Tooth",
    "dataset_path": r"./datasets/3D-CBCT-Tooth",
    "create_data": False,
    "batch_size": 1,
    "num_workers": 2,
    # —————————————————————————————————————————————    Model     ——————————————————————————————————————————————————————
    "model_name": "LiteFANet",
    "in_channels": 1,
    "classes": 2,
    "scaling_version": "Basic",
    "dimension": "3d",
    "index_to_class_dict":
    {
        0: "background",
        1: "foreground"
    },
    "resume": None,
    "pretrain": None,
    # ——————————————————————————————————————————————    Optimizer     ——————————————————————————————————————————————————————
    "optimizer_name": "Adam",
    "learning_rate": 0.0005,
    "weight_decay": 0.00005,
    "momentum": 0.8,
    # ———————————————————————————————————————————    Learning Rate Scheduler     —————————————————————————————————————————————————————
    "lr_scheduler_name": "ReduceLROnPlateau",
    "gamma": 0.1,
    "step_size": 9,
    "milestones": [1, 3, 5, 7, 8, 9],
    "T_max": 2,
    "T_0": 2,
    "T_mult": 2,
    "mode": "max",
    "patience": 1,
    "factor": 0.5,
    # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
    "metric_names": ["DSC"],
    "loss_function_name": "DiceLoss",
    "class_weight": [0.00551122, 0.99448878],
    "sigmoid_normalization": False,
    "dice_loss_mode": "extension",
    "dice_mode": "standard",
    # —————————————————————————————————————————————   Training   ——————————————————————————————————————————————————————
    "optimize_params": False,
    "use_amp": False,
    "run_dir": r"./runs",
    "start_epoch": 0,
    "end_epoch": 20,
    "best_dice": 0,
    "update_weight_freq": 32,
    "terminal_show_freq": 256,
    "save_epoch_freq": 4,
    # ————————————————————————————————————————————   Testing   ———————————————————————————————————————————————————————
    "crop_stride": [32, 32, 32]
}

params_MMOTU = {
    # ——————————————————————————————————————————————     Launch Initialization    ———————————————————————————————————————————————————
    "CUDA_VISIBLE_DEVICES": "0",
    "seed": 1777777,
    "cuda": True,
    "benchmark": False,
    "deterministic": True,
    # —————————————————————————————————————————————     Preprocessing       ————————————————————————————————————————————————————
    "resize_shape": (224, 224),
    # ——————————————————————————————————————————————    Data Augmentation    ——————————————————————————————————————————————————————
    "augmentation_p": 0.12097393901893663,
    "color_jitter": 0.4203933474361258,
    "random_rotation_angle": 30,
    "normalize_means": (0.22250386, 0.21844882, 0.21521868),
    "normalize_stds": (0.21923075, 0.21622984, 0.21370508),
    # —————————————————————————————————————————————    Data Loading     ——————————————————————————————————————————————————————
    "dataset_name": "MMOTU",
    "dataset_path": r"./datasets/MMOTU",
    "batch_size": 1,  #64
    "num_workers": 4,
    # —————————————————————————————————————————————    Model     ——————————————————————————————————————————————————————
    "model_name": "LiteFANet",
    "in_channels": 3,
    "classes": 2,
    "scaling_version": "Ultra",
    "dimension": "2d",
    "index_to_class_dict":
    {
        0: "background",
        1: "foreground"
    },
    "resume": None,
    "pretrain": None,
    # ——————————————————————————————————————————————    Optimizer     ——————————————————————————————————————————————————————
    "optimizer_name": "AdamW",
    "learning_rate": 0.01,
    "weight_decay": 0.00001,
    "momentum": 0.7725414416309884,
    # ———————————————————————————————————————————    Learning Rate Scheduler     —————————————————————————————————————————————————————
    "lr_scheduler_name": "CosineAnnealingLR",
    "gamma": 0.8689275449032848,
    "step_size": 5,
    "milestones": [10, 30, 60, 100, 120, 140, 160, 170],
    "T_max": 200,
    "T_0": 10,
    "T_mult": 5,
    "mode": "max",
    "patience": 1,
    "factor": 0.97,
    # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
    "metric_names": ["DSC", "IoU"],
    "loss_function_name": "DiceLoss",
    "class_weight": [0.2350689696563569, 1-0.2350689696563569],
    "sigmoid_normalization": False,
    "dice_loss_mode": "extension",
    "dice_mode": "standard",
    # —————————————————————————————————————————————   Training   ——————————————————————————————————————————————————————
    "optimize_params": False,
    "run_dir": r"./runs",
    "start_epoch": 0,
    "end_epoch": 5000,
    "best_metric": 0,
    "terminal_show_freq": 8,
    "save_epoch_freq": 500,
}

params_ISIC_2018 = {
    # ——————————————————————————————————————————————     Launch Initialization    ———————————————————————————————————————————————————
    "CUDA_VISIBLE_DEVICES": "0",
    "seed": 1777777,
    "cuda": True,
    "benchmark": False,
    "deterministic": True,
    # —————————————————————————————————————————————     Preprocessing       ————————————————————————————————————————————————————
    "resize_shape": (224, 224),
    # ——————————————————————————————————————————————    Data Augmentation    ——————————————————————————————————————————————————————
    "augmentation_p": 0.1,
    "color_jitter": 0.37,
    "random_rotation_angle": 15,
    "normalize_means": (0.50297405, 0.54711632, 0.71049083),
    "normalize_stds": (0.18653496, 0.17118206, 0.17080363),
    # —————————————————————————————————————————————    Data Loading     ——————————————————————————————————————————————————————
    "dataset_name": "ISIC-2018",
    "dataset_path": r"./datasets/tusimple",
    "batch_size": 32,
    "num_workers": 2,
    # —————————————————————————————————————————————    Model     ——————————————————————————————————————————————————————
    "model_name": "LiteFANet",
    "in_channels": 3,
    "classes": 2,
    "scaling_version": "Ultra",
    "dimension": "2d",
    "index_to_class_dict":
    {
        0: "background",
        1: "foreground"
    },
    "resume": None,
    "pretrain": None,
    # ——————————————————————————————————————————————    Optimizer     ——————————————————————————————————————————————————————
    "optimizer_name": "AdamW",
    "learning_rate": 0.005,
    "weight_decay": 0.000001,
    "momentum": 0.9657205586290213,
    # ———————————————————————————————————————————    Learning Rate Scheduler     —————————————————————————————————————————————————————
    "lr_scheduler_name": "CosineAnnealingWarmRestarts",
    "gamma": 0.9582311026945434,
    "step_size": 20,
    "milestones": [1, 3, 5, 7, 8, 9],
    "T_max": 100,
    "T_0": 5,
    "T_mult": 5,
    "mode": "max",
    "patience": 20,
    "factor": 0.3,
    # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
    "metric_names": ["DSC", "IoU", "JI", "ACC"],
    "loss_function_name": "DiceLoss",
    "class_weight": [0.029, 1-0.029],
    "sigmoid_normalization": False,
    "dice_loss_mode": "extension",
    "dice_mode": "standard",
    # —————————————————————————————————————————————   Training   ——————————————————————————————————————————————————————
    "optimize_params": False,
    "run_dir": r"./runs",
    "start_epoch": 0,
    "end_epoch": 150,
    "best_metric": 0,
    "terminal_show_freq": 20,
    "save_epoch_freq": 50,
}

params_DRIVE = {
    # ——————————————————————————————————————————————     Launch Initialization    ———————————————————————————————————————————————————
    "CUDA_VISIBLE_DEVICES": "0",
    "seed": 1777777,
    "cuda": True,
    "benchmark": False,
    "deterministic": True,
    # —————————————————————————————————————————————     Preprocessing       ————————————————————————————————————————————————————
    "resize_shape": (512, 512),
    # ——————————————————————————————————————————————    Data Augmentation    ——————————————————————————————————————————————————————
    "augmentation_p": 0.31154673834507396,
    "color_jitter": 0.20193812578236442,
    "random_rotation_angle": 45,
    "normalize_means": (0.16155026, 0.26819696, 0.50784565),
    "normalize_stds": (0.10571646, 0.18532471, 0.35080457),
    # —————————————————————————————————————————————    Data Loading     ——————————————————————————————————————————————————————
    "dataset_name": "DRIVE",
    "dataset_path": r"./datasets/DRIVE",
    "batch_size": 2,
    "num_workers": 2,
    # —————————————————————————————————————————————    Model     ——————————————————————————————————————————————————————
    "model_name": "LiteFANet",
    "in_channels": 3,
    "classes": 2,
    "scaling_version": "Ultra",
    "dimension": "2d",
    "index_to_class_dict":
    {
        0: "background",
        1: "foreground"
    },
    "resume": None,
    "pretrain": None,
    # ——————————————————————————————————————————————    Optimizer     ——————————————————————————————————————————————————————
    "optimizer_name": "AdamW",
    "learning_rate": 0.005,
    "weight_decay": 0.001,
    "momentum": 0.705859940948433,
    # ———————————————————————————————————————————    Learning Rate Scheduler     —————————————————————————————————————————————————————
    "lr_scheduler_name": "CosineAnnealingWarmRestarts",
    "gamma": 0.9251490005593288,
    "step_size": 10,
    "milestones": [1, 3, 5, 7, 8, 9],
    "T_max": 50,
    "T_0": 10,
    "T_mult": 4,
    "mode": "max",
    "patience": 5,
    "factor": 0.8,
    # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
    "metric_names": ["DSC", "IoU", "JI", "ACC"],
    "loss_function_name": "DiceLoss",
    "class_weight": [0.08631576554733908, 0.913684234452661],
    "sigmoid_normalization": False,
    "dice_loss_mode": "extension",
    "dice_mode": "standard",
    # —————————————————————————————————————————————   Training   ——————————————————————————————————————————————————————
    "optimize_params": False,
    "run_dir": r"./runs",
    "start_epoch": 0,
    "end_epoch": 200,
    "best_metric": 0,
    "terminal_show_freq": 5,
    "save_epoch_freq": 50,
}

params_STARE = {
    # ——————————————————————————————————————————————     Launch Initialization    ———————————————————————————————————————————————————
    "CUDA_VISIBLE_DEVICES": "0",
    "seed": 1777777,
    "cuda": True,
    "benchmark": False,
    "deterministic": True,
    # —————————————————————————————————————————————     Preprocessing       ————————————————————————————————————————————————————
    "resize_shape": (512, 512),
    # ——————————————————————————————————————————————    Data Augmentation    ——————————————————————————————————————————————————————
    "augmentation_p": 0.31154673834507396,
    "color_jitter": 0.20193812578236442,
    "random_rotation_angle": 45,
    "normalize_means": (0.11336552, 0.33381058, 0.58892505),
    "normalize_stds": (0.10905356, 0.19210595, 0.35295892),
    # —————————————————————————————————————————————    Data Loading     ——————————————————————————————————————————————————————
    "dataset_name": "STARE",
    "dataset_path": r"./datasets/STARE",
    "batch_size": 2,
    "num_workers": 2,
    # —————————————————————————————————————————————    Model     ——————————————————————————————————————————————————————
    "model_name": "LiteFANet",
    "in_channels": 3,
    "classes": 2,
    "scaling_version": "Ultra",
    "dimension": "2d",
    "index_to_class_dict":
    {
        0: "background",
        1: "foreground"
    },
    "resume": None,
    "pretrain": None,
    # ——————————————————————————————————————————————    Optimizer     ——————————————————————————————————————————————————————
    "optimizer_name": "AdamW",
    "learning_rate": 0.005,
    "weight_decay": 0.001,
    "momentum": 0.705859940948433,
    # ———————————————————————————————————————————    Learning Rate Scheduler     —————————————————————————————————————————————————————
    "lr_scheduler_name": "CosineAnnealingWarmRestarts",
    "gamma": 0.9251490005593288,
    "step_size": 10,
    "milestones": [1, 3, 5, 7, 8, 9],
    "T_max": 50,
    "T_0": 10,
    "T_mult": 4,
    "mode": "max",
    "patience": 5,
    "factor": 0.8,
    # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
    "metric_names": ["DSC", "IoU", "JI", "ACC"],
    "loss_function_name": "DiceLoss",
    "class_weight": [0.07542384887839432, 0.9245761511216056],
    "sigmoid_normalization": False,
    "dice_loss_mode": "extension",
    "dice_mode": "standard",
    # —————————————————————————————————————————————   Training   ——————————————————————————————————————————————————————
    "optimize_params": False,
    "run_dir": r"./runs",
    "start_epoch": 0,
    "end_epoch": 200,
    "best_metric": 0,
    "terminal_show_freq": 2,
    "save_epoch_freq": 50,
}

params_CHASE_DB1 = {
    # ——————————————————————————————————————————————     Launch Initialization    ———————————————————————————————————————————————————
    "CUDA_VISIBLE_DEVICES": "0",
    "seed": 1777777,
    "cuda": True,
    "benchmark": False,
    "deterministic": True,
    # —————————————————————————————————————————————     Preprocessing       ————————————————————————————————————————————————————
    "resize_shape": (512, 512),
    # ——————————————————————————————————————————————    Data Augmentation    ——————————————————————————————————————————————————————
    "augmentation_p": 0.31154673834507396,
    "color_jitter": 0.20193812578236442,
    "random_rotation_angle": 45,
    "normalize_means": (0.02789665, 0.16392259, 0.45287978),
    "normalize_stds": (0.03700363, 0.14539037, 0.36542216),
    # —————————————————————————————————————————————    Data Loading     ——————————————————————————————————————————————————————
    "dataset_name": "CHASE-DB1",
    "dataset_path": r"./datasets/CHASE-DB1",
    "batch_size": 2,
    "num_workers": 2,
    # —————————————————————————————————————————————    Model     ——————————————————————————————————————————————————————
    "model_name": "LiteFANet",
    "in_channels": 3,
    "classes": 2,
    "scaling_version": "Ultra",
    "dimension": "2d",
    "index_to_class_dict":
    {
        0: "background",
        1: "foreground"
    },
    "resume": None,
    "pretrain": None,
    # ——————————————————————————————————————————————    Optimizer     ——————————————————————————————————————————————————————
    "optimizer_name": "AdamW",
    "learning_rate": 0.005,
    "weight_decay": 0.001,
    "momentum": 0.705859940948433,
    # ———————————————————————————————————————————    Learning Rate Scheduler     —————————————————————————————————————————————————————
    "lr_scheduler_name": "CosineAnnealingWarmRestarts",
    "gamma": 0.9251490005593288,
    "step_size": 10,
    "milestones": [1, 3, 5, 7, 8, 9],
    "T_max": 50,
    "T_0": 10,
    "T_mult": 4,
    "mode": "max",
    "patience": 5,
    "factor": 0.8,
    # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
    "metric_names": ["DSC", "IoU", "JI", "ACC"],
    "loss_function_name": "DiceLoss",
    "class_weight": [0.07186707540874207, 0.928132924591258],
    "sigmoid_normalization": False,
    "dice_loss_mode": "extension",
    "dice_mode": "standard",
    # —————————————————————————————————————————————   Training   ——————————————————————————————————————————————————————
    "optimize_params": False,
    "run_dir": r"./runs",
    "start_epoch": 0,
    "end_epoch": 200,
    "best_metric": 0,
    "terminal_show_freq": 5,
    "save_epoch_freq": 50,
}

params_Kvasir_SEG = {
    # ——————————————————————————————————————————————     Launch Initialization    ———————————————————————————————————————————————————
    "CUDA_VISIBLE_DEVICES": "0",
    "seed": 1777777,
    "cuda": True,
    "benchmark": False,
    "deterministic": True,
    # —————————————————————————————————————————————     Preprocessing       ————————————————————————————————————————————————————
    "resize_shape": (224, 224),
    # ——————————————————————————————————————————————    Data Augmentation    ——————————————————————————————————————————————————————
    "augmentation_p": 0.22448543324157222,
    "color_jitter": 0.3281010563062837,
    "random_rotation_angle": 30,
    "normalize_means": (0.24398195, 0.32772844, 0.56273),
    "normalize_stds": (0.18945072, 0.2217485, 0.31491405),
    # —————————————————————————————————————————————    Data Loading     ——————————————————————————————————————————————————————
    "dataset_name": "Kvasir-SEG",
    "dataset_path": r"./datasets/Kvasir-SEG",
    "batch_size": 32,
    "num_workers": 2,
    # —————————————————————————————————————————————    Model     ——————————————————————————————————————————————————————
    "model_name": "LiteFANet",
    "in_channels": 3,
    "classes": 2,
    "scaling_version": "Ultra",
    "dimension": "2d",
    "index_to_class_dict":
    {
        0: "background",
        1: "foreground"
    },
    "resume": None,
    "pretrain": None,
    # ——————————————————————————————————————————————    Optimizer     ——————————————————————————————————————————————————————
    "optimizer_name": "AdamW",
    "learning_rate": 0.0005,
    "weight_decay": 0.000001,
    "momentum": 0.7781834740942233,
    # ———————————————————————————————————————————    Learning Rate Scheduler     —————————————————————————————————————————————————————
    "lr_scheduler_name": "CosineAnnealingLR",
    "gamma": 0.8079569870480704,
    "step_size": 20,
    "milestones": [10, 30, 60, 100, 120, 140, 160, 170],
    "T_max": 500,
    "T_0": 10,
    "T_mult": 2,
    "mode": "max",
    "patience": 5,
    "factor": 0.91,
    # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
    "metric_names": ["DSC", "IoU", "JI", "ACC"],
    "loss_function_name": "DiceLoss",
    "class_weight": [0.1557906849111095, 0.8442093150888904],
    "sigmoid_normalization": False,
    "dice_loss_mode": "extension",
    "dice_mode": "standard",
    # —————————————————————————————————————————————   Training   ——————————————————————————————————————————————————————
    "optimize_params": False,
    "run_dir": r"./runs",
    "start_epoch": 0,
    "end_epoch": 500,
    "best_metric": 0,
    "terminal_show_freq": 8,
    "save_epoch_freq": 150,
}

params_acdc = {
    # ——————————————————————————————————————————————     Launch Initialization    ———————————————————————————————————————————————————
    "CUDA_VISIBLE_DEVICES": "0",
    "seed": 1777777,
    "cuda": True,
    "benchmark": False,
    "deterministic": True,
    # —————————————————————————————————————————————     Preprocessing       ————————————————————————————————————————————————————
    "resize_shape": (256, 256),
    # ——————————————————————————————————————————————    Data Augmentation    ——————————————————————————————————————————————————————
    "augmentation_p": 0.22448543324157222,
    "color_jitter": 0.3281010563062837,
    "random_rotation_angle": 30,
    "normalize_means": (31.129348754882812, 31.129348754882812, 31.129348754882812),
    "normalize_stds": (42.57401657104492, 42.57401657104492, 42.57401657104492),
    # —————————————————————————————————————————————    Data Loading     ——————————————————————————————————————————————————————
    "dataset_name": "acdc",
    "dataset_path": r"./datasets/acdc",
    "batch_size": 8,
    "num_workers": 4,
    # —————————————————————————————————————————————    Model     ——————————————————————————————————————————————————————
    "model_name": "CENet",
    "in_channels": 1,
    "classes": 4,
    # "scaling_version": "Ultra",
    "dimension": "2d",
    "index_to_class_dict":
    {
        0: "background",
        1: "LV",
        2:"Myo",
        3:"RV"
    },
    "resume": None,
    "pretrain": None,
    # ——————————————————————————————————————————————    Optimizer     ——————————————————————————————————————————————————————
    "optimizer_name": "AdamW",
    "learning_rate": 0.001,
    "weight_decay": 0.00005,
    "momentum": 0.7781834740942233,
    # ———————————————————————————————————————————    Learning Rate Scheduler     —————————————————————————————————————————————————————
    "lr_scheduler_name": "CosineAnnealingLR",
    "gamma": 0.8079569870480704,
    "step_size": 20,
    "milestones": [10, 30, 60, 100, 120, 140, 160, 170],
    "T_max": 500,
    "T_0": 10,
    "T_mult": 2,
    "mode": "max",
    "patience": 5,
    "factor": 0.91,
    # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
    "metric_names": ["DSC", "IoU", "JI", "ACC"],
    "loss_function_name": "DiceLoss",
    "class_weight": [0.1557906849111095, 0.8442093150888904],
    "sigmoid_normalization": False,
    "dice_loss_mode": "extension",
    "dice_mode": "standard",
    # —————————————————————————————————————————————   Training   ——————————————————————————————————————————————————————
    "optimize_params": False,
    "run_dir": r"./runs",
    "start_epoch": 0,
    "end_epoch": 500,
    "best_metric": 0,
    "terminal_show_freq": 8,
    "save_epoch_freq": 150,
}

# （省略其余 params 定义，实际脚本中保留）

def parse_args():
    parser = argparse.ArgumentParser(description='Train with metrics')
    parser.add_argument('-d', '--dataset', type=str, default='3D-CBCT-Tooth')
    parser.add_argument('-m', '--model', type=str, default='LiteFANet')
    parser.add_argument('-pre', '--pretrain_weight', type=str, default=None)
    parser.add_argument('-dim', '--dimension', type=str, default='2d')
    parser.add_argument('-s', '--scaling_version', type=str, default='Basic')
    parser.add_argument('--epoch', type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    # select params dict
    if args.dataset == "3D-CBCT-Tooth":
        params = params_3D_CBCT_Tooth
    elif args.dataset == "MMOTU":
        params = params_MMOTU
    elif args.dataset == "ISIC-2018":
        params = params_ISIC_2018
    elif args.dataset == "DRIVE":
        params = params_DRIVE
    elif args.dataset == "STARE":
        params = params_STARE
    elif args.dataset == "CHASE-DB1":
        params = params_CHASE_DB1
    elif args.dataset == "Kvasir-SEG":
        params = params_Kvasir_SEG
    elif args.dataset == "acdc":
        params = params_acdc
    else:
        raise RuntimeError(f"No {args.dataset} dataset available")

    # update params
    params["dataset_name"] = args.dataset
    params["dataset_path"] = os.path.join(r"./datasets", args.dataset)
    params["model_name"] = args.model
    if args.pretrain_weight is not None:
        params["pretrain"] = args.pretrain_weight
    params["dimension"] = args.dimension
    params["scaling_version"] = args.scaling_version
    if args.epoch is not None:
        params["end_epoch"] = args.epoch
        params["save_epoch_freq"] = max(1, args.epoch // 4)

    if params.get("optimize_params", False):
        tuner_params = nni.get_next_parameter()
        params.update(tuner_params)

    os.environ["CUDA_VISIBLE_DEVICES"] = params["CUDA_VISIBLE_DEVICES"]
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    utils.reproducibility(params["seed"], params["deterministic"], params["benchmark"])

    print("Built with CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    device = torch.device("cuda:0" if params["cuda"] and torch.cuda.is_available() else "cpu")
    params["device"] = device
    print("Using device:", device)

    train_loader, valid_loader = dataloaders.get_dataloader(params)
    print("Complete the initialization of dataloader")

    model, optimizer, lr_scheduler = models.get_model_optimizer_lr_scheduler(params)
    model.to(device)
    print(f"Complete the initialization of model:{params['model_name']}, optimizer:{params['optimizer_name']}, and lr_scheduler:{params['lr_scheduler_name']}")

    loss_function = losses.get_loss_function(params)
    print("Complete the initialization of loss function")

    metric = metrics.get_metric(params)
    print("Complete the initialization of metrics")

    # -----------------------------------------------
    # 计算模型参数量、FLOPs 和 FPS
    # 参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params/1e6:.8f} M")

    # 确定输入尺寸
    if params["dimension"] == "3d":
        dummy_shape = (1, params["in_channels"], *params.get("crop_size", (160,160,96)))
    else:
        dummy_shape = (1, params["in_channels"], *params.get("resize_shape", (256,256)))

    # —— 1. FLOPs（在 CPU 上） ——
    try:
        from thop import profile

        model_cpu = model.cpu()
        dummy_cpu = torch.randn(dummy_shape)

        if hasattr(model_cpu, 'forward_dummy'):
            backup_fwd = model_cpu.forward
            model_cpu.forward = model_cpu.forward_dummy

        macs, _ = profile(model_cpu, inputs=(dummy_cpu,), verbose=False)

        if hasattr(model_cpu, 'forward_dummy'):
            model_cpu.forward = backup_fwd

        flops = 2 * macs
        print(f"Model FLOPs: {flops/1e9:.8f} G")

        model.to(device)
    except ImportError:
        print("Please install thop: pip install thop to compute FLOPs")
    except Exception as e:
        print(f"FLOPs calculation failed: {e}")

    # —— 2. FPS（在 GPU 上） ——
    try:
        dummy_gpu = torch.randn(dummy_shape, device=device)

        model.eval()
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_gpu)

        torch.cuda.synchronize()
        repeats = 100
        start = time.time()
        for _ in range(repeats):
            _ = model(dummy_gpu)
        torch.cuda.synchronize()
        fps = repeats / (time.time() - start)

        print(f"Inference speed: {fps:.1f} FPS (batch size=1)")
        model.train()
    except Exception as e:
        print(f"FPS measurement failed: {e}")
    # -----------------------------------------------

    trainer = trainers.get_trainer(params, train_loader, valid_loader,
                                     model, optimizer, lr_scheduler,
                                     loss_function, metric)

    if params.get("resume") or params.get("pretrain"):
        trainer.load()
    print("Complete the initialization of trainer")

    trainer.training()

if __name__ == '__main__':
    main()
