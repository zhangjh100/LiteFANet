# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh
@Contact  :   yykzhjh@163.com
@DateTime :   2024/01/01 00:33
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch

from torchvision import transforms


class ISIC2018Tester:
    """
    Tester class
    """

    def __init__(self, opt, model, metrics=None):
        self.opt = opt
        self.model = model
        self.metrics = metrics
        self.device = self.opt["device"]

        self.statistics_dict = self.init_statistics_dict()

    def inference(self, image_path):
        test_transforms = transforms.Compose([
            transforms.Resize(self.opt["resize_shape"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.opt["normalize_means"], std=self.opt["normalize_stds"])
        ])

        # 加载原图用于叠加
        image_pil = Image.open(image_path).convert("RGB")
        w, h = image_pil.size
        image_tensor = test_transforms(image_pil)

        dir_path, image_name = os.path.split(image_path)
        file_name = os.path.splitext(image_name)[0]
        segmentation_image_path = os.path.join(dir_path, file_name + "_segmentation.png")
        overlay_image_path = os.path.join(dir_path, file_name + "_overlay.png")

        self.model.eval()
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            output = self.model(image_tensor)

        # 获取预测 mask（0 或 1）
        segmented_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # print("Segmented mask (top-left 5x5):")
        # print(segmented_mask[:50, :50])

        # 放大到 0/255 便于可视化保存
        binary_mask = (segmented_mask * 255).astype(np.uint8)

        # resize 回原图尺寸
        binary_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(segmentation_image_path, binary_mask)
        print(f"✅ Saved segmentation mask to: {segmentation_image_path}")

        # 叠加可视化图像（原图 + 半透明 mask）
        base_image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        color_mask = np.zeros_like(base_image)
        color_mask[:, :, 2] = binary_mask  # 红色通道显示分割区域

        overlay = cv2.addWeighted(base_image, 0.7, color_mask, 0.3, 0)
        cv2.imwrite(overlay_image_path, overlay)
        print(f"✅ Saved overlay visualization to: {overlay_image_path}")






    def evaluation(self, dataloader):
        self.reset_statistics_dict()
        self.model.eval()

        total_images = 0  # 👈 统计总图像数

        with torch.no_grad():
            for input_tensor, target in tqdm(dataloader, leave=True):
                input_tensor, target = input_tensor.to(self.device), target.to(self.device)
                output = self.model(input_tensor)
                self.calculate_metric_and_update_statistcs(output.cpu(), target.cpu(), len(target))
                total_images += len(target)  # 👈 每个 batch 的样本数加起来

        class_IoU = self.statistics_dict["total_area_intersect"] / self.statistics_dict["total_area_union"]
        class_IoU = np.nan_to_num(class_IoU)
        dsc = self.statistics_dict["DSC_sum"] / self.statistics_dict["count"]
        JI = self.statistics_dict["JI_sum"] / self.statistics_dict["count"]
        ACC = self.statistics_dict["ACC_sum"] / self.statistics_dict["count"]

        print("\n✅ Total test images evaluated:", total_images)  # ✅ 打印总共评估了多少张
        print("valid_DSC:{:.6f}  valid_IoU:{:.6f}  valid_ACC:{:.6f}  valid_JI:{:.6f}".format(dsc, class_IoU[1], ACC, JI))

    def evaluate_all_metrics(self, dataloader):
        metrics_dict = {
            metric_name: []
            for metric_name in self.opt["metric_names"]
        }

        self.model.eval()
        with torch.no_grad():
            for input_tensor, target in dataloader:
                input_tensor, target = input_tensor.to(self.device), target.to(self.device)
                output = self.model(input_tensor).cpu().float()
                target = target.cpu().float()
                # calculate metrics
                for metric_name, metric_func in self.metrics.items():
                    if metric_name == "IoU":
                        area_intersect, area_union, _, _ = metric_func(output, target)
                        class_IoU = area_intersect.numpy() / area_union.numpy()
                        class_IoU = np.nan_to_num(class_IoU)
                        metrics_dict[metric_name].append(class_IoU[1])
                    elif metric_name == "DSC":
                        metrics_dict[metric_name].append(metric_func(output, target))
        return metrics_dict["DSC"], metrics_dict["IoU"]

    def calculate_metric_and_update_statistcs(self, output, target, cur_batch_size):
        mask = torch.zeros(self.opt["classes"])
        unique_index = torch.unique(target).int()
        for index in unique_index:
            mask[index] = 1
        self.statistics_dict["count"] += cur_batch_size
        for i, class_name in self.opt["index_to_class_dict"].items():
            if mask[i] == 1:
                self.statistics_dict["class_count"][class_name] += cur_batch_size
        for metric_name, metric_func in self.metrics.items():
            if metric_name == "IoU":
                area_intersect, area_union, _, _ = metric_func(output, target)
                self.statistics_dict["total_area_intersect"] += area_intersect.numpy()
                self.statistics_dict["total_area_union"] += area_union.numpy()
            elif metric_name == "ACC":
                batch_mean_ACC = metric_func(output, target)
                self.statistics_dict["ACC_sum"] += batch_mean_ACC * cur_batch_size
            elif metric_name == "JI":
                batch_mean_JI = metric_func(output, target)
                self.statistics_dict["JI_sum"] += batch_mean_JI * cur_batch_size
            elif metric_name == "DSC":
                batch_mean_DSC = metric_func(output, target)
                self.statistics_dict["DSC_sum"] += batch_mean_DSC * cur_batch_size
            else:
                per_class_metric = metric_func(output, target)
                per_class_metric = per_class_metric * mask
                self.statistics_dict[metric_name]["avg"] += (torch.sum(per_class_metric) / torch.sum(mask)).item() * cur_batch_size
                for j, class_name in self.opt["index_to_class_dict"].items():
                    self.statistics_dict[metric_name][class_name] += per_class_metric[j].item() * cur_batch_size

    def init_statistics_dict(self):
        statistics_dict = {
            metric_name: {class_name: 0.0 for _, class_name in self.opt["index_to_class_dict"].items()}
            for metric_name in self.opt["metric_names"]
        }
        statistics_dict["total_area_intersect"] = np.zeros((self.opt["classes"],))
        statistics_dict["total_area_union"] = np.zeros((self.opt["classes"],))
        statistics_dict["JI_sum"] = 0.0
        statistics_dict["ACC_sum"] = 0.0
        statistics_dict["DSC_sum"] = 0.0
        for metric_name in self.opt["metric_names"]:
            statistics_dict[metric_name]["avg"] = 0.0
        statistics_dict["class_count"] = {class_name: 0 for _, class_name in self.opt["index_to_class_dict"].items()}
        statistics_dict["count"] = 0

        return statistics_dict

    def reset_statistics_dict(self):
        self.statistics_dict["count"] = 0
        self.statistics_dict["total_area_intersect"] = np.zeros((self.opt["classes"],))
        self.statistics_dict["total_area_union"] = np.zeros((self.opt["classes"],))
        self.statistics_dict["JI_sum"] = 0.0
        self.statistics_dict["ACC_sum"] = 0.0
        self.statistics_dict["DSC_sum"] = 0.0
        for _, class_name in self.opt["index_to_class_dict"].items():
            self.statistics_dict["class_count"][class_name] = 0
        for metric_name in self.opt["metric_names"]:
            self.statistics_dict[metric_name]["avg"] = 0.0
            for _, class_name in self.opt["index_to_class_dict"].items():
                self.statistics_dict[metric_name][class_name] = 0.0

    def load(self):
        pretrain_state_dict = torch.load(self.opt["pretrain"], map_location=lambda storage, loc: storage.cuda(self.device))
        model_state_dict = self.model.state_dict()
        load_count = 0
        for param_name in model_state_dict.keys():
            if (param_name in pretrain_state_dict) and (model_state_dict[param_name].size() == pretrain_state_dict[param_name].size()):
                model_state_dict[param_name].copy_(pretrain_state_dict[param_name])
                load_count += 1
        self.model.load_state_dict(model_state_dict, strict=True)
        print("{:.2f}% of model parameters successfully loaded with training weights".format(100 * load_count / len(model_state_dict)))
