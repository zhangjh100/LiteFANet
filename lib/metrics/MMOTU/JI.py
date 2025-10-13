# -*- encoding: utf-8 -*-
import math
import torch
import torch.nn as nn
import numpy as np

import sys
sys.path.append(r"D:\Projects\Python\3D-tooth-segmentation\PMFS-Net：Polarized Multi-scale Feature Self-attention Network For CBCT Tooth Segmentation\my-2d")
from lib.utils import *



class JI(object):
    def __init__(self, num_classes=33, sigmoid_normalization=False):
        """
        定义JI评价指标计算器

        :param num_classes: 类别数
        :param sigmoid_normalization: 对网络输出采用sigmoid归一化方法，否则采用softmax
        """
        super(JI, self).__init__()
        # 初始化参数
        self.num_classes = num_classes
        # 初始化sigmoid或者softmax归一化方法
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)


    def __call__(self, input, target):
        """
        JI

        :param input: 网络模型输出的预测图,(B, C, H, W)
        :param target: 标注图像,(B, H, W)
        :return:
        """
        # 对预测图进行Sigmiod或者Sofmax归一化操作
        input = self.normalization(input)

        # 将预测图像进行分割
        seg = torch.argmax(input, dim=1)
        # 判断预测图和真是标签图的维度大小是否一致
        assert seg.shape == target.shape, "seg和target的维度大小不一致"
        # 转换seg和target数据类型为numpy.ndarray
        seg = seg.numpy().astype(float)
        target = target.numpy().astype(float)

        return cal_jaccard_index(seg, target)




if __name__ == '__main__':
    random.seed(123)  # 为python设置随机种子
    os.environ['PYTHONHASHSEED'] = str(123)
    np.random.seed(123)  # 为numpy设置随机种子
    torch.manual_seed(123)  # 为CPU设置随机种子

    pred = torch.randn((4, 2, 32, 32))
    gt = torch.randint(2, (4, 32, 32))

    JI_metric = JI(num_classes=2)

    ji = JI_metric(pred, gt)

    print(ji)




















