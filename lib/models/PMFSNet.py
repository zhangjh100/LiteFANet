# -*- encoding: utf-8 -*-
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))

import torch
import torch.nn as nn

from lib.models.modules.ConvBlock import ConvBlock
from lib.models.modules.LocalPMFSBlock import DownSampleWithLocalPMFSBlock
from lib.models.modules.GlobalPMFSBlock import GlobalPMFSBlock_AP_Separate





class LiteFANet(nn.Module):
    def __init__(self, in_channels=1, out_channels=35, dim="3d", scaling_version="Basic",
                 Ultra_module=DownSampleWithLocalPMFSBlock,
                 global_module=GlobalPMFSBlock_AP_Separate):
        super(LiteFANet, self).__init__()

        self.scaling_version = scaling_version

        # 根据模型规模定义参数（越大越重）
        if scaling_version == "Ultra":
            base_channels = [24, 48, 64]
            skip_channels = [24, 48, 64]
            units = [5, 10, 10]
            pmfs_ch = 64
        elif scaling_version == "Pro":
            base_channels = [24, 24, 24]
            skip_channels = [12, 24, 24]
            units = [5, 10, 10]
            pmfs_ch = 48
        elif scaling_version == "Basic":
            base_channels = [24, 24, 24]
            skip_channels = [12, 24, 24]
            units = [3, 5, 5]
            pmfs_ch = 48
        else:
            raise RuntimeError(f"{scaling_version} scaling version is not available")

        # 设置上采样方式
        if dim == "3d":
            upsample_mode = 'trilinear'
        elif dim == "2d":
            upsample_mode = 'bilinear'
        else:
            raise RuntimeError(f"{dim} dimension is error")

        # 设置每层卷积的参数
        kernel_sizes = [5, 3, 3]
        growth_rates = [4, 8, 16]
        downsample_channels = [base_channels[i] + units[i] * growth_rates[i] for i in range(len(base_channels))]

        # 构建下采样编码器部分
        self.down_convs = nn.ModuleList()
        for i in range(3):
            self.down_convs.append(
                Ultra_module(
                    in_channel=(in_channels if i == 0 else downsample_channels[i - 1]),
                    base_channel=base_channels[i],
                    kernel_size=kernel_sizes[i],
                    skip_channel=skip_channels[i],
                    unit=units[i],
                    growth_rate=growth_rates[i],
                    downsample=True,
                    skip=((i < 2) if scaling_version == "Ultra" else True),
                    dim=dim
                )
            )

        # 全局特征建模模块
        self.Global = global_module(
            in_channels=downsample_channels,
            max_pool_kernels=[4, 2, 1],
            ch=pmfs_ch,
            ch_k=pmfs_ch,
            ch_v=pmfs_ch,
            br=3,
            dim=dim
        )

        # 解码器结构，根据模型类型不同有所区分
        if scaling_version == "Ultra":
            self.up2 = torch.nn.Upsample(scale_factor=2, mode=upsample_mode)
            self.up_conv2 = Ultra_module(in_channel=downsample_channels[2] + skip_channels[1],
                                         base_channel=base_channels[1],
                                         kernel_size=3,
                                         unit=units[1],
                                         growth_rate=growth_rates[1],
                                         downsample=False,
                                         skip=False,
                                         dim=dim)

            self.up1 = torch.nn.Upsample(scale_factor=2, mode=upsample_mode)
            self.up_conv1 = Ultra_module(in_channel=downsample_channels[1] + skip_channels[0],
                                         base_channel=base_channels[0],
                                         kernel_size=3,
                                         unit=units[0],
                                         growth_rate=growth_rates[0],
                                         downsample=False,
                                         skip=False,
                                         dim=dim)
        else:
            self.bottle_conv = ConvBlock(
                in_channel=downsample_channels[2] + skip_channels[2],
                out_channel=skip_channels[2],
                kernel_size=3,
                stride=1,
                batch_norm=True,
                preactivation=True,
                dim=dim
            )

            self.upsample_1 = torch.nn.Upsample(scale_factor=2, mode=upsample_mode)
            self.upsample_2 = torch.nn.Upsample(scale_factor=4, mode=upsample_mode)

        # 输出卷积层
        self.out_conv = ConvBlock(
            in_channel=(downsample_channels[0] if scaling_version == "Ultra" else sum(skip_channels)),
            out_channel=out_channels,
            kernel_size=3,
            stride=1,
            batch_norm=True,
            preactivation=True,
            dim=dim
        )
        self.upsample_out = torch.nn.Upsample(scale_factor=2, mode=upsample_mode)

    def forward(self, x):
        # 前向传播：分 Ultra 和其他类型路径
        if self.scaling_version == "Ultra":
            #输入进入第1层下采样模块；
            #返回主路径特征 x1 和用于跳跃连接的特征 x1_skip
            x1, x1_skip = self.down_convs[0](x)
            #第2层继续提取特征，获取 x2 和 x2_skip。
            x2, x2_skip = self.down_convs[1](x1)
            #最底部的编码器（无 skip 输出），提取高层抽象语义特征
            x3 = self.down_convs[2](x2)

            # print(f"x1.shape = {x1.shape}, x1_skip.shape = {x1_skip.shape}") #x1.shape = torch.Size([64, 44, 112, 112]), x1_skip.shape = torch.Size([64, 24, 112, 112])
            # print(f"x2.shape = {x2.shape}, x2_skip.shape = {x2_skip.shape}") #x2.shape = torch.Size([64, 128, 56, 56]), x2_skip.shape = torch.Size([64, 48, 56, 56])
            # print(f"x3.shape = {x3.shape}") #x3.shape = torch.Size([64, 224, 28, 28])

            #将所有层的主路径特征送入全局注意力模块 Global；
            # 进行跨层信息融合（提高上下文感受野）。
            d3 = self.Global([x1, x2, x3])
            # print(d3.shape) #torch.Size([64, 224, 28, 28])

            #解码
            #上采样 d3（x3 的全局融合结果），空间尺寸翻倍
            d2 = self.up2(d3)
            # print(d2.shape) #torch.Size([64, 224, 56, 56])
            #与之前保存的 x2_skip 拼接，恢复局部细节
            d2 = torch.cat((x2_skip, d2), dim=1)
            # print(d2.shape) #torch.Size([64, 272, 56, 56])
            d2 = self.up_conv2(d2)
            # print(d2.shape) #torch.Size([64, 128, 56, 56])
            #拼接结果进入下采样模块（但此处作用为上采样处理）
            d1 = self.up1(d2)
            # print(d1.shape) #torch.Size([64, 128, 112, 112])
            d1 = torch.cat((x1_skip, d1), dim=1)
            # print(d1.shape) #torch.Size([64, 152, 112, 112])
            d1 = self.up_conv1(d1)
            # print(d1.shape)#torch.Size([64, 44, 112, 112])

            #最后输出层
            #进一步卷积后通道压缩为 out_channels；
            # 最后上采样一次（如果需要恢复原始图像尺寸）。
            out = self.out_conv(d1)
            # print(out.shape) #torch.Size([64, 2, 112, 112])
            out = self.upsample_out(out)
            # print(out.shape) #torch.Size([64, 2, 224, 224])
        else:      #非 Ultra 结构（拼接 skip 特征合并输出）
            #每一层输出主路径结果和 skip 特征。
            x1, skip1 = self.down_convs[0](x)
            x2, skip2 = self.down_convs[1](x1)
            x3, skip3 = self.down_convs[2](x2)

            #同样送入全局注意力模块整合多层特征。
            x3 = self.Global([x1, x2, x3])
            #将融合结果 x3 与 skip3 拼接后做卷积，起到特征压缩融合的作用。
            skip3 = self.bottle_conv(torch.cat([x3, skip3], dim=1))
            #skip2 上采样 2 倍，skip3 上采样 4 倍，统一分辨率以便拼接。
            skip2 = self.upsample_1(skip2)
            skip3 = self.upsample_2(skip3)

            #多尺度 skip 拼接 → 卷积 → 输出通道调整 → 上采样。
            out = self.out_conv(torch.cat([skip1, skip2, skip3], dim=1))
            out = self.upsample_out(out)

        return out





if __name__ == '__main__':

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dims = ["3d", "2d"]
    channels = [1, 3]

    scaling_versions = ["Ultra", "Pro", "Basic"]

    xs = [torch.randn((1, 1, 160, 160, 96)).to(device), torch.randn((1, 3, 224, 224)).to(device)]

    for i, dim in enumerate(dims):
        for scaling_version in scaling_versions:
            model = LiteFANet(in_channels=channels[i], out_channels=2, dim=dim, scaling_version=scaling_version).to(device)
            y = model(xs[i])
            print(dim + "-" + scaling_version, ":")
            print(xs[i].size())
            print(y.size())
            print("params: {:.6f}M".format(count_parameters(model)))
