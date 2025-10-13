import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../"))

import torch
import torch.nn as nn

## 导入自定义的可分离卷积模块
from lib.models.modules.ConvBlock import DepthWiseSeparateConvBlock
import typing as t
import torch
import torch.nn as nn
from einops import rearrange

import typing as t
import torch
import torch.nn as nn
from einops import rearrange
import math


# ------- LDConv 模块定义（2D / 3D 支持）------- #
# class LDConv(nn.Module):
#     def __init__(self, inc, outc, num_param, stride=1, bias=None, dims='2d'):
#         super(LDConv, self).__init__()
#         assert dims in ['2d', '3d'], "dims must be '2d' or '3d'"
#         self.num_param = num_param
#         self.stride = stride
#         self.dims = dims
#
#         if dims == '2d':
#             self.conv = nn.Sequential(
#                 nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias),
#                 nn.BatchNorm2d(outc),
#                 nn.SiLU())
#             self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
#         else:
#             self.conv = nn.Sequential(
#                 nn.Conv3d(inc, outc, kernel_size=(num_param, 1, 1), stride=(num_param, 1, 1), bias=bias),
#                 nn.BatchNorm3d(outc),
#                 nn.SiLU())
#             self.p_conv = nn.Conv3d(inc, 3 * num_param, kernel_size=3, padding=1, stride=stride)
#
#         nn.init.constant_(self.p_conv.weight, 0)
#         self.p_conv.register_full_backward_hook(self._set_lr)
#
#     @staticmethod
#     def _set_lr(module, grad_input, grad_output):
#         grad_input = (g * 0.1 for g in grad_input)
#         grad_output = (g * 0.1 for g in grad_output)
#
#     def forward(self, x):
#         offset = self.p_conv(x)
#         dtype = offset.data.type()
#         N = self.num_param
#
#         if self.dims == '2d':
#             b, _, h, w = offset.shape
#             p = self._get_p_2d(offset, dtype, h, w)
#             p = p.permute(0, 2, 3, 1)
#             q_coords = self._get_interp_coords_2d(p, x, N)
#             x_offset = self._interpolate_2d(x, q_coords, N)
#             x_offset = self._reshape_x_offset(x_offset, N, dims='2d')
#         else:
#             b, _, d, h, w = offset.shape
#             p = self._get_p_3d(offset, dtype, d, h, w)
#             p = p.permute(0, 2, 3, 4, 1)
#             q_coords = self._get_interp_coords_3d(p, x, N)
#             x_offset = self._interpolate_3d(x, q_coords, N)
#             x_offset = self._reshape_x_offset(x_offset, N, dims='3d')
#
#         out = self.conv(x_offset)
#         return out
#
#     def _get_p_2d(self, offset, dtype, h, w):
#         N = offset.size(1) // 2
#         p_n = self._get_p_n(N, dtype)
#         p_0 = self._get_p_0_2d(h, w, N, dtype)
#         return p_0 + p_n + offset
#
#     def _get_p_3d(self, offset, dtype, d, h, w):
#         N = offset.size(1) // 3
#         p_n = self._get_p_n(N, dtype, dims='3d')
#         p_0 = self._get_p_0_3d(d, h, w, N, dtype)
#         return p_0 + p_n + offset
#
#     def _get_p_n(self, N, dtype, dims='2d'):
#         base = round(math.pow(self.num_param, 1/3 if dims == '3d' else 1/2))
#         size = [base] * (3 if dims == '3d' else 2)
#         while math.prod(size) < self.num_param:
#             size[0] += 1
#
#         coords = torch.stack(torch.meshgrid(*[torch.arange(s) for s in size]), dim=-1)
#         coords = coords.reshape(-1, coords.shape[-1])[:self.num_param]
#
#         p_n = coords.transpose(0, 1).contiguous().reshape(1, -1, 1, 1)
#         if dims == '3d':
#             p_n = p_n.unsqueeze(-1)
#         return p_n.type(dtype)
#
#     def _get_p_0_2d(self, h, w, N, dtype):
#         p_x, p_y = torch.meshgrid(
#             torch.arange(0, h * self.stride, self.stride),
#             torch.arange(0, w * self.stride, self.stride))
#         p_0_x = p_x.reshape(1, 1, h, w).repeat(1, N, 1, 1)
#         p_0_y = p_y.reshape(1, 1, h, w).repeat(1, N, 1, 1)
#         return torch.cat([p_0_x, p_0_y], 1).type(dtype)
#
#     def _get_p_0_3d(self, d, h, w, N, dtype):
#         p_z, p_x, p_y = torch.meshgrid(
#             torch.arange(0, d * self.stride, self.stride),
#             torch.arange(0, h * self.stride, self.stride),
#             torch.arange(0, w * self.stride, self.stride))
#         p_0_z = p_z.reshape(1, 1, d, h, w).repeat(1, N, 1, 1, 1)
#         p_0_x = p_x.reshape(1, 1, d, h, w).repeat(1, N, 1, 1, 1)
#         p_0_y = p_y.reshape(1, 1, d, h, w).repeat(1, N, 1, 1, 1)
#         return torch.cat([p_0_z, p_0_x, p_0_y], 1).type(dtype)
#
#     def _get_interp_coords_2d(self, p, x, N):
#         h_max, w_max = x.shape[2] - 1, x.shape[3] - 1
#         q = torch.cat([
#             torch.clamp(p[..., :N], 0, h_max),
#             torch.clamp(p[..., N:], 0, w_max)], dim=-1)
#         return q
#
#     def _get_interp_coords_3d(self, p, x, N):
#         d_max, h_max, w_max = x.shape[2] - 1, x.shape[3] - 1, x.shape[4] - 1
#         q = torch.cat([
#             torch.clamp(p[..., :N], 0, d_max),
#             torch.clamp(p[..., N:2*N], 0, h_max),
#             torch.clamp(p[..., 2*N:], 0, w_max)], dim=-1)
#         return q
#
#     def _interpolate_2d(self, x, q, N):
#         b, c, h, w = x.shape
#         index = q[..., :N] * w + q[..., N:]
#         index = index.long()
#         x_flat = x.reshape(b, c, -1)
#         idx = index.reshape(b, 1, -1).expand(-1, c, -1)
#         x_q = x_flat.gather(dim=-1, index=idx).reshape(b, c, h, w, N)
#         return x_q
#
#     def _interpolate_3d(self, x, q, N):
#         b, c, d, h, w = x.shape
#         index = q[..., :N] * h * w + q[..., N:2*N] * w + q[..., 2*N:]
#         index = index.long()
#         x_flat = x.reshape(b, c, -1)
#         idx = index.reshape(b, 1, -1).expand(-1, c, -1)
#         x_q = x_flat.gather(dim=-1, index=idx).reshape(b, c, d, h, w, N)
#         return x_q
#
#     @staticmethod
#     def _reshape_x_offset(x_offset, num_param, dims='2d'):
#         if dims == '2d':
#             b, c, h, w, n = x_offset.shape
#             return rearrange(x_offset, 'b c h w n -> b c (h n) w')
#         else:
#             b, c, d, h, w, n = x_offset.shape
#             return rearrange(x_offset, 'b c d h w n -> b c (d n) h w')



# ------- SCSA 模块定义，使用 LDConv 替换原始 conv ------- #
class SCSA(nn.Module):
    def __init__(
            self,
            dim: int,
            head_num: int,
            dims: str = '2d',
            window_size: int = 7,
            group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
            qkv_bias: bool = False,
            attn_drop_ratio: float = 0.,
            gate_layer: str = 'sigmoid',
            # down_sample_mode: str = 'avg_pool'
            down_sample_mode: str = 'max_pool'
    ):
        super(SCSA, self).__init__()
        self.dims = dims
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** -0.5
        self.group_kernel_sizes = group_kernel_sizes
        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.down_sample_mode = down_sample_mode

        assert self.dim % 4 == 0, 'The dimension of input feature should be divisible by 4.'
        self.group_chans = group_chans = self.dim // 4

        Conv1d = nn.Conv1d
        ConvND = nn.Conv2d if dims == '2d' else nn.Conv3d
        Pool = nn.AdaptiveAvgPool2d if dims == '2d' else nn.AdaptiveAvgPool3d

        # Spatial attention convs (1D along spatial dimensions)
        self.local_dwc = Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[0],
                                padding=group_kernel_sizes[0] // 2, groups=group_chans)
        self.global_dwc_s = Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[1],
                                   padding=group_kernel_sizes[1] // 2, groups=group_chans)
        self.global_dwc_m = Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[2],
                                   padding=group_kernel_sizes[2] // 2, groups=group_chans)
        self.global_dwc_l = Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[3],
                                   padding=group_kernel_sizes[3] // 2, groups=group_chans)
        self.local_dwc_1 = Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[0],
                                padding=group_kernel_sizes[0] // 2, groups=group_chans)
        self.global_dwc_s_1 = Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[1],
                                   padding=group_kernel_sizes[1] // 2, groups=group_chans)
        self.global_dwc_m_1 = Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[2],
                                   padding=group_kernel_sizes[2] // 2, groups=group_chans)
        self.global_dwc_l_1 = Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[3],
                                   padding=group_kernel_sizes[3] // 2, groups=group_chans)
        self.local_dwc_2 = Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[0],
                                padding=group_kernel_sizes[0] // 2, groups=group_chans)
        self.global_dwc_s_2 = Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[1],
                                   padding=group_kernel_sizes[1] // 2, groups=group_chans)
        self.global_dwc_m_2 = Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[2],
                                   padding=group_kernel_sizes[2] // 2, groups=group_chans)
        self.global_dwc_l_2 = Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[3],
                                   padding=group_kernel_sizes[3] // 2, groups=group_chans)

        # self.local_dwc_2 = Conv1d(self.dim, self.dim, kernel_size=group_kernel_sizes[0],
        #                         padding=group_kernel_sizes[0] // 2, groups=group_chans)
        # self.global_dwc_s_2 = Conv1d(self.dim, self.dim, kernel_size=group_kernel_sizes[1],
        #                            padding=group_kernel_sizes[1] // 2, groups=group_chans)
        # self.global_dwc_m_2 = Conv1d(self.dim, self.dim, kernel_size=group_kernel_sizes[2],
        #                            padding=group_kernel_sizes[2] // 2, groups=group_chans)
        # self.global_dwc_l_2 = Conv1d(self.dim, self.dim, kernel_size=group_kernel_sizes[3],
        #                            padding=group_kernel_sizes[3] // 2, groups=group_chans)


        self.sa_gate_1 = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()

        self.norm_h = nn.GroupNorm(4, dim)
        self.norm_w = nn.GroupNorm(4, dim)

        # self.sa_gate_3 = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.conv_d = nn.Identity()
        self.norm = nn.GroupNorm(1, dim)

        self.q = ConvND(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.k = ConvND(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.v = ConvND(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == 'softmax' else nn.Sigmoid()

        if window_size == -1:
            self.down_func = Pool(1)
        else:
            if down_sample_mode == 'avg_pool':
                self.down_func = nn.AvgPool2d(kernel_size=window_size, stride=window_size) if dims == '2d' else nn.AvgPool3d(kernel_size=window_size, stride=window_size)
            elif down_sample_mode == 'max_pool':
                self.down_func = nn.MaxPool2d(kernel_size=window_size, stride=window_size) if dims == '2d' else nn.MaxPool3d(kernel_size=window_size, stride=window_size)
            else:
                raise NotImplementedError(f"Unsupported down_sample_mode: {down_sample_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial Attention Path
        if self.dims == '2d':
            b, c, h, w = x.size()
            x_h = x.mean(dim=3)  # (B, C, H)
            x_w = x.mean(dim=2)  # (B, C, W)
        else:
            b, c, d, h, w = x.size()
            x_h = x.mean(dim=-1).flatten(2)  # (B, C, D*H)
            x_w = x.mean(dim=-2).flatten(2)  # (B, C, D*W)
            x_d = x.mean(dim=-3).flatten(2)  # (B, C, H*W)

        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)
        # print(x_h.shape)
        # print(l_x_h.shape)
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)
        if self.dims == '3d':
            l_x_d, g_x_d_s, g_x_d_m, g_x_d_l = torch.split(x_d, self.group_chans, dim=1)

        x_h_attn = self.sa_gate_1(self.norm_h(torch.cat((
            self.local_dwc(l_x_h),
            self.global_dwc_s(g_x_h_s),
            self.global_dwc_m(g_x_h_m),
            self.global_dwc_l(g_x_h_l),
        ), dim=1)))
        x_w_attn = self.sa_gate_1(self.norm_w(torch.cat((
            self.local_dwc_1(l_x_w),
            self.global_dwc_s_1(g_x_w_s),
            self.global_dwc_m_1(g_x_w_m),
            self.global_dwc_l_1(g_x_w_l),
        ), dim=1)))
        if self.dims == '3d':
            x_d_attn = self.sa_gate_1(self.norm_w(torch.cat((
            self.local_dwc_2(l_x_d),
            self.global_dwc_s_2(g_x_d_s),
            self.global_dwc_m_2(g_x_d_m),
            self.global_dwc_l_2(g_x_d_l),
            ), dim=1)))

        if self.dims == '2d':
            x_h_attn = x_h_attn.view(b, c, h, 1)
            x_w_attn = x_w_attn.view(b, c, 1, w)
        else:
            x_h_attn = x_h_attn.view(b, c, d, h, 1)
            x_w_attn = x_w_attn.view(b, c, d, 1, w)
            x_d_attn=x_d_attn.view(b, c, 1, h,w)
        if self.dims == '2d':
            x = x * x_h_attn * x_w_attn
        else:
            x = x * x_h_attn * x_w_attn * x_d_attn
        y = self.down_func(x)
        y = self.conv_d(y)
        y = self.norm(y)

        q = self.q(y)
        k = self.k(y)
        v = self.v(y)

        if self.dims == '2d':
            _, _, h_, w_ = y.size()
            q = rearrange(q, 'b (hn hd) h w -> b hn hd (h w)', hn=self.head_num, hd=self.head_dim)
            k = rearrange(k, 'b (hn hd) h w -> b hn hd (h w)', hn=self.head_num, hd=self.head_dim)
            v = rearrange(v, 'b (hn hd) h w -> b hn hd (h w)', hn=self.head_num, hd=self.head_dim)
        else:
            _, _, d_, h_, w_ = y.size()
            q = rearrange(q, 'b (hn hd) d h w -> b hn hd (d h w)', hn=self.head_num, hd=self.head_dim)
            k = rearrange(k, 'b (hn hd) d h w -> b hn hd (d h w)', hn=self.head_num, hd=self.head_dim)
            v = rearrange(v, 'b (hn hd) d h w -> b hn hd (d h w)', hn=self.head_num, hd=self.head_dim)

        attn = q @ k.transpose(-2, -1) * self.scaler
        attn = self.attn_drop(attn.softmax(dim=-1))
        attn = attn @ v

        if self.dims == '2d':
            attn = rearrange(attn, 'b hn hd (h w) -> b (hn hd) h w', h=h_, w=w_)
            attn = attn.mean((2, 3), keepdim=True)
        else:
            attn = rearrange(attn, 'b hn hd (d h w) -> b (hn hd) d h w', d=d_, h=h_, w=w_)
            attn = attn.mean((2, 3, 4), keepdim=True)

        attn = self.ca_gate(attn)
        return attn * x
        # return  x

class DWConvNd(nn.Module):
    """
    Depthwise conv for both 2D and 3D: groups=channels
    """
    def __init__(self, dim, dims='2d'):
        super().__init__()
        assert dims in ('2d', '3d'), "dims must be '2d' or '3d'"
        self.dims = dims
        if dims == '2d':
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,
                                     bias=True, groups=dim)
        else:
            self.dwconv = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1,
                                     bias=True, groups=dim)

    def forward(self, x, shape):
        # x: (B, N, C), shape: tuple of spatial dims
        B, N, C = x.shape
        if self.dims == '2d':
            H, W = shape
            assert N == H * W, f"Sequence length {N} mismatches H*W={H*W}"
            t = x.transpose(1, 2).reshape(B, C, H, W)
            t = self.dwconv(t)
            return t.flatten(2).transpose(1, 2)
        else:  # For 3D
            D, H, W = shape
            assert N == D * H * W, f"Sequence length {N} mismatches D*H*W={D*H*W}"
            t = x.transpose(1, 2).reshape(B, C, D, H, W)
            t = self.dwconv(t)
            return t.flatten(2).transpose(1, 2)

class CGLU(nn.Module):
    """
    Convolutional Gated Linear Unit supporting 4D (H,W) and 5D (D,H,W) inputs.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., dims='2d'):
        super().__init__()
        assert dims in ('2d', '3d'), "dims must be '2d' or '3d'"
        self.dims = dims
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)

        # projection to 2*hidden, then gated conv, then final proj
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConvNd(hidden_features, dims=dims)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # x: Tensor of shape
        #   4D: (B, C, H, W) if dims='2d'
        #   5D: (B, C, D, H, W) if dims='3d'
        B = x.shape[0]
        if self.dims == '2d' and x.dim() == 4:
            B, C, H, W = x.shape
            N = H * W
            seq = x.reshape(B, C, N).transpose(1, 2)
            shape = (H, W)
        elif self.dims == '3d' and x.dim() == 5:
            B, C, D, H, W = x.shape
            N = D * H * W
            seq = x.reshape(B, C, N).transpose(1, 2)
            shape = (D, H, W)
        else:
            raise RuntimeError(f"Unsupported input shape {x.shape} for dims={self.dims}")

        # gated linear unit with conv
        x2, gate = self.fc1(seq).chunk(2, dim=-1)
        # depthwise conv on spatial
        x2 = self.dwconv(x2, shape)
        x2 = self.act(x2) * gate
        x2 = self.drop(x2)
        x2 = self.fc2(x2)
        x2 = self.drop(x2)

        # reshape back to spatial tensor
        if self.dims == '2d':
            out = x2.transpose(1, 2).reshape(B, -1, H, W)
        else:
            out = x2.transpose(1, 2).reshape(B, -1, D, H, W)
        return out



class DepthWiseSeparateConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, batch_norm=True, preactivation=False, dim="3d"):
        super(DepthWiseSeparateConvBlock, self).__init__()

        if dim == "3d":
            constant_pad = torch.nn.ConstantPad3d
            conv = torch.nn.Conv3d
            bn = torch.nn.BatchNorm3d
        elif dim == "2d":
            constant_pad = torch.nn.ConstantPad2d
            conv = torch.nn.Conv2d
            bn = torch.nn.BatchNorm2d
        else:
            raise RuntimeError(f"{dim} dimension is error")

        padding = kernel_size - stride
        if padding % 2 != 0:
            pad = constant_pad(
                tuple([padding % 2, padding - padding % 2] * (3 if dim == "3d" else 2)), 0
            )
        else:
            pad = constant_pad(padding // 2, 0)

        if preactivation:
            layers = [
                torch.nn.ReLU(),
                pad,
                conv(
                    in_channels=in_channel,
                    out_channels=in_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    groups=in_channel,
                    bias=False
                ),
                conv(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    bias=True
                )
            ]
            if batch_norm:
                layers = [bn(in_channel)] + layers
        else:
            layers = [
                pad,
                conv(
                    in_channels=in_channel,
                    out_channels=in_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    groups=in_channel,
                    bias=False
                ),
                conv(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    bias=False
                )
            ]
            if batch_norm:
                layers.append(bn(out_channel))
            layers.append(torch.nn.ReLU())

        self.conv = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class oneConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation, dim="3d"):
        super().__init__()
        self.conv = DepthWiseSeparateConvBlock(
            in_channel=in_channels,
            out_channel=out_channels,
            kernel_size=kernel_size,
            stride=1,
            batch_norm=False,
            preactivation=False,
            dim=dim
        )

    def forward(self, x):
        return self.conv(x)


class MSFblock(nn.Module):
    def __init__(self, in_channels, dim="3d"):
        super(MSFblock, self).__init__()
        self.dim = dim
        out_channels = in_channels

        # Depthwise-separable project conv (1x1)
        self.project = DepthWiseSeparateConvBlock(
            in_channel=out_channels,
            out_channel=out_channels,
            kernel_size=1,
            stride=1,
            batch_norm=True,
            preactivation=False,
            dim=dim
        )
        # Adaptive pooling based on dimension
        if dim == "3d":
            self.gap = nn.AdaptiveAvgPool3d(1)
        else:
            self.gap = nn.AdaptiveAvgPool2d(1)
        # if dim == "3d":
        #     self.gap = nn.AdaptiveMaxPool3d(1)
        # else:
        #     self.gap = nn.AdaptiveMaxPool2d(1)

        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        # SE branches
        self.SE1 = oneConv(in_channels, in_channels, 1, 0, 1, dim=dim)
        self.SE2 = oneConv(in_channels, in_channels, 1, 0, 1, dim=dim)
        self.SE3 = oneConv(in_channels, in_channels, 1, 0, 1, dim=dim)

    def forward(self, x0, x1, x2):
        y0, y1, y2 = x0, x1, x2

        # Squeeze
        y0_w = self.SE1(self.gap(y0))  # [B, C, 1, 1, (1)]
        y1_w = self.SE2(self.gap(y1))
        y2_w = self.SE3(self.gap(y2))

        # Concatenate on squeeze dimension
        weight = torch.cat([y0_w, y1_w, y2_w], dim=2)  # shape [..., 3, ...]
        weight = self.softmax(self.sigmoid(weight))

        # Split weights
        w0 = weight[:, :, 0].unsqueeze(2)
        w1 = weight[:, :, 1].unsqueeze(2)
        w2 = weight[:, :, 2].unsqueeze(2)

        # Weighted sum and project
        x_att = w0 * y0 + w1 * y1 + w2 * y2
        return self.project(x_att)


## 定义GlobalPMFSBlock（使用全局多尺度注意力 + 可分离卷积）
class GlobalPMFSBlock_AP_Separate(nn.Module):
    def __init__(self, in_channels, max_pool_kernels, ch, ch_k, ch_v, br, dim="3d", scsa_heads: int = 8):
        super(GlobalPMFSBlock_AP_Separate, self).__init__()

        self.ch_bottle = in_channels[-1]
        self.ch = ch
        self.ch_k = ch_k
        self.ch_v = ch_v
        self.br = br
        self.ch_in = self.ch * self.br
        self.dim = dim

        if dim == "3d":
            max_pool = nn.MaxPool3d
            conv = nn.Conv3d
            bn = nn.BatchNorm3d
        elif dim == "2d":
            max_pool = nn.MaxPool2d
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        else:
            raise RuntimeError(f"{dim} dimension is error")

        self.ch_convs = nn.ModuleList([
            DepthWiseSeparateConvBlock(
                in_channel=in_channel,
                # out_channel=self.ch,
                out_channel=self.ch_in,
                kernel_size=3,
                stride=1,
                batch_norm=True,
                preactivation=True,
                dim=dim
            )
            for in_channel in in_channels
        ])

        self.max_pool_layers = nn.ModuleList([
            max_pool(kernel_size=k, stride=k)
            for k in max_pool_kernels
        ])

        self.sigmoid = nn.Sigmoid()

        self.output_conv = DepthWiseSeparateConvBlock(in_channel=self.ch_in, out_channel=self.ch_bottle, kernel_size=3, stride=1, batch_norm=True, preactivation=True, dim=dim)
        #特征融合1
        # self.msf_block = MSFblock(in_channels=self.ch_in, dim=self.dim)
        # self.scsa = SCSA(dim=self.ch_in, head_num=scsa_heads, dims=dim)
        # 将通道注意力替换为CGLU模块
        # self.ch_attention = CGLU(in_features=self.ch_in, hidden_features=self.ch_in, out_features=self.ch_in, dims=dim)
        # 添加一个额外的卷积层将通道数调整为 out_channels
        # self.adjust_channels = conv(self.ch, self.ch_in, kernel_size=1, stride=1, bias=False)

        # 打印各个参数值
        # print(f"self.ch_bottle: {self.ch_bottle}")
        # print(f"self.ch: {self.ch}")
        # print(f"self.ch_k: {self.ch_k}")
        # print(f"self.ch_v: {self.ch_v}")
        # print(f"self.br: {self.br}")
        # print(f"self.ch_in: {self.ch_in}")
        # print(f"self.dim: {self.dim}")
        #
        # # 打印 MaxPool 和 Conv 的选择
        # print(f"max_pool: {max_pool}")
        # print(f"conv: {conv}")
        # print(f"bn: {bn}")
        #
        # # 打印所有其他模块配置
        # print(f"self.ch_Wq: {self.ch_Wq}")
        # print(f"self.ch_Wk: {self.ch_Wk}")
        # print(f"self.ch_Wv: {self.ch_Wv}")
        # print(f"self.ch_score_conv: {self.ch_score_conv}")
        # print(f"self.ch_layer_norm: {self.ch_layer_norm}")
        # print(f"self.sp_Wq: {self.sp_Wq}")
        # print(f"self.sp_Wk: {self.sp_Wk}")
        # print(f"self.sp_Wv: {self.sp_Wv}")
        # print(f"self.sp_output_conv: {self.sp_output_conv}")
        # print(f"self.output_conv: {self.output_conv}")


    def forward(self, feature_maps):
        device = feature_maps[0].device  # 获取输入数据所在设备
        self.to(device)  # 确保模型权重被转移到正确的设备

        max_pool_maps = [
            max_pool_layer(feature_maps[i].to(device))  # 将数据移到同一设备
            for i, max_pool_layer in enumerate(self.max_pool_layers)
        ]

        ch_outs = [
            ch_conv(max_pool_maps[i].to(device))  # 将数据移到同一设备
            for i, ch_conv in enumerate(self.ch_convs)
        ]

        # 使用 MSFblock 进行多尺度特征融合
        # x = self.msf_block(ch_outs[0], ch_outs[1], ch_outs[2])  # 对三个尺度的特征图进行融合
        # 将通道数调整为192
        # x = self.adjust_channels(x)  # 调整输出的通道数为192
        x = ch_outs[0]
        if self.dim == "3d":
            bs, c, d, h, w = x.size()

            # x = self.scsa(x)
            # x = self.ch_attention(x)
            out = self.output_conv(x)
        else:
            # 这部分逻辑完全一致 (2D)
            bs, c, h, w = x.size()

            # x = self.scsa(x)
            # x = self.ch_attention(x)
            out = self.output_conv(x)
        return out

if __name__ == '__main__':
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #
    # x = [
    #     torch.randn((1, 32, 80, 80, 48)).to(device),
    #     torch.randn((1, 64, 40, 40, 24)).to(device),
    #     torch.randn((1, 128, 20, 20, 12)).to(device),
    # ]
    #
    # model = GlobalPMFSBlock_AP_Separate([32, 64, 128], [4, 2, 1], 64, 64, 64, 3, dim="3d").to(device)
    #
    # output = model(x)
    #
    # print(output.size())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x = [
        torch.randn((1, 32, 80, 80)).to(device),
        torch.randn((1, 64, 40, 40)).to(device),
        torch.randn((1, 128, 20, 20)).to(device),
    ]

    model = GlobalPMFSBlock_AP_Separate([32, 64, 128], [4, 2, 1], 64, 64, 64, 3, dim="2d").to(device)

    output = model(x)

    print(output.size())
