import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

# import mamba_ssm.selective_scan_fn (in which causal_conv1d is needed)
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except:
    pass

# an alternative for mamba_ssm
# try:
#     from selective_scan_interface import selective_scan_fn as selective_scan_fn_v1
# except:
#     pass

# cross selective scan ===============================
# if True:
    # import selective_scan_cuda_core as selective_scan_cuda


class Cross_Modal_Mamba(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=4,
            ssm_ratio=1,
            dt_rank="auto",
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # x proj; dt proj ============================
        self.x_proj_1 = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        self.x_proj_2 = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)

        self.dt_proj_1 = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                    **factory_kwargs)
        self.dt_proj_2 = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                    **factory_kwargs)

        # A, D =======================================
        self.A_log_1 = self.A_log_init(self.d_state, self.d_inner)  # (D, N)
        self.A_log_2 = self.A_log_init(self.d_state, self.d_inner)  # (D)
        self.D_1 = self.D_init(self.d_inner)  # (D)
        self.D_2 = self.D_init(self.d_inner)  # (D)

        # out norm ===================================
        self.out_norm_1 = nn.LayerNorm(self.d_inner)
        self.out_norm_2 = nn.LayerNorm(self.d_inner)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        selective_scan = selective_scan_fn
        B, d, L = x.shape
        # x = x.permute(0, 2, 1)
        # y = y.permute(0, 2, 1)
        x_dbl = self.x_proj_1(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        y_dbl = self.x_proj_2(rearrange(y, "b d l -> (b l) d"))  # (bl d)
        dt_x, B_x, C_x = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_y, B_y, C_y = torch.split(y_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_x = self.dt_proj_1.weight @ dt_x.t()
        dt_y = self.dt_proj_2.weight @ dt_y.t()
        dt_x = rearrange(dt_x, "d (b l) -> b d l", l=L)
        dt_y = rearrange(dt_y, "d (b l) -> b d l", l=L)
        A_x = -torch.exp(self.A_log_1.float())  # (k * d, d_state)
        A_y = -torch.exp(self.A_log_2.float())  # (k * d, d_state)
        B_x = rearrange(B_x, "(b l) dstate -> b dstate l", l=L).contiguous()
        B_y = rearrange(B_y, "(b l) dstate -> b dstate l", l=L).contiguous()
        C_x = rearrange(C_x, "(b l) dstate -> b dstate l", l=L).contiguous()
        C_y = rearrange(C_y, "(b l) dstate -> b dstate l", l=L).contiguous()

        x_out = selective_scan(
            x, dt_x,
            A_x, B_x, C_y, self.D_1.float(),
            delta_bias=self.dt_proj_1.bias.float(),
            delta_softplus=True,
        )
        y_out = selective_scan(
            y, dt_y,
            A_y, B_y, C_x, self.D_2.float(),
            delta_bias=self.dt_proj_2.bias.float(),
            delta_softplus=True,
        )
        # assert out_y.dtype == torch.float
        x_out = rearrange(x_out, "b d l -> b l d")
        x_out = self.out_norm_1(x_out)
        y_out = rearrange(y_out, "b d l -> b l d")
        y_out = self.out_norm_2(y_out)
        return x_out, y_out


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # H, W = self.input_resolution
        # B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"
        # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        # x = x.view(B, H, W, C)

        B,H,W,C = x.shape

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops
