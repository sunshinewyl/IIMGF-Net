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
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

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

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        selective_scan = selective_scan_fn
        B, L, d = x_rgb.shape
        x_rgb = x_rgb.permute(0, 2, 1)
        x_e = x_e.permute(0, 2, 1)
        x_dbl_rgb = self.x_proj_1(rearrange(x_rgb, "b d l -> (b l) d"))  # (bl d)
        x_dbl_e = self.x_proj_2(rearrange(x_e, "b d l -> (b l) d"))  # (bl d)
        dt_rgb, B_rgb, C_rgb = torch.split(x_dbl_rgb, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_e, B_e, C_e = torch.split(x_dbl_e, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_rgb = self.dt_proj_1.weight @ dt_rgb.t()
        dt_e = self.dt_proj_2.weight @ dt_e.t()
        dt_rgb = rearrange(dt_rgb, "d (b l) -> b d l", l=L)
        dt_e = rearrange(dt_e, "d (b l) -> b d l", l=L)
        A_rgb = -torch.exp(self.A_log_1.float())  # (k * d, d_state)
        A_e = -torch.exp(self.A_log_2.float())  # (k * d, d_state)
        B_rgb = rearrange(B_rgb, "(b l) dstate -> b dstate l", l=L).contiguous()
        B_e = rearrange(B_e, "(b l) dstate -> b dstate l", l=L).contiguous()
        C_rgb = rearrange(C_rgb, "(b l) dstate -> b dstate l", l=L).contiguous()
        C_e = rearrange(C_e, "(b l) dstate -> b dstate l", l=L).contiguous()

        y_rgb = selective_scan(
            x_rgb, dt_rgb,
            A_rgb, B_rgb, C_e, self.D_1.float(),
            delta_bias=self.dt_proj_1.bias.float(),
            delta_softplus=True,
        )
        y_e = selective_scan(
            x_e, dt_e,
            A_e, B_e, C_rgb, self.D_2.float(),
            delta_bias=self.dt_proj_2.bias.float(),
            delta_softplus=True,
        )
        # assert out_y.dtype == torch.float
        y_rgb = rearrange(y_rgb, "b d l -> b l d")
        y_rgb = self.out_norm_1(y_rgb)
        y_e = rearrange(y_e, "b d l -> b l d")
        y_e = self.out_norm_2(y_e)
        return y_rgb, y_e


class CMIGF_block(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            drop_path: float = 0,
            d_state=16,
            ssm_ratio=1,
            dt_rank="auto",
            # dwconv ===============
            # d_conv=-1, # < 2 means no conv
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            softmax_version=False,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj_modalx = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.act = nn.SiLU()

        self.out_proj_rgb = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj_e = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout_rgb = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.dropout_e = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.CMA_ssm = Cross_Modal_Mamba(
            d_model=self.d_model,
            d_state=self.d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            **kwargs,
        )

        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        resi_x, resi_y = x, y
        x = self.in_proj(x)
        y = self.in_proj_modalx(y)
        B, H, W, D = x.shape

        x_trans = x.permute(0, 3, 1, 2).contiguous()
        y_trans = y.permute(0, 3, 1, 2).contiguous()
        x_conv = self.act(self.conv2d(x_trans))  # (b, d, h, w)
        y_conv = self.act(self.conv2d(y_trans))  # (b, d, h, w)
        x_conv = rearrange(x_conv, "b d h w -> b (h w) d")
        y_conv = rearrange(y_conv, "b d h w -> b (h w) d")
        x_c, y_c = self.CMA_ssm(x_conv, y_conv)
        # to b, d, h, w
        x_c = x_c.view(B, H, W, -1)
        y_c = y_c.view(B, H, W, -1)

        out_x = self.dropout_rgb(self.out_proj_rgb(x_c))
        out_y = self.dropout_e(self.out_proj_e(y_c))

        out_x = resi_x + self.drop_path1(out_x)
        out_y = resi_y + self.drop_path2(out_y)
        return out_x, out_y


class CMIGF(nn.Module):
    def __init__(self, d_model=96, depth=3, drop_path=0.2,input_resolution=None):
        super().__init__()
        self.depth = depth

        self.cross_mamba = nn.ModuleList(
            CMIGF_block(d_model=d_model)
            for _ in range(self.depth)
        )

    def forward(self, cli, der):
        cli = rearrange(cli, 'b c h w -> b h w c')
        der = rearrange(der, 'b c h w -> b h w c')
        for block in self.cross_mamba:
            cli, der = block(cli, der)

        return cli, der