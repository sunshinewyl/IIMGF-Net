import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict
from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from einops import rearrange
from models.aux_model.mamba.csms6s import selective_scan_fn
from models.aux_model.mamba.function import Cross_mamba

class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

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
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
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
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
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
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        # dt proj ============================
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))  # (K, inner, rank)
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))  # (K, inner)
        del dt_projs

        # A, D =======================================
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True)  # (K * D)
        return A_logs, Ds, dt_projs_weight, dt_projs_bias


class cross_SS2D(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=1.0,
            dt_rank="auto",
            # ======================
            dropout=0.0,
            # ======================
            seq=False,
            force_fp32=True,
            **kwargs,
    ):
        if "channel_first" in kwargs:
            assert not kwargs["channel_first"]
        act_layer = nn.SiLU
        dt_min = 0.001
        dt_max = 0.1
        dt_init = "random"
        dt_scale = 1.0
        dt_init_floor = 1e-4
        bias = False
        conv_bias = True
        d_conv = 3
        k_group = 4
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        # in proj ============================
        self.in_proj_cli = nn.Linear(d_model, d_inner * 2, bias=bias)
        self.in_proj_der = nn.Linear(d_model, d_inner * 2, bias=bias)
        self.act: nn.Module = act_layer()
        self.conv2d = nn.Conv2d(
            in_channels=d_inner,
            out_channels=d_inner,
            groups=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        # x proj ============================
        self.x_proj_der = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight_der = nn.Parameter(torch.stack([t.weight for t in self.x_proj_der], dim=0)) # (K, N, inner)
        del self.x_proj_der
        self.x_proj_cli = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight_cli = nn.Parameter(torch.stack([t.weight for t in self.x_proj_cli], dim=0))  # (K, N, inner)
        del self.x_proj_cli

        # dt proj, A, D ============================
        self.A_logs_cli, self.Ds_cli, self.dt_projs_weight_cli, self.dt_projs_bias_cli = mamba_init.init_dt_A_D(
            d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4,
        )
        self.A_logs_der, self.Ds_der, self.dt_projs_weight_der, self.dt_projs_bias_der = mamba_init.init_dt_A_D(
            d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4,
        )

        # out proj =======================================
        self.out_norm_1 = nn.LayerNorm(d_inner)
        self.out_norm_2 = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def traverse_four(self, x, x_proj_weight, dt_projs_weight, A_logs, Ds, dt_projs_bias):
        B ,C, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W
        """ 四个不同的遍历路径 """
        # 堆叠输入张量 x 的两个视角（原始和转置）, [b, 2, d, l]
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        # 拼接 x_hwwh 和 其翻转
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        # 将 xs 通过权重矩阵 self.x_proj_weight 进行投影
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        if hasattr(self, "x_proj_bias"):
            x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        """ x --投影-> delta, B, C 矩阵 """
        # 由投影后的x分别得到 delta, B, C 矩阵, '(B, L, D) -> (B, L, N)'
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        # 将 dts（delta） 通过权重矩阵 self.dt_projs_weight 进行投影, '(B, L, N) -> (B, L, D)'
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        xs = xs.view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.contiguous()  # (b, k, d_state, l)
        Cs = Cs.contiguous()  # (b, k, d_state, l)

        As = -A_logs.float().exp()  # (k * d, d_state)
        Ds = Ds.float()  # (k * d)
        dt_projs_bias = dt_projs_bias.float().view(-1)  # (k * d)

        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        return xs, dts, Bs, Cs, As, Ds, dt_projs_bias

    def path_overlay(self, out_y, B, H, W, L):
        """ 四种遍历路径叠加 (Mamba之后) """
        # token位置还原
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        # 四种状态叠加
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y

        # 还原形状，方便输出
        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y = y.view(B, H, W, -1)
        return y


    def forward(self, x: torch.Tensor, y: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        selective_scan = partial(selective_scan_fn, backend="mamba")
        xs_der, dts_der, Bs_der, Cs_der, As_der, Ds_der, dt_projs_bias_der = (
            self.traverse_four(x,
                               self.x_proj_weight_der,
                               self.dt_projs_weight_der,
                               self.A_logs_der,
                               self.Ds_der,
                               self.dt_projs_bias_der))
        xs_cli, dts_cli, Bs_cli, Cs_cli, As_cli, Ds_cli, dt_projs_bias_cli = (
            self.traverse_four(y,
                                self.x_proj_weight_cli,
                                self.dt_projs_weight_cli,
                                self.A_logs_cli,
                                self.Ds_cli,
                                self.dt_projs_bias_cli))

        y_der = selective_scan(
            xs_der, dts_der,
            As_der, Bs_der, Cs_cli, Ds_der,
            delta_bias=dt_projs_bias_der,
            delta_softplus=True
        ).view(B, K, -1, L)

        y_cli = selective_scan(
            xs_cli, dts_cli,
            As_cli, Bs_cli, Cs_der, Ds_cli,
            delta_bias=dt_projs_bias_cli,
            delta_softplus=True
        ).view(B, K, -1, L)

        y_der = self.path_overlay(y_der, B, H, W, L)
        y_cli = self.path_overlay(y_cli, B, H, W, L)

        y_der = self.out_norm_1(y_der).permute(0, 3, 1, 2)
        y_cli = self.out_norm_2(y_cli).permute(0, 3, 1, 2)
        return y_der, y_cli





