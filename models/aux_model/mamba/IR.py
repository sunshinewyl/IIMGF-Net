import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
from timm.layers import DropPath, trunc_normal_

def index_reverse(index):
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1]).to(index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r


def semantic_neighbor(x, index):
    dim = index.dim()
    assert x.shape[:dim] == index.shape, "x ({:}) and index ({:}) shape incompatible".format(x.shape, index.shape)

    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)

    shuffled_x = torch.gather(x, dim=dim - 1, index=index)
    return shuffled_x


class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class Gate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim)  # DW Conv

    def forward(self, x, H, W):
        x1, x2 = x.chunk(2, dim=-1)
        B, N, C = x.shape
        x2 = self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C // 2, H, W)).flatten(2).transpose(-1, -2).contiguous()
        return x1 * x2



class ASSM(nn.Module):
    def __init__(self, dim, d_state, input_resolution, num_tokens=64, inner_rank=128, mlp_ratio=2.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_tokens = num_tokens
        self.inner_rank = inner_rank

        # Mamba params
        self.expand = mlp_ratio
        hidden = int(self.dim * self.expand)
        self.d_state = d_state
        self.selectiveScan = Selective_Scan(d_model=hidden, d_state=self.d_state, expand=1)
        self.out_norm = nn.LayerNorm(hidden)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(hidden, dim, bias=True)

        self.in_proj = nn.Sequential(
            nn.Conv2d(self.dim, hidden, 1, 1, 0),
        )

        self.CPE = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden),
        )

        self.embeddingB = nn.Embedding(self.num_tokens, self.inner_rank)  # [64,32] [32, 48] = [64,48]
        self.embeddingB.weight.data.uniform_(-1 / self.num_tokens, 1 / self.num_tokens)

        self.route = nn.Sequential(
            nn.Linear(self.dim, self.dim // 3),
            nn.GELU(),
            nn.Linear(self.dim // 3, self.num_tokens),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, x_size, token):
        B, n, C = x.shape
        H, W = x_size

        full_embedding = self.embeddingB.weight @ token.weight  # [128, C]

        pred_route = self.route(x)  # [B, HW, num_token]
        cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)  # [B, HW, num_token]

        prompt = torch.matmul(cls_policy, full_embedding).view(B, n, self.d_state)

        detached_index = torch.argmax(cls_policy.detach(), dim=-1, keepdim=False).view(B, n)  # [B, HW]
        x_sort_values, x_sort_indices = torch.sort(detached_index, dim=-1, stable=False)
        x_sort_indices_reverse = index_reverse(x_sort_indices)

        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x = self.in_proj(x)
        x = x * torch.sigmoid(self.CPE(x))
        cc = x.shape[1]
        x = x.view(B, cc, -1).contiguous().permute(0, 2, 1)  # b,n,c

        semantic_x = semantic_neighbor(x, x_sort_indices) # SGN-unfold
        y = self.selectiveScan(semantic_x, prompt)
        y = self.out_proj(self.out_norm(y))
        x = semantic_neighbor(y, x_sort_indices_reverse) # SGN-fold

        return x


class Selective_Scan(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            expand=1.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.x_proj_rgb = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight_rgb = nn.Parameter(torch.stack([t.weight for t in self.x_proj_rgb], dim=0))  # (K=4, N, inner)
        del self.x_proj_rgb

        self.x_proj_e = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight_e = nn.Parameter(torch.stack([t.weight for t in self.x_proj_e], dim=0))  # (K=4, N, inner)
        del self.x_proj_e

        self.dt_projs_rgb = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight_rgb = nn.Parameter(torch.stack([t.weight for t in self.dt_projs_rgb], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias_rgb = nn.Parameter(torch.stack([t.bias for t in self.dt_projs_rgb], dim=0))  # (K=4, inner)
        del self.dt_projs_rgb

        self.dt_projs_e = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight_e = nn.Parameter(torch.stack([t.weight for t in self.dt_projs_e], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias_e = nn.Parameter(torch.stack([t.bias for t in self.dt_projs_e], dim=0))  # (K=4, inner)
        del self.dt_projs_e

        self.A_logs_rgb = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.A_logs_e = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.Ds_rgb = self.D_init(self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.Ds_e = self.D_init(self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.selective_scan = selective_scan_fn

        self.inner_rank = math.ceil(self.d_model / 16)
        self.num_tokens = 64

        self.embedding_rgb = nn.Embedding(self.num_tokens, self.inner_rank)  # [64,32] [32, 48] = [64,48]
        self.embedding_rgb.weight.data.uniform_(-1 / self.num_tokens, 1 / self.num_tokens)

        self.embedding_e = nn.Embedding(self.num_tokens, self.inner_rank)  # [64,32] [32, 48] = [64,48]
        self.embedding_e.weight.data.uniform_(-1 / self.num_tokens, 1 / self.num_tokens)

        self.route_rgb = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 3),
            nn.GELU(),
            nn.Linear(self.d_model // 3, self.num_tokens),
            nn.LogSoftmax(dim=-1)
        )

        self.route_e = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 3),
            nn.GELU(),
            nn.Linear(self.d_model // 3, self.num_tokens),
            nn.LogSoftmax(dim=-1)
        )
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
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x_rgb, x_e, prompt_rgb, prompt_e):
        B, L, C = x_rgb.shape
        K = 1  # mambairV2 needs noly 1 scan
        x_rgb = x_rgb.permute(0, 2, 1).view(B, 1, C, L).contiguous()  # B, 1, C ,L
        x_e = x_e.permute(0, 2, 1).view(B, 1, C, L).contiguous()

        x_dbl_rgb = torch.einsum("b k d l, k c d -> b k c l", x_rgb.view(B, K, -1, L), self.x_proj_weight_rgb)
        x_dbl_e = torch.einsum("b k d l, k c d -> b k c l", x_e.view(B, K, -1, L), self.x_proj_weight_e)
        d_rgb, B_rgb, C_rgb = torch.split(x_dbl_rgb, [self.dt_rank, self.d_state, self.d_state], dim=2)
        d_e, B_e, C_e = torch.split(x_dbl_e, [self.dt_rank, self.d_state, self.d_state], dim=2)
        d_rgb = torch.einsum("b k r l, k d r -> b k d l", d_rgb.view(B, K, -1, L), self.dt_projs_weight_rgb)
        d_e = torch.einsum("b k r l, k d r -> b k d l", d_e.view(B, K, -1, L), self.dt_projs_weight_e)
        x_rgb = x_rgb.float().view(B, -1, L)
        x_e = x_e.float().view(B, -1, L)
        d_rgb = d_rgb.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        d_e = d_e.contiguous().float().view(B, -1, L)
        B_rgb = B_rgb.float().view(B, K, -1, L)
        B_e = B_e.float().view(B, K, -1, L)
        #  our ASE here ---
        C_rgb = C_rgb.float().view(B, K, -1, L) + prompt_rgb  # (b, k, d_state, l)
        C_e = C_e.float().view(B, K, -1, L) + prompt_e
        D_rgb = self.Ds_rgb.float().view(-1)
        D_e = self.Ds_e.float().view(-1)
        A_rgb = -torch.exp(self.A_logs_rgb.float()).view(-1, self.d_state)
        A_e = -torch.exp(self.A_logs_e.float()).view(-1, self.d_state)
        dt_projs_bias_rgb = self.dt_projs_bias_rgb.float().view(-1)  # (k * d)
        dt_projs_bias_e = self.dt_projs_bias_e.float().view(-1)
        out_y_rgb = self.selective_scan(
            x_rgb, d_rgb,
            A_rgb, B_rgb, C_e, D_rgb, z=None,
            delta_bias=dt_projs_bias_rgb,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y_rgb.dtype == torch.float

        out_y_e = self.selective_scan(
            x_e, d_e,
            A_e, B_e, C_rgb, D_e, z=None,
            delta_bias=dt_projs_bias_e,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y_rgb.dtype == torch.float

        out_y_rgb = out_y_rgb.float().view(B, -1, L)
        out_y_e = out_y_e.float().view(B, -1, L)
        y_rgb = rearrange(out_y_rgb, "b d l -> b l d")
        y_rgb = self.out_norm_1(y_rgb)
        y_e = rearrange(out_y_e, "b d l -> b l d")
        y_e = self.out_norm_2(y_e)

        return y_rgb, y_e

    def forward(self, x_rgb, x_e, token_rgb, token_e):
        B, n, C = x_rgb.shape

        full_embedding = self.embedding_rgb.weight @ token_rgb.weight  # [128, C]
        pred_route = self.route_rgb(x_rgb)  # [B, HW, num_token]
        cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)  # [B, HW, num_token]
        prompt_rgb = torch.matmul(cls_policy, full_embedding).view(B, n, self.d_state)

        full_embedding = self.embedding_e.weight @ token_e.weight  # [128, C]
        pred_route = self.route_e(x_e)  # [B, HW, num_token]
        cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)  # [B, HW, num_token]
        prompt_e = torch.matmul(cls_policy, full_embedding).view(B, n, self.d_state)

        b, l, c = prompt_rgb.shape
        prompt_rgb = prompt_rgb.permute(0, 2, 1).contiguous().view(b, 1, c, l)
        prompt_e = prompt_e.permute(0, 2, 1).contiguous().view(b, 1, c, l)
        x1, x2 = self.forward_core(x_rgb, x_e, prompt_rgb, prompt_e)  # [B, L, C]
        x1 = x1.permute(0, 2, 1).contiguous()
        x2 = x2.permute(0, 2, 1).contiguous()
        return x1, x2



class Cross_mamba(nn.Module):
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

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj_modalx = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

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

        self.CMA_ssm = Selective_Scan(
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

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor, token_rgb, token_e):
        # resi_rgb, resi_e = x_rgb, x_e
        x_rgb = self.in_proj(x_rgb)
        x_e = self.in_proj_modalx(x_e)
        x_rgb, resi_rgb = x_rgb.chunk(2, dim=-1)
        x_e, resi_e = x_e.chunk(2, dim=-1)
        resi_rgb = self.act(resi_rgb)
        resi_e = self.act(resi_e)
        B, H, W, D = x_rgb.shape

        x_rgb_trans = x_rgb.permute(0, 3, 1, 2).contiguous()
        x_e_trans = x_e.permute(0, 3, 1, 2).contiguous()
        x_rgb_conv = self.act(self.conv2d(x_rgb_trans))  # (b, d, h, w)
        x_e_conv = self.act(self.conv2d(x_e_trans))  # (b, d, h, w)
        x_rgb_conv = rearrange(x_rgb_conv, "b d h w -> b (h w) d")
        x_e_conv = rearrange(x_e_conv, "b d h w -> b (h w) d")
        y_rgb, y_e = self.CMA_ssm(x_rgb_conv, x_e_conv, token_rgb, token_e)
        # to b, d, h, w
        y_rgb = y_rgb.view(B, H, W, -1)
        y_e = y_e.view(B, H, W, -1)

        out_rgb = self.dropout_rgb(self.out_proj_rgb(y_rgb))
        out_e = self.dropout_e(self.out_proj_e(y_e))

        out_rgb = resi_rgb + self.drop_path1(out_rgb)
        out_e = resi_e + self.drop_path2(out_e)
        return out_rgb, out_e

class fuse_mamba_test2(nn.Module):
    def __init__(self, d_model=96, depth=3, d_state=16, drop_path=0.2,input_resolution=None):
        super().__init__()
        self.depth = depth

        self.cross_mamba = nn.ModuleList(
            Cross_mamba(d_model=d_model)
            for _ in range(self.depth)
        )
        self.inner_rank = math.ceil(d_model / 16)

        self.embedding_rgb = nn.Embedding(self.inner_rank, d_state)
        self.embedding_rgb.weight.data.uniform_(-1 / self.inner_rank, 1 / self.inner_rank)

        self.embedding_e = nn.Embedding(self.inner_rank, d_state)
        self.embedding_e.weight.data.uniform_(-1 / self.inner_rank, 1 / self.inner_rank)

    def forward(self, cli, der):
        cli = rearrange(cli, 'b c h w -> b h w c')
        der = rearrange(der, 'b c h w -> b h w c')
        for block in self.cross_mamba:
            cli, der = block(cli, der, self.embedding_rgb, self.embedding_e)

        return cli, der