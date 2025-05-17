import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
# from mamba_ssm.modules.mamba_simple import Mamba
# from models.aux_model.mamba_block.mamba_simple import Mamba
from mamba_1p1p1.mamba_ssm.modules.mamba_simple import Mamba

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        # h, w = x.shape[-2:]
        return self.body(x)

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self,patch_size=4, stride=4,in_chans=36, embed_dim=32*32*32, norm_layer=None, flatten=True):
        super().__init__()
        # patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = LayerNorm(embed_dim,'BiasFree')

    def forward(self, x):
        #（b,c,h,w)->(b,c*s*p,h//s,w//s)
        #(b,h*w//s**2,c*s**2)
        B, C, H, W = x.shape
        # x = F.unfold(x, self.patch_size, stride=self.patch_size)
        # x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        # x = self.norm(x)
        return x

class PatchUnEmbed(nn.Module):
    def __init__(self,basefilter) -> None:
        super().__init__()
        self.nc = basefilter
    def forward(self, x,x_size):
        B,HW,C = x.shape
        x = x.transpose(1, 2).view(B, self.nc, x_size[0], x_size[1])
        return x

class TokenSwapMamba(nn.Module):
    def __init__(self, dim):
        super(TokenSwapMamba, self).__init__()
        self.msencoder = Mamba(dim,bimamba_type=None)
        self.panencoder = Mamba(dim,bimamba_type=None)
        self.norm1 = LayerNorm(dim,'with_bias')
        self.norm2 = LayerNorm(dim,'with_bias')
    def forward(self, ms,pan,ms_residual,pan_residual):
        # ms (B,N,C)
        #pan (B,N,C)
        ms_residual = ms + ms_residual
        pan_residual = pan + pan_residual
        ms = self.norm1(ms_residual)
        pan = self.norm2(pan_residual)
        B,N,C = ms.shape
        ms_first_half = ms[:, :, :C//2]
        pan_first_half = pan[:, :, :C//2]
        ms_swap= torch.cat([pan_first_half,ms[:,:,C//2:]],dim=2)
        pan_swap= torch.cat([ms_first_half,pan[:,:,C//2:]],dim=2)
        ms_swap = self.msencoder(ms_swap) + ms_residual
        pan_swap = self.panencoder(pan_swap) + pan_residual
        return ms_swap,pan_swap,ms_residual,pan_residual


class CrossMamba(nn.Module):
    def __init__(self, dim, input_resolution):
        super(CrossMamba, self).__init__()
        self.input_resolution = input_resolution
        self.cross_mamba = Mamba(dim,bimamba_type="v3")
        self.norm1 = LayerNorm(dim,'with_bias')
        self.norm2 = LayerNorm(dim,'with_bias')
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward_cross(self, x, x_resi, y):
        x_resi = x + x_resi
        x = self.norm1(x_resi)
        y = self.norm2(y)
        global_f = self.cross_mamba(self.norm1(x), extra_emb=self.norm2(y))
        B,HW,C = global_f.shape
        x = global_f.transpose(1, 2).view(B, C, self.input_resolution[0], self.input_resolution[0])
        x = (self.dwconv(x)+x).flatten(2).transpose(1, 2)
        return x, x_resi
    def forward(self,ms,ms_resi,pan,pan_res):
        ms, ms_resi = self.forward_cross(ms, ms_resi, pan)
        pan, pan_res = self.forward_cross(pan, pan_res, ms)
        return ms,ms_resi, pan, pan_res



class mamba_fusion(nn.Module):
    def __init__(self, dim, input_resolution, depth=[2,5]):
        super(mamba_fusion, self).__init__()
        self.resolution = input_resolution
        self.token_fusions = nn.ModuleList()

        for i in range(depth[0]):
            self.token_fusions.append(TokenSwapMamba(dim))
        self.cross_mambas = nn.ModuleList()
        for i in range(depth[1]):
            self.cross_mambas.append(CrossMamba(dim, input_resolution=self.resolution))

    def forward(self, cli, der):
        _, _, h, w = cli.shape
        cli = to_3d(cli)
        der = to_3d(der)
        cli_resi = 0
        der_resi = 0
        for token_fusion in self.token_fusions:
            cli, der, cli_resi, der_resi = token_fusion(cli, der, cli_resi, der_resi)

        cli_resi = 0
        der_resi = 0
        for cross_mamba in self.cross_mambas:
            cli, cli_resi, der, der_resi = cross_mamba(cli, cli_resi, der, der_resi)
        return cli, der

