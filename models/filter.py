import torch
import torch.nn as nn
import torch.nn.functional as F


class CMDF_block(nn.Module):
    def __init__(self, in_channels, reduction=4, kernel_size=3):
        super(CMDF_block, self).__init__()
        """
        in_channels: 输入通道数 c
        reduction: 通道压缩率 r
        kernel_size: 静态和动态卷积的卷积核大小 k
        """
        self.kernel_size = kernel_size
        self.mid_channels = in_channels // reduction

        # Static Filter: Depth-wise convolution
        self.static_dwconv = nn.Conv2d(in_channels, in_channels,
                                       kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                                       groups=in_channels, bias=False)

        # Context Interaction: 1x1 Conv -> AvgPool -> 1x1 Conv -> ReLU -> 1x1 Conv
        self.context_conv1 = nn.Conv2d(in_channels * 2, self.mid_channels * 2, kernel_size=1, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # Spatial average pooling
        self.context_conv2 = nn.Conv2d(self.mid_channels * 2, self.mid_channels * 2, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.context_conv3 = nn.Conv2d(self.mid_channels * 2, (in_channels) * kernel_size * kernel_size, kernel_size=1,
                                       bias=False)

        # Spatial-wise filter generation: simple 1x1 conv
        self.spatial_conv = nn.Conv2d(in_channels * 2, kernel_size * kernel_size, kernel_size=1, bias=False)

        # Final 1x1 fusion
        self.fusion_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)

    def forward(self, X2, Y2):
        """
        X2, Y2: 输入的两个模态 (b, c, h, w)
        """
        B, C, H, W = X2.size()

        # Static Filter (Depth-wise conv)
        X2_static = self.static_dwconv(X2)  # (B, C, H, W)

        # Cross-Modal Contextual Interaction
        F_concat = torch.cat([X2_static, Y2], dim=1)  # (B, 2C, H, W)

        # Channel-wise filter generation
        context = self.context_conv1(F_concat)  # (B, 2c/r, H, W)
        context = self.avgpool(context)  # (B, 2c/r, 1, 1)
        context = self.context_conv2(context)  # (B, 2c/r, 1, 1)
        context = self.relu(context)
        channel_filter = self.context_conv3(context)  # (B, C*k^2, 1, 1)
        channel_filter = channel_filter.view(B, C, self.kernel_size * self.kernel_size, 1,
                                             1)  # (B, C, k^2, 1, 1)

        # Spatial-wise filter generation
        spatial_filter = self.spatial_conv(F_concat)  # (B, k^2, H, W)
        spatial_filter = spatial_filter.view(B, 1, self.kernel_size * self.kernel_size, H, W)  # (B, 1, k^2, H, W)

        # Combine Channel-wise and Spatial-wise filters
        dynamic_filter = channel_filter + spatial_filter  # broadcasting

        # Apply Dynamic Filter (depth-wise convolution per pixel)
        unfolded_X2 = F.unfold(X2, kernel_size=self.kernel_size, padding=self.kernel_size // 2)  # (B, C*k^2, H*W)
        unfolded_X2 = unfolded_X2.view(B, C, self.kernel_size * self.kernel_size, H, W)  # (B, C, k^2, H, W)
        dynamic_out = (unfolded_X2 * dynamic_filter).sum(dim=2)  # (B, C, H, W)

        # Concatenate static & dynamic outputs
        out = torch.cat([X2_static, dynamic_out], dim=1)  # (B, 2C, H, W)

        # Final fusion
        out = self.fusion_conv(out)  # (B, C, H, W)

        return out

class CMDF(nn.Module):
    def __init__(self, in_channels, reduction=4, kernel_size=3):
        super(CMDF, self).__init__()
        self.CMDF_cli = CMDF_block(in_channels, reduction=reduction, kernel_size=kernel_size)
        self.CMDF_der = CMDF_block(in_channels, reduction=reduction, kernel_size=kernel_size)

    def forward(self, cli, der):
        cli = cli.permute(0, 3, 1, 2).contiguous()
        der = der.permute(0, 3, 1, 2).contiguous()
        cli = self.CMDF_cli(cli, der)
        der = self.CMDF_der(der, cli)
        cli = cli.permute(0, 2, 3, 1).contiguous()
        der = der.permute(0, 2, 3, 1).contiguous()
        return cli, der


