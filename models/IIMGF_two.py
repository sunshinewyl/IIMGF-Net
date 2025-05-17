import torch
import torch.nn as nn
from models.mca import PatchMerging,CrossTransformer_meta
import torch.nn.functional as F
from models.aux_model.mamba.vmamba import vmamba_small_s2l15, vmamba_tiny_s1l8
from timm.layers import DropPath, trunc_normal_
from models.aux_model.mamba.function import CMIGF
from models.aux_model.mamba.function_last import Cross_Mamba_Attention_SSM

class IIMGF(nn.Module):
    def __init__(self,num_classes=5, embed_dims=[192, 384, 768, 768], input_resolution=[[28,28], [14,14], [7,7]],depth=[1,1,2,1],
                 re_size=[14,7,7], N=[784,196,49]):
        super(IIMGF, self).__init__()
        self.mamba = vmamba_tiny_s1l8()
        checkpoint = torch.load('/home/wyl/work/mamba_code/mamba_test/vssm1_tiny_0230s_ckpt_epoch_264.pth')
        self.mamba.load_state_dict(checkpoint['model'])
        self.re_size = re_size
        self.pos_embed = self._pos_embed(96, 56, 56)

        self.fusion_block0 = CMIGF(d_model=embed_dims[0],depth=depth[0], input_resolution=input_resolution[0])
        # self.fusion_block0 = cross_mamba_fusion(in_channel=embed_dims[0])
        self.downsample_cli_0 = PatchMerging(input_resolution=input_resolution[0], dim=embed_dims[0])
        self.downsample_der_0 = PatchMerging(input_resolution=input_resolution[0], dim=embed_dims[0])

        self.fusion_block1 = CMIGF(d_model=embed_dims[1],depth=depth[1],input_resolution=input_resolution[1])
        # self.fusion_block1 = cross_mamba_fusion(in_channel=embed_dims[1])
        self.downsample_cli = PatchMerging(input_resolution=input_resolution[1],dim=embed_dims[1])
        self.downsample_der = PatchMerging(input_resolution=input_resolution[1], dim=embed_dims[1])

        self.fusion_block2 = CMIGF(d_model=embed_dims[2],depth=depth[2], input_resolution=input_resolution[2])
        # self.fusion_block2 = cross_mamba_fusion(in_channel=embed_dims[2])
        # self.fusion_block3 = fuse_mamba_test1(d_model=embed_dims[3], input_resolution=input_resolution[2], depth=depth[3])

        self.pool_cli = nn.AdaptiveAvgPool2d((1,1))
        self.pool_der = nn.AdaptiveAvgPool2d((1,1))
        self.pool_fuse = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(embed_dims[3])
        self.norm_f = nn.LayerNorm(embed_dims[3]*2)

        self.C2F = nn.ModuleList([])
        self.C2F_cross = nn.ModuleList([
            Cross_Mamba_Attention_SSM(d_model=embed_dims[0] * 2),
            Cross_Mamba_Attention_SSM(d_model=embed_dims[1] * 2),
        ])
        self.F2C = nn.ModuleList([])
        self.F2C_cross = nn.ModuleList([
            Cross_Mamba_Attention_SSM(d_model=embed_dims[1] * 2),
            Cross_Mamba_Attention_SSM(d_model=embed_dims[2] * 2),
        ])
        self.C2F.append(nn.ModuleList([
            nn.Linear(N[1], N[0]),
            nn.ReLU(inplace=True),
            nn.Conv1d(embed_dims[1] * 2, embed_dims[0] * 2, kernel_size=1),
            nn.ReLU(inplace=True),
        ]))
        self.C2F.append(nn.ModuleList([
            nn.Linear(N[2], N[1]),
            nn.ReLU(inplace=True),
            nn.Conv1d(embed_dims[2] * 2, embed_dims[1] * 2, kernel_size=1),
        ]))
        self.F2C.append(nn.ModuleList([
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(embed_dims[0] * 2, embed_dims[1] * 2, kernel_size=1),
            nn.ReLU(inplace=True),
        ]))
        self.F2C.append(nn.ModuleList([
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(embed_dims[1] * 2, embed_dims[2] * 2, kernel_size=1),
        ]))

        self.num_classes = num_classes
        # DIAG fc
        hidden_num = 3072
        self.fc_diag = nn.Linear(hidden_num,
                                 self.num_classes)  # only use for three modalities

        self.fc_pn = nn.Linear(hidden_num, 3)
        self.fc_bwn = nn.Linear(hidden_num, 2)
        self.fc_vs = nn.Linear(hidden_num, 3)
        self.fc_pig = nn.Linear(hidden_num, 3)
        self.fc_str = nn.Linear(hidden_num, 3)
        self.fc_dag = nn.Linear(hidden_num, 3)
        self.fc_rs = nn.Linear(hidden_num, 2)

    def _pos_embed(self, embed_dims, patch_height, patch_width):
        pos_embed = nn.Parameter(torch.zeros(1, embed_dims, patch_height, patch_width))
        trunc_normal_(pos_embed, std=0.02)
        return pos_embed

    def loss_orth(self, share, distinct):
        return F.cosine_similarity(share, distinct, dim=-1).mean()

    def forward(self,cli_x,der_x):
        f = []
        B = cli_x.size(0)

        cli_h = self.mamba.patch_embed(cli_x)
        cli_h = cli_h + self.pos_embed
        der_h = self.mamba.patch_embed(der_x)
        der_h = der_h + self.pos_embed

        cli_h = self.mamba.layers[0](cli_h)
        der_h = self.mamba.layers[0](der_h)

        # fusion
        cli_f0, der_f0 = self.fusion_block0(cli_h, der_h)
        f.append([cli_f0, der_f0])
        cli_f0 = self.downsample_cli_0(cli_f0)
        der_f0 = self.downsample_der_0(der_f0)
        cli_f0 = cli_f0.transpose(1, 2).reshape(B, -1, self.re_size[0], self.re_size[0])
        der_f0 = der_f0.transpose(1, 2).reshape(B, -1, self.re_size[0], self.re_size[0])

        cli_h = self.mamba.layers[1](cli_h) #(320,196)
        der_h = self.mamba.layers[1](der_h)

        #fusion
        cli_f1, der_f1 = self.fusion_block1(cli_h + cli_f0, der_h + der_f0)
        f.append([cli_f1, der_f1])
        cli_f1 = self.downsample_cli(cli_f1)
        der_f1 = self.downsample_der(der_f1)
        cli_f1 = cli_f1.transpose(1, 2).reshape(B, -1, self.re_size[1], self.re_size[1])
        der_f1 = der_f1.transpose(1, 2).reshape(B, -1, self.re_size[1], self.re_size[1])

        cli_h = self.mamba.layers[2](cli_h) #(640, 49)
        der_h = self.mamba.layers[2](der_h)

        #fusion 1
        cli_f2, der_f2 = self.fusion_block2(cli_h + cli_f1, der_h + der_f1)
        f.append([cli_f2, der_f2])
        cli_f2 = cli_f2.transpose(1, 2).reshape(B, -1, self.re_size[2], self.re_size[2])
        der_f2 = der_f2.transpose(1, 2).reshape(B, -1, self.re_size[2], self.re_size[2])

        # der_h = self.mamba.layers[3](der_h) #(640, 49)
        # cli_h = self.mamba.layers[3](cli_h)

        for i in range(len(f)):
            f[i] = torch.cat(f[i], dim=-1)
            f[i] = f[i].permute(0, 3, 1, 2).flatten(2)

        ##C2F
        for j in range(len(f)):
            if j < len(f) - 1:
                input = f[j + 1]
                for module in self.C2F[j]:
                    temp = module(input)
                    input = temp
                f[j], _ = self.C2F_cross[j](f[j], temp)
                f[j] = f[j].transpose(1, 2)
            else:
                f[j] = f[j]
        ##F2C
        for k in range(len(f)):
            if k == 0:
                f[k] = f[k]
            else:
                input = f[k - 1]
                for module in self.F2C[k - 1]:
                    temp = module(input)
                    input = temp
                out, _ = self.F2C_cross[k - 1](f[k], temp)
                f[k] = out.transpose(1, 2) + f[k]

        Z3 = f[-1]

        cli = self.pool_cli(cli_f2 + cli_h)
        cli = torch.flatten(cli, start_dim=1)
        der = self.pool_der(der_f2 + der_h)
        der = torch.flatten(der, start_dim=1)
        fuse = self.pool_fuse(Z3)
        fuse = torch.flatten(fuse, start_dim=1)

        x = torch.cat((cli, der, fuse), dim=-1)

        # fc
        diag = self.fc_diag(x)

        pn = self.fc_pn(x)
        bwv = self.fc_bwn(x)
        vs = self.fc_vs(x)
        pig = self.fc_pig(x)
        str = self.fc_str(x)
        dag = self.fc_dag(x)
        rs = self.fc_rs(x)

        return[diag,pn,bwv,vs,pig,str,dag,rs]

    def criterion(self, logit, truth,weight=None):
        if weight == None:
            loss = F.cross_entropy(logit, truth)
        else:
            loss = F.cross_entropy(logit, truth,weight=weight)

        return loss