from typing import Any
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .modules.LKAs import DLKAFormer_EncoderBlock, DLKAFormer_DecoderBlock
from .modules.dynunet_blocks import UnetResBlock, UnetOutBlock
from .modules.vit.transformers import (
    TransformerBlock,
    TransformerBlock_3D_LKA,
    TransformerBlock_LKA_Channel,
    TransformerBlock_SE,
    TransformerBlock_Deform_LKA_Channel,
    TransformerBlock_Deform_LKA_Channel_sequential,
    TransformerBlock_3D_LKA_3D_conv,
    TransformerBlock_LKA_Channel_norm,
    TransformerBlock_Deform_LKA_Spatial_sequential,
    TransformerBlock_Deform_LKA_Spatial,
    TransformerBlock_3D_single_deform_LKA,
    TransformerBlock_Deform_LKA_Channel_V2,
    TransformerBlock_Deform_LKA_Spatial_V2,
    TransformerBlock_3D_single_deform_LKA_V2,
)

from timm.models.layers import trunc_normal_
from monai.networks.layers.utils import get_norm_layer
from .modules.layers import LayerNorm
from .modules.dynunet_blocks import get_conv_layer, UnetResBlock
from .modules.deform_conv import DeformConvPack
from .modules.dynunet_blocks import get_padding
from functools import partial


class BridgeModule(nn.Module):
    def __init__(
        self,
        feats: list[int],
        c_attn_block: nn.Module = nn.Identity,
        s_attn_block: nn.Module = nn.Identity,
        m_attn_block: nn.Module = nn.Identity,
        use_weigths=False,
    ):
        super().__init__()
        ifeats = feats[::-1]

        self.use_c = False if c_attn_block == nn.Identity else True
        self.use_s = False if s_attn_block == nn.Identity else True
        self.use_m = False if m_attn_block == nn.Identity else True

        self.use_weigths = use_weigths
        if use_weigths:
            self.c_w = nn.Parameter(torch.zeros(len(feats)))
            self.s_w = nn.Parameter(torch.zeros(len(feats)))
            self.m_w = nn.Parameter(torch.zeros(len(feats)))

        self.act = nn.GELU()
        self.m_atts, self.c_atts, self.s_atts = (
            nn.ModuleList(),
            nn.ModuleList(),
            nn.ModuleList(),
        )
        self.norms = nn.ModuleList()
        # self.norms_atts = nn.ModuleList()
        # self.norms_x = nn.ModuleList()
        for feat in ifeats:
            self.c_atts.append(c_attn_block(feat))
            self.s_atts.append(s_attn_block(feat))
            self.m_atts.append(m_attn_block(feat))
            self.norms.append(nn.BatchNorm3d(feat))
            # self.norms_atts.append(nn.BatchNorm3d(feat))
            # self.norms_x.append(nn.BatchNorm3d(feat))

        self.ups = nn.ModuleList()
        for df, uf in zip(ifeats[:-1], ifeats[1:]):
            self.ups.append(
                nn.Sequential(
                    nn.Conv3d(df, uf, kernel_size=1, padding=0),
                    nn.GELU(),
                    nn.Upsample(scale_factor=2, mode="trilinear"),
                )
            )

        self.projs = nn.ModuleDict()
        self.proj_norms = nn.ModuleDict()
        for f1 in feats:
            for f2 in feats:
                self.projs[f"{f1}->{f2}"] = (
                    nn.Sequential(
                        nn.Conv3d(f1, f2, kernel_size=1, padding=0), nn.GELU()
                    )
                    #                     if f1 != f2
                    #                     else nn.Identity()
                )

    #             self.proj_norms[f"n{f1}"] = (
    #                 nn.BatchNorm3d(f1)
    #             )

    def _aggregate(self, skips, out_shape):
        feat = out_shape[1]
        special_shape = out_shape[2:]
        agg = torch.zeros(out_shape).to(skips[0].device)
        for skip in skips:
            #             if (skip.shape == out_shape).all(): continue
            ps = self.projs[f"{skip.shape[1]}->{feat}"](skip)
            ps = F.interpolate(ps, size=special_shape)
            #             ps = self.proj_norms[f"{skip.shape[1]}->{feat}"](ps)
            agg = agg + ps
        #         agg = self.proj_norms[f"n{feat}"](agg)
        return agg

    def forward(self, *skips):
        i = 0
        last_x = None
        outs = []
        for x, ca, sa, ma, norm in zip(
            skips[::-1],
            self.c_atts,
            self.s_atts,
            self.m_atts,
            self.norms,  # , self.norms_atts, self.norms_x
        ):
            _x = x.clone()
            if i > 0:
                x = x + self.ups[i - 1](last_x)

            c_att = ca(_x) if self.use_c else 0
            s_att = sa(x) if self.use_c else 0
            m_att = ma(_x + self._aggregate(skips, x.shape)) if self.use_m else 0

            if self.use_weigths:
                att = self.c_w[i] * c_att + self.s_w[i] * s_att + self.m_w[i] * m_att
            else:
                att = c_att + s_att + m_att

            #             att = F.layer_norm(att, normalized_shape=att.shape[2:])
            #             x = F.layer_norm(x, normalized_shape=x.shape[2:])

            x = norm(att + x)
            outs.append(x)

            last_x = x.clone()
            i += 1

        return outs[::-1]
