from typing import Optional, Sequence, Tuple, Union, Any
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..modules.vit.transformers import (
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

from .base import BaseBlock, get_conv_layer
from .cnn import get_cnn_block


__all__ = ["HybridEncoder", "HybridDecoder"]


def get_vit_block(code):
    if code == "c":
        return TransformerBlock_Deform_LKA_Channel_V2
    elif code == "s":
        return TransformerBlock_Deform_LKA_Spatial_V2
    elif code == "C":
        return TransformerBlock_Deform_LKA_Channel_sequential
    elif code == "S":
        return TransformerBlock_Deform_LKA_Spatial_sequential
    elif code == "R":
        return TransformerBlock_3D_single_deform_LKA
    else:
        raise NotImplementedError(f"Not implemented cnn-block for code:<{code}>")


class HybridEncoder(BaseBlock):
    def __init__(
        self,
        in_channels: int,
        features: list[int],
        cnn_kernel_sizes: list[int | tuple],
        cnn_strides: list[int | tuple],
        cnn_maxpools: list[bool],
        cnn_dropouts: list[float] | float,
        vit_input_sizes: list[int],
        vit_proj_sizes: list[int],
        vit_repeats: list[int],
        vit_num_heads: list[int],
        vit_dropouts: list[float] | float,
        spatial_dims=3,
        cnn_blocks="b",
        vit_blocks="c",
        vit_sandwich=True,
        norm_name="batch",  # ("group", {"num_groups": in_channels}),
        act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        *args: Any,
        **kwds: Any,
    ) -> Any:
        super().__init__()

        # >>> checking
        if not isinstance(cnn_dropouts, list):
            cnn_dropouts = [cnn_dropouts for _ in features]
        if not isinstance(vit_dropouts, list):
            vit_dropouts = [vit_dropouts for _ in features]
        io_channles = [in_channels] + features
        io_channles = [(i, o) for i, o in zip(io_channles[:-1], io_channles[1:])]
        # <<< checking
        self

        self.encoder_blocks = nn.ModuleList()
        infos = zip(
            cnn_blocks,
            vit_blocks,
            io_channles,
            cnn_kernel_sizes,
            cnn_strides,
            cnn_maxpools,
            cnn_dropouts,
            vit_input_sizes,
            vit_proj_sizes,
            vit_repeats,
            vit_num_heads,
            vit_dropouts,
        )
        for (
            c_blkc,
            t_blkc,
            (ich, och),
            c_ks,
            c_st,
            c_mp,
            c_do,
            t_is,
            t_ps,
            t_rp,
            t_nh,
            t_do,
        ) in infos:
            conv_block = get_cnn_block(code=c_blkc)(
                spatial_dims=spatial_dims,
                in_channels=ich,
                out_channels=och,
                kernel_size=c_ks,
                stride=1 if c_mp else c_st,
                norm_name=norm_name,
                act_name=act_name,
                dropout=c_do,
            )
            # vit_block = get_vit_block(code=t_blkc)(
            #     input_size=och,
            #     hidden_size=och*2,
            #     proj_size=t_ps,
            #     num_heads=t_nh,
            #     dropout_rate=t_do,
            #     pos_embed=True,
            # )

            vits = [
                get_vit_block(code=t_blkc)(
                    input_size=t_is,
                    hidden_size=och,
                    proj_size=t_ps,
                    num_heads=t_nh,
                    dropout_rate=t_do,
                    pos_embed=True,
                )
                for _ in range(t_rp)
            ]

            if vit_sandwich:
                vit_block = nn.Sequential(
                    *vits,
                    get_cnn_block(code=c_blkc)(
                        spatial_dims=spatial_dims,
                        in_channels=och,
                        out_channels=och,
                        kernel_size=c_ks,
                        stride=1,
                        norm_name=norm_name,
                        act_name=act_name,
                        dropout=c_do,
                    ),
                )
            else:
                vit_block = nn.Sequential(*vits)

            if c_mp:
                maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
                self.encoder_blocks.append(
                    nn.Sequential(conv_block, maxpool, vit_block)
                )
            else:
                self.encoder_blocks.append(nn.Sequential(conv_block, vit_block))

        self.apply(self._init_weights)

    def forward(self, x):
        layer_features = []
        for block in self.encoder_blocks:
            x = block(x)
            layer_features.append(x.clone())
        return x, layer_features


class HybridDecoder(BaseBlock):
    def __init__(
        self,
        in_channels,
        skip_channels,
        features,
        cnn_kernel_sizes: list[int | tuple],
        cnn_dropouts: list[float] | float,
        vit_input_sizes: list[int],
        vit_proj_sizes: list[int],
        vit_repeats: list[int],
        vit_num_heads: list[int],
        vit_dropouts: list[float] | float,
        tcv_kernel_sizes,
        tcv_strides,
        tcv_bias=False,
        spatial_dims=3,
        cnn_blocks="b",
        vit_blocks="c",
        vit_sandwich=True,
        norm_name="batch",  # ("group", {"num_groups": in_channels}),
        act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        *args: Any,
        **kwds: Any,
    ) -> Any:
        super().__init__()

        # >>> checking
        if len(cnn_blocks) == 1:
            cnn_blocks *= len(cnn_kernel_sizes)
        if len(vit_blocks) == 1:
            vit_blocks *= len(vit_input_sizes)
        assert isinstance(cnn_kernel_sizes, list), "cnn_kernel_sizes must be a list"
        assert isinstance(features, list), "features must be a list"
        assert (
            len(cnn_blocks) == len(cnn_kernel_sizes) == len(features)
        ), "cnn_blocks, cnn_kernel_sizes, and features must have the same length"
        if not isinstance(cnn_dropouts, list):
            cnn_dropouts = [cnn_dropouts for _ in features]
        if not isinstance(vit_dropouts, list):
            vit_dropouts = [vit_dropouts for _ in features]
        assert (
            len(cnn_kernel_sizes) == len(tcv_strides) == len(features)
        ), "kernel_sizes, features, and tcv_strides must have the same length"
        # <<< checking
        self._data = []

        io_channles = [in_channels] + features
        io_channles = [(i, o) for i, o in zip(io_channles[:-1], io_channles[1:])]

        self.ups, self.convs, self.vits, self.convs_e = (
            nn.ModuleList(),
            nn.ModuleList(),
            nn.ModuleList(),
            nn.ModuleList(),
        )
        info = zip(
            cnn_blocks,
            vit_blocks,
            io_channles,
            skip_channels,
            tcv_kernel_sizes,
            tcv_strides,
            cnn_kernel_sizes,
            cnn_dropouts,
            vit_input_sizes,
            vit_proj_sizes,
            vit_repeats,
            vit_num_heads,
            vit_dropouts,
        )

        for (
            c_blkc,
            t_blkc,
            (ich, och),
            skch,
            tcv_ks,
            tcv_st,
            c_ks,
            c_do,
            t_is,
            t_ps,
            t_rp,
            t_nh,
            t_do,
        ) in info:
            transp_conv = get_conv_layer(
                spatial_dims=spatial_dims,
                in_channels=ich,
                out_channels=och,
                kernel_size=tcv_ks,
                stride=tcv_st,
                dropout=0,
                bias=tcv_bias,
                conv_only=True,
                is_transposed=True,
            )
            self.ups.append(transp_conv)

            conv_block = get_cnn_block(code=c_blkc)(
                spatial_dims=spatial_dims,
                in_channels=och + skch,
                out_channels=och,
                kernel_size=c_ks,
                stride=1,
                dropout=c_do,
                norm_name=norm_name,
                act_name=act_name,
            )
            self.convs.append(conv_block)

            vits = [
                get_vit_block(code=t_blkc)(
                    input_size=t_is,
                    hidden_size=och,
                    proj_size=t_ps,
                    num_heads=t_nh,
                    dropout_rate=t_do,
                    pos_embed=True,
                )
                for _ in range(t_rp)
            ]

            if vit_sandwich:
                self.vits.append(
                    nn.Sequential(
                        *vits,
                        get_cnn_block(code=c_blkc)(
                            spatial_dims=spatial_dims,
                            in_channels=och,
                            out_channels=och,
                            kernel_size=c_ks,
                            stride=1,
                            dropout=c_do,
                            norm_name=norm_name,
                            act_name=act_name,
                        ),
                    )
                )
            else:
                self.vits.append(nn.Sequential(*vits))

        self.apply(self._init_weights)

    def forward(self, x, skips: list, return_outs=False, skip_sum=False):
        outs = []
        for up, conv, vit in zip(self.ups, self.convs, self.vits):
            # print(f"\nx: {x.shape}, skip: {skips[-1].shape}\n")
            x = up(x)

            if skip_sum:
                x = x + skips.pop()
            else:
                x = torch.cat((x, skips.pop()), dim=1)

            x = conv(x)
            x = vit(x)
            if return_outs:
                outs.append(x.clone())
        return (x, outs) if return_outs else x
