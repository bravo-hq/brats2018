from typing import Optional, Sequence, Tuple, Union, Any
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ....modules.vit.transformers import (
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


__all__ = ["Bottleneck"]


def get_block(code):
    if code.lower() == "c":
        return TransformerBlock_Deform_LKA_Channel_V2
    elif code.lower() == "s":
        return TransformerBlock_Deform_LKA_Spatial_V2
    else:
        raise NotImplementedError(f"Not implemented cnn-block for code:<{code}>")


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        features,
        cnn_kernel_sizes: list[int | tuple],
        cnn_strides: list[int | tuple],
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
        assert isinstance(cnn_strides, list), "strides must be a list"
        assert (
            len(cnn_blocks)
            == len(cnn_kernel_sizes)
            == len(cnn_strides)
            == len(features)
        ), "cnn_blocks, cnn_kernel_sizes, features, and strides must have the same length"
        if not isinstance(cnn_dropouts, list):
            cnn_dropouts = [cnn_dropouts for _ in features]
        if not isinstance(vit_dropouts, list):
            vit_dropouts = [vit_dropouts for _ in features]
        assert (
            len(cnn_kernel_sizes) == len(tcv_strides) == len(features)
        ), "kernel_sizes, features, and tcv_strides must have the same length"
        # <<< checking

        io_channles = [in_channels] + features
        io_channles = [(i, o) for i, o in zip(io_channles[:-1], io_channles[1:])]

        self.ups, self.convs, self.vits = (
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
            cnn_strides,
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
            c_st,
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

            vit_block = get_vit_block(code=t_blkc)(
                input_size=och,
                hidden_size=och,
                proj_size=t_ps,
                num_heads=t_nh,
                dropout_rate=t_do,
                pos_embed=True,
            )
            self.vits.append(vit_block)

        self.apply(self._init_weights)

    def forward(self, x, skips: list, return_outs=False):
        outs = []
        for up, conv, vit in zip(self.ups, self.convs, self.vits):
            out = up(x)
            out = torch.cat((out, skips.pop()), dim=1)
            out = conv(out)
            out = vit(out)
            if return_outs:
                outs.append(out.clone())
        return out, outs if return_outs else out
