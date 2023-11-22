from typing import Any
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ..modules.LKAs import DLKAFormer_EncoderBlock, DLKAFormer_DecoderBlock
from ..modules.dynunet_blocks import UnetResBlock, UnetOutBlock
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
)

from timm.models.layers import trunc_normal_
from monai.networks.layers.utils import get_norm_layer
from ..modules.layers import LayerNorm
from ..modules.dynunet_blocks import get_conv_layer, UnetResBlock
from ..modules.deform_conv import DeformConvPack
from ..modules.dynunet_blocks import get_padding


class BaseBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class CNNBlock(BaseBlock):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(5, 5, 5),
        stride=(2, 2, 2),
        deform_cnn=False,
        **kwargs
    ):
        super().__init__()
        self.deform_cnn = deform_cnn
        if deform_cnn:
            self.conv = nn.Sequential(
                DeformConvPack(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=get_padding(kernel_size, stride),
                ),
                nn.BatchNorm3d(out_channels),
                nn.PReLU(),
            )
        else:
            self.conv = UnetResBlock(
                3,
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_name="batch",
            )

        self.cnorm = get_norm_layer(
            name=("group", {"num_groups": in_channels}), channels=out_channels
        )

        self.apply(self._init_weights)

    def forward(self, x):
        r = x.clone()
        if self.deform_cnn:
            x = x.contiguous()
        x = self.conv(x)
        x = self.cnorm(x)
        return x


class CNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        kernel_sizes,
        features,
        strides,
        maxpools,
        dropouts,
        deform_cnn=False,
    ) -> Any:
        super().__init__()
        assert isinstance(kernel_sizes, list), "kernel_sizes must be a list"
        assert isinstance(features, list), "features must be a list"
        assert isinstance(strides, list), "strides must be a list"
        assert (
            len(kernel_sizes) == len(strides) == len(features)
        ), "kernel_sizes, features, and strides must have the same length"

        if not isinstance(dropouts, list):
            dropouts = [dropouts for _ in features]

        in_out_channles = [in_channels] + features
        in_out_channles = [
            (i, o) for i, o in zip(in_out_channles[:-1], in_out_channles[1:])
        ]

        self.encoder_blocks = nn.ModuleList()

        for (ich, och), ks, st, mp, do in zip(
            in_out_channles, kernel_sizes, strides, maxpools, dropouts
        ):
            encoder = CNNBlock(
                in_channels=ich,
                out_channels=och,
                kernel_size=ks,
                stride=1 if mp else st,
                deform_cnn=deform_cnn,
                dropout=do,
            )
            if mp:
                maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
                self.encoder_blocks.append(nn.Sequential(encoder, maxpool))
            else:
                self.encoder_blocks.append(encoder)

    def forward(self, x):
        layer_features = []
        for block in self.encoder_blocks:
            x = block(x)
            layer_features.append(x.clone())
        return x, layer_features


class HybEncBlk(BaseBlock):
    def __init__(
        self,
        in_channels=4,
        out_channels=32,
        kernel_size=(5, 5, 5),
        stride=(2, 2, 2),
        dropout=0.10,
        tf_input_size=32 * 32 * 32,
        tf_proj_size=64,
        tf_repeats=3,
        tf_num_heads=4,
        tf_dropout=0.15,
        deform_cnn=False,
        trans_block=TransformerBlock,
        **kwargs
    ):
        super().__init__()
        self.deform_cnn = deform_cnn

        if deform_cnn:
            self.conv = nn.Sequential(
                DeformConvPack(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=get_padding(kernel_size, stride),
                ),
                nn.BatchNorm3d(out_channels),
                nn.PReLU(),
            )
        else:
            self.conv = UnetResBlock(
                3,
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_name="batch",
            )

        self.norm = get_norm_layer(
            name=("group", {"num_groups": in_channels}), channels=out_channels
        )
        self.stages = None
        if trans_block:
            self.stages = nn.Sequential(
                *[
                    trans_block(
                        input_size=tf_input_size,
                        hidden_size=out_channels,
                        proj_size=tf_proj_size,
                        num_heads=tf_num_heads,
                        dropout_rate=tf_dropout,
                        pos_embed=True,
                    )
                    for _ in range(tf_repeats)
                ]
            )

        self.apply(self._init_weights)

    def forward(self, x):
        if self.deform_cnn:
            x = x.contiguous()
        x = self.conv(x)
        x = self.norm(x)
        if self.stages:
            x = self.stages(x)
        return x


class HybridEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        features: list[int],
        cnn_kernel_sizes: list[int | tuple],
        cnn_strides: list[int | tuple],
        cnn_maxpools: list[bool],
        cnn_dropouts: list[float] | float,
        tf_input_sizes: list[int],
        tf_proj_sizes: list[int],
        tf_repeats: list[int],
        tf_num_heads: list[int],
        tf_dropouts: list[float] | float,
        spatial_dims=3,
        trans_block=TransformerBlock,
        *args: Any,
        **kwds: Any
    ) -> Any:
        super().__init__()

        if not isinstance(cnn_dropouts, list):
            cnn_dropouts = [cnn_dropouts for _ in features]
        if not isinstance(tf_dropouts, list):
            tf_dropouts = [tf_dropouts for _ in features]
        io_channles = [in_channels] + features
        io_channles = [(i, o) for i, o in zip(io_channles[:-1], io_channles[1:])]

        self.encoder_blocks = nn.ModuleList()
        infos = zip(
            io_channles,
            cnn_kernel_sizes,
            cnn_strides,
            cnn_maxpools,
            cnn_dropouts,
            tf_input_sizes,
            tf_proj_sizes,
            tf_repeats,
            tf_num_heads,
            tf_dropouts,
        )
        for (
            (ich, och),
            cnnks,
            cnnst,
            cnnmp,
            cnndo,
            tfis,
            tfps,
            tfr,
            tfnh,
            tfdo,
        ) in infos:
            block = HybEncBlk(
                in_channels=ich,  # 4,
                out_channels=och,  # 32,
                # deform_cnn=True,
                kernel_size=cnnks,  # 5,
                stride=cnnst,  # 2,
                dropout=cnndo,  # 0.1,
                tf_input_size=tfis,  # ,
                tf_proj_size=tfps,  # ,
                tf_repeats=tfr,  # 3,
                tf_num_heads=tfnh,  # ,
                tf_dropout=tfdo,  # ,
                trans_block=trans_block,
            )
            self.encoder_blocks.append(block)

    def forward(self, x):
        layer_features = []
        for block in self.encoder_blocks:
            x = block(x)
            layer_features.append(x.clone())
        return x, layer_features


class CNNDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        features,
        skip_channels,
        cnn_kernel_sizes,
        cnn_dropouts,
        tcv_kernel_sizes,
        tcv_strides,
        deform_cnn=False,
        spatial_dims=3,
    ) -> Any:
        super().__init__()
        assert isinstance(cnn_kernel_sizes, list), "cnn_kernel_sizes must be a list"
        assert isinstance(features, list), "features must be a list"
        assert isinstance(tcv_strides, list), "strides must be a list"
        assert (
            len(cnn_kernel_sizes) == len(tcv_strides) == len(features)
        ), "cnn_kernel_sizes, features, and tcv_strides must have the same length"

        if not isinstance(cnn_dropouts, list):
            cnn_dropouts = [cnn_dropouts for _ in features]

        in_out_channles = [in_channels] + features
        in_out_channles = [
            (i, o) for i, o in zip(in_out_channles[:-1], in_out_channles[1:])
        ]

        self.decoder_blocks = nn.ModuleList()

        info = zip(
            in_out_channles,
            skip_channels,
            cnn_kernel_sizes,
            cnn_dropouts,
            tcv_kernel_sizes,
            tcv_strides,
        )
        for (ich, och), skch, cnnks, cnndo, tcvks, tcvst in info:
            cnn_block = UnetResBlock
            decoder = DLKAFormer_DecoderBlock(
                spatial_dims=spatial_dims,
                in_channels=ich,
                out_channels=och,
                skip_channels=skch,
                cnn_block=cnn_block,
                cnn_kernel_size=cnnks,
                cnn_dropout=cnndo,
                tcv_kernel_size=tcvks,
                tcv_stride=tcvst,
                deform_cnn=deform_cnn,
                trans_block=None,
            )
            self.decoder_blocks.append(decoder)

    def forward(self, x, skips: list):
        for block in self.decoder_blocks:
            skip = skips.pop()
            x = block(x, skip)
        return x


class HybridDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        features: list[int],
        skip_channels: list[int],
        cnn_kernel_sizes: list[int | tuple],
        cnn_dropouts: list[float] | float,
        tcv_strides: list[int | tuple],
        tcv_kernel_sizes: list[int | tuple],
        tf_input_sizes: list[int],
        tf_proj_sizes: list[int],
        tf_repeats: list[int],
        tf_num_heads: list[int],
        tf_dropouts: list[float] | float,
        spatial_dims=3,
        trans_block=TransformerBlock,
        *args: Any,
        **kwds: Any
    ) -> Any:
        super().__init__()

        if not isinstance(cnn_dropouts, list):
            cnn_dropouts = [cnn_dropouts for _ in features]
        if not isinstance(tf_dropouts, list):
            tf_dropouts = [tf_dropouts for _ in features]
        io_channles = [in_channels] + features
        io_channles = [(i, o) for i, o in zip(io_channles[:-1], io_channles[1:])]

        self.decoder_blocks = nn.ModuleList()
        infos = zip(
            io_channles,
            skip_channels,
            cnn_kernel_sizes,
            cnn_dropouts,
            tcv_kernel_sizes,
            tcv_strides,
            tf_input_sizes,
            tf_proj_sizes,
            tf_repeats,
            tf_num_heads,
            tf_dropouts,
        )
        for (
            (ich, och),
            skch,
            cnnks,
            cnndo,
            tcvks,
            tcvst,
            tfis,
            tfps,
            tfr,
            tfnh,
            tfdo,
        ) in infos:
            cnn_block = UnetResBlock
            decoder = DLKAFormer_DecoderBlock(
                spatial_dims=spatial_dims,
                in_channels=ich,
                out_channels=och,
                skip_channels=skch,
                cnn_block=cnn_block,
                cnn_kernel_size=cnnks,
                cnn_dropout=cnndo,
                tcv_kernel_size=tcvks,
                tcv_stride=tcvst,
                tf_input_size=tfis,
                tf_proj_size=tfps,
                tf_repeats=tfr,
                tf_num_heads=tfnh,
                tf_dropout=tfdo,
                trans_block=trans_block,
            )
            self.decoder_blocks.append(decoder)

    def forward(self, x, skips=None):
        first = True
        for block in self.decoder_blocks:
            if first:
                x = block(x)
                first = False
            else:
                x = block(x, skips.pop())
        return x


class Model(nn.Module):
    def __init__(
        self,
        spatial_shapes,
        in_channels=4,
        out_channels=3,
        # encoder params
        cnn_kernel_sizes=[5, 3],
        cnn_features=[32, 64],
        cnn_strides=[2, 2],
        cnn_maxpools=[0, 1],
        cnn_dropouts=0.1,
        hyb_kernel_sizes=[3, 3, 3],
        hyb_features=[128, 256, 512],
        hyb_strides=[2, 2, 2],
        hyb_maxpools=[0, 0, 0],
        hyb_cnn_dropouts=0.1,
        hyb_tf_proj_sizes=[32, 64, 64],
        hyb_tf_repeats=[3, 3, 3],
        hyb_tf_num_heads=[4, 4, 4],
        hyb_tf_dropouts=0.15,
        # decoder params
        dec_hyb_tcv_kernel_sizes=[5, 5, 5],
        dec_cnn_tcv_kernel_sizes=[5, 5, 5],
        dec_hyb_kernel_sizes=None,
        dec_hyb_features=None,
        dec_hyb_cnn_dropouts=None,
        dec_hyb_tf_proj_sizes=None,
        dec_hyb_tf_repeats=None,
        dec_hyb_tf_num_heads=None,
        dec_hyb_tf_dropouts=None,
        dec_cnn_kernel_sizes=None,
        dec_cnn_features=None,
        dec_cnn_dropouts=None,
    ):
        super().__init__()

        # ------------------------------------- Vars Prepration --------------------------------
        spatial_dims = len(spatial_shapes)
        init_features = cnn_features[0] // 2
        enc_cnn_in_channels = in_channels
        enc_cnn_out_channels = cnn_features[-1]
        enc_hyb_in_channels = enc_cnn_out_channels
        enc_hyb_out_channels = hyb_features[-1]

        # check dropouts
        cnn_dropouts = (
            [cnn_dropouts for _ in spatial_shapes]
            if not isinstance(cnn_dropouts, list)
            else cnn_dropouts
        )
        hyb_cnn_dropouts = (
            [hyb_cnn_dropouts for _ in spatial_shapes]
            if not isinstance(hyb_cnn_dropouts, list)
            else hyb_cnn_dropouts
        )
        hyb_tf_dropouts = (
            [hyb_tf_dropouts for _ in spatial_shapes]
            if not isinstance(hyb_tf_dropouts, list)
            else hyb_tf_dropouts
        )

        # check strides
        cnn_strides = [
            [st for _ in spatial_shapes] if not isinstance(st, list) else st
            for st in cnn_strides
        ]
        hyb_strides = [
            [st for _ in spatial_shapes] if not isinstance(st, list) else st
            for st in hyb_strides
        ]

        # check dec params
        dec_hyb_skip_channels = [0] + hyb_features[::-1][1:]
        dec_cnn_skip_channels = cnn_features[::-1]
        if not dec_hyb_features:
            dec_hyb_features = hyb_features[::-1][1:] + [enc_hyb_in_channels]
        if not dec_cnn_features:
            dec_cnn_features = cnn_features[::-1][1:] + [init_features]

        if not dec_hyb_kernel_sizes:
            dec_hyb_kernel_sizes = hyb_kernel_sizes[::-1]
        if not dec_hyb_cnn_dropouts:
            dec_hyb_cnn_dropouts = hyb_cnn_dropouts[::-1]
        if not dec_hyb_tf_proj_sizes:
            dec_hyb_tf_proj_sizes = hyb_tf_proj_sizes[::-1]
        if not dec_hyb_tf_repeats:
            dec_hyb_tf_repeats = hyb_tf_repeats[::-1]
        if not dec_hyb_tf_num_heads:
            dec_hyb_tf_num_heads = hyb_tf_num_heads[::-1]
        if not dec_hyb_tf_dropouts:
            dec_hyb_tf_dropouts = hyb_tf_dropouts[::-1]
        if not dec_cnn_kernel_sizes:
            dec_cnn_kernel_sizes = cnn_kernel_sizes[::-1]
        if not dec_cnn_dropouts:
            dec_cnn_dropouts = cnn_dropouts[::-1]

        # calculate spatial_shapes in encoder and decoder diferent layers
        enc_spatial_shaps = [spatial_shapes]
        for stride in cnn_strides + hyb_strides:
            enc_spatial_shaps.append(
                [int(np.ceil(ss / st)) for ss, st in zip(enc_spatial_shaps[-1], stride)]
            )
        dec_spatial_shaps = [enc_spatial_shaps[-1]]
        for stride in hyb_strides[::-1] + cnn_strides[::-1]:
            dec_spatial_shaps.append(
                [int(np.ceil(ss * st)) for ss, st in zip(dec_spatial_shaps[-1], stride)]
            )
        enc_cnn_spatial_shaps = enc_spatial_shaps[: len(cnn_kernel_sizes)]
        enc_hyb_spatial_shaps = enc_spatial_shaps[
            len(cnn_kernel_sizes) + 1 :
        ]  # we need output channels of cnn before tf
        dec_hyb_spatial_shaps = dec_spatial_shaps[: len(hyb_kernel_sizes)]
        dec_cnn_spatial_shaps = dec_spatial_shaps[
            len(hyb_kernel_sizes) :
        ]  # we need input channels of block before cnn

        # calc hyb_tf_input_sizes corresponding cnn_strides and hyb_strides
        enc_hyb_tf_input_sizes = [
            np.prod(ss, dtype=int) for ss in enc_hyb_spatial_shaps
        ]
        dec_hyb_tf_input_sizes = [
            np.prod(ss, dtype=int) for ss in dec_hyb_spatial_shaps
        ]

        # ------------------------------------- Initialization --------------------------------
        self.init = nn.Sequential(
            nn.Conv3d(in_channels, init_features, 1),
            nn.BatchNorm3d(init_features),
            nn.PReLU(),
        )

        # ------------------------------------- Encoder --------------------------------
        self.cnn_encoder = CNNEncoder(
            in_channels=init_features,
            kernel_sizes=cnn_kernel_sizes,
            features=cnn_features,
            strides=cnn_strides,
            maxpools=cnn_maxpools,
            dropouts=cnn_dropouts,
            deform_cnn=True,
        )

        self.hyb_encoder = HybridEncoder(
            in_channels=cnn_features[-1],
            features=hyb_features,
            cnn_kernel_sizes=hyb_kernel_sizes,
            cnn_strides=hyb_strides,
            cnn_maxpools=hyb_maxpools,
            # deform_cnn=True,
            cnn_dropouts=hyb_cnn_dropouts,
            tf_input_sizes=enc_hyb_tf_input_sizes,
            tf_proj_sizes=hyb_tf_proj_sizes,
            tf_repeats=hyb_tf_repeats,
            tf_num_heads=hyb_tf_num_heads,
            tf_dropouts=hyb_tf_dropouts,
            trans_block=TransformerBlock_Deform_LKA_Channel,
        )

        # ------------------------------------- Decoder --------------------------------
        self.hyb_decoder = HybridDecoder(
            spatial_dims=spatial_dims,
            in_channels=enc_hyb_out_channels,
            features=dec_hyb_features,
            skip_channels=dec_hyb_skip_channels,
            cnn_kernel_sizes=dec_hyb_kernel_sizes,
            cnn_dropouts=dec_cnn_dropouts,
            tcv_kernel_sizes=dec_hyb_tcv_kernel_sizes,
            tcv_strides=hyb_strides[::-1],
            tf_input_sizes=dec_hyb_tf_input_sizes,
            tf_proj_sizes=hyb_tf_proj_sizes,
            tf_repeats=hyb_tf_repeats,
            tf_num_heads=hyb_tf_num_heads,
            tf_dropouts=hyb_tf_dropouts,
            trans_block=TransformerBlock_Deform_LKA_Channel,
        )

        self.cnn_decoder = CNNDecoder(
            spatial_dims=spatial_dims,
            in_channels=dec_hyb_features[-1],
            skip_channels=dec_cnn_skip_channels,
            cnn_kernel_sizes=dec_cnn_kernel_sizes,
            cnn_dropouts=dec_cnn_dropouts,
            features=dec_cnn_features,
            tcv_kernel_sizes=dec_cnn_tcv_kernel_sizes,
            tcv_strides=cnn_strides[::-1],
            # deform_cnn=True
        )

        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=dec_cnn_features[-1] + init_features,
            out_channels=out_channels,
            dropout=0,
        )

    def forward(self, x):
        x = self.init(x)
        r = x.clone()

        x, cnn_skips = self.cnn_encoder(x)
        # print(f"after x, cnn_skips = self.cnn_encoder(x) | x:{x.shape}")
        x, hyb_skips = self.hyb_encoder(x)
        # print(f"after x, hyb_skips = self.hyb_encoder(x) | x:{x.shape}")

        x = self.hyb_decoder(x, hyb_skips[:-1])
        # print(f"after x = self.hyb_decoder(x, hyb_skips) | x:{x.shape}")
        x = self.cnn_decoder(x, cnn_skips)
        # print(f"after x = self.cnn_decoder(x, cnn_skips) | x:{x.shape}")

        x = torch.concatenate([x, r], dim=1)
        x = self.out(x)
        return x
