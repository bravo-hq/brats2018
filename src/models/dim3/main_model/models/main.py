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
    TransformerBlock_Deform_LKA_Channel_V2,
    TransformerBlock_Deform_LKA_Spatial_V2,
    TransformerBlock_3D_single_deform_LKA_V2,
)

from timm.models.layers import trunc_normal_
from monai.networks.layers.utils import get_norm_layer
from ..modules.layers import LayerNorm
from ..modules.dynunet_blocks import get_conv_layer, UnetResBlock
from ..modules.deform_conv import DeformConvPack
from ..modules.dynunet_blocks import get_padding
from functools import partial


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
        **kwargs,
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
        self, in_channels, kernel_sizes, features, strides, maxpools, dropouts, deforms
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

        for (ich, och), ks, st, mp, do, dfrm in zip(
            in_out_channles, kernel_sizes, strides, maxpools, dropouts, deforms
        ):
            encoder = CNNBlock(
                in_channels=ich,
                out_channels=och,
                kernel_size=ks,
                stride=1 if mp else st,
                deform_cnn=dfrm,
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
        use_conv=True,
        trans_block=TransformerBlock,
        **kwargs,
    ):
        super().__init__()
        self.deform_cnn = deform_cnn

        if use_conv:
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
        if hasattr(self, "conv"):
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
        use_cnn: list[bool],
        cnn_kernel_sizes: list[int | tuple],
        cnn_strides: list[int | tuple],
        cnn_maxpools: list[bool],
        cnn_dropouts: list[float] | float,
        tf_input_sizes: list[int],
        tf_proj_sizes: list[int],
        tf_repeats: list[int],
        tf_num_heads: list[int],
        tf_dropouts: list[float] | float,
        cnn_deforms: list[bool],
        spatial_dims=3,
        trans_block=TransformerBlock,
        *args: Any,
        **kwds: Any,
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
            use_cnn,
            cnn_kernel_sizes,
            cnn_strides,
            cnn_maxpools,
            cnn_dropouts,
            cnn_deforms,
            tf_input_sizes,
            tf_proj_sizes,
            tf_repeats,
            tf_num_heads,
            tf_dropouts,
        )
        for (
            (ich, och),
            ucnn,
            cnnks,
            cnnst,
            cnnmp,
            cnndo,
            cnndfrm,
            tfis,
            tfps,
            tfr,
            tfnh,
            tfdo,
        ) in infos:
            block = HybEncBlk(
                in_channels=ich,  # 4,
                out_channels=och,  # 32,
                kernel_size=cnnks,  # 5,
                stride=cnnst,  # 2,
                dropout=cnndo,  # 0.1,
                deform_cnn=cnndfrm,  # False
                tf_input_size=tfis,  # ,
                tf_proj_size=tfps,  # ,
                tf_repeats=tfr,  # 3,
                tf_num_heads=tfnh,  # ,
                tf_dropout=tfdo,  # ,
                trans_block=trans_block,
                use_conv=ucnn,
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
        deforms,
        tcv_kernel_sizes,
        tcv_strides,
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
            deforms,
            tcv_kernel_sizes,
            tcv_strides,
        )
        for (ich, och), skch, cnnks, cnndo, cnndfrm, tcvks, tcvst in info:
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
                deform_cnn=cnndfrm,
                trans_block=None,
            )
            self.decoder_blocks.append(decoder)

    def forward(self, x, skips: list, return_outs=False):
        outs = []
        for block in self.decoder_blocks:
            skip = skips.pop()
            x = block(x, skip)
            if return_outs:
                outs.append(x.clone())
        if return_outs:
            return x, outs
        else:
            return x


class HybridDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        features: list[int],
        skip_channels: list[int],
        use_cnn: list[bool],
        cnn_kernel_sizes: list[int | tuple],
        cnn_dropouts: list[float] | float,
        cnn_deforms: list[bool],
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
        **kwds: Any,
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
            use_cnn,
            cnn_kernel_sizes,
            cnn_dropouts,
            cnn_deforms,
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
            ucnn,
            cnnks,
            cnndo,
            cnndfrm,
            tcvks,
            tcvst,
            tfis,
            tfps,
            tfr,
            tfnh,
            tfdo,
        ) in infos:
            cnn_block = UnetResBlock if ucnn else None
            decoder = DLKAFormer_DecoderBlock(
                spatial_dims=spatial_dims,
                in_channels=ich,
                out_channels=och,
                skip_channels=skch,
                cnn_block=cnn_block,
                cnn_kernel_size=cnnks,
                cnn_dropout=cnndo,
                deform_cnn=cnndfrm,
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

    def forward(self, x, skips=None, return_outs=False):
        first = True
        outs = []
        for block in self.decoder_blocks:
            if first:
                x = block(x)
                first = False
            else:
                x = block(x, skips.pop())
            if return_outs:
                outs.append(x.clone())

        if return_outs:
            return x, outs
        else:
            return x


class Model_Base(nn.Module):
    def __init__(
        self,
        spatial_shapes,
        do_ds=False,
        in_channels=4,
        out_channels=3,
        # encoder params
        cnn_kernel_sizes=[5, 3],
        cnn_features=[32, 64],
        cnn_strides=[2, 2],
        cnn_maxpools=[0, 1],
        cnn_dropouts=0.1,
        cnn_deforms=[True, True],
        hyb_use_cnn=[True, True],
        hyb_kernel_sizes=[3, 3, 3],
        hyb_features=[128, 256, 512],
        hyb_strides=[2, 2, 2],
        hyb_maxpools=[0, 0, 0],
        hyb_cnn_dropouts=0.1,
        hyb_tf_proj_sizes=[32, 64, 64],
        hyb_tf_repeats=[3, 3, 3],
        hyb_tf_num_heads=[4, 4, 4],
        hyb_tf_dropouts=0.15,
        hyb_deforms=[False, False],
        # decoder params
        dec_hyb_tcv_kernel_sizes=[5, 5, 5],
        dec_cnn_tcv_kernel_sizes=[5, 5, 5],
        dec_cnn_deforms=None,
        dec_hyb_use_cnn=None,
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
        dec_hyb_deforms=None,
    ):
        super().__init__()
        self.do_ds = do_ds

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

        if not dec_hyb_use_cnn:
            dec_hyb_use_cnn = hyb_use_cnn[::-1]
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
        if not dec_hyb_deforms:
            dec_hyb_deforms = hyb_deforms[::-1]
        if not dec_cnn_kernel_sizes:
            dec_cnn_kernel_sizes = cnn_kernel_sizes[::-1]
        if not dec_cnn_dropouts:
            dec_cnn_dropouts = cnn_dropouts[::-1]
        if not dec_cnn_deforms:
            dec_cnn_deforms = cnn_deforms[::-1]

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
            nn.PReLU(),
            nn.BatchNorm3d(init_features),
        )

        # ------------------------------------- Encoder --------------------------------
        self.cnn_encoder = CNNEncoder(
            in_channels=init_features,
            kernel_sizes=cnn_kernel_sizes,
            features=cnn_features,
            strides=cnn_strides,
            maxpools=cnn_maxpools,
            dropouts=cnn_dropouts,
            deforms=cnn_deforms,
        )

        self.hyb_encoder = HybridEncoder(
            in_channels=cnn_features[-1],
            features=hyb_features,
            use_cnn=hyb_use_cnn,
            cnn_kernel_sizes=hyb_kernel_sizes,
            cnn_strides=hyb_strides,
            cnn_maxpools=hyb_maxpools,
            cnn_deforms=hyb_deforms,
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
            # return_outs=self.use_ds,
            spatial_dims=spatial_dims,
            in_channels=enc_hyb_out_channels,
            features=dec_hyb_features,
            skip_channels=dec_hyb_skip_channels,
            use_cnn=dec_hyb_use_cnn,
            cnn_kernel_sizes=dec_hyb_kernel_sizes,
            cnn_dropouts=dec_cnn_dropouts,
            cnn_deforms=dec_hyb_deforms,
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
            # return_outs=self.use_ds,
            spatial_dims=spatial_dims,
            in_channels=dec_hyb_features[-1],
            skip_channels=dec_cnn_skip_channels,
            cnn_kernel_sizes=dec_cnn_kernel_sizes,
            cnn_dropouts=dec_cnn_dropouts,
            features=dec_cnn_features,
            tcv_kernel_sizes=dec_cnn_tcv_kernel_sizes,
            tcv_strides=cnn_strides[::-1],
            deforms=dec_cnn_deforms,
        )

        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=dec_cnn_features[-1] + init_features,
            out_channels=out_channels,
            dropout=0,
        )

        # self.cnn_context_bridge = CNNContextBridge(
        #     ...
        # )

        self.num_classes = out_channels

    def forward(self, x):
        x = self.init(x)
        r = x.clone()

        x, cnn_skips = self.cnn_encoder(x)
        # print(f"after x, cnn_skips = self.cnn_encoder(x) | x:{x.shape}")
        x, hyb_skips = self.hyb_encoder(x)
        # print(f"after x, hyb_skips = self.hyb_encoder(x) | x:{x.shape}")

        if self.do_ds:
            x, hyb_outs = self.hyb_decoder(x, hyb_skips[:-1], return_outs=True)
            x, cnn_outs = self.cnn_decoder(x, cnn_skips, return_outs=True)

        else:
            x = self.hyb_decoder(x, hyb_skips[:-1])
            # print(f"after x = self.hyb_decoder(x, hyb_skips) | x:{x.shape}")
            # x, cnn_skips = self.cnn_context_bridge(x, cnn_skips)
            x = self.cnn_decoder(x, cnn_skips)
            # print(f"after x = self.cnn_decoder(x, cnn_skips) | x:{x.shape}")

        x = torch.concatenate([x, r], dim=1)
        x = self.out(x)

        if self.do_ds:
            return x, hyb_outs + cnn_outs

        return x


# ================================================================================================================================================================================================


class GateChannelAttentionModule(nn.Module):
    def __init__(self, channels, c=3, eps=1e-5):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.eps = eps
        self.c = c

    def forward(self, x):
        y = self.avgpool(x)
        mean = y.mean(dim=1, keepdim=True)
        mean_x2 = (y**2).mean(dim=1, keepdim=True)
        var = mean_x2 - mean**2
        y_norm = (y - mean) / torch.sqrt(var + self.eps)
        y_transform = torch.exp(-(y_norm**2 / 2 * self.c))
        return x * y_transform.expand_as(x)


class PositionalAttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.b = nn.Conv3d(dim, dim, 1)
        self.c = nn.Conv3d(dim, dim, 1)
        self.d = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        n, c, d, h, w = x.shape
        B = self.b(x).flatten(2).transpose(1, 2)
        C = self.c(x).flatten(2)
        D = self.d(x).flatten(2).transpose(1, 2)
        attn = (B @ C).softmax(dim=-1)
        y = (attn @ D).transpose(1, 2).reshape(n, c, d, h, w)
        out = y + x
        return out


class SKAttentionModule(nn.Module):
    def __init__(self, inplanes, planes=None, groups=32, ratio=16):
        super().__init__()
        if not planes:
            planes = inplanes
        d = max(planes // ratio, 32)
        self.planes = planes
        self.split_3x3x3 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=3, padding=1, groups=groups),
            nn.BatchNorm3d(planes),
            nn.LeakyReLU(),
        )
        self.split_5x5x5 = nn.Sequential(
            nn.Conv3d(
                inplanes, planes, kernel_size=3, padding=2, dilation=2, groups=groups
            ),
            nn.BatchNorm3d(planes),
            nn.LeakyReLU(),
        )
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        # self.fc = nn.Sequential(
        #     nn.Linear(planes, d),
        #     nn.BatchNorm1d(d),
        #     nn.LeakyReLU()
        # )
        self.fc = nn.Linear(planes, d)
        self.fc_norm = nn.BatchNorm1d(d)
        self.fc_act = nn.LeakyReLU()

        self.fc1 = nn.Linear(d, planes)
        self.fc2 = nn.Linear(d, planes)

    def forward(self, x):
        batch_size = x.shape[0]
        u1 = self.split_3x3x3(x)
        u2 = self.split_5x5x5(x)
        u = u1 + u2
        s = self.avgpool(u).flatten(1)

        z = self.fc(s)

        z = self.fc_act(z if z.shape[0] == 1 else self.fc_norm(z))

        attn_scores = torch.cat([self.fc1(z), self.fc2(z)], dim=1)
        attn_scores = attn_scores.view(batch_size, 2, self.planes)
        attn_scores = attn_scores.softmax(dim=1)
        a = attn_scores[:, 0].view(batch_size, self.planes, 1, 1, 1)
        b = attn_scores[:, 1].view(batch_size, self.planes, 1, 1, 1)
        u1 = u1 * a.expand_as(u1)
        u2 = u2 * b.expand_as(u2)
        x = u1 + u2
        return x


class MultiScaleLKA3DModule(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()

        kernels = [3, 5, 7]
        paddings = [1, 4, 9]
        dilations = [1, 2, 3]
        self.conv0 = nn.Conv3d(
            in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels
        )
        self.spatial_convs = nn.ModuleList(
            [
                nn.Conv3d(
                    in_channels,
                    in_channels,
                    kernel_size=kernels[i],
                    stride=1,
                    padding=paddings[i],
                    groups=in_channels,
                    dilation=dilations[i],
                )
                for i in range(len(kernels))
            ]
        )
        self.conv1 = nn.Conv3d(3 * in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        original_input = x.clone()
        attn = self.conv0(x)
        spatial_attns = [conv(attn) for conv in self.spatial_convs]
        attn = torch.cat(spatial_attns, dim=1)
        attn = self.conv1(attn)
        return original_input * attn


class MSLKA3AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.proj_1x1_1 = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.activation = nn.GELU()
        self.spatial_gating = MultiScaleLKA3DModule(in_channels)
        self.proj_1x1_2 = nn.Conv3d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        x = self.proj_1x1_1(x)
        x = self.activation(x)
        x = self.spatial_gating(x)
        x = self.proj_1x1_2(x)
        return x


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


class TransformerBlock_Deform_LKA_SC(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.dlka_s = TransformerBlock_Deform_LKA_Spatial(*args, **kwargs)
        self.dlka_c = TransformerBlock_Deform_LKA_Channel(*args, **kwargs)

    def forward(self, x):
        s_a = self.dlka_s(x)
        c_a = self.dlka_c(x)
        sc_a = s_a + c_a
        return F.layer_norm(sc_a, normalized_shape=sc_a.shape[2:])


class TransformerBlock_Deform_LKA_SC_V2(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.dlka_s = TransformerBlock_Deform_LKA_Spatial_V2(*args, **kwargs)
        self.dlka_c = TransformerBlock_Deform_LKA_Channel_V2(*args, **kwargs)

    def forward(self, x):
        s_a = self.dlka_s(x)
        c_a = self.dlka_c(x)
        sc_a = s_a + c_a
        return F.layer_norm(sc_a, normalized_shape=sc_a.shape[2:])


class Model_Bridge(nn.Module):
    def __init__(
        self,
        spatial_shapes,
        do_ds=False,
        in_channels=4,
        out_channels=3,
        # encoder params
        cnn_kernel_sizes=[5, 3],
        cnn_features=[32, 64],
        cnn_strides=[2, 2],
        cnn_maxpools=[0, 1],
        cnn_dropouts=0.1,
        cnn_deforms=[True, True],
        hyb_use_cnn=[True, True],
        hyb_kernel_sizes=[3, 3, 3],
        hyb_features=[128, 256, 512],
        hyb_strides=[2, 2, 2],
        hyb_maxpools=[0, 0, 0],
        hyb_cnn_dropouts=0.1,
        hyb_tf_proj_sizes=[32, 64, 64],
        hyb_tf_repeats=[3, 3, 3],
        hyb_tf_num_heads=[4, 4, 4],
        hyb_tf_dropouts=0.15,
        hyb_deforms=[False, False],
        hyb_tf_block=0,
        # bridge params
        br_use=True,
        br_skip_levels=[0, 1, 2, 3],
        br_c_attn_use=True,
        br_s_att_use=True,
        br_m_att_use=True,
        br_use_p_ttn_w=True,
        # decoder params
        dec_hyb_tcv_kernel_sizes=[5, 5, 5],
        dec_cnn_tcv_kernel_sizes=[5, 5, 5],
        dec_cnn_deforms=None,
        dec_hyb_use_cnn=None,
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
        dec_hyb_deforms=None,
        dec_hyb_tf_block=None,
    ):
        super().__init__()
        self.do_ds = do_ds

        # tf attn blocks
        tf_blocks = {
            "0": TransformerBlock_Deform_LKA_Channel,
            "1": TransformerBlock_Deform_LKA_Channel_V2,
            "2": TransformerBlock_Deform_LKA_Spatial,
            "3": TransformerBlock_Deform_LKA_Spatial_V2,
            "4": TransformerBlock_3D_single_deform_LKA,
            "5": TransformerBlock_3D_single_deform_LKA_V2,
            "6": TransformerBlock_Deform_LKA_SC,
            "7": TransformerBlock_Deform_LKA_SC_V2,
        }

        # bridge params
        self.br_use = br_use
        self.br_skip_levels = br_skip_levels
        self.br_c_attn_use = br_c_attn_use
        self.br_s_att_use = br_s_att_use
        self.br_m_att_use = br_m_att_use
        self.br_use_p_ttn_w = br_use_p_ttn_w

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

        if not dec_hyb_use_cnn:
            dec_hyb_use_cnn = hyb_use_cnn[::-1]
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
        if not dec_hyb_deforms:
            dec_hyb_deforms = hyb_deforms[::-1]
        if not dec_cnn_kernel_sizes:
            dec_cnn_kernel_sizes = cnn_kernel_sizes[::-1]
        if not dec_cnn_dropouts:
            dec_cnn_dropouts = cnn_dropouts[::-1]
        if not dec_cnn_deforms:
            dec_cnn_deforms = cnn_deforms[::-1]
        if dec_hyb_tf_block == None:
            dec_hyb_tf_block = hyb_tf_block

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
            nn.PReLU(),
            nn.BatchNorm3d(init_features),
        )

        # ------------------------------------- Encoder --------------------------------
        self.cnn_encoder = CNNEncoder(
            in_channels=init_features,
            kernel_sizes=cnn_kernel_sizes,
            features=cnn_features,
            strides=cnn_strides,
            maxpools=cnn_maxpools,
            dropouts=cnn_dropouts,
            deforms=cnn_deforms,
        )

        self.hyb_encoder = HybridEncoder(
            in_channels=cnn_features[-1],
            features=hyb_features,
            use_cnn=hyb_use_cnn,
            cnn_kernel_sizes=hyb_kernel_sizes,
            cnn_strides=hyb_strides,
            cnn_maxpools=hyb_maxpools,
            cnn_deforms=hyb_deforms,
            cnn_dropouts=hyb_cnn_dropouts,
            tf_input_sizes=enc_hyb_tf_input_sizes,
            tf_proj_sizes=hyb_tf_proj_sizes,
            tf_repeats=hyb_tf_repeats,
            tf_num_heads=hyb_tf_num_heads,
            tf_dropouts=hyb_tf_dropouts,
            trans_block=tf_blocks[str(hyb_tf_block)],
        )

        # ------------------------------------- Decoder --------------------------------
        self.hyb_decoder = HybridDecoder(
            # return_outs=self.use_ds,
            spatial_dims=spatial_dims,
            in_channels=enc_hyb_out_channels,
            features=dec_hyb_features,
            skip_channels=dec_hyb_skip_channels,
            use_cnn=dec_hyb_use_cnn,
            cnn_kernel_sizes=dec_hyb_kernel_sizes,
            cnn_dropouts=dec_cnn_dropouts,
            cnn_deforms=dec_hyb_deforms,
            tcv_kernel_sizes=dec_hyb_tcv_kernel_sizes,
            tcv_strides=hyb_strides[::-1],
            tf_input_sizes=dec_hyb_tf_input_sizes,
            tf_proj_sizes=hyb_tf_proj_sizes,
            tf_repeats=hyb_tf_repeats,
            tf_num_heads=hyb_tf_num_heads,
            tf_dropouts=hyb_tf_dropouts,
            trans_block=tf_blocks[str(dec_hyb_tf_block)],
        )

        self.cnn_decoder = CNNDecoder(
            # return_outs=self.use_ds,
            spatial_dims=spatial_dims,
            in_channels=dec_hyb_features[-1],
            skip_channels=dec_cnn_skip_channels,
            cnn_kernel_sizes=dec_cnn_kernel_sizes,
            cnn_dropouts=dec_cnn_dropouts,
            features=dec_cnn_features,
            tcv_kernel_sizes=dec_cnn_tcv_kernel_sizes,
            tcv_strides=cnn_strides[::-1],
            deforms=dec_cnn_deforms,
        )

        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=dec_cnn_features[-1] + init_features,
            out_channels=out_channels,
            dropout=0,
        )

        if self.br_use:
            feats = cnn_features + hyb_features[:-1]
            self.bridge = BridgeModule(
                feats=[feats[i] for i in self.br_skip_levels],
                c_attn_block=GateChannelAttentionModule
                if self.br_c_attn_use
                else nn.Identity,
                s_attn_block=partial(SKAttentionModule, groups=8)
                if self.br_s_att_use
                else nn.Identity,
                m_attn_block=MultiScaleLKA3DModule
                if self.br_m_att_use
                else nn.Identity,
                use_weigths=self.br_use_p_ttn_w,
            )

        self.num_classes = out_channels

    def forward(self, x):
        x = self.init(x)
        r = x.clone()

        x, cnn_skips = self.cnn_encoder(x)
        # print(f"after x, cnn_skips = self.cnn_encoder(x) | x:{x.shape}")
        x, hyb_skips = self.hyb_encoder(x)
        # print(f"after x, hyb_skips = self.hyb_encoder(x) | x:{x.shape}")

        if self.br_use:
            skips = cnn_skips + hyb_skips[:-1]
            r_skips = self.bridge(*[skips[i] for i in self.br_skip_levels])

            _skips = []
            for i in range(len(skips)):
                _skips.append(r_skips[i] if i in self.br_skip_levels else skips[i])

            cnn_skips = _skips[: len(cnn_skips)]
            hyb_skips[:-1] = _skips[len(cnn_skips) :]

        if self.do_ds:
            x, hyb_outs = self.hyb_decoder(x, hyb_skips[:-1], return_outs=True)
            x, cnn_outs = self.cnn_decoder(x, cnn_skips, return_outs=True)
        else:
            x = self.hyb_decoder(x, hyb_skips[:-1])
            # print(f"after x = self.hyb_decoder(x, hyb_skips) | x:{x.shape}")
            # x, cnn_skips = self.cnn_context_bridge(x, cnn_skips)
            x = self.cnn_decoder(x, cnn_skips)
            # print(f"after x = self.cnn_decoder(x, cnn_skips) | x:{x.shape}")

        x = torch.concatenate([x, r], dim=1)
        x = self.out(x)

        if self.do_ds:
            return x, hyb_outs + cnn_outs

        return x
