import torch
from torch import nn
from timm.models.layers import trunc_normal_
from typing import Sequence, Tuple, Union
from monai.networks.layers.utils import get_norm_layer
from monai.utils import optional_import

# import sys
# sys.path.append("..")


from .layers import LayerNorm
from .vit.transformers import (
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
from .dynunet_blocks import get_conv_layer, UnetResBlock
from .deform_conv import DeformConvPack
from .dynunet_blocks import get_padding


einops, _ = optional_import("einops")


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


class DLKAFormer_EncoderBlock(BaseBlock):
    def __init__(
        self,
        spatial_dims=3,
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
            self.conv = get_conv_layer(
                3,
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dropout=dropout,
                conv_only=False,
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


class DLKAFormer_DecoderBlock(BaseBlock):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 64,
        out_channels: int = 32,
        skip_channels: int = 0,
        cnn_kernel_size: list[int] | int = 3,
        cnn_dropout: float | int = 0,
        cnn_block: nn.Module | None = UnetResBlock,
        deform_cnn: bool = False,
        tf_input_size: int = 32 * 32 * 32,
        tf_proj_size: int = 64,
        tf_repeats: int = 3,
        tf_num_heads: int = 4,
        tf_dropout: float | int = 0.15,
        trans_block: nn.Module = TransformerBlock,
        tcv_kernel_size: list[int] | int | None = 5,
        tcv_stride: list[int] | int = 2,
        norm_name: tuple | str = "batch",
        use_skip: bool = True,
    ) -> None:
        super().__init__()
        self.use_skip = use_skip
        in_channels += skip_channels

        tf_hidden_size = in_channels if cnn_block or tcv_kernel_size else out_channels
        cnn_out_channels = in_channels if tcv_kernel_size else out_channels
        self.blocks = nn.ModuleList()

        if trans_block:
            self.stages = nn.Sequential(
                *[
                    trans_block(
                        input_size=tf_input_size,
                        hidden_size=tf_hidden_size,
                        proj_size=tf_proj_size,
                        num_heads=tf_num_heads,
                        dropout_rate=tf_dropout,
                        pos_embed=True,
                    )
                    for _ in range(tf_repeats)
                ]
            )
            self.blocks.append(self.stages)

        if cnn_block and cnn_kernel_size:
            if deform_cnn:
                self.cnn = nn.Sequential(
                    DeformConvPack(
                        in_channels,
                        cnn_out_channels,
                        kernel_size=cnn_kernel_size,
                        stride=1,
                        padding=get_padding(cnn_kernel_size, stride=1),
                    ),
                    nn.BatchNorm3d(cnn_out_channels),
                    # nn.LayerNorm(cnn_out_channels),
                    # nn.BatchNorm3d(cnn_out_channels),
                    # nn.BatchNorm3d(out_channels),
                    nn.PReLU(),
                )
            else:
                self.cnn = cnn_block(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=cnn_out_channels,
                    kernel_size=cnn_kernel_size,
                    stride=1,
                    dropout=cnn_dropout,
                    norm_name=norm_name,
                )

            self.blocks.append(self.cnn)

        if tcv_kernel_size:
            self.tcv = get_conv_layer(
                spatial_dims=spatial_dims,
                in_channels=cnn_out_channels,
                out_channels=out_channels,
                kernel_size=tcv_kernel_size,
                stride=tcv_stride,
                conv_only=True,
                is_transposed=True,
            )
            self.blocks.append(self.tcv)
            self.blocks.append(nn.PReLU())
            self.blocks.append(nn.BatchNorm3d(out_channels))

        self.apply(self._init_weights)

    def forward(self, x, skip=None):
        # print(f"input x: {x.shape}, skip: {0 if skip==None else skip.shape}")
        if skip is not None:
            x = torch.concatenate([x, skip], dim=1)
        for block in self.blocks:
            x = block(x)
        # print(f"\t after x: {x.shape}\n")
        return x


class D_LKA_Former_Encoder(nn.Module):
    def __init__(
        self,
        input_size=[32 * 32 * 32, 16 * 16 * 16, 8 * 8 * 8, 4 * 4 * 4],
        dims=[32, 64, 128, 256],
        proj_size=[64, 64, 64, 32],
        depths=[3, 3, 3, 3],
        num_heads=4,
        spatial_dims=3,
        in_channels=1,
        dropout=0.0,
        transformer_dropout_rate=0.15,
        trans_block=TransformerBlock,
        **kwargs
    ):
        super().__init__()

        self.downsample_layers = (
            nn.ModuleList()
        )  # stem and 3 intermediate downsampling conv layers
        stem_layer = nn.Sequential(
            get_conv_layer(
                spatial_dims,
                in_channels,
                dims[0],
                kernel_size=(2, 4, 4),
                stride=(2, 4, 4),
                dropout=dropout,
                conv_only=True,
            ),
            get_norm_layer(
                name=("group", {"num_groups": in_channels}), channels=dims[0]
            ),
        )
        self.downsample_layers.append(stem_layer)
        for i in range(3):
            downsample_layer = nn.Sequential(
                get_conv_layer(
                    spatial_dims,
                    dims[i],
                    dims[i + 1],
                    kernel_size=(2, 2, 2),
                    stride=(2, 2, 2),
                    dropout=dropout,
                    conv_only=True,
                ),
                get_norm_layer(
                    name=("group", {"num_groups": dims[i]}), channels=dims[i + 1]
                ),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = (
            nn.ModuleList()
        )  # 4 feature resolution stages, each consisting of multiple Transformer blocks
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(
                    trans_block(
                        input_size=input_size[i],
                        hidden_size=dims[i],
                        proj_size=proj_size[i],
                        num_heads=num_heads,
                        dropout_rate=transformer_dropout_rate,
                        pos_embed=True,
                    )
                )  ## LEON HERE, change pos_embed back to True
            self.stages.append(nn.Sequential(*stage_blocks))
        self.hidden_states = []
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        hidden_states = []

        x = self.downsample_layers[0](x)
        x = self.stages[0](x)

        hidden_states.append(x)

        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i == 3:  # Reshape the output of the last stage
                x = einops.rearrange(x, "b c h w d -> b (h w d) c")
            hidden_states.append(x)
        return x, hidden_states

    def forward(self, x):
        x, hidden_states = self.forward_features(x)
        return x, hidden_states


class D_LKA_FormerUpBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        proj_size: int = 64,
        num_heads: int = 4,
        out_size: int = 0,
        depth: int = 3,
        conv_decoder: bool = False,
        trans_block=TransformerBlock,
        use_skip=True,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of heads inside each EPA module.
            out_size: spatial size for each decoder.
            depth: number of blocks for the current decoder stage.
        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )
        self.use_skip = use_skip
        print("Using skip connection in decoder: {}".format(use_skip))

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.decoder_block = nn.ModuleList()

        # If this is the last decoder, use ConvBlock(UnetResBlock) instead of D-LKA
        if conv_decoder == True:
            self.decoder_block.append(
                UnetResBlock(
                    spatial_dims,
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    norm_name=norm_name,
                )
            )
        else:
            stage_blocks = []
            for j in range(depth):
                stage_blocks.append(
                    trans_block(
                        input_size=out_size,
                        hidden_size=out_channels,
                        proj_size=proj_size,
                        num_heads=num_heads,
                        dropout_rate=0.15,
                        pos_embed=True,
                    )
                )  ## LEON here, change pos_embed to True
            self.decoder_block.append(nn.Sequential(*stage_blocks))

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inp, skip):
        out = self.transp_conv(inp)
        if self.use_skip == True:
            # print("Adding skip connection")
            out = out + skip
        out = self.decoder_block[0](out)

        return out
