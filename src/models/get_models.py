from models.dim2._unet.unet import get_unet
from models.dim2._multiresunet.multiresunet import get_multiresunet
from models.dim3.unets.model import UNet3D, ResidualUNet3D, ResidualUNetSE3D
import platform
from monai.networks.nets import SwinUNETR, UNETR, SegResNetVAE
import torch


class SegResNetVAEModified(SegResNetVAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x, loss = super().forward(x)
        if self.training:
            return x, loss
        else:
            return x


def get_transunet(config):
    from models.dim2._transunet.vit_seg_modeling_c4 import (
        VisionTransformer as TransUnet,
    )
    from models.dim2._transunet.vit_seg_modeling_c4 import CONFIGS as CONFIGS_ViT_seg

    config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
    config_vit.n_classes = 3
    config_vit.n_skip = 3
    if "R50-ViT-B_16".find("R50") != -1:
        config_vit.patches.grid = (
            int(config["dataset"]["input_size"][0] / 16),
            int(config["dataset"]["input_size"][1] / 16),
        )
    model = TransUnet(config_vit, **config["model"]["params"])

    torch.cuda.empty_cache()
    return model


def get_missformer(config):
    from models.dim2._missformer.MISSFormer import MISSFormer

    return MISSFormer(**config["model"]["params"])


def get_resunet(config):
    from models.dim2._resunet.res_unet import ResUnet

    return ResUnet(**config["model"]["params"])


def get_uctransnet(config):
    from models.dim2._uctransnet.UCTransNet import UCTransNet
    import models.dim2._uctransnet.Config as uct_config

    config_vit = uct_config.get_CTranS_config()
    return UCTransNet(config_vit, **config["model"]["params"])


def get_attunet(config):
    from models.dim2._attunet.attunet import AttU_Net as AttUnet

    return AttUnet(**config["model"]["params"])


def get_unet3d(config):
    return UNet3D(**config["model"]["params"])


def get_resunet3d(config):
    return ResidualUNet3D(**config["model"]["params"])


def get_resunetse3d(config):
    return ResidualUNetSE3D(**config["model"]["params"])


def get_transunet3d(config):
    from models.dim3.transunet3d.transunet3d_model import (
        Generic_TransUNet_max_ppbp as TransUnet3D,
    )

    return TransUnet3D(**config["model"]["params"])


def get_main_model(config):
    from models.dim3.main_model.models.dLKA import Model as MainModel_per
    from models.dim3.main_model.models.main import Model_Base as MainModel

    return MainModel(**config["model"]["params"])


def get_lhunet_model(config):
    from models.dim3.lhunet.models.v4 import LHUNet as model

    return model(**config["model"]["params"])


def get_main_bridge_model(config):
    from models.dim3.main_model.models.main import Model_Bridge

    return Model_Bridge(**config["model"]["params"])


def get_swinunetr(config):
    return SwinUNETR(**config["model"]["params"])


def get_unetr(config):
    return UNETR(**config["model"]["params"])


def get_segresnetvae(config):
    return SegResNetVAEModified(**config["model"]["params"])


def get_nnformer(config):
    from models.dim3.nnformer.nnFormer_tumor import nnFormer

    return nnFormer(**config["model"]["params"])


def get_unetrpp(config):
    from models.dim3.untrpp.tumor.unetr_pp_tumor import UNETR_PP as UNETRPP

    return UNETRPP(**config["model"]["params"])


def d_lka_net_synapse(config):
    from models.dim3.d_lka_former.d_lka_net_synapse import D_LKA_Net
    from models.dim3.d_lka_former.transformerblock import (
        TransformerBlock_3D_single_deform_LKA,
        TransformerBlock,
    )

    return D_LKA_Net(
        trans_block=TransformerBlock_3D_single_deform_LKA, **config["model"]["params"]
    )


def get_vnet(config):
    from models.dim3.vnet.vnet import VNet

    return VNet(**config["model"]["params"])


MODEL_FACTORY = {
    "unet": get_unet,
    "transunet": get_transunet,
    "missformer": get_missformer,
    "multiresunet": get_multiresunet,
    "resunet": get_resunet,
    "uctransnet": get_uctransnet,
    "attunet": get_attunet,
    "unet3d": get_unet3d,
    "resunet3d": get_resunet3d,
    "resunetse3d": get_resunetse3d,
    "mainmodel": get_main_model,
    "mainmodel-bridge": get_main_bridge_model,
    "transunet3d": get_transunet3d,
    "swinunetr": get_swinunetr,
    "swinunetr3d": get_swinunetr,
    "swinunetr3d-v2": get_swinunetr,
    "unetr": get_unetr,
    "unetr3d": get_unetr,
    "segresnetvae3d": get_segresnetvae,
    "nnformer3d": get_nnformer,
    "unetrpp3d": get_unetrpp,
    "dlk-former": d_lka_net_synapse,
    "vnet": get_vnet,
    "lhunet": get_lhunet_model,
}


def get_model(config):
    model_name = (
        config["model"]["name"].lower().split("_")[0]
    )  # Get the base name (e.g., unet from unet_variant1)
    if model_name in MODEL_FACTORY:
        return MODEL_FACTORY[model_name](config)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented.")
