class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


# from termcolor import colored
import torch
import torch.nn.functional as F
import torchvision
import os
from monai.transforms import (
    ResizeWithPadOrCropd,
)
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
import glob
from concurrent.futures import ProcessPoolExecutor
import platform
from tqdm import tqdm
import h5py
import pandas as pd
from joblib import Parallel, delayed

resize_transform = ResizeWithPadOrCropd(
    keys=["preds_wt", "preds_tc", "preds_et", "mask_wt", "mask_tc", "mask_et"],
    spatial_size=[240, 240, 155],
    mode="constant",
    constant_values=0,
    value=0,
    allow_missing_keys=True,
)


def _print(string, p=None):
    if not p:
        print(string)
        return
    pre = f"{bcolors.ENDC}"

    if "bold" in p.lower():
        pre += bcolors.BOLD
    elif "underline" in p.lower():
        pre += bcolors.UNDERLINE
    elif "header" in p.lower():
        pre += bcolors.HEADER

    if "warning" in p.lower():
        pre += bcolors.WARNING
    elif "error" in p.lower():
        pre += bcolors.FAIL
    elif "ok" in p.lower():
        pre += bcolors.OKGREEN
    elif "info" in p.lower():
        if "blue" in p.lower():
            pre += bcolors.OKBLUE
        else:
            pre += bcolors.OKCYAN

    print(f"{pre}{string}{bcolors.ENDC}")


import yaml


def load_config(config_filepath):
    try:
        with open(config_filepath, "r") as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError:
        _print(f"Config file not found! <{config_filepath}>", "error_bold")
        exit(1)


import json


def print_config(config, logger=None):
    conf_str = json.dumps(config, indent=2)
    if logger:
        logger.info(f"\n{' Config '.join(2*[10*'>>',])}\n{conf_str}\n{28*'>>'}")
    else:
        _print("Config:", "info_underline")
        print(conf_str)
        print(30 * "~-", "\n")


def show_sbs(im1, im2, figsize=[8, 4], im1_title="Image", im2_title="Mask", show=True):
    if im1.shape[0] < 4:
        im1 = np.array(im1)
        im1 = np.transpose(im1, [1, 2, 0])

    if im2.shape[0] < 4:
        im2 = np.array(im2)
        im2 = np.transpose(im2, [1, 2, 0])

    _, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].imshow(im1)
    axs[0].set_title(im1_title)
    axs[1].imshow(im2, cmap="gray")
    axs[1].set_title(im2_title)
    if show:
        plt.show()


def overlay(img, mask, pred):
    overlay = img.clone()
    # seg in green
    overlay[:, 1, :, :].masked_fill_(mask[:, 1, :, :] == 1, 1)
    # pres in red
    overlay[:, 0, :, :].masked_fill_(pred[:, 0, :, :] == 1, 1)
    return overlay


def convert_to_rgb_and_stack(tensor):
    rgb_list = [tensor.repeat(1, 3, 1, 1) for tensor in torch.split(tensor, 1, dim=1)]
    return torch.cat(rgb_list, dim=2)


def prepare_tensor(tensor, sigmoid_threshold=None):
    tensor = tensor[:, 0, :, :, :]
    if sigmoid_threshold:
        tensor = F.sigmoid(tensor) > sigmoid_threshold
    return tensor.type(torch.int8)


def make_grids(imgs, *masks_preds):
    imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())

    modalities_stacked = [
        convert_to_rgb_and_stack(imgs[:, i, :, :, :]) for i in range(4)
    ]

    masks_preds_stacked = [
        convert_to_rgb_and_stack(prepare_tensor(tensor, 0.5 if i % 2 else None))
        for i, tensor in enumerate(masks_preds)
    ]

    overlays = [
        overlay(
            modalities_stacked[0], masks_preds_stacked[i], masks_preds_stacked[i + 1]
        )
        for i in range(0, len(masks_preds_stacked), 2)
    ]

    concat_list = [
        torch.cat(
            (
                *modalities_stacked,
                masks_preds_stacked[i],
                masks_preds_stacked[i + 1],
                overlays[i // 2],
            ),
            dim=3,
        )
        for i in range(0, len(masks_preds_stacked), 2)
    ]

    grids = [torchvision.utils.make_grid(concat, nrow=1) for concat in concat_list]

    return tuple(grids)


def save_nifty(
    name,
    preds,
    masks,
    save_dir,
    device="cpu",
):
    """
    inputs are 4D tensors (channel,depth,height,width)
    """
    preds = preds.permute(0, 2, 3, 1)
    masks = masks.permute(0, 2, 3, 1)

    pred_mask_3d = {
        "preds_wt": (F.sigmoid(preds[0:1]) > 0.5),
        "preds_tc": (F.sigmoid(preds[1:2]) > 0.5),
        "preds_et": (F.sigmoid(preds[2:3]) > 0.5),
        "mask_wt": (masks[0:1]),
        "mask_tc": (masks[1:2]),
        "mask_et": (masks[2:3]),
    }
    transformed = resize_transform(pred_mask_3d)
    transformed = pred_mask_3d
    mask_multilabel = get_multilable(transformed, device=device)
    preds_multilabel = get_multilable(transformed, is_pred=True, device=device)
    mask_img = nib.Nifti1Image(mask_multilabel.detach().cpu().numpy(), affine=np.eye(4))
    preds_img = nib.Nifti1Image(
        preds_multilabel.detach().cpu().numpy(), affine=np.eye(4)
    )
    name_dir = os.path.join(save_dir, "nifty predictions", name)
    os.makedirs(name_dir, exist_ok=True)
    nib.save(mask_img, os.path.join(name_dir, f"{name}-seg.nii.gz"))
    nib.save(preds_img, os.path.join(name_dir, f"{name}-pred.nii.gz"))
    save_with_different_thresholds(preds, name, name_dir, threshold=0.5, device=device)
    save_with_different_thresholds(preds, name, name_dir, threshold=0.4, device=device)


def get_multilable(transformed, is_pred=False, device="cpu"):
    key = "preds" if is_pred else "mask"
    multilabel = torch.zeros(transformed["preds_wt"].shape[-3:], dtype=torch.uint8).to(
        device
    )
    multilabel[transformed[f"{key}_et"][0] == 1] = 3
    multilabel[(transformed[f"{key}_tc"][0] == 1) & (multilabel != 3)] = 1
    multilabel[
        (transformed[f"{key}_wt"][0] == 1) & (multilabel != 3) & (multilabel != 1)
    ] = 2
    return multilabel


def save_with_different_thresholds(preds, name, name_dir, threshold=0.4, device="cpu"):
    for i, _type in enumerate(["wt", "tc", "et"]):
        preds_multilabel = (F.sigmoid(preds[i : i + 1]) > threshold).to(device)
        preds_multilabel = resize_transform({f"preds_{_type}": preds_multilabel})[
            f"preds_{_type}"
        ].type(torch.uint8)
        preds_img = nib.Nifti1Image(
            preds_multilabel.squeeze().detach().cpu().numpy(), affine=np.eye(4)
        )
        nib.save(
            preds_img, os.path.join(name_dir, f"{name}-pred-{_type}-{threshold}.nii.gz")
        )
