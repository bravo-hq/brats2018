#!/usr/bin/env python3
# encoding: utf-8
# Code modified from https://github.com/Wangyixinxin/ACN
import glob
import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import random
import monai.transforms as T


class ACDC(Dataset):
    def __init__(
        self,
        data_root: str,
        crop_size: tuple or list = [160, 160, 14],
        normalization: bool = True,
        augmentation: bool = True,
        mode: str = "tr",
        vl_split: float = 0.1,
        te_split: float = 0.2,
        p: float = 0.5,
    ):  # , train=True,):
        super(ACDC, self).__init__()
        self.data_root = data_root
        # self.train = train
        self.crop_size = crop_size
        self.normalization = normalization
        self.vl_split = vl_split
        self.te_split = te_split
        self.mode = mode
        self.test_split = te_split
        self.val_split = vl_split
        self.p = p

        self.img_plus_seg_dirs = []
        patients_dir = glob.glob(os.path.join(data_root, "patient*"))
        patients_dir.sort()
        self.split_dataset(patients_dir)

        for p in self.patients_list:
            data_files_train = [
                f
                for f in glob.glob(os.path.join(p, "*.nii.gz"))
                if "_gt" not in f and "_4d" not in f
            ]
            corresponding_seg_files = [f[:-7] + "_gt.nii.gz" for f in data_files_train]
            for d, s in zip(data_files_train, corresponding_seg_files):
                self.img_plus_seg_dirs.append((d, s))

        self.define_transforms()

    def __len__(self):
        return len(self.img_plus_seg_dirs)

    def __getitem__(self, index):
        data_dict = {
            "volume": 0,
            "seg-volume": 0,
            "patient_name": os.path.basename(self.img_plus_seg_dirs[index][0])[:-7],
            "affinity": 0,
        }

        img_seg_pair = self.img_plus_seg_dirs[index]
        volumes, seg_volume, data_dict["affinity"] = self._load_data(img_seg_pair)

        volumes = np.stack(volumes, axis=0).astype(np.float32)  # [C, H, W, D]
        seg_volume = np.expand_dims(seg_volume, axis=0).astype(np.uint8)  # [1, H, W, D]
        
        # correct the spacing
        data_dict = self.sapcing_transform({"volume": volumes, "seg-volume": seg_volume})
        volumes = data_dict["volume"]
        seg_volume = data_dict["seg-volume"]

        if self.mode == "tr":
            volumes, seg_volume = self._aug_sample(volumes, seg_volume)

        data_dict["volume"] = (
            torch.as_tensor(volumes).permute(0, 3, 1, 2).float()
        )  # [C, D, H, W]
        data_dict["seg-volume"] = (
            torch.as_tensor(seg_volume).permute(0, 3, 1, 2).type(torch.uint8)
        )  # [1, D, H, W]

        return data_dict

    def define_transforms(self):
        keys = ["volume", "seg-volume"]
        self.pad_or_crop_transform = T.ResizeWithPadOrCropd(
            keys=keys,
            spatial_size=self.crop_size,
            mode="constant",
            constant_values=0,
            value=0,
        )
        self.aug_sample_transform = T.Compose(
            [
                T.RandSpatialCropd(
                    keys=keys,
                    random_center=True,
                    roi_size=self.crop_size,
                ),
                T.SpatialPadd(
                    keys=keys,
                    spatial_size=self.crop_size,
                    mode="constant",
                    value=0,
                    constant_values=0,
                ),  # we have to add this line because the crop size depth might be bigger than the original depth
                T.RandFlipd(
                    keys=keys,
                    prob=self.p,
                    spatial_axis=0,
                ),  # vertical flip
                T.RandFlipd(
                    keys=keys,
                    prob=self.p,
                    spatial_axis=1,
                ),  # horizontal flip
                T.RandFlipd(
                    keys=keys,
                    prob=self.p,
                    spatial_axis=2,
                ),  # depth flip
            ]
        )
        
        self.sapcing_transform = T.Spacingd(keys=keys, pixdim=(1.25, 1.25, 10), mode="nearest")

    def _load_data(self, img_seg_pair):
        volumes = []
        for i, path in enumerate(img_seg_pair):
            nib_file = nib.load(path)
            volume = nib_file.get_fdata()
            affinity = nib_file.affine
            if not i == len(img_seg_pair) - 1 and self.normalization:
                volume = self.normlize(volume)  # [0, 1.0]
            volumes.append(volume)  # [h, w, d]
        return volumes[:-1], volumes[-1], affinity

    def split_dataset(self, patients_dir):
        vl_index = int(len(patients_dir) * self.vl_split)
        te_index = int(len(patients_dir) * self.te_split)
        if self.mode == "tr":
            self.patients_list = patients_dir[te_index + vl_index :]
        elif self.mode == "te":
            self.patients_list = patients_dir[:te_index]
        elif self.mode == "vl":
            self.patients_list = patients_dir[te_index : te_index + vl_index]
        else:
            raise ValueError("mode must be tr, vl or te")

    def _aug_sample(self, volumes, mask):
        data_dict = {"volume": volumes, "seg-volume": mask}
        transformed = self.aug_sample_transform(data_dict)
        return transformed["volume"], transformed["seg-volume"]

    def normlize(self, x):
        if x.max() == x.min():
            return x
        return (x - x.min()) / (x.max() - x.min())


if __name__ == "__main__":
    data_root = "D:\\DATASETS\\ACDC\\ACDC\\database\\training"
    crop_size = [160, 160, 14]
    dataset = ACDC(data_root, crop_size=crop_size, mode="tr")
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, batch in enumerate(dataloader):
        x = batch["volume"]
        y = batch["seg-volume"]
        print(f"patient name: {batch['patient_name']}")
        print(x.shape, y.shape)
        break
