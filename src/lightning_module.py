import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from metrics import get_binary_metrics
from utils import make_grids
import numpy as np
import nibabel as nib
from monai.losses import DiceCELoss
from monai.inferers import SlidingWindowInferer
from optimizers import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Union, Tuple, Dict


class SemanticSegmentation3D(pl.LightningModule):
    def __init__(self, config: dict, model=None):
        super(SemanticSegmentation3D, self).__init__()
        self.config = config
        self.model = model
        lambda_dice = config["training"]["criterion"]["params"].get("dice_weight", 0.5)
        lambda_ce = config["training"]["criterion"]["params"].get("bce_weight", 0.5)
        self.global_loss_weight = config["training"]["criterion"]["params"].get(
            "global_weight", 1.0
        )
        self.criterion_dice_ce = DiceCELoss(
            sigmoid=False,
            softmax=True,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
            to_onehot_y=True,
        ).to(self.device)

        ## Initialize metrics for each type and mode
        modes = ["tr", "vl", "te"]
        self.modes_dict = {"tr": "train", "vl": "val", "te": "test"}
        
        self.datatype = (config['dataset']['name'].split('_')[0]).lower()
        
        self.types = ["wt", "tc", "et"] if self.datatype != 'acdc' else ['rv', 'myo', 'lv']

        self.metrics = {}
        for mode in modes:
            self.metrics[mode] = {}
            for type_ in self.types:
                metric_name = f"metrics_{self.modes_dict[mode]}_{type_}"
                self.metrics[mode][type_] = (
                    get_binary_metrics(mode=mode)
                    .clone(prefix=f"{metric_name}/")
                    .to(self.device)
                )

        self.lr = self.config["training"]["optimizer"]["params"]["lr"]
        self.log_pictures = config["checkpoints"]["log_pictures"]
        self.save_nifty = config["checkpoints"].get("save_nifty", True)
        self.test_batch_size = config["data_loader"]["test"]["batch_size"]
        if self.test_batch_size != config["data_loader"]["validation"]["batch_size"]:
            raise ValueError("Test batch size must be equal to validation batch size")

        self.use_sliding_window = config.get("use_sliding_window", False)
        if self.use_sliding_window:
            roi_size = config["dataset"]["input_size"]
            # need to chaneg the location to channel first
            roi_size = [roi_size[2], roi_size[0], roi_size[1]]
            self.slider = SlidingWindowInferer(
                roi_size=roi_size,
                sw_batch_size=config["data_loader"]["train"]["batch_size"],
                **config["sliding_window_params"],
            )
        else:
            print(
                "NOT USING SLIDING WINDOW !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )

        self.model_name = config["model"]["name"].split("_")[0]
        self.deep_supervision = False
        if hasattr(self.model, "do_ds"):
            self.deep_supervision = self.model.do_ds
        self.save_hyperparameters(config)

    def forward(self, x):
        return self.model(x)

    def _extract_data(self, batch: dict) -> (torch.Tensor, torch.Tensor):
        imgs = batch["volume"].float()
        msks = batch["seg-volume"].type(torch.uint8)
        return imgs, msks

    def on_epoch_end(self, stage: str):
        for type_ in self.types:
            metric = self.metrics[stage][type_].compute()
            self.log_dict({f"{k}": v for k, v in metric.items()})
            self.metrics[stage][type_].reset()

    def on_train_epoch_start(self) -> None:
        if self.use_sliding_window:
            if self.deep_supervision:
                self.model.do_ds = True

    def on_validation_epoch_start(self) -> None:
        if self.use_sliding_window:
            if self.deep_supervision:
                self.model.do_ds = False

    def on_test_epoch_start(self) -> None:
        if self.use_sliding_window:
            if self.deep_supervision:
                self.model.do_ds = False

    def _shared_step(self, batch, stage: str, batch_idx=None) -> torch.Tensor:
        imgs, gts = self._extract_data(batch)
        preds = self._forward_pass(imgs, stage)

        loss, preds = self._calculate_losses(preds, gts)

        self._update_metrics(preds, gts, stage)
        self._log_losses(loss, stage)

        if stage == "te":
            self._save_nifty_or_picture(batch, imgs, preds, gts, batch_idx)

        return loss

    # batch have values of size (batch ,modalities, channels, height, width)
    def training_step(self, batch, batch_idx):
        return {"loss": self._shared_step(batch, "tr", batch_idx)}

    def on_train_epoch_end(self) -> None:
        self.on_epoch_end("tr")

    def validation_step(self, batch, batch_idx):
        return {"val_loss": self._shared_step(batch, "vl", batch_idx)}

    def on_validation_epoch_end(self) -> None:
        self.on_epoch_end("vl")

    def test_step(self, batch, batch_idx):
        return {"test_loss": self._shared_step(batch, "te", batch_idx)}

    def on_test_epoch_end(self) -> None:
        self.on_epoch_end("te")

    def configure_optimizers(self):
        optimizer_cls = getattr(
            torch.optim, self.config["training"]["optimizer"]["name"]
        )
        del self.config["training"]["optimizer"]["params"]["lr"]
        optimizer = optimizer_cls(
            self.model.parameters(),
            lr=self.lr,
            **self.config["training"]["optimizer"]["params"],
        )
        scheduler_cls = globals().get(
            self.config["training"]["scheduler"]["name"], None
        )
        if scheduler_cls is None:
            scheduler = None
        else:
            scheduler = scheduler_cls(
                optimizer, **self.config["training"]["scheduler"]["params"]
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
                "name": "lr_scheduler",
            },
        }

    def _calculate_losses(
        self,
        preds: Union[torch.Tensor, list, tuple],
        gts: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        if isinstance(preds, (list, tuple)):  # TODO: for the supervision case
            # just doing it for the SegResNetVAE case, where the output is a list of 2 elements
            weights = self._cal_loss_weights(preds)
            loss_dict = self._cal_loss_for_supervision(preds, gts, weights)
            preds = preds[0]
        else:
            loss_dict = self._cal_losses(preds, gts)

        return loss_dict, preds

    def _cal_losses(
        self,
        preds: torch.Tensor,
        gts: torch.Tensor,
    ) -> dict:
        loss = self._cal_global_loss(preds, gts)

        return loss

    def _cal_loss_weights(self, preds: list | tuple) -> list:
        if self.model_name == "SegResNetVAE3D":
            weights = [1.0, 0.1]
        else:
            weights = np.array([1 / (2**i) for i in range(len(preds))])
            if self.model_name == "TransUnet3D":
                mask = np.array(
                    [True]
                    + [
                        True if i < len(preds) - 1 else False
                        for i in range(1, len(preds))
                    ]
                )
                weights[~mask] = 0
            weights = weights / weights.sum()
        return weights

    def _cal_global_loss(self, preds: torch.Tensor, gts: torch.Tensor) -> torch.Tensor:
        loss = self.criterion_dice_ce(
            preds.permute(0, 1, 3, 4, 2).float(), gts.permute(0, 1, 3, 4, 2).float()
        )
        return loss

    def _log_losses(self, losses: torch.Tensor, stage: str) -> None:
        self.log(
            f"{self.modes_dict[stage]}_loss",
            losses,
            on_epoch=True,
            prog_bar=True,
        )

    def _update_metrics(
        self, preds: torch.Tensor, gts: torch.Tensor, stage: str
    ) -> None:
        metrics = self.metrics[stage]
        if self.datatype == 'acdc':
            preds_list, gts_list = self._contstruct_rv_myo_lv(preds, gts)
        else:
            preds_list, gts_list = self._contstruct_wt_tc_et(preds, gts)
        for index, type_ in enumerate(self.types):
            pred = preds_list[index].float()
            gt = gts_list[index].type(torch.uint8)
            metrics[type_].to(self.device)
            metrics[type_].update(pred, gt)

    def _save_nifty_or_picture(self, batch, imgs, preds, gts, batch_idx):
        save_dir = self.logger.log_dir
        if self.log_pictures or self.save_nifty:
            for patient_idx in range(imgs.shape[0]):
                if self.save_nifty:
                    self.save_nifty_from_logits_preds(
                        batch["patient_name"][patient_idx],
                        batch["affinity"][patient_idx],
                        preds[patient_idx],
                        gts[patient_idx],
                        save_dir,
                    )
                if self.log_pictures:
                    self._log_pictures(
                        imgs[patient_idx : patient_idx + 1],
                        preds[patient_idx : patient_idx + 1],
                        gts[patient_idx : patient_idx + 1],
                        batch_idx,
                        patient_idx,
                    )

    def save_nifty_from_logits_preds(
        self,
        name: str,
        affinity: torch.Tensor,
        preds: torch.Tensor,
        masks: torch.Tensor,
        save_dir: str,
    ) -> None:
        """
        inputs are 4D tensors (channel,depth,height,width)
        """
        affinity = affinity.detach().cpu().numpy()
        preds = preds.permute(0, 2, 3, 1)
        masks = masks.permute(0, 2, 3, 1).squeeze().type(torch.uint8)
        preds_labels = (
            torch.argmax(F.softmax(preds, dim=0), dim=0).squeeze().type(torch.uint8)
        )
        mask_img = nib.Nifti1Image(masks.detach().cpu().numpy(), affine=affinity)
        preds_img = nib.Nifti1Image(
            preds_labels.detach().cpu().numpy(), affine=affinity
        )
        name_dir = os.path.join(save_dir, "nifty predictions", name)
        os.makedirs(name_dir, exist_ok=True)
        nib.save(mask_img, os.path.join(name_dir, f"{name}-seg.nii.gz"))
        nib.save(preds_img, os.path.join(name_dir, f"{name}-pred.nii.gz"))

    def _contstruct_rv_myo_lv(
        self, preds: torch.Tensor, gts: torch.Tensor
    ) -> (list, list):
        preds_list = []
        gts_list = []
        preds_labels = torch.argmax(F.softmax(preds, dim=1), dim=1).unsqueeze(1)
        for i, type in enumerate(["rv", "myo", "lv"]):
            preds_list.append((preds_labels == i+1).type(torch.uint8))
            gts_list.append((gts == i+1).type(torch.uint8))
        return preds_list, gts_list


    def _contstruct_wt_tc_et(
        self, preds: torch.Tensor, gts: torch.Tensor
    ) -> (list, list):
        preds_list = []
        gts_list = []
        preds_labels = torch.argmax(F.softmax(preds, dim=1), dim=1).unsqueeze(1)
        for type in ["wt", "tc", "et"]:
            constructor = getattr(self, f"_construct_{type}")
            preds_list.append(constructor(preds_labels))
            gts_list.append(constructor(gts))
        return preds_list, gts_list

    def _construct_et(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor == 3).type(torch.uint8)

    def _construct_tc(self, tensor: torch.Tensor) -> torch.Tensor:
        return ((tensor == 3) | (tensor == 1)).type(torch.uint8)

    def _construct_wt(self, tensor: torch.Tensor) -> torch.Tensor:
        return (((tensor == 3) | (tensor == 1)) | (tensor == 2)).type(torch.uint8)

    def _log_pictures(
        self,
        imgs,
        preds,
        gts,
        batch_idx,
        patient_idx,
    ):
        grid_wt, grid_tc, grid_et = make_grids(
            imgs,
            gts[:, 0:1],
            preds[:, 0:1],
            gts[:, 1:2],
            preds[:, 1:2],
            gts[:, 2:3],
            preds[:, 2:3],
        )
        self.logger.experiment.add_image(
            "pictures_wt",
            grid_wt,
            batch_idx * self.test_batch_size + patient_idx,
        )
        self.logger.experiment.add_image(
            "pictures_tc",
            grid_tc,
            batch_idx * self.test_batch_size + patient_idx,
        )
        self.logger.experiment.add_image(
            "pictures_et",
            grid_et,
            batch_idx * self.test_batch_size + patient_idx,
        )

    def _cal_loss_for_supervision(
        self,
        preds: list[torch.Tensor],
        gts: torch.Tensor,
        dilated_masks: torch.Tensor | None,
        instance_gts: torch.Tensor | None,
        weights: np.ndarray,
    ):
        assert len(preds) == len(weights), "preds and weights must have the same length"

        if self.model_name == "SegResNetVAE3D":
            return self._handle_seg_res_net_vae(
                preds, gts, dilated_masks, instance_gts, weights
            )

        gts = gts.type(torch.uint8)
        losses = {
            "total_global_loss": torch.tensor([0.0]).to(self.device),
            "global_loss_wt": None,
            "global_loss_tc": None,
            "global_loss_et": None,
            "total_blob_loss": torch.tensor([0.0]).to(self.device),
            "blob_loss_wt": None,
            "blob_loss_tc": None,
            "blob_loss_et": None,
            "total_loss": None,
        }

        for i, pred in enumerate(preds):
            losses = self._handle_supervision_loss_update(
                pred, gts, dilated_masks, instance_gts, weights[i], losses
            )
        return losses

    def _handle_seg_res_net_vae(self, preds, gts, dilated_masks, instance_gts, weights):
        """
        Handle loss calculation for SegResNetVAE3D model.

        This is a separate method for special handling of the "SegResNetVAE3D" model.
        """
        losses = self._cal_losses(preds[0], gts, dilated_masks, instance_gts)
        losses["total_loss"] = losses["total_loss"] * weights[0] + preds[1] * weights[1]
        return losses

    def _handle_supervision_loss_update(
        self, pred, gts, dilated_masks, instance_gts, weight, losses
    ):
        """
        Update the losses dictionary with losses from a single prediction.
        """
        gt_downsampled = F.interpolate(gts, size=pred.shape[2:], mode="nearest-exact")

        # Only downsample if blob loss weight is not zero
        if self.blob_loss_weight != 0.0:
            dilated_downsampled = F.interpolate(
                dilated_masks, size=pred.shape[2:], mode="nearest-exact"
            )
            instance_downsampled = F.interpolate(
                instance_gts, size=pred.shape[2:], mode="nearest-exact"
            )
        else:
            dilated_downsampled = None
            instance_downsampled = None

        local_loss_dict = self._cal_losses(
            pred, gt_downsampled, dilated_downsampled, instance_downsampled
        )

        for key, value in local_loss_dict.items():
            if value is not None:
                losses[key] = (
                    (losses[key] + value * weight)
                    if losses[key] is not None
                    else value * weight
                )

        return losses

    def _forward_pass(self, imgs: torch.Tensor, stage: str) -> torch.Tensor | list:
        if self.use_sliding_window:
            if stage == "tr":
                preds = self(imgs)
            else:
                preds = self.slider(imgs, self.model)
        else:
            preds = self(imgs)
        return preds
