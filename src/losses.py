import torch
from torch.nn import functional as F
from monai.losses import DiceLoss, HausdorffDTLoss, MaskedDiceLoss, DiceCELoss


EPSILON = 1e-6


# class DiceLoss(torch.nn.Module):
#     def __init__(
#         self,
#     ):
#         super().__init__()

#     def forward(self, pred, mask):
#         pred = pred.flatten()
#         mask = mask.flatten()

#         intersect = (mask * pred).sum()
#         dice_score = 2 * intersect / (pred.sum() + mask.sum() + EPSILON)
#         dice_loss = 1 - dice_score
#         return dice_loss


class DiceLossWithLogtis(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, pred, mask):
        if not torch.is_tensor(pred):
            raise TypeError(
                "Input type is not a torch.Tensor. Got {}".format(type(pred))
            )
        if not len(pred.shape) == 5:
            raise ValueError(
                "Invalid input shape, we expect BxMxCxHxW. Got: {}".format(pred.shape)
            )
        if not pred.shape[-2:] == mask.shape[-2:]:
            raise ValueError(
                "input and target shapes must be the same. Got: {}".format(
                    pred.shape, mask.shape
                )
            )
        if not pred.device == mask.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}".format(
                    pred.device, mask.device
                )
            )

        prob = F.sigmoid(pred)
        true_1_hot = mask.type(prob.type())

        dims = (0,) + tuple(range(2, true_1_hot.ndimension()))
        intersection = torch.sum(prob * true_1_hot, dims)
        cardinality = torch.sum(prob + true_1_hot, dims)
        dice_loss = (2.0 * intersection / (cardinality + EPSILON)).mean()
        return 1.0 - dice_loss


# class BlobLossWithLogits(torch.nn.Module):
#     def __init__(self, dice=True, hausdorff=True):
#         super().__init__()
#         self.dice = DiceLoss(sigmoid=True) if dice else None
#         self.hd = HausdorffDTLoss(sigmoid=True) if hausdorff else None

#     def forward(self, pred, mask, blob_masks):
#         assert pred.shape == mask.shape
#         assert pred.dim() == 5
#         assert pred.device == mask.device
#         dice_total = []
#         hd_total = []
#         for batch_idx in range(pred.shape[0]): # on the batch dimension
#             blob_masks_patient = blob_masks[batch_idx : batch_idx + 1]
#             pred_patient = pred[batch_idx : batch_idx + 1]
#             mask_patient = mask[batch_idx : batch_idx + 1]
#             dice_patient = []
#             hd_patient = []
#             for blob_mask in range(blob_masks_patient.shape[1]): # on number of blob instances
#                 pred_blob = pred_patient * blob_masks_patient[:, blob_mask : blob_mask + 1]
#                 if self.dice:
#                     dice_patient.append(self.dice(pred_blob, blob_mask))
#                 if self.hd:
#                     hd_patient.append(self.hd(pred_blob, blob_mask))
#             if self.dice:
#                 dice_total.append(torch.mean(torch.stack(dice_patient)))
#             if self.hd:
#                 hd_total.append(torch.mean(torch.stack(hd_patient)))
#         if self.dice and self.hd:
#             return (torch.mean(torch.stack(dice_total)) + torch.mean(torch.stack(hd_total)))/2
#         elif self.dice:
#             return torch.mean(torch.stack(dice_total))
#         elif self.hd:
#             return torch.mean(torch.stack(hd_total))


class BlobLossWithLogits(torch.nn.Module):
    def __init__(self, dice=True, hausdorff=True, device="cuda", sigmoid=True):
        super().__init__()
        self.device = device
        self.losses = {
            "dice": DiceCELoss(sigmoid=sigmoid, lambda_dice=0.5, lambda_ce=0.5).to(
                self.device
            )
            if dice
            else None,
            # "hausdorff": HausdorffDTLoss(sigmoid=True).to(self.device) if hausdorff else None,
        }
        if not any(self.losses.values()):
            raise ValueError("At least one loss must be True")
        for key, loss_fn in self.losses.items():
            if loss_fn:
                print(f"Using {key}_blob loss")

    def forward(
        self, pred: torch.Tensor, instance_gts: torch.Tensor, blob_masks: torch.Tensor
    ):
        assert (
            pred.device == blob_masks.device
        ), "All tensors must be on the same device"
        assert (
            pred.device == instance_gts.device
        ), "All tensors must be on the same device"
        assert (
            pred.dim() == blob_masks.dim() and pred.dim() == 5
        ), "Shape or dimension mismatch"
        assert pred.dim() == instance_gts.dim(), "Shape mismatch"
        assert instance_gts.shape == blob_masks.shape, "Shape mismatch"
        assert (
            pred.shape[0] == blob_masks.shape[0] == instance_gts.shape[0]
        ), "Batch size mismatch"
        assert (
            blob_masks.dtype == torch.uint8 and instance_gts.dtype == torch.uint8
        ), "Blob masks must be uint8 tensors"

        total_losses = {
            key: [] for key in self.losses.keys() if self.losses[key] is not None
        }

        for batch_idx in range(pred.shape[0]):  # on the batch dimension
            blob_masks_patient, pred_patient, gts_patient = [
                tensor[batch_idx : batch_idx + 1]
                for tensor in [blob_masks, pred, instance_gts]
            ]

            # assert (
            #     blob_masks_patient.max() == gts_patient.max()
            # ), "Blob masks and instance gts must have the same number of instances"

            patient_losses = {
                key: [] for key in self.losses.keys() if self.losses[key] is not None
            }
            for unique_label in torch.unique(
                gts_patient
            ):  # on number of blob instances
                if unique_label == 0:
                    continue
                blob_temp = (blob_masks_patient == unique_label).type(torch.bool)
                gt_temp = (gts_patient == unique_label).float()
                for key, loss_fn in self.losses.items():
                    if loss_fn:
                        patient_losses[key].append(
                            loss_fn(pred_patient * blob_temp, gt_temp * blob_temp)
                        )
            for key, val in patient_losses.items():
                if val:  # only when all instance are not in the slices selected
                    total_losses[key].append(torch.stack(val).mean())
                else:  # temporary fix
                    for key, loss_fn in self.losses.items():
                        if loss_fn:
                            temp_loss = loss_fn(pred_patient, gts_patient.float())
                            total_losses[key].append(temp_loss)

        total_losses_mean = [
            torch.stack(val).mean() for _, val in total_losses.items() if val
        ]
        # if len(total_losses_mean) == 0:
        #     return None
        return torch.stack(total_losses_mean).sum() / len(total_losses_mean)
