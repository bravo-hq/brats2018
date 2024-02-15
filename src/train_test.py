import os
import time
import glob
import warnings
import pytorch_lightning as pl
import numpy as np
from matplotlib import pyplot as plt
from torchinfo import summary
import yaml
import platform
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.tuner.tuning import Tuner
import argparse

from loader.dataloaders import get_dataloaders
from utils import load_config, print_config
from models.get_models import get_model
from lightning_module import SemanticSegmentation3D

warnings.filterwarnings("ignore")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Your script description here")
    # parser.add_argument(
    #     "-m", "--model_name", type=str, required=True, help="model name"
    # )
    # parser.add_argument("-o", "--optim", type=str, required=True, help="optmizer")
    parser.add_argument("-c", "--config", type=str, required=True, help="config file")
    return parser.parse_args()


def set_seed(seed=0):
    pl.seed_everything(seed)


def get_base_directory():
    return (
        os.getcwd()
        if os.path.basename(os.getcwd()) != "src"
        else os.path.dirname(os.getcwd())
    )


def configure_logger(config, parent_dir):
    if config["LRZ_node"]:
        path = os.path.join(
            f"/cabinet/yousef/{(config['dataset']['name'].split('_')[0])}/",
            "tb_logs",
        )
    else:
        path = os.path.join(parent_dir, "tb_logs")
    return TensorBoardLogger(path, name=config["model"]["name"])


def configure_trainer(config, logger):
    checkpoint_callback = ModelCheckpoint(
        monitor="val_total_dice",
        dirpath=logger.log_dir,
        filename=f"{config['model']['name']}-{{epoch:02d}}-{{val_total_dice:.6f}}",
        save_top_k=3,
        mode="max",
        save_last=True,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.0001, patience=25, verbose=True, mode="min"
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    check_val_every_n_epoch = config.get("check_val_every_n_epoch", 1)
    return Trainer(
        logger=logger,
        # callbacks=[checkpoint_callback, early_stop_callback],
        callbacks=[checkpoint_callback, lr_monitor],
        max_epochs=config["training"]["epochs"],
        check_val_every_n_epoch=check_val_every_n_epoch,
        accelerator="gpu",
        devices=1,
    )


def get_platform():
    lrz_node = False
    if platform.system() == "Linux":
        lrz_node = True
    return lrz_node


def get_input_size_and_module(config):
    summary_input_size = (
        config["data_loader"]["train"]["batch_size"],
        4 if (config["dataset"]["name"].split("_")[0]) != "acdc" else 1,
        config["dataset"]["input_size"][2],
        config["dataset"]["input_size"][0],
        config["dataset"]["input_size"][1],
    )
    return summary_input_size, SemanticSegmentation3D


def main():
    set_seed()

    BASE_DIR = get_base_directory()

    LRZ_NODE = get_platform()
    # debug_mode = True
    # if not debug_mode:
    #     args = parse_arguments()
    #     CONFIG_NAME = f"metastasis_seg/{args.config.split('_')[1]}/{args.config}.yaml"
    # else:
    #     CONFIG_NAME = "metastasis_seg/nnformer3d/met_nnformer3d_sgd.yaml"

    # CONFIG_NAME = (
    #     f"metastasis_seg/{args.model_name}/met_{args.model_name}_{args.optim}.yaml"
    # )

    args = parse_arguments()
    CONFIG_NAME = (
        f"{args.config.split('_')[0]}/{args.config.split('_')[1]}/{args.config}.yaml"
    )
    # CONFIG_NAME = "metastasis_seg/unet/met_unet_adam.yaml"
    CONFIG_FILE_PATH = os.path.join(BASE_DIR, "configs", CONFIG_NAME)

    config = load_config(CONFIG_FILE_PATH)
    print_config(config)

    summary_input_size, lightning_module = get_input_size_and_module(config)

    tr_loader, vl_loader, te_loader = get_dataloaders(config, ["tr", "vl", "te"])

    network = get_model(config)
    # do summary of the model
    summary(
        network,
        input_size=summary_input_size,
        col_names=["input_size", "output_size", "num_params", "mult_adds", "trainable"],
        mode="train",
    )
    config["LRZ_node"] = LRZ_NODE
    logger = configure_logger(config, BASE_DIR)
    trainer = configure_trainer(config, logger)

    just_test = config["checkpoints"]["test_mode"]
    auto_lr_find = config.get("find_lr", False)
    ckpt_path = None
    if config["checkpoints"]["continue_training"]:
        ckpt_path = config["checkpoints"]["ckpt_path"]

    if not just_test:
        model = lightning_module(config, model=network)

        if auto_lr_find:
            Tuner(trainer).lr_find(
                model, train_dataloaders=tr_loader, val_dataloaders=vl_loader
            )
            config["training"]["optimizer"]["params"]["lr"] = model.lr

            # lr_find_files = glob.glob(".lr_find*")
            # if lr_find_files:
            #     os.remove(lr_find_files[0])
            with open(os.path.join(logger.log_dir, "hpram.yaml"), "w") as yaml_file:
                yaml.dump(config, yaml_file)
            if ckpt_path:
                if not os.path.exists(ckpt_path):
                    raise UserWarning(f'Checkpoint path "{ckpt_path}" does not exist!!')
                print(f"Try to resume from {ckpt_path}")

            trainer.fit(
                model,
                train_dataloaders=tr_loader,
                val_dataloaders=te_loader,
                ckpt_path=ckpt_path,
            )
        else:
            os.makedirs(logger.log_dir, exist_ok=True)
            with open(os.path.join(logger.log_dir, "hpram.yaml"), "w") as yaml_file:
                yaml.dump(config, yaml_file)
            trainer.fit(
                model,
                train_dataloaders=tr_loader,
                val_dataloaders=te_loader,
                ckpt_path=ckpt_path,
            )

        print(f"testing {CONFIG_NAME}")
        trainer.test(
            model, dataloaders=te_loader, ckpt_path="best"
        )  # uses the best model to do the test

    else:
        ckpt_path = config["checkpoints"]["ckpt_path"]
        if not os.path.exists(ckpt_path):
            raise UserWarning(f'Checkpoint path "{ckpt_path}" does not exist!!')
        print(f"Try to test from {ckpt_path}")
        try:
            model = lightning_module.load_from_checkpoint(
                checkpoint_path=ckpt_path,
                map_location="cpu",
                config=config,
                model=network,
            )
        except:
            checkpoint = torch.load(ckpt_path)
            network.load_state_dict(checkpoint["MODEL_STATE"], strict=False)
            model = lightning_module(config, model=network)

        print("loaded the checkpoint.")
        trainer.test(model, dataloaders=te_loader)


if __name__ == "__main__":
    main()
