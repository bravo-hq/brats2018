#!/bin/bash
#SBATCH -N 1
#SBATCH -p lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH -o swinunetr3d-v2.out%J
#SBATCH -e swinunetr3d-v2.err%J
#SBATCH -t 3-00:00:00
#SBATCH --mail-user=yousef.sadegheih@ur.de
#SBATCH --mail-type=END,FAIL,BEGIN


python3 --version
nvidia-smi
srun --container-image=/dss/dssfs04/lwp-dss-0002/pn36fu/pn36fu-dss-0000/env/pytorch-env.sqsh --container-mounts=/dss/dssfs04/lwp-dss-0002/pn36fu/pn36fu-dss-0000:/cabinet bash -c "python3 --version && cd && ./env.sh && cd deeplearning/brats2018 && python src/train_test.py -c brats-2021_swinunetr3d-v2_adamw"

