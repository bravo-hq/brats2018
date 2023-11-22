cd ~/deeplearning/Metastasis-segmentation/src

python train_test.py -c met_unetr3d_adamw
python train_test.py -c met_swinunetr3d_adamw
python train_test.py -c met_swinunetr3d-v2_adamw

