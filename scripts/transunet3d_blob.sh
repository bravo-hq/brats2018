cd ~/deeplearning/Metastasis-segmentation/src
python train_test.py -c met_transunet3d_sgd_blob
python train_test.py -c met_transunet3d_adam_blob
python train_test.py -c met_transunet3d_adamw_blob
