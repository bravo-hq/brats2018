cd /dss/dsshome1/05/di93sip/deeplearning/Metastasis-segmentation/src
python train_test.py -c met_unet3d_sgd
python train_test.py -c met_resunet3d_sgd
python train_test.py -c met_resunetse3d_sgd

