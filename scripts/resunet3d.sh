cd /dss/dsshome1/05/di93sip/deeplearning/Metastasis-segmentation/src
python train_test.py -c met_resunet3d_adam
python train_test.py -c met_resunet3d_adamw
python train_test.py -c met_resunet3d_sgd

python train_test.py -c met_resunet3d_adam_blob
python train_test.py -c met_resunet3d_adamw_blob
python train_test.py -c met_resunet3d_sgd_blob
