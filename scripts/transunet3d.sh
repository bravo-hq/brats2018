cd /dss/dsshome1/05/di93sip/deeplearning/Metastasis-segmentation/src
python train_test.py -c met_transunet3d_sgd 
python train_test.py -c met_transunet3d_adam 
python train_test.py -c met_transunet3d_adamw
