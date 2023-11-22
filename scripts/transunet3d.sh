cd ~/deeplearning/brats2018/src
python train_test.py -c met_transunet3d_sgd 
python train_test.py -c met_transunet3d_adam 
python train_test.py -c met_transunet3d_adamw
