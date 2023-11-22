cd ~/deeplearning/brats2018/src
python train_test.py -c met_resunetse3d_adam
python train_test.py -c met_resunetse3d_adamw
python train_test.py -c met_resunetse3d_sgd

python train_test.py -c met_resunetse3d_adam_blob
python train_test.py -c met_resunetse3d_adamw_blob
python train_test.py -c met_resunetse3d_sgd_blob