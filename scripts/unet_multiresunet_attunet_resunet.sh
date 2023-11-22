cd ~/deeplearning/Metastasis-segmentation/src
python train_test.py -c met_unet_adam
python train_test.py -c met_unet_adamw
python train_test.py -c met_unet_sgd
python train_test.py -c met_multiresunet_adam
python train_test.py -c met_multiresunet_adamw
python train_test.py -c met_multiresunet_sgd
python train_test.py -c met_attunet_adam
python train_test.py -c met_attunet_adamw
python train_test.py -c met_attunet_sgd
python train_test.py -c met_resunet_adam
python train_test.py -c met_resunet_adamw
python train_test.py -c met_resunet_sgd

python train_test.py -c met_unet_adam_blob
python train_test.py -c met_unet_adamw_blob
python train_test.py -c met_unet_sgd_blob
python train_test.py -c met_multiresunet_adam_blob
python train_test.py -c met_multiresunet_adamw_blob
python train_test.py -c met_multiresunet_sgd_blob
python train_test.py -c met_attunet_adam_blob
python train_test.py -c met_attunet_adamw_blob
python train_test.py -c met_attunet_sgd_blob
python train_test.py -c met_resunet_adam_blob
python train_test.py -c met_resunet_adamw_blob
python train_test.py -c met_resunet_sgd_blob