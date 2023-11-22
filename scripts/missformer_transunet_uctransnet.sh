cd /dss/dsshome1/05/di93sip/deeplearning/Metastasis-segmentation/src/
python train_test.py -c met_missformer_adam
python train_test.py -c met_missformer_adamw
python train_test.py -c met_missformer_sgd
python train_test.py -c met_transunet_adam
python train_test.py -c met_transunet_adamw
python train_test.py -c met_transunet_sgd
python train_test.py -c met_uctransnet_adam
python train_test.py -c met_uctransnet_adamw
python train_test.py -c met_uctransnet_sgd

python train_test.py -c met_missformer_adam_blob
python train_test.py -c met_missformer_adamw_blob
python train_test.py -c met_missformer_sgd_blob
python train_test.py -c met_transunet_adam_blob
python train_test.py -c met_transunet_adamw_blob
python train_test.py -c met_transunet_sgd_blob
python train_test.py -c met_uctransnet_adam_blob
python train_test.py -c met_uctransnet_adamw_blob
python train_test.py -c met_uctransnet_sgd_blob