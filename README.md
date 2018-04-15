This project is a Chest X Ray Image classification algorithm build on NIH chest_ray_14 dataset<br />
The baseline model is based on CheXNet published by Andrew Ng el, the paper is here: https://arxiv.org/pdf/1711.05225.pdf <br />
Quick Start:<br />
1.Change dir to ChestRayXNet<br />
2.Download chest_ray_14 dataset and unzip them in one floder, defult folder of this project is ChestRayXNet/data/image<br />
3.Modify the image path in ChestRayXNet/shell/write_all.sh<br />
4.Run shell/write_all.sh under ChestRayXNet, then you start convert 112,120 chest x ray images into 30 TFRecord file.(train, validation,test set included)<br />
5.Decide which kinds of network you want to train on, <br />
  (1)If you want to train densenet121, modify shell/train_generate.sh --model_type to densenet121 and --log_dir to log/dense121<br />
  (2)If you want to train densenet161, modify shell/train_generate.sh --model_type to densenet161 and --log_dir to log/dense161<br />
  (3)If you want to train vgg16, modify shell/train_generate.sh --model_type to vgg16 and --log_dir to log/vgg16<br />
6.To get the overview of train, run the fellowing command on terminal:<br />
tensorbord --logdir='LOG_DIR' --port=6006<br />
The perameter 'LOG_DIR' is the path where you save your training log, can be found in train_generate.sh<br />
7.To evaluate the model performance, run shell/eval_muti.sh, remember to check the model type in eval_muti.py<br />

In the futher, I will merge all the training file onto a single .py file, and you can simply change the perameter in shell <br />
script for different network.<br />
I will modify the eval_muti.py to make all network arcitecture aviliable.<br />
This README file write in a hurry and I may forget some improtant information, so please let me know any point that makes you confused.<br />

This project authored by Ruiqi Sun.<br />
