This project is a Chest X Ray Image classification algorithm build on NIH chest_ray_14 dataset
The baseline model is based on CheXNet published by Andrew Ng el, the paper is here: https://arxiv.org/pdf/1711.05225.pdf 
Quick Start:
1.Change dir to ChestRayXNet
2.Download chest_ray_14 dataset and unzip them in one floder, defult folder of this project is ChestRayXNet/data/image
3.Modify the image path in ChestRayXNet/shell/write_all.sh
4.Run shell/write_all.sh under ChestRayXNet, then you start convert 112,120 chest x ray images into 30 TFRecord file.
5.Decide which kinds of network you want to train on, 
  (1)If you want to train densenet121, run shell/train_densenet_121.sh
  (2)If you want to train densenet161, run shell/train_densenet.sh
  (3)If you want to train vgg16, run shell/train_vgg16.sh
6.To evaluate the model performance, run shell/eval_muti.sh, remember to check the model type in eval_muti.py

In the futher, I will merge all the training file onto a single .py file, and you can simply change the perameter in shell 
script for different network.
I will modify the eval_muti.py to make all network arcitecture aviliable.
This README file write in a hurry and I may forget some improtant information, so please let me know any point that makes you confused.

This project authored by Ruiqi Sun.
