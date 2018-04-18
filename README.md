# Diagnosis of pulmonary disease based on NIH ChestX-ray14
This project is a chest X ray images muti-label classification algorithm with high performance which have potential to help radiologist diagnosis pulmonary disease better.<br />
The baseline of this algorithm is [CheXNet](https://arxiv.org/pdf/1711.05225.pdf), I learn lots of things the from the paper which help me train my DenseNet121 well.<br />
All the model trained on NIH ChestX-ray14 dataset, you can download the data [here](https://github.com/zoogzog/chexnet)<br />
## Quick Start
* Download ChestX-ray14 dataset [here](https://github.com/zoogzog/chexnet), unzip all 14 xxx.tar.gz file to a single floder, you can run      'tar -xvzf *FILENAME*' 
* Modify the image path in shell/write_all.sh, image path fellowed by --dataset_dir tag.
* Change your work directory to ChestRayXNet.
* Run shell/write_all.sh to convert all 112,120 chest X ray images into .tfrecord format.
* Run shell/train_densenet_121.sh to start training a network named CheXNet, all settings is same as [CheXNet](https://arxiv.org/pdf/1711.05225.pdf) published by Andrew Ng et al.
* Futher, if you want to try different networks, modify the network type in shell/train_generate.sh fellowed by --model_type tag and log directory. If you get OOM error during training, then modify the batch size fellowed by --batch_size tag.
* To Visualize training procedure, run fellowing command in terminal, LOG_DIR is the dir set after --log_dir tag in the shell script you running.
  tensorbord --logdir='*LOG_DIR*' --port=6006
* To evaluate the model performance, run shell/eval_muti.sh, if your train different network other than DenseNet121, modify the network type and log dir in eval_muti.sh file.

## Last but not least
This README file write in a hurry and I belive I must make some mistake, please let me know if thier is any word that made you confused.<br />
I update this README.md file 3 days after first commit but I think it's still not good and clear enough, maybe next version would be better.<br />

This project is writen by **Ruiqi Sun**(**孙瑞琦**）<br />
