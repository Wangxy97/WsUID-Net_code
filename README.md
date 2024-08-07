# Learning Scribbles for Dense Depth:Weakly-Supervised Single Underwater Image Depth Estimation Boosted by Multi-Task Learning
[[Project]](https://wangxy97.github.io/WsUIDNet) [[Paper]](https://ieeexplore.ieee.org/document/10415086?source=authoralert) [[Dataset]](https://github.com/Wangxy97/SUIM-SDA_Dataset)

This repository is the official PyTorch implementation of WsUID-Net.

## Package requirements
* The following packages are required to run the codes. Note: the version of the packages were the ones we used and are suggestive, not mandatory.
    * python = 3.6
    * opencv-python = 4.5.3
    * torch = 1.10.1
    * easydict = 1.9
    * visdom = 0.1.8.9

## Dataset preparation 
* You need to prepare datasets for following training and testing activities. You can download SUIM-SDA dataset from [Google Drive](https://drive.google.com/file/d/19HGObIYPAZzNVR0OA3phzCUfag8WEk84/view?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/11PDmq-_ifb7801TnahKDPg?pwd=qa3m).
    * Decompress the SUIM-SDA package to the ./datasets folder
    * You can calculate the edge graph according to the semantic segmentation mask in SUIM-SDA, or download the edge graph directly from [here](https://pan.baidu.com/s/181yT4CDmmttSP9MJM5QbrA?pwd=1dag).
* Run ./data/Conver_data.py.  Convert the .csv file that stores the depth-rank samples to the .pkl file used for training.
``` 
python Conver_data.py
```

## Train
* You need to modify the input and output paths in train.py depending on where your data set is stored on disk, then run the following code in the terminal:

```
python3 -m visdom.server -port=8007
python train.py 
```
## Test
* You need to change ckpt_path in the test.py file, then run the following code in the terminal:
```
python test.py 
```
## Using the pre-trained model

You can download the trained model from [here](https://pan.baidu.com/s/1od3fPW2s4hqabVGxkfnLOA?pwd=4g4p).
To test on the pre-trained models, change ckpt_path in the test.py file.

## Citation
```
@ARTICLE{10415086,
  author={Li, Kunqian and Wang, Xiya and Liu, Wenjie and Qi, Qi and Hou, Guojia and Zhang, Zhiguo and Sun, Kun},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Learning Scribbles for Dense Depth: Weakly Supervised Single Underwater Image Depth Estimation Boosted by Multitask Learning}, 
  year={2024}
  doi={10.1109/TGRS.2024.3358892}}
```

## Acknowledgements
- https://github.com/xahidbuffon/SUIM
