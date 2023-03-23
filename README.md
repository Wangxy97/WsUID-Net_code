# Learning Scribbles for Dense Depth:Weakly-Supervised Single Underwater Image Depth Estimation Boosted by Multi-Task Learning
[[Project]](https://wangxy97.github.io/WsUIDNet) [[Paper]](https://arxiv.org/XXX) [[Dataset]](https://github.com/Wangxy97/SUIM-SDA_Dataset)

This repository is the official PyTorch implementation of WsUID-Net.

## The final version of the code and related descriptions are constantly being refined and updated……

## Package requirements
* The following packages are required to run the codes. Note: the version of the packages were the ones we used and are suggestive, not mandatory.
    * python = 3.6
    * opencv-python = 4.5.3
    * torch = 1.10.1
    * easydict = 1.9
    * visdom = 0.1.8.9

## Dataset preparation 
* You need to prepare datasets for following training and testing activities. You can download SUIM-SDA dataset from [Google Drive](https://drive.google.com/file/d/****) or [Baidu Netdisk](http://).
    * Decompress the SUIM-SDA package to the ./datasets folder
    * You can calculate the edge graph according to the semantic segmentation mask in SUIM-SDA, or download the edge graph directly from [here](http://).
* Run ./data/Conver_data.py.  Convert the .csv file that stores the depth-rank samples to the .pkl file used for training.
``` 
python Conver_data.py
```

## Train
You need to modify the input and output paths in train.py depending on where your data set is stored on disk, then run the following code in the terminal:

```
python3 -m visdom.server -port=8007
python train.py 
```
## Test
```
python test.py 
```
## Using the pre-trained model

You can download the trained model from [here](https://drive.google.com/file/d/****).
To test on the pre-trained models, change ckpt_path in the test.py file.

## Citation
```
@article{w2023wsuid,
  title={Learning Scribbles for Dense Depth:Weakly-Supervised Single Underwater Image Depth Estimation Boosted by Multi-Task Learning},
  author={Kunqian Li and Xiya Wang and Wenjie Liu and Guojia Hou and Zhiguo Zhang and Kun Sun},
  journal={arXiv preprint arXiv:xxxx.xxxx},
  year={2023}
}
```

## Acknowledgements
- https://github.com/xahidbuffon/SUIM
