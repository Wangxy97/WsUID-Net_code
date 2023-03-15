# Learning Scribbles for Dense Depth:Weakly-Supervised Single Underwater Image Depth Estimation Boosted by Multi-Task Learning
This repository is the official PyTorch implementation of WsUID-Net.
## Dataset preparation 
You need to prepare datasets for following training and testing activities, the detailed information is at [Dataset Setup](data/README.md).

## Train
``` 
python train.py --dataroot /path_to_data --name train_name --model SGUIENet --display_env display_env_name
```
## Test
```
python test.py --dataroot /path_to_data --name test_name --model test_SGUIE 
```
You can download the trained model from [here](https://drive.google.com/file/d/1vbY4GZ5-AwVKouDFHvFj9nL-grnIB2d3/view?usp=sharing).

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
