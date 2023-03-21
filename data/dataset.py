import os
import cv2
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

def default_loader(path):
    return Image.open(path).convert('RGB')

def get_semMap(mask_path):
    mask = cv2.imread(mask_path)
    hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)

    BW = cv2.inRange(hsv, lowerb=np.array([0, 0, 0]), upperb=np.array([180, 255, 46]))  # black
    SR = cv2.inRange(hsv, lowerb=np.array([0, 0, 221]), upperb=np.array([180, 30, 255]))  # white
    FV = cv2.inRange(hsv, lowerb=np.array([26, 43, 46]), upperb=np.array([34, 255, 255]))  # yellow
    HD = cv2.inRange(hsv, lowerb=np.array([100, 43, 46]), upperb=np.array([124, 255, 255]))  # blue
    PF = cv2.inRange(hsv, lowerb=np.array([35, 43, 46]), upperb=np.array([77, 255, 255]))  # green
    WR = cv2.inRange(hsv, lowerb=np.array([78, 43, 46]), upperb=np.array([99, 255, 255]))  # Sky blue
    RO = cv2.inRange(hsv, lowerb=np.array([0, 43, 46]), upperb=np.array([10, 255, 255]))  # red
    RI = cv2.inRange(hsv, lowerb=np.array([125, 43, 46]), upperb=np.array([155, 255, 255]))  # purple
    return np.stack((BW, HD, PF, WR, RO, RI, FV, SR), -1)


class UDE_Dataset(Dataset):
    def __init__(self, path_img, path_edge, path_mask, path_target,
                 loader=default_loader, transforms=None):
        super(UDE_Dataset, self).__init__()
        #
        with open(path_target, 'rb') as f:
            self.targets = pickle.load(f)
        self.path_img = path_img
        self.path_edge = path_edge
        self.path_mask = path_mask
        self.imgs = [target['name'] for target in self.targets]

        self.transforms = transforms
        self.loader = loader

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.path_img, self.imgs[index]))
        edge = Image.open(os.path.join(self.path_edge, self.imgs[index]))
        mask = get_semMap(os.path.join(self.path_mask, self.imgs[index].strip('.jpg') + ".bmp"))
        img_resize = img.resize((256, 256))
        target = self.targets[index]

        if self.transforms:
            img_resize = self.transforms(img_resize)
            edge = self.transforms(edge)
            mask = self.transforms(mask)
        return img_resize, edge, mask, target

    def __len__(self):
        return len(self.imgs)


class TEST_Dataset(Dataset):
    def __init__(self, path_img, transforms=None):
        self.path_img = path_img
        self.imgs = os.listdir(path_img)
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.path_img, self.imgs[index]))
        img_resize = img.resize((256, 256))

        if self.transforms:
            img_resize = self.transforms(img_resize)
        return img_resize


