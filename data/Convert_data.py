"""Program description:
Convert the .csv file that stores the depth-rank samples to the .pkl file used for training.
"""

import numpy as np
import os
from PIL import Image
import pandas as pd
import pickle as pkl


def convert_point(path, name, H, W, number):
    raw_labels = pd.read_csv(path, header=None, skiprows=0)
    point = raw_labels[:len(raw_labels)].values.astype('float32')
    if len(point) <= number:
        row_rand_array = np.arange(point.shape[0])
        np.random.shuffle(row_rand_array)
        row_rand = point[row_rand_array[0:(number-len(point))]]
        points = np.vstack((point, row_rand))
        while len(points) < number:
            row_rand_array = np.arange(point.shape[0])
            np.random.shuffle(row_rand_array)
            row_rand = point[row_rand_array[0:(number - len(points))]]
            points = np.vstack((points, row_rand))
    else:
        row_rand_array = np.arange(point.shape[0])
        np.random.shuffle(row_rand_array)
        points = point[row_rand_array[0:number]]
    label = {}
    label['name'] = name
    label['x_A'] = points[:, 0] * (256/H)
    label['y_A'] = points[:, 1] * (256/W)
    label['x_B'] = points[:, 2] * (256/H)
    label['y_B'] = points[:, 3] * (256/W)
    label['ordinal_relation'] = points[:, 4]
    return label


def save(obj, path):
    with open(path, 'wb') as f:
        pkl.dump(obj, f)

def load(path):
    with open(path, 'rb') as f:
        return pkl.load(f)


if __name__ == '__main__':
    csv_path = './datasets/SUIM-SDA/train_dataset/Annotations/Ranked_sample/'
    path_img = './datasets/SUIM-SDA/train_dataset/Raw/'
    filename_list = os.listdir(csv_path)
    labels = []
    for i in range(len(filename_list)):
        filename = filename_list[i].strip('.csv') + ".jpg"
        img = Image.open(os.path.join(path_img, filename))
        H = img.height
        W = img.widt
        csv_ = csv_path + filename_list[i]
        label = convert_point(csv_, filename, H, W, number=10000)
        labels.append(label)

    save(labels, './dataset/target.pkl')
    print('finish')




