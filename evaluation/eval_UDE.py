'''
Function: Quantitative evaluation of depth prediction results
Remarks: Calculation of WHDR (Weighted Human Disagreement Rate)、
SI-MSE and Pearson correlation coefficient
'''

import csv
import cv2
import os
import skimage.io as io
from scipy.stats import pearsonr
import numpy as np


def load_target(path):   #加载csv文件，返回列的值list
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    csvfile = open(path, 'r')
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        list1.append(row[0])
        list2.append(row[1])
        list3.append(row[2])
        list4.append(row[3])
        list5.append(row[4])
    return list1, list2, list3, list4, list5


def relations(z_A, z_B, gt, thresh):
    order = 1
    if (z_A / z_B) > 1 + thresh:
        order = -1
    else:
        if (z_B / z_A) > 1 + thresh:
            order = 1
        else:
            order = 0
    if order == gt:
        return True

    else:
        return False

def WHDR(predDepth, target_path, thresh, H, W):
    y_c, x_c, y_f, x_f, lable = load_target(target_path)
    i = 0
    for index in range(len(lable)):
        y_A = int(y_c[index]) * (256/H)
        x_A = int(x_c[index]) * (256/W)
        y_B = int(y_f[index]) * (256/H)
        x_B = int(x_f[index]) * (256/W)
        gt = int(lable[index])

        z_A = predDepth[int(y_A), int(x_A)]  # "A" points (row, col)
        z_B = predDepth[int(y_B), int(x_B)]  # "B" points

        result = relations(z_A, z_B, gt, thresh)
        if result:
            i += 1
    whdr = 1-(i / (len(lable)))
    # print("correct_samples / total_samples:{} / {}".format(i, len(lable)))
    # print('thresh:', thresh)
    # print("Evaluation result: WHDR = ", whdr)
    return whdr



def sq_sinv(y,y_):
    # To avoid log(0) = -inf
    y_[y_==0] = 1
    y[y==0] = 1
    alpha = np.mean(np.log(y_) - np.log(y))
    err = (np.log(y) - np.log(y_) + alpha) ** 2
    return (np.mean(err[:]) / 2)

def pear_coeff(y,y_):
    y = y.ravel()
    y_ = y_.ravel()
    err = pearsonr(y, y_)
    if np.isnan(err[0]):
        return 0
    else:
        return err[0]

def eval_whdr(result_path, target_path):
    path_img = '/data/wangxy/UDE_new/UDE/dataset/SUIM_TEST/images/'
    file_list = os.listdir(path_img)
    whdr_list = []
    for file in file_list:
        depth = cv2.imread(result_path + file, cv2.IMREAD_GRAYSCALE)
        raw = cv2.imread(path_img + file)
        H = raw.shape[0]
        W = raw.shape[1]
        csv_file = file.strip('.jpg') + ".csv"
        whdr = WHDR(depth, target_path + csv_file, 0.02, H, W)
        whdr_list.append(whdr)
    average = np.mean(whdr_list)
    print('WHDR-average:', average)



def calculate_metrics(gt_path, results_path):
    l1 = os.listdir(gt_path)
    l1.sort()
    l2 = os.listdir(results_path)
    l2.sort()

    score = []
    names = []
    for i in range(len(l1)):
        g1 = (io.imread(os.path.join(gt_path, l1[i]))).astype(np.uint8)
        t1 = io.imread(os.path.join(results_path, l1[i])).astype(np.uint8)
        # If depth value is nan or invalid, change it to zero.
        for j in range(g1.shape[0]):
            for k in range(g1.shape[1]):
                if g1[j, k] >= 255:
                    g1[j, k] = 0
                    t1[j, k] = 0

        score.append([sq_sinv(t1, g1), pear_coeff(g1, t1)])
        # print(l1[i])
        # print('RMSE Squared log Scale-invariant error:', score[-1][0], ', Pearson Coefficient', score[-1][1])
        names.append(l1[i])

    m = np.mean(np.array(score), axis=0)
    print('Mean RMSE Squared log Scale-invariant error', m[0])
    print('Mean Pearson Coefficient', m[1])


if __name__=='__main__':

# Path of Ground True
    squid_gt = 'The path of SQUID depth GT'
    suim_target = 'The path of SUIM test dataset relative depth sample'

# path of test result
    pre_suim = '../out/Ours/suim/'
    pre_squid = '../out/Ours/squid/'

    eval_whdr(result_path=pre_suim, target_path=suim_target)
    calculate_metrics(gt_path=squid_gt, results_path=pre_squid)
