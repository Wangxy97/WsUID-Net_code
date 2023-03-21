import cv2
import os

pre_depth = '../out/'

if __name__ == '__main__':
    img_path = pre_depth + '/result/'
    output_path = pre_depth + "heat_result/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    file_list = os.listdir(img_path)
    for file in file_list:
        img = cv2.imread(img_path + file, cv2.IMREAD_GRAYSCALE)
        heat_img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # Note that the three-channel heat map
                                                             # here is a GBR arrangement specific to cv2
        cv2.imwrite(output_path + file, heat_img)
    print('finish')