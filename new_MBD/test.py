import cv2
import sys

import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
import MBD
import MBD_gray
import copy
import os
import time

src_folder = '/media/minh/MEDIA/Study/dataset/Interative_Dataset/images/images/'
label_folder = '/media/minh/MEDIA/Study/dataset/Interative_Dataset/images-labels/images-labels/'
output_folder1 = '/media/minh/MEDIA/Study/Results/MBD_gray_1/'
output_folder = '/media/minh/MEDIA/Study/Results/MBD_gray/'
gt_folder = '/media/minh/MEDIA/Study/dataset/Interative_Dataset/images-gt/images-gt/'

input_file = os.listdir(src_folder)

print(len(input_file))

runtime_total = 0
runtime_total1 = 0

IoU_total1 = 0
IoU_total = 0


for entry in input_file:
    print(entry)

    # if entry == '189080.jpg':

    parts = entry.split(".")

    src_name = src_folder + entry
    label_name = label_folder + parts[0] + '-anno.png'

    img = cv2.imread(src_name)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    label_gray = cv2.cvtColor(cv2.imread(label_name), cv2.COLOR_BGR2GRAY)
    h, w = label_gray.shape


    list_bg= []
    list_fg= []

    f_fg = np.zeros((h,w))
    f_bg = np.zeros((h,w))

    for i in range(0, h):
        for j in range(0, w):
            if i == 0 or j == 0 or i == h-1 or j == w-1:
                f_bg[i][j] = 255



    f_fg = np.array(f_fg, dtype="uint8") # convert to uint8
    f_bg = np.array(f_bg, dtype="uint8") # convert to uint8

    start_time = time.time()
    dmap_scalar_fg = MBD.MBD_waterflow(img_gray, f_bg)
    end_time1 = time.time()

    dist_map1 = MBD_gray.waterflow_marker(img_gray, f_bg)
    end_time2 = time.time()


    ret, thresh = cv2.threshold(dmap_scalar_fg, 90, 255, cv2.THRESH_BINARY)
    ret1, thresh1 = cv2.threshold(dist_map1, 90, 255, cv2.THRESH_BINARY)



    parts = entry.split('.')
    part = parts[0]
    gt_file = gt_folder + part + '.png'
    gt_img = cv2.imread(gt_file)
    gt_gray =  cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
    gt_gray = gt_gray /255

    thresh = thresh / 255
    thresh1 = thresh1 / 255


    A = thresh +gt_gray
    B = (A == 2)
    C = (A>0)
    inter = np.sum(B)
    union = np.sum(C)
    IoU = inter/(union+ 0.0001)
    IoU_total = IoU_total + IoU


    A1 = thresh1 +gt_gray
    B1 = (A1 == 2)
    C1 = (A1>0)
    inter1 = np.sum(B1)
    union1 = np.sum(C1)
    IoU1 = inter1/(union1+ 0.0001)
    IoU_total1 = IoU_total1 + IoU1


    initial_name = output_folder + entry
    initial_name1 = output_folder1 +entry

    cv2.imwrite(initial_name, dist_map1)
    cv2.imwrite(initial_name1, dmap_scalar_fg.astype(np.uint8))

print('IoU_total: ',IoU_total)
print('IoU_total1: ',IoU_total1)