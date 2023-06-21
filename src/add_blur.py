import numpy as np
import cv2
import torch
from tqdm import tqdm
from types import SimpleNamespace
from process_hdr import exr2hdr, save_exr, normalizeRaw, bayer2rgb, save_hdr, print_min_max, bayer2RGB, RandomNoiseAdder
import os
import shutil


gt_list = [
    "/home/luoleyouluole/Image-Restoration-Experiments/data/train",
    "/home/luoleyouluole/Image-Restoration-Experiments/data/valid",
    "/home/luoleyouluole/Image-Restoration-Experiments/data/test",
]
gt_list.sort()

blur_list = [
    "/home/luoleyouluole/Image-Restoration-Experiments/data/train_blur",
    "/home/luoleyouluole/Image-Restoration-Experiments/data/valid_blur",
    "/home/luoleyouluole/Image-Restoration-Experiments/data/test_blur",
]
blur_list.sort()



for gt_folder, blur_folder in tqdm(zip(gt_list, blur_list)):
    if os.path.exists(blur_folder):
        shutil.rmtree(blur_folder)
        os.mkdir(blur_folder)
    else:
        os.mkdir(blur_folder)


for gt_folder, blur_folder in tqdm(zip(gt_list, blur_list)):
    print(gt_folder, blur_folder)
    gt_imgs = os.listdir(gt_folder)
    gt_imgs.sort()
    for gt in tqdm(gt_imgs):
        img = cv2.imread(os.path.join(gt_folder, gt), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
        blur = cv2.GaussianBlur(img, (51, 51), 0)
        save_exr(blur, blur_folder, gt.replace(".hdr", "_blur.hdr"))
