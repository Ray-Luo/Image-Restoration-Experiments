import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from types import SimpleNamespace
import OpenEXR, Imath
import os
from process_hdr import save_hdr
from tqdm import tqdm
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

SEED = 12345
NUMBER_AUG = 5

def augment(img_folder: str, name: str):
    img = cv2.imread(os.path.join(img_folder, name), -1).astype(np.float32)

    seen = set()
    random.seed(SEED)
    rand_num = random.uniform(0.1, 0.9)

    while len(seen) < NUMBER_AUG:
        if rand_num not in seen:
            seen.add(rand_num)
            img *= rand_num
            new_name = name.split('.')[0] + '_aug_' + str(len(seen)) + '.hdr'
            save_hdr(img, img_folder, new_name)
            rand_num = random.uniform(0.1, 0.9)

img_folder = "/home/luoleyouluole/Image-Restoration-Experiments/data/test_aug"
file_list = os.listdir(img_folder)
file_list.sort()

for file_name in tqdm(file_list):
    augment(img_folder, file_name)
