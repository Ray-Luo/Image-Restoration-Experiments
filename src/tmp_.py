from eval_util import pu_psnr, pu_ssim
import torch.nn.functional as F
from process_hdr import save_hdr, print_min_max
import torch
import numpy as np
import torchvision.transforms as transforms
import os
import cv2
from skimage import transform

def visualize(img: np.array,root, name):
    img = img / 4000.0
    img = np.clip(img, 0., 1.)
    img = np.power(img, 1/2.2)
    save_hdr(img, root, name)

img = cv2.imread("/home/luoleyouluole/Image-Restoration-Experiments/data/res/Ahwahnee_Great_Lounge_raw_GT.hdr", -1).astype(np.float32)
# img2 = cv2.imread("/home/luoleyouluole/Image-Restoration-Experiments/data/res/Ahwahnee_Great_Lounge_naive.hdr", -1).astype(np.float32)

downscaled = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
res_naive = cv2.resize(downscaled, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)

save_hdr(res_naive, "/home/luoleyouluole/Image-Restoration-Experiments/data/res/", "Ahwahnee_Great_Lounge" + "_raw_naive.hdr")
# visualize(res_naive, "/home/luoleyouluole/Image-Restoration-Experiments/data/res/", "Ahwahnee_Great_Lounge" + "_naive.hdr")

img = cv2.imread("/home/luoleyouluole/Image-Restoration-Experiments/data/res/Frontier_raw_GT.hdr", -1).astype(np.float32)
# img2 = cv2.imread("/home/luoleyouluole/Image-Restoration-Experiments/data/res/Ahwahnee_Great_Lounge_naive.hdr", -1).astype(np.float32)

downscaled = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
res_naive = cv2.resize(downscaled, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)

save_hdr(res_naive, "/home/luoleyouluole/Image-Restoration-Experiments/data/res/", "Frontier" + "_raw_naive.hdr")
