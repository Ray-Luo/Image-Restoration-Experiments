from eval_util import pu_psnr, pu_ssim
import torch.nn.functional as F
from process_hdr import save_hdr, print_min_max
import torch
import numpy as np
import torchvision.transforms as transforms
import os
import cv2
from skimage import transform


img_path = "/home/luoleyouluole/Image-Restoration-Experiments/data/hdr_data/train_patchify/Letchworth_Tea_Table_1_s190.hdr"
res_dir = "/home/luoleyouluole/Image-Restoration-Experiments/res.hdr"

img = cv2.imread(img_path, -1).astype(np.float32)

print_min_max(img)

ds = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)

upsample = cv2.resize(ds, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)

print_min_max(upsample)

upsample /= np.max(upsample)
upsample = np.power(upsample, 1/1.3)

cv2.imwrite(res_dir, upsample, [cv2.IMWRITE_HDR_COMPRESSION, 0])
