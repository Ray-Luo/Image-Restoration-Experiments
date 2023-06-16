import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np
from process_hdr import print_min_max, save_hdr
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor, tensor2img
from torchvision.transforms.functional import (
    adjust_brightness,
    adjust_contrast,
    adjust_hue,
    adjust_saturation,
    normalize,
)
import torch

img = cv2.imread("/home/luoleyouluole/Image-Restoration-Experiments/1_enhanced.png", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32) / 255.0
print_min_max(img)


img = img2tensor(img, bgr2rgb=True, float32=True)
normalize(img, torch.tensor([0.5, 0.5, 0.5]), torch.tensor([0.5, 0.5, 0.5]), inplace=True)
img = tensor2img(img, rgb2bgr=True, min_max=(0, 1))

# Convert RGB to YUV
yuv_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV) / 255.0

print_min_max(yuv_img)
img_gt = torch.tensor(yuv_img[:, :, 0]).unsqueeze(0)
print(img_gt.shape)



cv2.imwrite("/home/luoleyouluole/Image-Restoration-Experiments/1_enhanced_.png", yuv_img[:,:,0]*255.0)
