import numpy as np
import cv2
import torch
from process_hdr import rgb2bayer, addNoise, normalizeRaw, bayer2rgb
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

profile = np.load("/home/luoleyouluole/Image-Restoration-Experiments/Milan_NoiseProfile.npy")

# print(profile.shape, profile)

img = cv2.imread("/home/luoleyouluole/Image-Restoration-Experiments/data/1/A004C015_121104AF.00000000.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
img = img * 65535
img = img.astype(np.int32)
img = torch.from_numpy(img).unsqueeze(0)

gt_bayer, wb_gain = rgb2bayer(img)
noisy_bayer = addNoise(profile, gt_bayer)

noisy_tensors = normalizeRaw(noisy_bayer)
gt_tensors = normalizeRaw(gt_bayer)

wb_gain = np.sqrt(wb_gain)
gt_tensors[0, 0, :2, :2] = wb_gain
noisy_tensors[0, 0, :2, :2] = wb_gain

if gt_tensors.shape[0] == 1:
    gt_tensors = np.squeeze(gt_tensors, axis=0)
    noisy_tensors = np.squeeze(noisy_tensors, axis=0)

print(type(noisy_tensors))
noisy_tensors = cv2.cvtColor(noisy_tensors, cv2.COLOR_BAYER_BG2BGR_EA)

noisy_tensors = torch.from_numpy(noisy_tensors)
gt_tensors = torch.from_numpy(gt_tensors)

# noisy_tensors = bayer2rgb(noisy_tensors)

print(noisy_tensors.shape)
