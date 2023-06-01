import numpy as np
import rawpy
import imageio
import cv2
import torch
from process_hdr import rgb2bayer, addNoise, normalizeRaw, bayer2rgb, save_hdr, print_min_max, bayer2RGB
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

profile = np.load("/home/luoleyouluole/Image-Restoration-Experiments/Milan_NoiseProfile.npy")

# print(profile.shape, profile)

img = cv2.imread("/home/luoleyouluole/Image-Restoration-Experiments/data/1/A004C015_121104AF.00000000.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
print_min_max(img)
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

noisy_tensors = np.transpose(noisy_tensors, (1, 2, 0))#.astype(np.uint16)
noisy_tensors = np.squeeze(noisy_tensors, axis=-1)
# noisy_tensors.tofile("./noise.raw")
# noisy_tensors = np.transpose(noisy_tensors, (1, 2, 0)).astype(np.uint16)
# noisy_tensors = cv2.cvtColor(noisy_tensors, cv2.COLOR_BayerBG2BGR)
noisy_tensors = bayer2RGB(noisy_tensors)
noisy_tensors = noisy_tensors * 255


# imageio.imsave('./noise.jpg', noisy_tensors)

print_min_max(noisy_tensors)

save_hdr(noisy_tensors, "/home/luoleyouluole/Image-Restoration-Experiments/", "noise.hdr")

print(noisy_tensors.shape)

noisy_tensors = torch.from_numpy(noisy_tensors)
gt_tensors = torch.from_numpy(gt_tensors)

# noisy_tensors = bayer2rgb(noisy_tensors)
