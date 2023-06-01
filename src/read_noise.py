import numpy as np
import cv2
import torch
from process_hdr import rgb2bayer

noise = np.load("/home/luoleyouluole/Image-Restoration-Experiments/Milan_NoiseProfile.npy")

print(noise.shape, noise)

img = cv2.imread("/home/luoleyouluole/Image-Restoration-Experiments/data/HDR_VIDEO_FRAME_20/A001C096_141231KD_00091764.hdr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)

img = torch.from_numpy(img).unsqueeze(0)

gt_bayer, wb_gain = rgb2bayer(img)
