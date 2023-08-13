import numpy as np
import cv2
import torch
from tqdm import tqdm
from types import SimpleNamespace
from process_hdr import exr2hdr, save_exr, normalizeRaw, bayer2rgb, save_hdr, print_min_max, bayer2RGB, RandomNoiseAdder
import os
import shutil

hdr_folder = "/home/luoleyouluole/cargo/"
exr_folder = "/home/luoleyouluole/cargo_exr/"

hdr_imgs = os.listdir(hdr_folder)
hdr_imgs.sort()

for hdr in hdr_imgs:
    if not ".hdr" in hdr:
        continue

    hdr_img = cv2.imread(os.path.join(hdr_folder, hdr), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)

    save_exr(hdr_img, exr_folder, hdr.replace(".hdr", ".exr"))
