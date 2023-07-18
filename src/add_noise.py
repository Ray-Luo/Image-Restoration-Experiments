import numpy as np
import cv2
import torch
from tqdm import tqdm
from types import SimpleNamespace
from process_hdr import exr2hdr, save_exr, normalizeRaw, bayer2rgb, save_hdr, print_min_max, bayer2RGB, RandomNoiseAdder
import os
import shutil

# profile = np.load("/home/luoleyouluole/Image-Restoration-Experiments/Milan_NoiseProfile.npy")
noise = {
    "iso_range_start": 0.0,
    "iso_range_end": 1.0,
    "min_gain": 2.0,
    "max_gain": 15.5,
    "shot_noise_slope": [0.00011190175847216757],
    "shot_noise_intercept": [0.0004843544188776163],
    "shot_noise_stderr": [3.210092787806755e-06],
    "read_noise_slope": [4.738392326016082e-06],
    "read_noise_intercept": [-5.734861671087736e-06],
    "read_noise_stderr": [1.2087166682107576e-07]
}
noise_adder = RandomNoiseAdder(**noise)


gt_list = [
    # "/home/luoleyouluole/Image-Restoration-Experiments/data/train",
    # "/home/luoleyouluole/Image-Restoration-Experiments/data/valid",
    "/home/luoleyouluole/Image-Restoration-Experiments/data/test_aug",
]
gt_list.sort()

gt_exr_list = [
    # "/home/luoleyouluole/Image-Restoration-Experiments/data/train_exr",
    # "/home/luoleyouluole/Image-Restoration-Experiments/data/valid_exr",
    "/home/luoleyouluole/Image-Restoration-Experiments/data/test_exr_aug",
]
gt_exr_list.sort()

gt_hdr_list = [
    # "/home/luoleyouluole/Image-Restoration-Experiments/data/train_hdr",
    # "/home/luoleyouluole/Image-Restoration-Experiments/data/valid_hdr",
    "/home/luoleyouluole/Image-Restoration-Experiments/data/test_hdr_aug",
]
gt_hdr_list.sort()

noise_exr_list = [
    # "/home/luoleyouluole/Image-Restoration-Experiments/data/train_noise_exr",
    # "/home/luoleyouluole/Image-Restoration-Experiments/data/valid_noise_exr",
    "/home/luoleyouluole/Image-Restoration-Experiments/data/test_noise_exr_aug",
]
noise_exr_list.sort()

noise_hdr_list = [
    # "/home/luoleyouluole/Image-Restoration-Experiments/data/train_noise_hdr",
    # "/home/luoleyouluole/Image-Restoration-Experiments/data/valid_noise_hdr",
    "/home/luoleyouluole/Image-Restoration-Experiments/data/test_noise_hdr_aug",
]
noise_hdr_list.sort()

for gt_exr_folder, gt_hdr_folder, noise_exr_folder, noise_hdr_folder in tqdm(zip(gt_exr_list, gt_hdr_list, noise_exr_list, noise_hdr_list)):
    if os.path.exists(gt_exr_folder):
        shutil.rmtree(gt_exr_folder)
        os.mkdir(gt_exr_folder)
    else:
        os.mkdir(gt_exr_folder)

    if os.path.exists(gt_hdr_folder):
        shutil.rmtree(gt_hdr_folder)
        os.mkdir(gt_hdr_folder)
    else:
        os.mkdir(gt_hdr_folder)

    if os.path.exists(noise_exr_folder):
        shutil.rmtree(noise_exr_folder)
        os.mkdir(noise_exr_folder)
    else:
        os.mkdir(noise_exr_folder)

    if os.path.exists(noise_hdr_folder):
        shutil.rmtree(noise_hdr_folder)
        os.mkdir(noise_hdr_folder)
    else:
        os.mkdir(noise_hdr_folder)


for gt_folder, gt_exr_folder, gt_hdr_folder, noise_exr_folder, noise_hdr_folder in tqdm(zip(gt_list, gt_exr_list, gt_hdr_list, noise_exr_list, noise_hdr_list)):
    print(gt_folder, gt_exr_folder, gt_hdr_folder, noise_exr_folder, noise_hdr_folder)
    gt_imgs = os.listdir(gt_folder)
    gt_imgs.sort()
    for gt in tqdm(gt_imgs):
        if "cargo_boat" not in gt and "skyscraper" not in gt and "urban_land" not in gt:
            continue

        img = cv2.imread(os.path.join(gt_folder, gt), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
        img = img / 4000.0
        save_exr(img, gt_exr_folder, gt.replace(".hdr", ".exr"))

        img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2)
        data, shot_noise, read_noise = noise_adder(img)
        data = data.permute(0, 2, 3, 1).squeeze(0).numpy()

        save_exr(data, noise_exr_folder, gt.replace(".hdr", "_noise.exr"))

        gt_exr = cv2.imread(os.path.join(gt_exr_folder, gt.replace(".hdr", ".exr")), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
        gt_hdr = exr2hdr(gt_exr)
        save_hdr(gt_hdr, gt_hdr_folder, gt.replace(".hdr", "_exr2hdr.hdr"))

        noise_exr = cv2.imread(os.path.join(noise_exr_folder, gt.replace(".hdr", "_noise.exr")), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
        noise_hdr = exr2hdr(noise_exr)
        save_hdr(noise_hdr, noise_hdr_folder, gt.replace(".hdr", "_noise_exr2hdr.hdr"))
