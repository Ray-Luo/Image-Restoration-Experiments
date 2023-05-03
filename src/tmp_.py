from eval_util import pu_psnr, pu_ssim
import torch.nn.functional as F
from process_hdr import save_hdr, print_min_max
import torch
import numpy as np
import torchvision.transforms as transforms
import os
import cv2
from skimage import transform
import random

noise_profile = "/home/luoleyouluole/Image-Restoration-Experiments/data/OnePlus10Pro_NoiseProfile.npy"
noise_profile_variance = np.load(noise_profile)
print(noise_profile_variance.shape)
NOISE_ORDER = 1.0
AUG_BLACK_LEVEL = 4096
AUG_WHITE_LEVEL = 65535
BAYER_PATTERN = ""
focus_order = 3

def visualize(img: np.array,root, name):
    img = img / 4000.0
    img = np.clip(img, 0., 1.)
    img = np.power(img, 1/2.2)
    save_hdr(img, root, name)

def noiseFromVariance(img):
    np.random.seed(int.from_bytes(os.urandom(4), byteorder="little"))
    shape = img.shape
    var = noise_profile_variance[img]
    noise = var * np.random.randn(*shape)
    return noise

def getRandomMultiplier():
    np.random.seed()
    dice = np.random.randint(10)
    max_multf = (
        AUG_WHITE_LEVEL - AUG_BLACK_LEVEL
    ) * 1.0 / 255.0 - 1
    multf = max_multf + 1
    if dice == 1:
        multf = 1 + max_multf * np.random.random_sample()
    elif dice > 1:
        multf = 1 + max_multf * (np.random.random_sample() ** focus_order)
    return multf

def getRandomGain():
    multf = getRandomMultiplier()
    r_gain = random.uniform(1.0, 3.0)
    b_gain = random.uniform(1.0, 3.0)
    if BAYER_PATTERN == "MONO":
        r_gain = 1.0
        b_gain = 1.0
    return multf, r_gain, b_gain

def applyGain(rgb_tensor, multf, r_gain, b_gain):
    wb_gain = np.zeros((2, 2), dtype=np.float32)
    T, H, W, C = rgb_tensor.shape
    rc = np.clip(
        rgb_tensor[:, :, :, 0] * multf / r_gain + AUG_BLACK_LEVEL,
        0,
        AUG_WHITE_LEVEL,
    )
    gc = np.clip(
        rgb_tensor[:, :, :, 1] * multf + AUG_BLACK_LEVEL,
        0,
        AUG_WHITE_LEVEL,
    )
    bc = np.clip(
        rgb_tensor[:, :, :, 2] * multf / b_gain + AUG_BLACK_LEVEL,
        0,
        AUG_WHITE_LEVEL,
    )

    bayer_tensor = np.zeros((T, 1, H, W), dtype=np.int32)

    if BAYER_PATTERN == "MONO":
        bayer_tensor[:, 0, :, :] = (rc * 0.299 + gc * 0.587 + bc * 0.114).astype(
            np.uint32
        )
    elif BAYER_PATTERN == "RGGB":
        bayer_tensor[:, 0, 0:H:2, 0:W:2] = rc[:, 0:H:2, 0:W:2]
        bayer_tensor[:, 0, 0:H:2, 1:W:2] = gc[:, 0:H:2, 1:W:2]
        bayer_tensor[:, 0, 1:H:2, 0:W:2] = gc[:, 1:H:2, 0:W:2]
        bayer_tensor[:, 0, 1:H:2, 1:W:2] = bc[:, 1:H:2, 1:W:2]
        wb_gain = np.array([[r_gain, 1.0], [1.0, b_gain]])
    elif BAYER_PATTERN == "BGGR":
        bayer_tensor[:, 0, 0:H:2, 0:W:2] = bc[:, 0:H:2, 0:W:2]
        bayer_tensor[:, 0, 0:H:2, 1:W:2] = gc[:, 0:H:2, 1:W:2]
        bayer_tensor[:, 0, 1:H:2, 0:W:2] = gc[:, 1:H:2, 0:W:2]
        bayer_tensor[:, 0, 1:H:2, 1:W:2] = rc[:, 1:H:2, 1:W:2]
        wb_gain = np.array([[b_gain, 1.0], [1.0, r_gain]])
    elif BAYER_PATTERN == "GRBG":
        bayer_tensor[:, 0, 0:H:2, 0:W:2] = gc[:, 0:H:2, 0:W:2]
        bayer_tensor[:, 0, 0:H:2, 1:W:2] = rc[:, 0:H:2, 1:W:2]
        bayer_tensor[:, 0, 1:H:2, 0:W:2] = bc[:, 1:H:2, 0:W:2]
        bayer_tensor[:, 0, 1:H:2, 1:W:2] = gc[:, 1:H:2, 1:W:2]
        wb_gain = np.array([[1.0, r_gain], [b_gain, 1.0]])
    elif BAYER_PATTERN == "GBRG":
        bayer_tensor[:, 0, 0:H:2, 0:W:2] = gc[:, 0:H:2, 0:W:2]
        bayer_tensor[:, 0, 0:H:2, 1:W:2] = bc[:, 0:H:2, 1:W:2]
        bayer_tensor[:, 0, 1:H:2, 0:W:2] = rc[:, 1:H:2, 0:W:2]
        bayer_tensor[:, 0, 1:H:2, 1:W:2] = gc[:, 1:H:2, 1:W:2]
        wb_gain = np.array([[1.0, b_gain], [r_gain, 1.0]])
    else:
        raise NotImplementedError

    return bayer_tensor, wb_gain

def rgb2bayer(rgb_tensor):
    multf, r_gain, b_gain = getRandomGain()
    return applyGain(rgb_tensor, multf, r_gain, b_gain)

def add_noise(img):
    bayer, wb_gain = rgb2bayer(img)
    noise = noiseFromVariance(bayer)

    noise_order = NOISE_ORDER - (
        NOISE_ORDER - 1.0
    ) * np.power(
        np.clip(
            (img.astype(np.float32) - AUG_BLACK_LEVEL)
            / AUG_WHITE_LEVEL,
            0,
            1.0,
        ),
        0.2,
    )
    noisy_bayer = np.clip(
        noise * noise_order + img.astype(np.float32),
        0,
        AUG_WHITE_LEVEL,
    )
    return noisy_bayer

img = cv2.imread("/home/luoleyouluole/Image-Restoration-Experiments/data/rit_hdr4000/Ahwahnee_Great_Lounge.hdr", -1).astype(np.float32)
print_min_max(img)
noisy = add_noise(img)
print_min_max(noisy)
