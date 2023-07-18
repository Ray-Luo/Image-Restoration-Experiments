import hydra
import pyrootutils
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from data.components.rit_dataset import RITDataset
from eval_util import pu_psnr, pu_ssim
from tqdm import tqdm
import torch.nn.functional as F
from process_hdr import save_hdr, print_min_max, process_rit
import torch
import numpy as np
import torchvision.transforms as transforms
import os
import cv2
import matplotlib.pyplot as plt
import warnings


A_COEFF = 0.456520040846940
B_COEFF = 1.070672820603428
L_MAX = 10000
N = 0.1593017578125
M = 78.84375
C1 = 0.8359375
C2 = 18.8515625
C3 = 18.6875
MAX = 10.8354  # PU(4000)

def pq2original(V):
    Lmax = 10000
    n    = 0.15930175781250000
    m    = 78.843750000000000
    c1   = 0.83593750000000000
    c2   = 18.851562500000000
    c3   = 18.687500000000000

    im_t = np.power(np.maximum(V,0),1/m)
    L = Lmax * np.power(np.maximum(im_t-c1,0)/(c2-c3*im_t), 1/n)
    return L



images = "/home/luoleyouluole/Image-Restoration-Experiments/data/video"
for file_name in os.listdir(images):
    name = file_name.split('.')[0] + ".hdr"
    path = os.path.join(images, file_name)
    img = cv2.imread(path, -1).astype(np.float32)
    print_min_max(img)
    linear = pq2original(img)
    # hdr = process_rit(linear)
    print_min_max(linear)
    save_hdr(linear, "/home/luoleyouluole/Image-Restoration-Experiments/data/test/", name)
