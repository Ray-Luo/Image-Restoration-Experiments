from eval_util import pu_psnr, pu_ssim
import torch.nn.functional as F
from process_hdr import save_hdr
import torch
import numpy as np
import torchvision.transforms as transforms
import os
import cv2
from skimage import transform



lq_path = "/home/luoleyouluole/Image-Restoration-Experiments/data/test_3/lq"
hq_path = "/home/luoleyouluole/Image-Restoration-Experiments/data/test_3/hq"
file_list = os.listdir(lq_path)
file_list.sort()


def identity(x):
    return x

transform_fn, inverse_fn = identity, identity
transform_hdr = transforms.Compose([
    transforms.Lambda(lambda img: torch.from_numpy(img.transpose((2, 0, 1)))),
    transforms.Lambda(lambda img: transform_fn(img)),
])

for file_name in file_list:
    lq_img_path = os.path.join(lq_path, file_name)
    lq = cv2.imread(lq_img_path, -1).astype(np.float32)
    lq_tensor = transform_hdr(lq).unsqueeze(0)

    hq_img_path = os.path.join(hq_path, file_name.replace("_4x", ""))
    gt = cv2.imread(hq_img_path, -1).astype(np.float32)

    lq = transform.resize(lq, (lq.shape[0] * 4, lq.shape[1] * 4), order=0)
    save_hdr(lq, "/home/luoleyouluole/Image-Restoration-Experiments/data/", file_name)

    lq_tensor = F.interpolate(lq_tensor, scale_factor=4, mode='nearest', align_corners=None)
    lq_tensor = inverse_fn(lq_tensor).squeeze(0).cpu().permute(1,2,0).detach().numpy()

    diff = lq_tensor - lq
    print("diff", np.sum(diff))


    psnr = pu_psnr(lq, gt)
    ssim = pu_ssim(lq, gt)

    print("scikit-image", file_name, psnr, ssim)

    psnr = pu_psnr(lq_tensor, gt)
    ssim = pu_ssim(lq_tensor, gt)
    print("torch", file_name, psnr, ssim)
