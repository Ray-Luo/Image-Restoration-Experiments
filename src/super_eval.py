import hydra
import pyrootutils
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from data.components.rit_dataset import RITDataset
from eval_util import ssim, psnr
from tqdm import tqdm
import torch.nn.functional as F
from process_hdr import save_hdr, print_min_max
import torch
import numpy as np
import torchvision.transforms as transforms
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

import warnings


# Ignore all warnings
warnings.filterwarnings("ignore")


pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from src import utils

log = utils.get_pylogger(__name__)

A_COEFF = 0.456520040846940
B_COEFF = 1.070672820603428
L_MAX = 10000
N = 0.1593017578125
M = 78.84375
C1 = 0.8359375
C2 = 18.8515625
C3 = 18.6875

def original2linear(x):
    return x / 4000.0

def linear2original(x):
    return x * 4000.0

def identity(x):
    return x

def original2log(x):
    return torch.log(torch.max(x, torch.ones_like(x) * 1e-5))

def log2original(x):
    x = torch.exp(x)
    return x

def original2pu(x):
    return torch.log2(A_COEFF * x + B_COEFF)

def pu2original(x):
    A_COEFF = 0.456520040846940
    B_COEFF = 1.070672820603428
    return (torch.pow(2.0, x) - B_COEFF) / A_COEFF

def original2pq(x):
    im_t = torch.pow(torch.clip(x, 0, L_MAX) / L_MAX, N)
    out = torch.pow((C2 * im_t + C1) / (1 + C3 * im_t), M)
    return out

def pq2original(x):
    im_t = torch.pow(torch.maximum(x, torch.zeros_like(x)),1 / M)
    out = L_MAX * torch.pow(torch.maximum(im_t - C1, torch.zeros_like(x))/(C2 - C3 * im_t), 1 / N)
    return out

def draw_histogram(array, mode, save_path):
    array += 1e-5
    array = np.log(array)
    fig, ax = plt.subplots()
    sns.distplot(array.flatten(), bins=100, kde=False)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Log Histogram of {}'.format(mode))
    plt.savefig(os.path.join(save_path + '{}'.format(mode)))

def visualize(img: np.array,root, name):
    img /= 4000.0
    img = np.clip(img, 0., 1.)
    img = np.power(img, 1/2.2)
    save_hdr(img, root, name)
    # print_min_max(img)

def cal_psnr(pred: np.array, gt: np.array):
    return psnr(pred, gt)

def cal_ssim(pred: np.array, gt: np.array):
    return ssim(pred, gt)

def check_if_load_correct(experiemnt_signiture: str, tag_file_path: str):
    with open(tag_file_path, "r") as file:
        contents = file.read()

    for item in experiemnt_signiture.split('_'):
        if item not in contents:
            return False

    return True

def get_transform(experiemnt_signiture: str):
    representation, loss = experiemnt_signiture.split('_')
    if representation == "linear":
        return original2linear, linear2original

    elif representation == "log":
        return original2log, log2original

    elif representation == "pu":
        return original2pu, pu2original

    elif representation == "pq":
        return original2pq, pq2original

    else:
        raise NotImplementedError

transform_hdr = transforms.Compose([
    transforms.Lambda(lambda img: torch.from_numpy(img.transpose((2, 0, 1)))),
    transforms.Lambda(lambda img: original2pq(img)),
])


@utils.task_wrapper
def evaluate(cfg: DictConfig):

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log_path = hydra.utils.instantiate(cfg.log_path)
    results_save_path = cfg.results_save_path

    report = ""

    for experiment, path in tqdm(log_path.items()):
        report += "**************************  " + experiment + "  **************************\n"
        tag_file = os.path.join(path,"tags.log")
        assert check_if_load_correct(experiment, tag_file) is True, "Checkpoint file is not correct!"

        model_path = os.path.join(path,"checkpoints/last.ckpt")
        model = model.load_from_checkpoint(model_path)
        net = model.cuda()
        net.eval()

        file_list = os.listdir(cfg.data.lq_path)
        file_list.sort()

        transform_fn, inverse_fn = get_transform(experiment)
        transform_hdr = transforms.Compose([
            transforms.Lambda(lambda img: torch.from_numpy(img.transpose((2, 0, 1)))),
            transforms.Lambda(lambda img: transform_fn(img)),
        ])

        for file_name in file_list:
            lq_path = os.path.join(cfg.data.lq_path, file_name)
            lq_img = cv2.imread(lq_path, -1).astype(np.float32)
            lq = transform_hdr(lq_img).unsqueeze(0).cuda()

            hq_path = os.path.join(cfg.data.hq_path, file_name.replace("_4x", ""))
            gt = cv2.imread(hq_path, -1).astype(np.float32)
            file_name = file_name.replace("_4x", "").split('.')[0]
            report += "*********************  " + file_name + "  *********************\n"
            if not os.path.exists(os.path.join(results_save_path, file_name + "_GT.hdr")):
                visualize(gt, results_save_path, file_name + "_GT.hdr")
                draw_histogram(gt, file_name + "_GT", results_save_path)


            res_naive = F.interpolate(lq, size=(lq.shape[2]*4, lq.shape[3]*4), mode='nearest', align_corners=None)
            res_naive = inverse_fn(res_naive).squeeze(0).cpu().permute(1,2,0).detach().numpy()
            psnr = cal_psnr(res_naive, gt)
            ssim = cal_ssim(res_naive, gt)
            report += "navie -- PSNR = {:.5f}, ssim = {:.5f}\n".format(psnr, ssim)
            if not os.path.exists(os.path.join(results_save_path, file_name + "_naive.hdr")):
                draw_histogram(res_naive, file_name + "_nearest-neighbor", results_save_path)
                visualize(res_naive, results_save_path, file_name + "_naive.hdr")

            with torch.no_grad():
                pred = net(lq)
                res_img = inverse_fn(pred).squeeze(0).cpu().permute(1,2,0).detach().numpy()
                psnr = cal_psnr(res_img, gt)
                ssim = cal_ssim(res_img, gt)
                report += "{} -- PSNR = {:.5f}, ssim = {:.5f}\n".format(experiment, psnr, ssim)
                draw_histogram(res_img, file_name + "_Nets_pq", results_save_path)
                visualize(res_img, results_save_path, file_name + "_{}.hdr".format(experiment))

        report += "\n\n"

    with open(os.path.join(results_save_path, "report.log"), "w") as file:
        # Write a string to the file
        file.write(report)

    res_dict = {
    }

    object_dict = {
        "cfg": cfg,
        "res": res_dict
    }

    return {}, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
