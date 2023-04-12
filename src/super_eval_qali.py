import hydra
import pyrootutils
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from data.components.rit_dataset import RITDataset
from eval_util import pu_ssim, pu_psnr
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


def pu2linear(x):
    A_COEFF = 0.456520040846940
    B_COEFF = 1.070672820603428
    return (torch.pow(2.0, x) - B_COEFF) / A_COEFF

def linear2original(x):
    return x * 4000.0

def original2linear(x):
    return x / 4000.0

def identity(x):
    return x

transform_hdr = transforms.Compose([
    transforms.Lambda(lambda img: torch.from_numpy(img.transpose((2, 0, 1)))),
    transforms.Lambda(lambda img: original2linear(img)),
])


@utils.task_wrapper
def evaluate(cfg: DictConfig):
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    net = model.net
    net.eval()

    representation = hydra.utils.instantiate(cfg.representation)
    loss_fn = hydra.utils.instantiate(cfg.loss)

    dataset = RITDataset(representation, cfg.data.hq_path, cfg.data.lq_path)
    dataloader = DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=False)
    dataloader = tqdm(dataloader)

    res_img = torch.ones(1, 3, 2868, 4312)
    res_naive = torch.ones(1, 3, 2868, 4312)
    x_index = 0
    y_index = 0

    res_list = []
    navie_list = []

    file_list = os.listdir(cfg.data.lq_path)
    file_list.sort()

    for file_name in tqdm(file_list):
        lq_path = os.path.join(cfg.data.lq_path, file_name)
        lq_img = cv2.imread(lq_path, -1).astype(np.float32)
        lq = transform_hdr(lq_img).unsqueeze(0)

        pred = net(lq)
        # pred = pu2linear(pred)
        bicubic_pred = F.interpolate(lq, size=(lq.shape[2]*4, lq.shape[3]*4), mode='nearest', align_corners=None)
        # bicubic_pred = pu2linear(bicubic_pred)

        res_list.append(pred)
        navie_list.append(bicubic_pred)


    h = 2868
    w = 4312
    crop_size = 384
    step = 192
    thresh_size = 0
    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)

    index = 0
    for x in h_space:
        for y in w_space:
            res_img[:,:, x:x + crop_size, y:y + crop_size] = res_list[index]
            res_naive[:, :, x:x + crop_size, y:y + crop_size] = navie_list[index]
            index += 1


    res_img = linear2original(res_img.squeeze(0).permute(1,2,0).detach().numpy())
    res_naive = linear2original(res_naive.squeeze(0).permute(1,2,0).detach().numpy())
    print_min_max(res_img)
    print_min_max(res_naive)
    save_hdr(res_img, "/home/luoleyouluole/Image-Restoration-Experiments/data", "res_img.hdr")
    save_hdr(res_naive, "/home/luoleyouluole/Image-Restoration-Experiments/data", "res_naive.hdr")

    test = res_img.flatten()
    fig, ax = plt.subplots()
    sns.distplot(test, bins=100, kde=False)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Nets Prediction')
    plt.savefig('./nets_prediction.png')

    test = res_naive.flatten()
    fig, ax = plt.subplots()
    sns.distplot(test, bins=100, kde=False)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Nearest-neighbor Prediction')
    plt.savefig('./nearest_prediction.png')


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
