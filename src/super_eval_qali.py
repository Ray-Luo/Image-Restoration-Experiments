import hydra
import pyrootutils
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from data.components.rit_dataset import RITDataset
from eval_util import pu_ssim, pu_psnr
from tqdm import tqdm
import torch.nn.functional as F
from process_hdr import save_hdr
import torch
import numpy as np

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

def linear2linear(x):
    return x


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
    res_naive = torch.zeros(1, 3, 2868, 4312)
    x_index = 0
    y_index = 0

    res_list = []
    navie_list = []

    save_hdr(res_img.numpy(), "/home/luoleyouluole/Image-Restoration-Experiments/data/", "res_img.hdr")
    save_hdr(res_naive.numpy(), "/home/luoleyouluole/Image-Restoration-Experiments/data/", "res_naive.hdr")

    for data in dataloader:
        hq = data['hq']
        lq = data['lq']
        hq_path = data['hq_path']


        pred = linear2linear(net(lq))
        bicubic_pred = F.interpolate(lq, scale_factor=4, mode='bilinear', align_corners=True).unsqueeze(0)

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


    res_img = res_img.squeeze(0).permute(1,2,0).detach().numpy()
    res_naive = res_naive.squeeze(0).permute(1,2,0).detach().numpy()
    save_hdr(res_img, "/home/luoleyouluole/Image-Restoration-Experiments/data", "res_img.hdr")
    save_hdr(res_naive, "/home/luoleyouluole/Image-Restoration-Experiments/data", "res_naive.hdr")

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
