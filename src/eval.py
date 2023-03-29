import hydra
import pyrootutils
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from data.components.rit_dataset import RITDataset
from eval_util import pu_ssim, pu_psnr
from tqdm import tqdm
import torch.nn.functional as F

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

    bicubic_loss = 0
    bicubic_psnr = 0
    bicubic_ssim = 0

    loss = 0
    psnr = 0
    ssim = 0
    for data in dataloader:
        hq = data['hq']
        lq = data['lq']

        pred = net(lq)
        bicubic_pred = F.interpolate(lq, scale_factor=2, mode='bicubic', align_corners=True)

        loss += loss_fn(pred, hq)
        bicubic_loss += loss_fn(bicubic_pred, hq)

        pred = pred.detach().numpy()
        hq = hq.detach().numpy()
        bicubic_pred = bicubic_pred.detach().numpy()

        psnr += pu_ssim(pred, hq)
        ssim += pu_psnr(pred, hq)

        bicubic_psnr += pu_ssim(bicubic_pred, hq)
        bicubic_ssim += pu_psnr(bicubic_pred, hq)

    loss /= len(dataset)
    psnr /= len(dataset)
    ssim /= len(dataset)

    bicubic_loss /= len(dataset)
    bicubic_psnr /= len(dataset)
    bicubic_ssim /= len(dataset)

    print("Averge loss -- bicubic vs net: ", bicubic_loss.item(), loss.item())
    print("Averge psnr -- bicubic vs net: ", bicubic_psnr, psnr)
    print("Averge ssim -- bicubic vs net: ", bicubic_ssim, ssim)

    res_dict = {
        "loss": loss.item(),
        "psnr": psnr,
        "loss": ssim,
        "bicubic_loss": bicubic_loss.item(),
        "bicubic_psnr": bicubic_psnr,
        "bicubic_loss": bicubic_ssim,
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

"""
Averge loss -- bicubic vs net:  12.168191909790039 1613.23681640625
Averge psnr -- bicubic vs net:  0.9997592414247578 0.6082884422196864
Averge ssim -- bicubic vs net:  77.08040852691732 32.634897858715235
"""
