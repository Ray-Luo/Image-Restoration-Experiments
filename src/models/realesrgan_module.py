from typing import Any, List

import hydra

import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric


class RealESRGANModule(LightningModule):
    """
    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        loss: torch.nn.Module,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.model = net

        # loss function
        self.criterion = loss
        self.loss = self.model.loss

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, x: torch.Tensor):
        return self.model.net_g(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

    def model_step(self, batch: Any):
        x, gt = batch["lq"], batch["hq"]
        loss = 0
        pred = self.forward(x)
        loss = self.criterion(pred, gt)

        return loss

    def training_step(
        self, batch: Any, batch_idx: int, optimizer_idx: int, *args, **kwargs
    ):
        x, gt = batch["lq"], batch["hq"]

        pred = self.forward(x)

        # Train generator
        if optimizer_idx == 0:

            # pixel loss
            l_g_pix = self.loss.pixel_loss(pred, gt)
            self.log("pixel_loss", l_g_pix, logger=True)

            # perceptual_loss
            l_g_precep, l_g_style = self.loss.perceptual_loss(pred, gt)
            self.log("percep_loss", l_g_precep, logger=True)
            self.log("style_loss", l_g_style, logger=True)

            # GAN loss
            fake_g_pred = self.model.net_d(pred)
            l_g_gan = self.loss.gan_loss(fake_g_pred, True, is_disc=False)
            self.log("g_gan_loss", l_g_gan, logger=True)

            # Total loss
            l_g_total = l_g_pix + l_g_precep + l_g_style + l_g_gan

            self.train_loss(l_g_total)
            self.log(
                "train/total_gen_loss",
                self.train_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
            return {"loss": l_g_total}

        # Train discriminator
        if optimizer_idx == 1:

            # real
            real_d_pred = self.model.net_d(gt)
            l_d_real = self.loss.gan_loss(real_d_pred, True, is_disc=True)

            # fake
            fake_d_pred = self.model.net_d(pred)
            l_d_fake = self.loss.gan_loss(fake_d_pred, False, is_disc=True)

            d_loss = (l_d_real + l_d_fake) / 2

            self.train_loss(d_loss)
            self.log(
                "train/d_loss",
                self.train_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )

            return {"loss": d_loss}

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        x, gt = batch["lq"], batch["hq"]
        pred = self.forward(x)
        loss = self.criterion(pred, gt)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        pass

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer_g = hydra.utils.instantiate(
            self.hparams.optimizer, self.model.net_g.parameters()
        )
        optimizer_d = hydra.utils.instantiate(
            self.hparams.optimizer, self.model.net_d.parameters()
        )
        lr_scheduler_g = hydra.utils.instantiate(self.hparams.scheduler, optimizer_g)
        lr_scheduler_d = hydra.utils.instantiate(self.hparams.scheduler, optimizer_d)
        return [optimizer_g, optimizer_d], [lr_scheduler_g, lr_scheduler_d]
