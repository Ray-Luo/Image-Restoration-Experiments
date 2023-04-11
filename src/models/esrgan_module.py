from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric


class ESRGANLitModule(LightningModule):
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
        self.save_hyperparameters(logger=False)

        self.net = net.net_d
        self.net_d = net.net_g

        # loss function
        self.criterion = loss

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()


    def forward(self, x: torch.Tensor, degrade: bool = False) -> torch.Tensor:
        return self.net(x, degrade)

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

    # def training_step(self, batch: Any, batch_idx: int):
    #     loss = self.model_step(batch)

    #     # update and log metrics
    #     self.train_loss(loss)
    #     self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)

    #     # we can return here dict with any tensors
    #     # and then read it in some callback or in `training_epoch_end()` below
    #     # remember to always return loss from `training_step()` or backpropagation will fail!
    #     return {"loss": loss}

    def training_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        optimizer_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:

        output, _ = self.forward(batch, degrade=False)

        # Log tensorboard images
        if self.global_step % 100 == 0:
            try:
                self.logger.experiment.add_images(
                    "hr_image", batch["hr_image"], self.global_step
                )
                self.logger.experiment.add_images(
                    "lr_image", batch["lr_image"], self.global_step
                )
                self.logger.experiment.add_images(
                    "degraded", degraded, self.global_step
                )
                self.logger.experiment.add_images("output", output, self.global_step)
            except AttributeError:
                print(self.logger)
                print(self.logger.experiment)

        # Train generator
        if optimizer_idx == 0:

            # pixel loss
            l_g_pix = self.criterion(output, batch["hq"])
            self.log("pixel_loss", l_g_pix, logger=True)

            # perceptual_loss
            l_g_precep, l_g_style = 0, 0 #self.loss.perceptual_loss(output, batch["hq"])
            self.log("percep_loss", l_g_precep, logger=True)
            self.log("style_loss", l_g_style, logger=True)

            # GAN loss
            fake_g_pred = self.model.net_d(output)
            l_g_gan = self.loss.gan_loss(fake_g_pred, True, is_disc=False)
            self.log("g_gan_loss", l_g_gan, logger=True)

            # Total loss
            l_g_total = l_g_pix + l_g_precep + l_g_style + l_g_gan
            self.log("total_gen_loss", l_g_total, logger=True)
            # return l_g_total
            return {"l_g_total": l_g_total}

        # Train discriminator
        if optimizer_idx == 1:

            # real
            real_d_pred = self.model.net_d(batch["hq"])
            l_d_real = self.loss.gan_loss(real_d_pred, True, is_disc=True)

            # fake
            fake_d_pred = self.model.net_d(output)
            l_d_fake = self.loss.gan_loss(fake_d_pred, False, is_disc=True)

            d_loss = (l_d_real + l_d_fake) / 2
            self.log("d_loss", d_loss, logger=True)

            # return d_loss
            return {"d_loss": d_loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from all batches of the epoch
        # this may not be an issue when training on mnist
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs

        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)

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
        lr_scheduler_g = hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer_g)
        lr_scheduler_d = hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer_d)
        return [optimizer_g, optimizer_d], [lr_scheduler_g, lr_scheduler_d]



        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = EDSRLitModule(None, None, None)
