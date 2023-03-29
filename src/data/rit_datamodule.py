from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule

from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST

from .components.rit_dataset import RITDataset


class RITDataModule(LightningDataModule):

    def __init__(
        self,
        representation,
        hq_path,
        lq_path,
        seed,
        train_val_split: Tuple[int, int] = (0.9, 0.1),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.seed = seed

        dataset = RITDataset(representation, self.hparams.hq_path, self.hparams.lq_path)

        train_size = int(len(dataset) * train_val_split[0])
        val_size = len(dataset) - train_size

        self.data_train, self.data_val = random_split(
            dataset=dataset,
            lengths=[train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )


    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


if __name__ == "__main__":
    _ = RITDataModule()
