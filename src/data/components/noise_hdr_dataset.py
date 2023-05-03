import os

import cv2
import numpy as np
import torch
import imageio
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class RITDataset(Dataset):

    def __init__(self, representation, noise_profile, hq_path, lq_path):

        self.representation = representation

        self.hq = hq_path # os.path.join(data_root, 'rit_all/rit_processed_patchify')
        self.lq = lq_path # os.path.join(data_root, 'rit_all/rit_processed_2x_patchify')

        file_list = os.listdir(self.lq)
        self.img_names = [file_name for file_name in file_list]

        self.noise_profile_variance = imageio.imread(noise_profile)

    def noiseFromVariance(self, img):
        np.random.seed(int.from_bytes(os.urandom(4), byteorder="little"))
        shape = img.shape
        var = self.noise_profile_variance[img]
        noise = var * np.random.randn(*shape)
        return noise

    def add_noise(self, img):
        noise = self.noiseFromVariance(img)

        noise_order = self.data_cfg.NOISE_ORDER - (
            self.data_cfg.NOISE_ORDER - 1.0
        ) * np.power(
            np.clip(
                (img.astype(np.float32) - self.data_cfg.AUG_BLACK_LEVEL)
                / self.data_cfg.AUG_WHITE_LEVEL,
                0,
                1.0,
            ),
            0.2,
        )
        noisy_bayer = np.clip(
            noise * noise_order + img.astype(np.float32),
            0,
            self.data_cfg.AUG_WHITE_LEVEL,
        )
        return noisy_bayer

    def __getitem__(self, index):

        hq_image_path = os.path.join(self.hq, self.img_names[index]).replace('4x_', '')
        lq_image_path = os.path.join(self.lq, self.img_names[index])

        hq_img = cv2.imread(hq_image_path, -1).astype(np.float32)
        lq_img = cv2.imread(lq_image_path, -1).astype(np.float32)

        # transforms.ToTensor() is used for 8-bit [0, 255] range images; can't be used for [0, ∞) HDR images
        transform_hdr = transforms.Compose([
            transforms.Lambda(lambda img: torch.from_numpy(img.transpose((2, 0, 1)))),
            transforms.Lambda(lambda img: self.representation(img)),
        ])
        hq_tensor = transform_hdr(hq_img)
        lq_tensor = transform_hdr(lq_img)

        sample_dict = {
            "hq": hq_tensor,
            "lq": lq_tensor,
            "hq_path": hq_image_path,
            "lq_path": lq_image_path,
        }

        return sample_dict

    def __len__(self):
        return len(self.img_names)
