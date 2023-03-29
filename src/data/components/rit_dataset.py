import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class RITDataset(Dataset):

    def __init__(self, representation, data_root):

        self.representation = representation

        self.hq = os.path.join(data_root, 'rit_all/rit_processed_patchify')
        self.lq = os.path.join(data_root, 'rit_all/rit_processed_2x_patchify')

        file_list = os.listdir(self.lq)
        self.img_names = [file_name for file_name in file_list]

    def __getitem__(self, index):

        hq_image_path = os.path.join(self.hq, self.img_names[index]).replace('2x_', '')
        lq_image_path = os.path.join(self.lq, self.img_names[index])

        hq_img = cv2.imread(hq_image_path, -1).astype(np.float32)
        lq_img = cv2.imread(lq_image_path, -1).astype(np.float32)

        # transforms.ToTensor() is used for 8-bit [0, 255] range images; can't be used for [0, ∞) HDR images
        transform_hdr = transforms.Compose([
            transforms.Lambda(lambda img: torch.from_numpy(img.transpose((2, 0, 1)))),
            transforms.Lambda(lambda img: img / 4000.0),
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
