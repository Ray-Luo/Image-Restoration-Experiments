import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

IMG_SIZE = (256, 256)

class RITDataset(Dataset):

    def __init__(self, representation, data_root):

        self.representation = representation
        self.data_root = data_root

        file_list = os.listdir(self.data_root)
        self.img_names = [file_name for file_name in file_list]

    def __getitem__(self, index):

        image_path = os.path.join(
            self.data_root, self.img_names[index]
        )

        hdr_img = cv2.imread(image_path, -1).astype(np.float32)

        # transforms.ToTensor() is used for 8-bit [0, 255] range images; can't be used for [0, ∞) HDR images
        transform_hdr = transforms.Compose([
            transforms.Lambda(lambda img: torch.from_numpy(img.transpose((2, 0, 1)))),
        ])
        hdr_tensor = transform_hdr(hdr_img)

        noised = transforms.Compose([
            transforms.RandomNoise(mean=0, stddev=0.1, p=1.0),
        ])
        noised_tensor = noised(hdr_tensor)

        representation = transforms.Compose([
            transforms.Lambda(lambda img: self.representation(img)),
        ])

        hdr_tensor = representation(hdr_tensor)
        noised_tensor = representation(noised_tensor)

        sample_dict = {
            "noised_image": noised_tensor,
            "hdr_image": hdr_tensor,
            "path": image_path,
        }

        return sample_dict

    def __len__(self):
        return len(self.img_names)
