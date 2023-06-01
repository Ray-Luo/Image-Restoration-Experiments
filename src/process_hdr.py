import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from types import SimpleNamespace

SEED = 12345
NUMBER_AUG = 10
FOCUS_ORDER = 3

cfg = {
    "enable_dpc": False,
    "bayer_binning": False,
    "NOISE_MODE": "VAR", #option: CDF, VAR
    "CROP_HEIGHT": 512,
    "CROP_WIDTH": 512,
    "RAW_HEIGHT": 3024,
    "RAW_WIDTH" : 4038,
    "RAW_BITS" : 10,
    "meta_padding": 8076,
    "AUG_WHITE_LEVEL": 65535,
    "AUG_BLACK_LEVEL": 4096,
    "BLACK_LEVEL": 256,
    "WHITE_LEVEL": 4095,
    "BLACK_OFFSET": 16,
    "BAYER_PATTERN": "RGGB",
    "NOISE_ORDER": 1.0
}

cfg = SimpleNamespace(**cfg)

def getRandomMultiplier():
    np.random.seed()
    dice = np.random.randint(10)
    max_multf = (
        cfg.AUG_WHITE_LEVEL - cfg.AUG_BLACK_LEVEL
    ) * 1.0 / 255.0 - 1
    multf = max_multf + 1
    if dice == 1:
        multf = 1 + max_multf * np.random.random_sample()
    elif dice > 1:
        multf = 1 + max_multf * (np.random.random_sample() ** FOCUS_ORDER)
    return multf

def getRandomGain():
    multf = getRandomMultiplier()
    r_gain = random.uniform(1.0, 3.0)
    b_gain = random.uniform(1.0, 3.0)
    if cfg.BAYER_PATTERN == "MONO":
        r_gain = 1.0
        b_gain = 1.0
    return multf, r_gain, b_gain

def applyGain(rgb_tensor, multf, r_gain, b_gain):
    wb_gain = np.zeros((2, 2), dtype=np.float32)
    T, H, W, C = rgb_tensor.shape
    rc = np.clip(
        rgb_tensor[:, :, :, 0] * multf / r_gain + cfg.AUG_BLACK_LEVEL,
        0,
        cfg.AUG_WHITE_LEVEL,
    )
    gc = np.clip(
        rgb_tensor[:, :, :, 1] * multf + cfg.AUG_BLACK_LEVEL,
        0,
        cfg.AUG_WHITE_LEVEL,
    )
    bc = np.clip(
        rgb_tensor[:, :, :, 2] * multf / b_gain + cfg.AUG_BLACK_LEVEL,
        0,
        cfg.AUG_WHITE_LEVEL,
    )

    bayer_tensor = np.zeros((T, 1, H, W), dtype=np.int32)

    if cfg.BAYER_PATTERN == "MONO":
        bayer_tensor[:, 0, :, :] = (rc * 0.299 + gc * 0.587 + bc * 0.114).astype(
            np.uint32
        )
    elif cfg.BAYER_PATTERN == "RGGB":
        bayer_tensor[:, 0, 0:H:2, 0:W:2] = rc[:, 0:H:2, 0:W:2]
        bayer_tensor[:, 0, 0:H:2, 1:W:2] = gc[:, 0:H:2, 1:W:2]
        bayer_tensor[:, 0, 1:H:2, 0:W:2] = gc[:, 1:H:2, 0:W:2]
        bayer_tensor[:, 0, 1:H:2, 1:W:2] = bc[:, 1:H:2, 1:W:2]
        wb_gain = np.array([[r_gain, 1.0], [1.0, b_gain]])
    elif cfg.BAYER_PATTERN == "BGGR":
        bayer_tensor[:, 0, 0:H:2, 0:W:2] = bc[:, 0:H:2, 0:W:2]
        bayer_tensor[:, 0, 0:H:2, 1:W:2] = gc[:, 0:H:2, 1:W:2]
        bayer_tensor[:, 0, 1:H:2, 0:W:2] = gc[:, 1:H:2, 0:W:2]
        bayer_tensor[:, 0, 1:H:2, 1:W:2] = rc[:, 1:H:2, 1:W:2]
        wb_gain = np.array([[b_gain, 1.0], [1.0, r_gain]])
    elif cfg.BAYER_PATTERN == "GRBG":
        bayer_tensor[:, 0, 0:H:2, 0:W:2] = gc[:, 0:H:2, 0:W:2]
        bayer_tensor[:, 0, 0:H:2, 1:W:2] = rc[:, 0:H:2, 1:W:2]
        bayer_tensor[:, 0, 1:H:2, 0:W:2] = bc[:, 1:H:2, 0:W:2]
        bayer_tensor[:, 0, 1:H:2, 1:W:2] = gc[:, 1:H:2, 1:W:2]
        wb_gain = np.array([[1.0, r_gain], [b_gain, 1.0]])
    elif cfg.BAYER_PATTERN == "GBRG":
        bayer_tensor[:, 0, 0:H:2, 0:W:2] = gc[:, 0:H:2, 0:W:2]
        bayer_tensor[:, 0, 0:H:2, 1:W:2] = bc[:, 0:H:2, 1:W:2]
        bayer_tensor[:, 0, 1:H:2, 0:W:2] = rc[:, 1:H:2, 0:W:2]
        bayer_tensor[:, 0, 1:H:2, 1:W:2] = gc[:, 1:H:2, 1:W:2]
        wb_gain = np.array([[1.0, b_gain], [r_gain, 1.0]])
    else:
        raise NotImplementedError

    return bayer_tensor, wb_gain

def rgb2bayer(rgb_tensor):
    multf, r_gain, b_gain = getRandomGain()
    return applyGain(rgb_tensor, multf, r_gain, b_gain)

def print_min_max(path: str, name: str):
    # Read the image using OpenCV
    img = cv2.imread(path, -1).astype(np.float32)

    # Get the minimum and maximum pixel values
    min_val = np.min(img)
    max_val = np.max(img)

    # Print the results
    print(min_val, "  ", max_val, "  ", img.shape, "  ", name)


def print_min_max(img: np.array):

    # Get the minimum and maximum pixel values
    min_val = np.min(img)
    max_val = np.max(img)

    # Print the results
    print(min_val, "  ", max_val, "  ", img.shape)


def save_hdr(img: np.array, img_folder: str, name: str):
    # print(os.path.join(img_folder, name))
    cv2.imwrite(os.path.join(img_folder, name), img, [cv2.IMWRITE_HDR_COMPRESSION, 0])


def augment(img_folder: str, name: str):
    img = cv2.imread(os.path.join(img_folder, name), -1).astype(np.float32)

    seen = set()
    random.seed(SEED)
    rand_num = random.uniform(1.0/8.0, 8.0)

    while len(seen) < NUMBER_AUG:
        if rand_num not in seen:
            seen.add(rand_num)
            img *= rand_num
            processed_img = process_rit(img)
            new_name = name.split('.')[0] + '_aug_' + str(len(seen)) + '.hdr'
            save_hdr(processed_img, img_folder, new_name)
            rand_num = random.uniform(1.0/8.0, 8.0)


def process_rit(img: np.array):
    print_min_max(img)
    img_g = img[:,:,1]
    value_at_99_percentile = np.percentile(img_g, 99)
    img /= value_at_99_percentile
    img *= 4000.0
    img = np.clip(img, 0.05, 4000.0)

    print_min_max(img)
    print("**********")

    return img


def process_save(img_folder: str, name: str, save_path: str):
    img = cv2.imread(os.path.join(img_folder, name), -1).astype(np.float32)
    print_min_max(img)
    # img_g = img[:,:,1]
    # value_at_99_percentile = np.percentile(img_g, 99)
    # img /= value_at_99_percentile
    # img *= 4000.0
    img = np.clip(img, 0.05, 4000.0)

    print_min_max(img)
    print("**********")

    new_name = name
    # new_name = name.split('.')[0] + "_processed.hdr"

    save_hdr(img, save_path, new_name)


def downsample2x(img_folder: str, name: str):
    img = cv2.imread(os.path.join(img_folder, name), -1).astype(np.float32)
    downscaled = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    new_name = name.split('.')[0] + "_2x.hdr"
    save_hdr(downscaled, img_folder, new_name)


def downsample4x(img_folder: str, name: str, save_path: str):
    img = cv2.imread(os.path.join(img_folder, name), -1).astype(np.float32)
    assert np.min(img) >= 0.0, print(name)
    downscaled = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
    new_name = name.split('.')[0] + "_4x.hdr"
    save_hdr(downscaled, save_path, new_name)
    print_min_max(downscaled)

def downsample4x(img):
    downscaled = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
    return downscaled

def print_info(img_folder: str, name: str):
    img = cv2.imread(os.path.join(img_folder, name), -1).astype(np.float32)
    print_min_max(img)


def compare_content(a_folder, b_folder):
    for file_name in file_list:
        print(file_name, os.path.exists(os.path.join(b_folder, file_name.replace('4x_', ''))))

def draw_histogram(array, mode, save_path):
    array = torch.log(array + torch.ones_like(array) * 1e-5)
    array = array.squeeze(0).cpu().permute(1,2,0).detach().numpy()
    fig, ax = plt.subplots()
    sns.distplot(array.flatten(), bins=100, kde=False)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Log Histogram of {} Prediction'.format(mode))
    plt.savefig(os.path.join(save_path + '{}_prediction.png'.format(mode)))

def exr2hdr(img):
    green = img[:,:,1]
    p = np.percentile(green, 99)
    img = img / p
    img = img * 4000.0
    img = np.clip(img, 0.05, 4000.0)
    return img


# folder_path = "/home/luoleyouluole/Image-Restoration-Experiments/data/hdr_data/test" # replace with the path to your image folder
# save_path = "/home/luoleyouluole/Image-Restoration-Experiments/data/hdr_data/test_d_4x"
# file_list = os.listdir(folder_path)
# file_list.sort()

# for file_name in file_list:
#     downsample4x(folder_path, file_name, save_path)

# img = cv2.imread("/home/luoleyouluole/Image-Restoration-Experiments/data/rit_hdr4000/Ahwahnee_Great_Lounge.hdr", -1).astype(np.float32)
# img = torch.tensor(img)

# img /= np.max(img)
# img = np.power(img, 1/2.2)
# save_hdr(img, "/home/luoleyouluole/Image-Restoration-Experiments/", "GT.hdr")


# draw_histogram(img, "GT", "./")
