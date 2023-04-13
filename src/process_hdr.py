import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
import torch

SEED = 12345
NUMBER_AUG = 10

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
    print(os.path.join(img_folder, name))
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
    downscaled = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    new_name = name.split('.')[0] + "_2x.hdr"
    save_hdr(downscaled, img_folder, new_name)


def downsample4x(img_folder: str, name: str, save_path: str):
    img = cv2.imread(os.path.join(img_folder, name), -1).astype(np.float32)
    downscaled = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
    new_name = name.split('.')[0] + "_4x.hdr"
    save_hdr(downscaled, save_path, new_name)


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


# folder_path = "/home/luoleyouluole/Image-Restoration-Experiments/data/rit_hdr4000/" # replace with the path to your image folder
# save_path = "/home/luoleyouluole/Image-Restoration-Experiments/data/rit_hdr4000_4x/"
# file_list = os.listdir(folder_path)

# for file_name in file_list:
    # print_info(folder_path, file_name)
    # process_save(folder_path, file_name, save_path)
    # augment(folder_path, file_name)
    # downsample4x(folder_path, file_name, save_path)
    # compare_content(save_path, folder_path)
    # break

# img = cv2.imread("/home/luoleyouluole/Image-Restoration-Experiments/data/rit_hdr4000/Ahwahnee_Great_Lounge.hdr", -1).astype(np.float32)
# img = torch.tensor(img)

# img /= np.max(img)
# img = np.power(img, 1/2.2)
# save_hdr(img, "/home/luoleyouluole/Image-Restoration-Experiments/", "GT.hdr")


# draw_histogram(img, "GT", "./")
