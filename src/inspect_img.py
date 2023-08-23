import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np
from process_hdr import print_min_max, save_hdr
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


A_COEFF = 0.456520040846940
B_COEFF = 1.070672820603428
L_MAX = 10000
N = 0.1593017578125
M = 78.84375
C1 = 0.8359375
C2 = 18.8515625
C3 = 18.6875
MAX = 10.8354  # PU(4000)


def original2pu(x):
    return np.log2(A_COEFF * x + B_COEFF) / MAX

def original2pq(x):
    im_t = np.power(np.clip(x, 0, L_MAX) / L_MAX, N)
    out = np.power((C2 * im_t + C1) / (1 + C3 * im_t), M)
    return out

def draw_histogram(array, mode, save_path):
    # array = array + 1e-5
    # array = np.log(array)
    fig, ax = plt.subplots()
    sns.distplot(array.flatten(), bins=100, kde=False)
    # if mode == "gt":
    #     plt.xlim(0, 1000)
    #     plt.ylim(0, 3e7)
    # else:
    #     plt.xlim(0, 1)
    #     plt.ylim(0, 3e7)
    # plt.xlim(-13, 900)
    # plt.ylim(0, 5.5e6)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of {}'.format(mode))
    plt.savefig(os.path.join(save_path + '{}'.format(mode)))

def visualize(img: np.array,root, name):
    img = img / 4000.0
    img = np.clip(img, 0., 1.)
    img = np.power(img, 1/2.2) * 255.0
    print(np.mean(img), np.min(img), np.max(img))
    # cv2.imwrite(os.path.join(root, name), img)

# folder = "/home/luoleyouluole/text/text_shot"
# save_folder = "/home/luoleyouluole/text/new_text_shot"
folder = "/home/luoleyouluole/Image-Restoration-Experiments/data/export/real"
imgs = os.listdir(folder)
# # img /= np.max(img)
# for img in imgs:
#     print(img)
#     image = cv2.imread(os.path.join(folder, img), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
#     if "mu_l1" in img:
#         image *= 4000.0
#         save_hdr(image,folder, img)

#     print_min_max(image)

# image = cv2.imread("/home/luoleyouluole/Image-Restoration-Experiments/data/test/Hancock_Kitchen_Inside.hdr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)

# blur = cv2.GaussianBlur(image, (51, 51), 0)

# save_hdr(blur, "/home/luoleyouluole/Image-Restoration-Experiments/data/export/gfn", "Hancock_Kitchen_Inside_blur.hdr")

for img in imgs:
    image = cv2.imread(os.path.join(folder, img), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)

    visualize(image, folder, img.split('.')[0] + "_gammma.png")


# draw_histogram(img, "GT", "./")
    # print_min_max(img)

"""
607 + 667 + 472 + 495 + 223
"""
