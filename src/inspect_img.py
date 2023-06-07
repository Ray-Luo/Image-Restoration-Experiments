import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np
from process_hdr import print_min_max, save_hdr
from tqdm import tqdm




# folder_path = "/home/luoleyouluole/Image-Restoration-Experiments/data/" # replace with the path to your image folder
# save_path = "/home/luoleyouluole/Image-Restoration-Experiments/data/hdr_data/test_d_4x"
# file_list = os.listdir(folder_path)
# file_list.sort()

# for file_name in file_list:
#     if not file_name.endswith(".hdr"):
#         continue
#     img = cv2.imread(os.path.join(folder_path, file_name), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
#     min_val = np.min(img)
#     max_val = np.max(img)
#     print(min_val, max_val, file_name, img.shape)
    # img = np.power(img, 1/2.2)
    # save_hdr(img, folder_path, file_name.replace(".hdr", "_view.hdr"))
    # assert min_val >= 0. and max_val <= 1.0, print(os.path.join(folder_path, file_name))


# img /= np.max(img)
img = cv2.imread("/home/luoleyouluole/Image-Restoration-Experiments/data/test_noise_hdr/Artist_Palette_noise_exr2hdr.hdr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
img = img / 4000.0
img = np.clip(img, 0., 1.)
img = np.power(img, 1/2.2)
save_hdr(img, "/home/luoleyouluole/Image-Restoration-Experiments/", "Artist_Palette_noise_exr2hdr.hdr")


# draw_histogram(img, "GT", "./")
    # print_min_max(img)

"""
607 + 667 + 472 + 495 + 223
"""
