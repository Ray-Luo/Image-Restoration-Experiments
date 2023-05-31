import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np
from process_hdr import print_min_max
from tqdm import tqdm




folder_path = "/home/luoleyouluole/Image-Restoration-Experiments/data/Night_Street" # replace with the path to your image folder
# save_path = "/home/luoleyouluole/Image-Restoration-Experiments/data/hdr_data/test_d_4x"
file_list = os.listdir(folder_path)
file_list.sort()

for file_name in tqdm(file_list):
    img = cv2.imread(os.path.join(folder_path, file_name), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
    min_val = np.min(img)
    max_val = np.max(img)
    assert min_val >= 0. and max_val <= 1.0, print(os.path.join(folder_path, file_name))


# img /= np.max(img)
# img = np.power(img, 1/2.2)
# save_hdr(img, "/home/luoleyouluole/Image-Restoration-Experiments/", "GT.hdr")


# draw_histogram(img, "GT", "./")
    # print_min_max(img)

"""
607 + 667 + 472 + 495 + 223
"""
