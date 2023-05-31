import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np
from process_hdr import print_min_max, exr2hdr, save_hdr, downsample4x
from tqdm import tqdm

save_path = "/home/luoleyouluole/Image-Restoration-Experiments/data/HDR_VIDEO_FRAME_4xd"


# folder_list = [
#     "/home/luoleyouluole/Image-Restoration-Experiments/data/1",
#     "/home/luoleyouluole/Image-Restoration-Experiments/data/2",
#     "/home/luoleyouluole/Image-Restoration-Experiments/data/Furniture",
#     "/home/luoleyouluole/Image-Restoration-Experiments/data/Nigh_Traffic_Flow",
#     "/home/luoleyouluole/Image-Restoration-Experiments/data/Night_Street",
# ]
folder_list = [
    "/home/luoleyouluole/Image-Restoration-Experiments/data/HDR_VIDEO_FRAME",
]
folder_list.sort()


for folder_path in tqdm(folder_list):
    file_list = os.listdir(folder_path)
    file_list.sort()
    for file_name in tqdm(file_list):
        img = cv2.imread(os.path.join(folder_path, file_name), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
        # img = exr2hdr(img)
        # file_name = file_name.replace(".exr", "**hdr").replace(".", "_").replace("**", ".")
        file_name = file_name.replace(".hdr", "_4dx.hdr")
        img = downsample4x(img)
        save_hdr(img, save_path, file_name)
    # assert min_val >= 0. and max_val <= 1.0, print(os.path.join(folder_path, file_name))


# img /= np.max(img)
# img = np.power(img, 1/2.2)
# save_hdr(img, "/home/luoleyouluole/Image-Restoration-Experiments/", "GT.hdr")


# draw_histogram(img, "GT", "./")
    # print_min_max(img)

"""
607 + 667 + 472 + 495 + 223
"""
