import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np
from process_hdr import print_min_max




# folder_path = "/home/luoleyouluole/Image-Restoration-Experiments/data/hdr_data/test" # replace with the path to your image folder
# save_path = "/home/luoleyouluole/Image-Restoration-Experiments/data/hdr_data/test_d_4x"
# file_list = os.listdir(folder_path)
# file_list.sort()

# for file_name in file_list:
#     downsample4x(folder_path, file_name, save_path)

img = cv2.imread("/home/luoleyouluole/Image-Restoration-Experiments/A009C050_1410316B.00043597.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)#.astype(np.float32)
# img = torch.tensor(img)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# img /= np.max(img)
# img = np.power(img, 1/2.2)
# save_hdr(img, "/home/luoleyouluole/Image-Restoration-Experiments/", "GT.hdr")


# draw_histogram(img, "GT", "./")
print_min_max(img)
