import cv2
import numpy as np

test_img = '/home/luoleyouluole/Image-Restoration-Experiments/data/res_dn_mu/Ben&Jerry\'s_exr2hdr_s001_raw_GT.hdr'
hdr_img = cv2.imread(test_img, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
print(hdr_img.shape)
