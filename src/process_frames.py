import os
import numpy as np
import cv2
from process_hdr import print_min_max

def pq2lin( V ):
    """ Convert from PQ-encoded values V (between 0 and 1) to absolute linear values (between 0.005 and 10000)
    """
    Lmax = 10000
    n    = 0.15930175781250000
    m    = 78.843750000000000
    c1   = 0.83593750000000000
    c2   = 18.851562500000000
    c3   = 18.687500000000000

    im_t = np.power(np.maximum(V,0),1/m)
    L = Lmax * np.power(np.maximum(im_t-c1,0)/(c2-c3*im_t), 1/n)
    L = np.clip(L, 0.005, 4000.0).astype('uint16')
    return L


# Directory where the frames are stored
input_dir = '/home/luoleyouluole/Image-Restoration-Experiments/data/LIVE_HDR_Public/frames'

# Directory where you want to save processed frames
output_dir = '/home/luoleyouluole/Image-Restoration-Experiments/data/LIVE_HDR_Public/processed_frames'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Loop over all jpg files in the directory
for filename in os.listdir(input_dir):
    if filename.endswith(".png"):
        # Construct full file path
        file_path = os.path.join(input_dir, filename)

        # Read the image data
        img = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)

        # Check if image is not already 16-bit per channel
        assert img.dtype == np.uint16

        try:
            # process the image
            img = pq2lin(img)

        assert img.dtype == np.uint16

        # Save the processed image data
        output_file_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_file_path, img)
