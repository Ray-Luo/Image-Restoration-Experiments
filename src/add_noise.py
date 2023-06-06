import numpy as np
import OpenEXR, Imath
import cv2
import torch
from tqdm import tqdm
from types import SimpleNamespace
from process_hdr import exr2hdr, addNoise, normalizeRaw, bayer2rgb, save_hdr, print_min_max, bayer2RGB, RandomNoiseAdder
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

# profile = np.load("/home/luoleyouluole/Image-Restoration-Experiments/Milan_NoiseProfile.npy")
noise = {
    "iso_range_start": 0.0,
    "iso_range_end": 1.0,
    "min_gain": 2.0,
    "max_gain": 15.5,
    "shot_noise_slope": [0.00011190175847216757],
    "shot_noise_intercept": [0.0004843544188776163],
    "shot_noise_stderr": [3.210092787806755e-06],
    "read_noise_slope": [4.738392326016082e-06],
    "read_noise_intercept": [-5.734861671087736e-06],
    "read_noise_stderr": [1.2087166682107576e-07]
}
noise_adder = RandomNoiseAdder(**noise)


gt_list = [
    "/home/luoleyouluole/Image-Restoration-Experiments/data/train",
    # "/home/luoleyouluole/Image-Restoration-Experiments/data/valid",
    # "/home/luoleyouluole/Image-Restoration-Experiments/data/test",
]
gt_list.sort()

noise_list = [
    "/home/luoleyouluole/Image-Restoration-Experiments/data/noise_train",
    # "/home/luoleyouluole/Image-Restoration-Experiments/data/noise_valid",
    # "/home/luoleyouluole/Image-Restoration-Experiments/data/noise_test",
]
noise_list.sort()

for gt_folder, noise_folder in tqdm(zip(gt_list, noise_list)):
    gt_imgs = os.listdir(gt_folder)
    gt_imgs.sort()
    for gt in tqdm(gt_imgs):
        img = cv2.imread(os.path.join(gt_folder, gt), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
        print_min_max(img, )
        gt_img = np.power(img, 1/2.2)
        save_hdr(gt_img, noise_folder, gt)

        img = img / 4000.0

        header = OpenEXR.Header(img.shape[1], img.shape[0])
        header['channels'] = dict([(c, Imath.Channel(Imath.PixelType(OpenEXR.FLOAT))) for c in "RGB"])

        # Create an OpenEXR file
        file = OpenEXR.OutputFile(os.path.join(noise_folder, gt), header)

        # Convert the numpy array data into a string
        red = (img[:,:,2].astype(np.float32)).tobytes()
        green = (img[:,:,1].astype(np.float32)).tobytes()
        blue = (img[:,:,0].astype(np.float32)).tobytes()

        # Write the image data to the exr file
        file.writePixels({'R': red, 'G': green, 'B': blue})

        img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2)

        data, shot_noise, read_noise = noise_adder(img)
        data = data.permute(0, 2, 3, 1).squeeze(0).numpy()

        header = OpenEXR.Header(data.shape[1], data.shape[0])
        header['channels'] = dict([(c, Imath.Channel(Imath.PixelType(OpenEXR.FLOAT))) for c in "RGB"])

        # Create an OpenEXR file
        file = OpenEXR.OutputFile(os.path.join(noise_folder, gt.replace(".hdr", "_noise.hdr")), header)

        # Convert the numpy array data into a string
        red = (data[:,:,2].astype(np.float32)).tobytes()
        green = (data[:,:,1].astype(np.float32)).tobytes()
        blue = (data[:,:,0].astype(np.float32)).tobytes()

        # Write the image data to the exr file
        file.writePixels({'R': red, 'G': green, 'B': blue})
        # data = exr2hdr(data)
        # print_min_max(data, )
        # noise_data = np.power(data, 1/2.2)
        # save_hdr(noise_data, noise_folder, gt.replace(".hdr", "_noise.hdr"))
        break
    break
