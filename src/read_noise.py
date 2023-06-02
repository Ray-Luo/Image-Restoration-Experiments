import numpy as np
import OpenEXR, Imath
import cv2
import torch
from types import SimpleNamespace
from process_hdr import rgb2bayer, addNoise, normalizeRaw, bayer2rgb, save_hdr, print_min_max, bayer2RGB, RandomNoiseAdder
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

img = cv2.imread("/home/luoleyouluole/Image-Restoration-Experiments/A004C015_121104AF.00000000.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2)
data, shot_noise, read_noise = noise_adder(img)
data = data.permute(0, 2, 3, 1).squeeze(0).numpy()

# Create header
header = OpenEXR.Header(data.shape[1], data.shape[0])
header['channels'] = dict([(c, Imath.Channel(Imath.PixelType(OpenEXR.FLOAT))) for c in "RGB"])

# Create an OpenEXR file
file = OpenEXR.OutputFile('output.exr', header)

# Convert the numpy array data into a string
red = (data[:,:,2].astype(np.float32)).tobytes()
green = (data[:,:,1].astype(np.float32)).tobytes()
blue = (data[:,:,0].astype(np.float32)).tobytes()

# Write the image data to the exr file
file.writePixels({'R': red, 'G': green, 'B': blue})
