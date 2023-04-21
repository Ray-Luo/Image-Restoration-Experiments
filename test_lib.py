# example.py

import os
import ctypes
import ctypes.util

libs_path = "/home/luoleyouluole/Image-Restoration-Experiments/shared_libs/"

for file_name in os.listdir(libs_path):
    print(file_name,"*****")
    ctypes.util.find_library(os.path.join(libs_path, file_name))

lib = ctypes.cdll.LoadLibrary("libxplat_compphoto_gpuEngine_mattingToolCpyFbcode.so")
lib.applyMatting.argtypes = [
    ctypes.c_char_p, # string maskPath,
    ctypes.c_char_p, # string guidancePath,
    ctypes.c_char_p, # string outputPath,
    ctypes.c_float, # float e,
    ctypes.c_float, # float s,
    ctypes.c_int, # int r,
    ctypes.c_bool, # bool gRgb,
    ctypes.c_bool # bool iRgb,
]
lib.applyMatting.restype = ctypes.c_void_p


maskPath = "/data/users/luoleyouluole/fbsource/xplat/compphoto/gpuEngine/test/testData/matting/person1_mask.png"
guidancePath = "/data/users/luoleyouluole/fbsource/xplat/compphoto/gpuEngine/test/testData/matting/person1.png"
outputPath = "./python_tres.png"
r = 10
s = 0.25
e = 1e-4
gRgb = True
iRgb = False

lib.applyMatting(maskPath.encode(), guidancePath.encode(), outputPath.encode(), e, s, r, gRgb, iRgb)
