import subprocess
import time

maskPath = "/data/sandcastle/boxes/fbsource/xplat/compphoto/gpuEngine/test/testData/matting/toy-mask.png"
guidancePath = "/data/sandcastle/boxes/fbsource/xplat/compphoto/gpuEngine/test/testData/matting/toy.png"
outputPath = "/home/luoleyouluole/Image-Restoration-Experiments/python_res.png"
r = 10
s = 0.25
e = 1e-4
gRgb = True
iRgb = False

start_time = time.time()
# Call the C++ binary with arguments
subprocess.run(["/home/luoleyouluole/fbsource/buck-out/gen/aab7ed39/xplat/compphoto/gpuEngine/mattingToolFbcode", "--mask", maskPath, "--guidance", guidancePath, "--output", outputPath, "--epsilon", str(e), "--scaleF", str(s), "--radius", str(r), "--guidanceRGB", str(gRgb), "--inputRGB", str(iRgb)], capture_output=True)

end_time = time.time()
elapsed_time = end_time - start_time

# print the latency in seconds
print('Latency: {:.2f} seconds'.format(elapsed_time))
