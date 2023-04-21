import subprocess

maskPath = "/home/luoleyouluole/fbsource/xplat/compphoto/gpuEngine/test/testData/matting/person1_mask.png"
guidancePath = "/home/luoleyouluole/fbsource/xplat/compphoto/gpuEngine/test/testData/matting/person1.png"
outputPath = "/home/luoleyouluole/Image-Restoration-Experiments/python_res.png"
r = 10
s = 0.25
e = 1e-4
gRgb = True
iRgb = False

# Call the C++ binary with arguments
subprocess.run(["/home/luoleyouluole/fbsource/buck-out/gen/aab7ed39/xplat/compphoto/gpuEngine/mattingToolFbcode", "--mask", maskPath, "--guidance", guidancePath, "--output", outputPath, "--epsilon", str(e), "--scaleF", str(s), "--radius", str(r), "--guidanceRGB", str(gRgb), "--inputRGB", str(iRgb)], capture_output=True)
