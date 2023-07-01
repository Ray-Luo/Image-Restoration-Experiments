import os
import subprocess

# directory containing the images
folder = '/home/luoleyouluole/Image-Restoration-Experiments/data/test_aug'


test_img_folder = '/home/luoleyouluole/Image-Restoration-Experiments/data/res_mir'

imgs = os.listdir(folder)
imgs.sort()

navie_psnr_rgb = []
navie_psnr_y = []
navie_cvvdp = []

linear_psnr_rgb = []
linear_psnr_y = []
linear_cvvdp = []

log_psnr_rgb = []
log_psnr_y = []
log_cvvdp = []

pu_psnr_rgb = []
pu_psnr_y = []
pu_cvvdp = []

pq_psnr_rgb = []
pq_psnr_y = []
pq_cvvdp = []

for filename in imgs:
    if "Artist_Palette" in filename or "Bigfoot_Pass" in filename:
        continue
    file_name = filename.split(".")[0]
    if "'" in file_name or "&" in file_name:
        file_name = file_name.replace("'", "\\'").replace("&", "\\&")
    reference_name = file_name + "_raw_GT.hdr"
    reference_img = os.path.join(test_img_folder, reference_name)
    test_names = [
        # file_name + "_raw_naive.hdr",
        file_name + "_raw_linear_l1.hdr",
        file_name + "_raw_pq_l1.hdr",
        file_name + "_raw_pu_l1.hdr",
    ]

    for test_name in test_names:
        test_img = os.path.join(test_img_folder, test_name)

        # print(test_img, reference_img)

        command = f"cvvdp --test {test_img} --ref {reference_img} --display standard_hdr_linear_zoom --metric pu-psnr-rgb pu-psnr-y cvvdp  --quiet"
        ret_value = subprocess.run(command, shell=True, capture_output=True, text=True)
        psnr_rgb, psnr_y, cvvdp = ret_value.stdout.split()
        print(test_name, psnr_rgb, psnr_y, cvvdp)

        if "naive" in test_name:
            navie_psnr_rgb.append(float(psnr_rgb))
            navie_psnr_y.append(float(psnr_y))
            navie_cvvdp.append(float(cvvdp))
        elif "linear" in test_name:
            linear_psnr_rgb.append(float(psnr_rgb))
            linear_psnr_y.append(float(psnr_y))
            linear_cvvdp.append(float(cvvdp))
        elif "log" in test_name:
            log_psnr_rgb.append(float(psnr_rgb))
            log_psnr_y.append(float(psnr_y))
            log_cvvdp.append(float(cvvdp))
        elif "pq" in test_name:
            pq_psnr_rgb.append(float(psnr_rgb))
            pq_psnr_y.append(float(psnr_y))
            pq_cvvdp.append(float(cvvdp))
        elif "pu" in test_name:
            pu_psnr_rgb.append(float(psnr_rgb))
            pu_psnr_y.append(float(psnr_y))
            pu_cvvdp.append(float(cvvdp))
        else:
            raise ValueError(f"Unknown test name: {test_name}")

    print("\n")

print("navie_psnr_rgb=", navie_psnr_rgb)
print("linear_psnr_rgb=", linear_psnr_rgb)
print("log_psnr_rg=", log_psnr_rgb)
print("pu_psnr_rgb=", pu_psnr_rgb)
print("pq_psnr_rgb=", pq_psnr_rgb)

print("navie_psnr_y=", navie_psnr_y)
print("linear_psnr_y=", linear_psnr_y)
print("log_psnr_y=", log_psnr_y)
print("pu_psnr_y=", pu_psnr_y)
print("pq_psnr_y=", pq_psnr_y)

print("navie_cvvdp=", navie_cvvdp)
print("linear_cvvdp=", linear_cvvdp)
print("log_cvvdp=", log_cvvdp)
print("pu_cvvdp=", pu_cvvdp)
print("pq_cvvdp=", pq_cvvdp)



# import scipy.stats as stats
