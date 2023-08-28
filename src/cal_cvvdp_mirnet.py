import os
import subprocess
import cv2
from process_hdr import save_exr, save_hdr
import numpy as np
from tqdm import tqdm


test_img_folder = '/home/luoleyouluole/Image-Restoration-Experiments/data/res_mir'

imgs = os.listdir(test_img_folder)
imgs.sort()
test_imgs = []
for filename in imgs:
    if "_raw_GT.hdr" in filename and "_aug_" not in filename:
        test_imgs.append(filename)

navie_psnr_rgb = []
navie_psnr_y = []
navie_cvvdp = []
navie_ssim = []

linear_psnr_rgb = []
linear_psnr_y = []
linear_cvvdp = []
linear_ssim = []

log_psnr_rgb = []
log_psnr_y = []
log_cvvdp = []

pu_psnr_rgb = []
pu_psnr_y = []
pu_cvvdp = []
pu_ssim = []

pq_psnr_rgb = []
pq_psnr_y = []
pq_cvvdp = []
pq_ssim = []


linear_smape_psnr_rgb = []
linear_smape_psnr_y = []
linear_smape_cvvdp = []
linear_smape_ssim = []

linear_pu_psnr_rgb = []
linear_pu_psnr_y = []
linear_pu_cvvdp = []
linear_pu_ssim = []

linear_pq_psnr_rgb = []
linear_pq_psnr_y = []
linear_pq_cvvdp = []
linear_pq_ssim = []

linear_mu_psnr_rgb = []
linear_mu_psnr_y = []
linear_mu_cvvdp = []
linear_mu_ssim = []

mu_psnr_rgb = []
mu_psnr_y = []
mu_cvvdp = []
mu_ssim = []

pu21_psnr_rgb = []
pu21_psnr_y = []
pu21_cvvdp = []
pu21_ssim = []

report  = ""


for file_name in tqdm(test_imgs):
    if "Artist_Palette" in file_name or "Bigfoot_Pass" in file_name or "Bandon_Sunset" in file_name:
        continue

    if "cargo_boat" in file_name or "skyscraper" in file_name or "urban_land" in file_name:
        continue

    reference_name = file_name
    reference_img = os.path.join(test_img_folder, reference_name)
    test_names = [
        file_name.replace("_GT", "_mu_l1"),
        file_name.replace("_GT", "_pu21_l1"),
        file_name.replace("_GT", "_naive"),
        file_name.replace("_GT", "_linear_l1"),
        file_name.replace("_GT", "_pu_l1"),
        file_name.replace("_GT", "_pq_l1"),
        file_name.replace("_GT", "_linear_pq"),
        file_name.replace("_GT", "_linear_pu"),
        file_name.replace("_GT", "_linear_smape"),
        file_name.replace("_GT", "_linear_mu"),
    ]

    img = cv2.imread(reference_img, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
    save_exr(img, test_img_folder, reference_name.replace(".hdr", ".exr"))
    reference_img_exr = os.path.join(test_img_folder, reference_name.replace(".hdr", ".exr"))

    for test_name in test_names:
        test_img = os.path.join(test_img_folder, test_name)


        # if "mu_l1" in test_img:
        #     hdr_img *= 4000.0
        # save_exr(hdr_img, test_img_folder, test_name.replace(".hdr", ".exr"))
        # exr_img = os.path.join(test_img_folder, test_name.replace(".hdr", ".exr"))

        if "_naive" in test_name:
            lq = cv2.GaussianBlur(img, (51, 51), 0)
            save_hdr(lq, test_img_folder, test_name)
            save_exr(lq, test_img_folder, test_name.replace(".hdr", ".exr"))
            exr_img = os.path.join(test_img_folder, test_name.replace(".hdr", ".exr"))

        else:
            hdr_img = cv2.imread(test_img, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
            save_exr(hdr_img, test_img_folder, test_name.replace(".hdr", ".exr"))
            exr_img = os.path.join(test_img_folder, test_name.replace(".hdr", ".exr"))

        exr_img_name = exr_img.replace("'", "\\'").replace("&", "\\&")
        reference_img_exr_name = reference_img_exr.replace("'", "\\'").replace("&", "\\&")

        command = f"cvvdp --test {exr_img_name} --ref {reference_img_exr_name} --display standard_hdr_linear_zoom --display standard_hdr_linear_zoom_4000 --config-paths /home/luoleyouluole/Image-Restoration-Experiments/src/display_models.json  --metric ssim pu-psnr-rgb pu-psnr-y cvvdp  --quiet"

        ret_value = subprocess.run(command, shell=True, capture_output=True, text=True)
        ssim, psnr_rgb, psnr_y, cvvdp = ret_value.stdout.split()
        print(test_name, ssim, psnr_rgb, psnr_y, cvvdp)

        os.remove(exr_img)

        report += test_name + " " + ssim + " " + psnr_rgb + " " + psnr_y + " " + cvvdp + "\n"


        if "naive" in test_name:
            navie_psnr_rgb.append(float(psnr_rgb))
            navie_psnr_y.append(float(psnr_y))
            navie_cvvdp.append(float(cvvdp))
            navie_ssim.append(float(ssim))
        elif "_linear_l1" in test_name:
            linear_psnr_rgb.append(float(psnr_rgb))
            linear_psnr_y.append(float(psnr_y))
            linear_cvvdp.append(float(cvvdp))
            linear_ssim.append(float(ssim))
        elif "log" in test_name:
            log_psnr_rgb.append(float(psnr_rgb))
            log_psnr_y.append(float(psnr_y))
            log_cvvdp.append(float(cvvdp))
        elif "_pq_l1" in test_name:
            pq_psnr_rgb.append(float(psnr_rgb))
            pq_psnr_y.append(float(psnr_y))
            pq_cvvdp.append(float(cvvdp))
            pq_ssim.append(float(ssim))
        elif "_pu_l1" in test_name:
            pu_psnr_rgb.append(float(psnr_rgb))
            pu_psnr_y.append(float(psnr_y))
            pu_cvvdp.append(float(cvvdp))
            pu_ssim.append(float(ssim))
        elif "_linear_smape" in test_name:
            linear_smape_psnr_rgb.append(float(psnr_rgb))
            linear_smape_psnr_y.append(float(psnr_y))
            linear_smape_cvvdp.append(float(cvvdp))
            linear_smape_ssim.append(float(ssim))
        elif "_linear_pu" in test_name:
            linear_pu_psnr_rgb.append(float(psnr_rgb))
            linear_pu_psnr_y.append(float(psnr_y))
            linear_pu_cvvdp.append(float(cvvdp))
            linear_pu_ssim.append(float(ssim))
        elif "_linear_pq" in test_name:
            linear_pq_psnr_rgb.append(float(psnr_rgb))
            linear_pq_psnr_y.append(float(psnr_y))
            linear_pq_cvvdp.append(float(cvvdp))
            linear_pq_ssim.append(float(ssim))
        elif "_linear_mu" in test_name:
            linear_mu_psnr_rgb.append(float(psnr_rgb))
            linear_mu_psnr_y.append(float(psnr_y))
            linear_mu_cvvdp.append(float(cvvdp))
            linear_mu_ssim.append(float(ssim))
        elif "_mu_l1" in test_name:
            mu_psnr_rgb.append(float(psnr_rgb))
            mu_psnr_y.append(float(psnr_y))
            mu_cvvdp.append(float(cvvdp))
            mu_ssim.append(float(ssim))
        elif "_pu21_l1" in test_name:
            pu21_psnr_rgb.append(float(psnr_rgb))
            pu21_psnr_y.append(float(psnr_y))
            pu21_cvvdp.append(float(cvvdp))
            pu21_ssim.append(float(ssim))
        else:
            raise ValueError(f"Unknown test name: {test_name}")

    report += "\n"
    print("\n")
    os.remove(reference_img_exr)


print("navie_psnr_rgb=", navie_psnr_rgb)
print("linear_psnr_rgb=", linear_psnr_rgb)
print("log_psnr_rgb=", log_psnr_rgb)
print("pu_psnr_rgb=", pu_psnr_rgb)
print("pq_psnr_rgb=", pq_psnr_rgb)
print("linear_smape_psnr_rgb=", linear_smape_psnr_rgb)
print("linear_pu_psnr_rgb=", linear_pu_psnr_rgb)
print("linear_pq_psnr_rgb=", linear_pq_psnr_rgb)
print("linear_mu_psnr_rgb=", linear_mu_psnr_rgb)
print("pu21_psnr_rgb=", pu21_psnr_rgb)
print("mu_psnr_rgb=", mu_psnr_rgb)

print("navie_psnr_y=", navie_psnr_y)
print("linear_psnr_y=", linear_psnr_y)
print("log_psnr_y=", log_psnr_y)
print("pu_psnr_y=", pu_psnr_y)
print("pq_psnr_y=", pq_psnr_y)
print("linear_smape_psnr_y=", linear_smape_psnr_y)
print("linear_pu_psnr_y=", linear_pu_psnr_y)
print("linear_pq_psnr_y=", linear_pq_psnr_y)
print("linear_mu_psnr_y=", linear_mu_psnr_y)
print("pu21_psnr_y=", pu21_psnr_y)
print("mu_psnr_y=", mu_psnr_y)

print("navie_cvvdp=", navie_cvvdp)
print("linear_cvvdp=", linear_cvvdp)
print("log_cvvdp=", log_cvvdp)
print("pu_cvvdp=", pu_cvvdp)
print("pq_cvvdp=", pq_cvvdp)
print("linear_smape_cvvdp=", linear_smape_cvvdp)
print("linear_pu_cvvdp=", linear_pu_cvvdp)
print("linear_pq_cvvdp=", linear_pq_cvvdp)
print("linear_mu_cvvdp=", linear_mu_cvvdp)
print("pu21_cvvdp=", pu21_cvvdp)
print("mu_cvvdp=", mu_cvvdp)

print("navie_ssim=", navie_ssim)
print("linear_ssim=", linear_ssim)
# print("log_ssim=", log_ssim)
print("pu_ssim=", pu_ssim)
print("pq_ssim=", pq_ssim)
print("linear_smape_ssim=", linear_smape_ssim)
print("linear_pu_ssim=", linear_pu_ssim)
print("linear_pq_ssim=", linear_pq_ssim)
print("linear_mu_ssim=", linear_mu_ssim)
print("pu21_ssim=", pu21_ssim)
print("mu_ssim=", mu_ssim)

report += "navie_psnr_rgb = " + str(navie_psnr_rgb) + "\n"
report += "linear_psnr_rgb = " + str(linear_psnr_rgb) + "\n"
report += "log_psnr_rgb = " + str(log_psnr_rgb) + "\n"
report += "pu_psnr_rgb = " + str(pu_psnr_rgb) + "\n"
report += "pq_psnr_rgb = " + str(pq_psnr_rgb) + "\n"
report += "linear_smape_psnr_rgb = " + str(linear_smape_psnr_rgb) + "\n"
report += "linear_pu_psnr_rgb = " + str(linear_pu_psnr_rgb) + "\n"
report += "linear_pq_psnr_rgb = " + str(linear_pq_psnr_rgb) + "\n"
report += "linear_mu_psnr_rgb = " + str(linear_mu_psnr_rgb) + "\n"
report += "mu_l1_psnr_rgb = " + str(mu_psnr_rgb) + "\n"
report += "pu21_psnr_rgb = " + str(pu21_psnr_rgb) + "\n"

report += "navie_psnr_y = " + str(navie_psnr_y) + "\n"
report += "linear_psnr_y = " + str(linear_psnr_y) + "\n"
report += "log_psnr_y = " + str(log_psnr_y) + "\n"
report += "pu_psnr_y = " + str(pu_psnr_y) + "\n"
report += "pq_psnr_y = " + str(pq_psnr_y) + "\n"
report += "linear_smape_psnr_y = " + str(linear_smape_psnr_y) + "\n"
report += "linear_pu_psnr_y = " + str(linear_pu_psnr_y) + "\n"
report += "linear_pq_psnr_y = " + str(linear_pq_psnr_y) + "\n"
report += "linear_mu_psnr_y = " + str(linear_mu_psnr_y) + "\n"
report += "mu_l1_psnr_y = " + str(mu_psnr_y) + "\n"
report += "pu21_psnr_y = " + str(pu21_psnr_y) + "\n"

report += "navie_cvvdp = " + str(navie_cvvdp) + "\n"
report += "linear_cvvdp = " + str(linear_cvvdp) + "\n"
report += "log_cvvdp = " + str(log_cvvdp) + "\n"
report += "pu_cvvdp = " + str(pu_cvvdp) + "\n"
report += "pq_cvvdp = " + str(pq_cvvdp) + "\n"
report += "linear_smape_cvvdp = " + str(linear_smape_cvvdp) + "\n"
report += "linear_pu_cvvdp = " + str(linear_pu_cvvdp) + "\n"
report += "linear_pq_cvvdp = " + str(linear_pq_cvvdp) + "\n"
report += "linear_mu_cvvdp = " + str(linear_mu_cvvdp) + "\n"
report += "mu_l1_psnr_cvvdp = " + str(mu_cvvdp) + "\n"
report += "pu21_cvvdp = " + str(pu21_cvvdp) + "\n"


report += "navie_ssim = " + str(navie_ssim) + "\n"
report += "linear_ssim = " + str(linear_ssim) + "\n"
report += "pu_ssim = " + str(pu_ssim) + "\n"
report += "pq_ssim = " + str(pq_ssim) + "\n"
report += "linear_smape_ssim = " + str(linear_smape_ssim) + "\n"
report += "linear_pu_ssim = " + str(linear_pu_ssim) + "\n"
report += "linear_pq_ssim = " + str(linear_pq_ssim) + "\n"
report += "linear_mu_ssim = " + str(linear_mu_ssim) + "\n"
report += "mu_l1_psnr_ssim = " + str(mu_ssim) + "\n"
report += "pu21_ssim = " + str(pu21_ssim) + "\n"

with open("/home/luoleyouluole/Image-Restoration-Experiments/src/report_mir_complete.txt", "w") as file:
    file.write(report)



# import scipy.stats as stats
