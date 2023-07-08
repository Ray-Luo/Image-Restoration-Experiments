import os
import subprocess

# directory containing the images
test_img_folder = '/home/luoleyouluole/Image-Restoration-Experiments/data/res_sad'
imgs = os.listdir(test_img_folder)
imgs.sort()
test_imgs = []
for filename in imgs:
    if "_raw_GT.hdr" in filename:
        test_imgs.append(filename)

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

linear_smape_psnr_rgb = []
linear_smape_psnr_y = []
linear_smape_cvvdp = []

linear_pu_psnr_rgb = []
linear_pu_psnr_y = []
linear_pu_cvvdp = []

linear_pq_psnr_rgb = []
linear_pq_psnr_y = []
linear_pq_cvvdp = []

linear_mu_psnr_rgb = []
linear_mu_psnr_y = []
linear_mu_cvvdp = []

report  = ""


for filename in test_imgs:
    if "Artist_Palette" in filename or "Bigfoot_Pass" in filename:
        continue
    file_name = filename.split("_raw_GT.hdr")[0]
    if "'" in file_name or "&" in file_name:
        file_name = file_name.replace("'", "\\'").replace("&", "\\&")
    reference_name = file_name + "_raw_GT.hdr"
    reference_img = os.path.join(test_img_folder, reference_name)
    test_names = [
        file_name + "_raw_linear_smape.hdr",
        file_name + "_raw_linear_mu.hdr",
        file_name + "_raw_linear_pu.hdr",
        file_name + "_raw_linear_pq.hdr",
        file_name + "_raw_linear_l1.hdr",
        file_name + "_raw_pq_l1.hdr",
        file_name + "_raw_pu_l1.hdr",
    ]

    # print(file_name)

    for test_name in test_names:
        test_img = os.path.join(test_img_folder, test_name)

        command = f"cvvdp --test {test_img} --ref {reference_img} --display standard_hdr_linear_zoom --metric pu-psnr-rgb pu-psnr-y cvvdp  --quiet"
        ret_value = subprocess.run(command, shell=True, capture_output=True, text=True)
        psnr_rgb, psnr_y, cvvdp = ret_value.stdout.split()
        print(test_name, psnr_rgb, psnr_y, cvvdp,)

        report += test_name + " " + psnr_rgb + " " + psnr_y + " " + cvvdp + "\n"

        if "naive" in test_name:
            navie_psnr_rgb.append(float(psnr_rgb))
            navie_psnr_y.append(float(psnr_y))
            navie_cvvdp.append(float(cvvdp))
        elif "_linear_l1" in test_name:
            linear_psnr_rgb.append(float(psnr_rgb))
            linear_psnr_y.append(float(psnr_y))
            linear_cvvdp.append(float(cvvdp))
        elif "log" in test_name:
            log_psnr_rgb.append(float(psnr_rgb))
            log_psnr_y.append(float(psnr_y))
            log_cvvdp.append(float(cvvdp))
        elif "_pq_l1" in test_name:
            pq_psnr_rgb.append(float(psnr_rgb))
            pq_psnr_y.append(float(psnr_y))
            pq_cvvdp.append(float(cvvdp))
        elif "_pu_l1" in test_name:
            pu_psnr_rgb.append(float(psnr_rgb))
            pu_psnr_y.append(float(psnr_y))
            pu_cvvdp.append(float(cvvdp))
        elif "_linear_smape" in test_name:
            linear_smape_psnr_rgb.append(float(psnr_rgb))
            linear_smape_psnr_y.append(float(psnr_y))
            linear_smape_cvvdp.append(float(cvvdp))
        elif "_linear_pu" in test_name:
            linear_pu_psnr_rgb.append(float(psnr_rgb))
            linear_pu_psnr_y.append(float(psnr_y))
            linear_pu_cvvdp.append(float(cvvdp))
        elif "_linear_pq" in test_name:
            linear_pq_psnr_rgb.append(float(psnr_rgb))
            linear_pq_psnr_y.append(float(psnr_y))
            linear_pq_cvvdp.append(float(cvvdp))
        elif "_linear_mu" in test_name:
            linear_mu_psnr_rgb.append(float(psnr_rgb))
            linear_mu_psnr_y.append(float(psnr_y))
            linear_mu_cvvdp.append(float(cvvdp))
        else:
            raise ValueError(f"Unknown test name: {test_name}")

    report += "\n"
    print("\n")

print("navie_psnr_rgb=", navie_psnr_rgb)
print("linear_psnr_rgb=", linear_psnr_rgb)
print("log_psnr_rgb=", log_psnr_rgb)
print("pu_psnr_rgb=", pu_psnr_rgb)
print("pq_psnr_rgb=", pq_psnr_rgb)
print("linear_smape_psnr_rgb=", linear_smape_psnr_rgb)
print("linear_pu_psnr_rgb=", linear_pu_psnr_rgb)
print("linear_pq_psnr_rgb=", linear_pq_psnr_rgb)
print("linear_mu_psnr_rgb=", linear_mu_psnr_rgb)

print("navie_psnr_y=", navie_psnr_y)
print("linear_psnr_y=", linear_psnr_y)
print("log_psnr_y=", log_psnr_y)
print("pu_psnr_y=", pu_psnr_y)
print("pq_psnr_y=", pq_psnr_y)
print("linear_smape_psnr_y=", linear_smape_psnr_y)
print("linear_pu_psnr_y=", linear_pu_psnr_y)
print("linear_pq_psnr_y=", linear_pq_psnr_y)
print("linear_mu_psnr_y=", linear_mu_psnr_y)

print("navie_cvvdp=", navie_cvvdp)
print("linear_cvvdp=", linear_cvvdp)
print("log_cvvdp=", log_cvvdp)
print("pu_cvvdp=", pu_cvvdp)
print("pq_cvvdp=", pq_cvvdp)
print("linear_smape_cvvdp=", linear_smape_cvvdp)
print("linear_pu_cvvdp=", linear_pu_cvvdp)
print("linear_pq_cvvdp=", linear_pq_cvvdp)
print("linear_mu_cvvdp=", linear_mu_cvvdp)

report += "navie_psnr_rgb = " + str(navie_psnr_rgb) + "\n"
report += "linear_psnr_rgb = " + str(linear_psnr_rgb) + "\n"
report += "log_psnr_rgb = " + str(log_psnr_rgb) + "\n"
report += "pu_psnr_rgb = " + str(pu_psnr_rgb) + "\n"
report += "pq_psnr_rgb = " + str(pq_psnr_rgb) + "\n"
report += "linear_smape_psnr_rgb = " + str(linear_smape_psnr_rgb) + "\n"
report += "linear_pu_psnr_rgb = " + str(linear_pu_psnr_rgb) + "\n"
report += "linear_pq_psnr_rgb = " + str(linear_pq_psnr_rgb) + "\n"
report += "linear_mu_psnr_rgb = " + str(linear_mu_psnr_rgb) + "\n"

report += "navie_psnr_y = " + str(navie_psnr_y) + "\n"
report += "linear_psnr_y = " + str(linear_psnr_y) + "\n"
report += "log_psnr_y = " + str(log_psnr_y) + "\n"
report += "pu_psnr_y = " + str(pu_psnr_y) + "\n"
report += "pq_psnr_y = " + str(pq_psnr_y) + "\n"
report += "linear_smape_psnr_y = " + str(linear_smape_psnr_y) + "\n"
report += "linear_pu_psnr_y = " + str(linear_pu_psnr_y) + "\n"
report += "linear_pq_psnr_y = " + str(linear_pq_psnr_y) + "\n"
report += "linear_mu_psnr_y = " + str(linear_mu_psnr_y) + "\n"

report += "navie_cvvdp = " + str(navie_cvvdp) + "\n"
report += "linear_cvvdp = " + str(linear_cvvdp) + "\n"
report += "log_cvvdp = " + str(log_cvvdp) + "\n"
report += "pu_cvvdp = " + str(pu_cvvdp) + "\n"
report += "pq_cvvdp = " + str(pq_cvvdp) + "\n"
report += "linear_smape_cvvdp = " + str(linear_smape_cvvdp) + "\n"
report += "linear_pu_cvvdp = " + str(linear_pu_cvvdp) + "\n"
report += "linear_pq_cvvdp = " + str(linear_pq_cvvdp) + "\n"
report += "linear_mu_cvvdp = " + str(linear_mu_cvvdp) + "\n"


with open("/home/luoleyouluole/Image-Restoration-Experiments/src/real_report.txt", "w") as file:
    file.write(report)



# import scipy.stats as stats
