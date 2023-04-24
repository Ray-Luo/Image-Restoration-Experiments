import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter

A_COEFF = 0.456520040846940
B_COEFF = 1.070672820603428
L_max = 10000
p = [234.0235618, 216.9339286, 0.0001091864237, 0.893206924, 0.06733984121, 1.444718567, 567.6315065]
MAX_PIXEL = p[6]*(((p[0] + p[1]*L_max**p[3])/(1 + p[2]*L_max**p[3]))**p[4] - p[5]) # --> 566.6339579284676

def pu(x):
    x = A_COEFF * x + B_COEFF
    x = np.clip(x, 1e-5, np.max(x))
    return np.log2(x)


def psnr(pred, gt):
    psnr = 0
    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        psnr = 100
    else:
        psnr = 20 * np.log10(MAX_PIXEL / np.sqrt(mse))

    return psnr


def pu_psnr(pred, gt):
    score = psnr(pu(pred), pu(gt))
    return score


def rgb2gray(rgb):
    # 0.212656, 0.715158, 0.072186 from REC.709 standard
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.212656 * r + 0.715158 * g + 0.072186 * b
    return gray


def ssim(img1, img2, k1=0.01, k2=0.03, sigma=1.5, L=MAX_PIXEL):
    # convert the input images to grayscale
    img1 = rgb2gray(img1)
    img2 = rgb2gray(img2)

    # calculate the mean, variance, and covariance of the two images
    mu1 = gaussian_filter(img1, sigma=sigma)
    mu2 = gaussian_filter(img2, sigma=sigma)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = gaussian_filter(img1 ** 2, sigma=sigma) - mu1_sq
    sigma2_sq = gaussian_filter(img2 ** 2, sigma=sigma) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sigma=sigma) - mu1_mu2

    # calculate the SSIM value
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    ssim = np.mean(ssim)
    return ssim


def pu_ssim(pred, gt):
    return ssim(pu(pred), pu(gt))
