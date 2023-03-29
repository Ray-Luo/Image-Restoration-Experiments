import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter

A_COEFF = 0.456520040846940
B_COEFF = 1.070672820603428
MAX_PU = 10.017749773073085 # pu(4000.0)

def pu(x):
    return np.log2(A_COEFF * x + 1e-5) + B_COEFF


def psnr(pred, gt):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.
    """
    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        return float('inf')
    else:
        psnr = 20 * np.log10(MAX_PU / np.sqrt(mse))
        return psnr


def pu_psnr(pred, gt):
    return psnr(pu(pred), pu(gt))


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def ssim(img1, img2, k1=0.01, k2=0.03, sigma=1.5, L=MAX_PU):
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

    return np.mean(ssim)


def pu_ssim(pred, gt):
    return ssim(pu(pred), pu(gt))
