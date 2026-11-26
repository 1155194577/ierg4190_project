import numpy as np


def denoise(x, name="none", kernel_size=3, sigma=1.0):
    ## TODO : implement denoising filters
    return np.asarray(x, dtype=np.float32)


def mse(clean, test):
    return float(np.mean((np.asarray(clean) - np.asarray(test)) ** 2))


def psnr(clean, test):
    m = mse(clean, test)
    return float("inf") if m == 0 else float(-10.0 * np.log10(m))
