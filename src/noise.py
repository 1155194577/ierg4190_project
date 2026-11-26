import numpy as np


def add_noise(x, sigma, seed=42):
    ## TODO : implement noise addition
    white_noise = np.random.RandomState(seed).normal(0, sigma, x.shape) 
    x = x + white_noise
    x = np.clip(x, 0, 255) ## prevent out of bound value
    return np.asarray(x, dtype=np.float32) / 255.0
