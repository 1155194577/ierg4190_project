import numpy as np


def extract(x, name="pixels"):
    if name != "pixels":
        raise NotImplementedError(f"feature {name!r} not implemented yet; use 'pixels'")
    return np.asarray(x, dtype=np.float32).reshape(len(x), -1)
