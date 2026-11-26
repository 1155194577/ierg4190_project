import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def _unpickle(data_dir, split):
    with open(Path(data_dir) / "cifar-100-python" / split, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    x = np.frombuffer(d[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
    return x.transpose(0, 2, 3, 1).copy(), np.asarray(d[b"fine_labels"], dtype=np.int64)


def get_arrays(data_dir, val_size=5000, seed=42):
    x, y = _unpickle(data_dir, "train")
    x_test, y_test = _unpickle(data_dir, "test")
    tr, va = train_test_split(np.arange(len(y)), test_size=val_size, stratify=y, random_state=seed)
    return (x[tr], y[tr]), (x[va], y[va]), (x_test, y_test)


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent / "data_tv"
    (xtr, ytr), (xva, yva), (xte, yte) = get_arrays(data_dir)
    print(xtr.shape, ytr.shape, xva.shape, xte.shape)
