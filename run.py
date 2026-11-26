import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from src.data import get_arrays
from src.denoise import denoise, mse, psnr
from src.features import extract
from src.models import LinearClassifier
from src.noise import add_noise


def evaluate(args, sigma, model, xt, yt):
    clean = xt.astype(np.float32) / 255.0
    den = denoise(add_noise(xt, sigma, args.seed), args.filter, args.kernel_size, args.filter_sigma)
    pred = model.predict(extract(den, args.feature))
    p = psnr(clean, den)
    return {
        "noise_std": sigma,
        "filter": args.filter,
        "feature": args.feature,
        "classifier": args.classifier,
        "seed": args.seed,
        "psnr": round(p, 2) if np.isfinite(p) else 100.0,
        "mse": round(mse(clean, den), 6),
        "accuracy": float((pred == yt).mean()),
        "macro_f1": float(f1_score(yt, pred, average="macro")),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--noise-std", type=float, default=0.10)
    p.add_argument("--filter", default="none")
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--filter-sigma", type=float, default=1.0)
    p.add_argument("--feature", default="pixels")
    p.add_argument("--classifier", default="svm")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--run-all", action="store_true")
    args = p.parse_args()

    (xs, ys), (_, _), (xt, yt) = get_arrays(args.data_dir, seed=args.seed)
    if args.max_samples:
        n = args.max_samples
        xs, ys, xt, yt = xs[:n], ys[:n], xt[:n], yt[:n]

    model = LinearClassifier(C=args.C).fit(extract(xs.astype(np.float32) / 255.0, args.feature), ys)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for sigma in (0.0, 0.05, 0.10, 0.20, 0.30) if args.run_all else (args.noise_std,):
        metrics = evaluate(args, sigma, model, xt, yt)
        dest = out / f"noise_{sigma:.2f}" if args.run_all else out
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics))


if __name__ == "__main__":
    main()
