# IERG4190: Multimedia Coding and Processing — Course Project

## Denoising and classification of noisy CIFAR-100 images

**Group-based project | 1–3 students | 15% of course mark | Due: 10 December 2026**

---

## 1. Overview and goals

This project connects spatial image filtering (Topic 3) with image representation and
classification (Topic 4) on a shared benchmark. Using the CIFAR-100 dataset, every group
corrupts images with additive white noise, denoises them using spatial filters, and classifies
the denoised images. The central goal is to study how noise levels and denoising filter settings
affect both image quality and classification performance. Because all groups work on the same
dataset and pipeline, effort focuses on controlled experimentation and clear technical analysis;
groups are encouraged to explore and compare different denoising settings, image features, and
classifier configurations.

Each group contains 1–3 students and submits a single `.zip` package containing the
reproducible code and the written report (PDF). All members should understand the complete
pipeline, and the report must state each member's contribution.

By completing the project, students should be able to:

- simulate controlled additive white noise and evaluate spatial denoising filters both
  quantitatively (PSNR/MSE) and visually;
- build an end-to-end denoising, feature-extraction, and classification pipeline on CIFAR-100
  without test-set leakage;
- measure and explain how noise strength, filter choice, image features, and classifier settings
  interact to determine recognition performance;
- present controlled comparisons, failure cases, and conclusions clearly in a concise technical
  report.

---

## 2. Dataset and pipeline

**Dataset.** Use the standard CIFAR-100 dataset, available from the official University of
Toronto dataset website at <https://www.cs.toronto.edu/~kriz/cifar.html> (Python archive
`cifar-100-python.tar.gz`; also mirrored on Hugging Face at
<https://huggingface.co/datasets/uoft-cs/cifar100>). CIFAR-100 contains 60,000 colour images of
size 32 × 32 across 100 fine classes (grouped into 20 coarse superclasses), split into 50,000
training images and 10,000 test images.

Hold out a validation subset (for example, 5,000 or 10,000 stratified images) from the 50,000
training images for tuning filter, feature, and classifier hyperparameters. Keep the official
10,000-image test set untouched until the final evaluation.

**Pipeline.** Every group should implement a reproducible pipeline of the form:

```
CIFAR-100 image → add white noise → spatial denoising → feature extraction → classifier → evaluation
```

**Noise model.** Normalise pixel intensities to [0, 1] (or state the [0, 255] scale explicitly) and
add zero-mean independent identically distributed (i.i.d.) Gaussian white noise to each pixel and
colour channel:

$$
\tilde{I}(x,y,c) = \mathrm{clip}\left(I(x,y,c) + \epsilon(x,y,c),\; 0,\; 1\right),
\qquad \epsilon(x,y,c) \sim \mathcal{N}(0, \sigma^2)
$$

Test the clean baseline ($\sigma = 0$) alongside at least three non-zero noise levels spanning
mild to strong corruption (for example, $\sigma \in \{0.05, 0.10, 0.20, 0.30\}$ on [0, 1]).

Fix random seeds so that all denoising filters, features, and classifiers are compared on the
exact same noisy images.

---

## 3. Core requirements

Every group must complete the following core study:

1. **Clean and noisy baselines.** Evaluate your classification pipeline on clean images
   ($\sigma = 0$) and on noisy images without denoising across all chosen noise levels.

2. **Spatial denoising comparison.** Apply at least two spatial denoising filters (such as a
   box/mean filter, Gaussian filter, median filter, or bilateral filter) and vary their key
   settings (such as kernel size or smoothing strength $\sigma_s$). Report both reconstruction
   quality (PSNR or MSE relative to the clean image) and downstream classification performance on
   the denoised images.

3. **Noise-level vs. denoising analysis.** Compare how classification accuracy (and
   macro-averaged F1) changes across the grid of noise levels and denoising settings. Clearly
   identify when denoising improves recognition, when over-smoothing hurts it, and whether
   higher PSNR always corresponds to higher classification accuracy.

4. **Feature and classifier comparison.** Compare at least two image feature representations, and
   compare a linear SVM (one-vs-all) against a softmax classifier with hyperparameters tuned on
   validation data.

5. **Qualitative inspection and error analysis.** Show representative clean, noisy, and denoised
   images alongside correctly and incorrectly classified examples. Discuss which classes or
   visual cues (such as fine texture or edges) are most affected by noise and smoothing.

6. **Reproducibility.** Provide clean code, a README, a dependency specification, fixed random
   seeds, and scripts or notebooks that reproduce the main tables and figures.

---

## 4. Encouraged explorations

Beyond the minimum baseline, groups are strongly encouraged to experiment across three design
dimensions and compare their performance systematically:

| Pipeline stage | Settings and variations to try | Questions to compare |
|---|---|---|
| **Spatial denoising** | Box/mean, Gaussian, median, and bilateral filters; kernel sizes (3 × 3, 5 × 5, 7 × 7); Gaussian $\sigma_s$ and bilateral range parameter $\sigma_r$; boundary padding modes; training on clean vs. noisy/denoised images | Which filter best balances noise suppression against edge and texture preservation at each $\sigma$? |
| **Image features** | Raw or normalised pixels, colour histograms, HOG descriptors, bag of visual words (with SIFT descriptors), spatial-pyramid histograms (optional), or combined colour + descriptor features | Which features remain discriminative under noise and blur, and how does feature dimensionality affect accuracy and runtime? |
| **Classifiers** | Linear SVM (one-vs-all) and softmax classifier (multinomial logistic regression); tuning regularisation strength ($C$) and feature standardisation on validation data | How do classifier choice and regularisation affect robustness to residual noise and filter blur? |

**Recommended software.** Students are recommended to implement their pipelines in Python using
scikit-image (<https://scikit-image.org/>) for noise generation, spatial denoising filters,
reconstruction metrics (PSNR/MSE), and image feature extraction (such as HOG and SIFT), together
with scikit-learn (<https://scikit-learn.org/>) for stratified train/validation splitting,
visual-vocabulary clustering ($k$-means), feature scaling, linear SVM and softmax classifiers,
hyperparameter tuning, and evaluation metrics (accuracy, macro-F1, and confusion matrices).
Supporting scientific libraries such as NumPy, SciPy, and Matplotlib may be used alongside them,
but the report must explain the methods, exact parameter settings, and empirical trade-offs
rather than treating library calls as black boxes. Evaluating on the 100 fine CIFAR-100 classes
is the default; groups may also report accuracy on the 20 coarse superclasses as an additional
diagnostic.

---

## 5. Report and submission

### 5.1 Written report

The report (`report.pdf`) should be **6–8 pages** (excluding references and optional appendix) and
include:

1. **Abstract:** problem, main methods compared, strongest result, and key finding.
2. **Experimental setup:** CIFAR-100 split protocol, white-noise model and noise levels, random
   seeds, and evaluation metrics (PSNR/MSE, top-1 accuracy, macro-F1).
3. **Methods:** denoising filters and settings, feature representations, and
   classifiers/hyperparameters evaluated.
4. **Results:** tables and plots comparing noise levels, denoising settings, features, and
   classifiers, plus visual examples of clean, noisy, and denoised images.
5. **Analysis and discussion:** trade-offs between denoising strength and classification
   accuracy, PSNR versus recognition accuracy, runtime costs, failure modes, and limitations.
6. **References and contribution statement:** citations for datasets, methods, and libraries,
   plus each group member's contribution.

### 5.2 Submission package and code structure

Submit a single `.zip` package (named `group_<id>.zip`) by **Thursday, 10 December 2026**,
containing both the written report and the reproducible Python codebase with the following
top-level layout:

```
group_<id>/
├── report.pdf            # Final written report (6–8 pages)
├── README.md             # Setup instructions, usage examples, and notes
├── requirements.txt      # Python dependencies (scikit-image, scikit-learn, etc.)
├── run.py                # Standardised command-line entry script
└── src/                  # Modular pipeline implementation
    ├── data.py           # CIFAR-100 loader and stratified train/val split
    ├── noise.py          # Seeded additive Gaussian white-noise generator
    ├── denoise.py        # Spatial denoising filters and PSNR/MSE helpers
    ├── features.py       # Feature extractors (pixels, HOG, BoVW-SIFT, etc.)
    └── models.py         # Linear SVM (one-vs-all) and softmax classifier
```

Additional helper modules or optional configuration files (e.g. under `configs/`) may be included
as needed. **Do not** bundle the CIFAR-100 data files or large cached artefacts inside the `.zip`
archive. Acknowledge any external code or AI-assisted work according to university policy.

### 5.3 Entry script and command-line interface (for auto-checking)

To facilitate automated verification during grading, the top-level entry script must be named
`run.py` and support non-interactive execution from the command line using Python's `argparse`:

```bash
python run.py --data-dir <PATH> --output-dir <PATH> \
  --noise-std 0.10 --filter gaussian --kernel-size 3 \
  --feature hog --classifier svm --C 1.0 --seed 42
```

**Required command-line arguments and behaviour:**

- `--data-dir <PATH>` (**required**): path to a local directory containing the extracted
  `cifar-100-python/` files (the grader supplies a pre-downloaded copy; `run.py` must not require
  internet access when this directory is present).
- `--output-dir <PATH>` (**required**): output directory (created if absent) where results are
  saved.
- `--noise-std <FLOAT>` (default `0.10`): Gaussian white noise standard deviation
  $\sigma \geq 0$ on [0, 1] (0 for clean images).
- `--filter <NAME>` (default `none`): spatial filter choice (e.g. `none`, `box`, `gaussian`,
  `median`, `bilateral`), paired with optional `--kernel-size <INT>` (default `3`) and
  `--filter-sigma <FLOAT>` (for Gaussian or bilateral smoothing strength).
- `--feature <NAME>` (default `hog`): image feature choice (e.g. `pixels`, `color_hist`, `hog`,
  `bovw`, `pyramid`).
- `--classifier <NAME>` (default `svm`): `svm` (one-vs-all linear SVM) or `softmax`
  (multinomial logistic regression), with `--C <FLOAT>` (default `1.0`) for regularisation
  strength.
- `--seed <INT>` (default `42`): random seed controlling the train/validation split, noise
  generation, and model fitting.
- `--max-samples <INT>` (optional): if given, limits the number of training and test images
  processed so the auto-checker can run a fast end-to-end smoke test.
- `--run-all` (optional flag): executes the full experiment sweep reported in `report.pdf` and
  regenerates the main tables and figures in `--output-dir`.

**Standardised output.** Every single-configuration run must exit with status 0 and write a
`metrics.json` file inside `--output-dir` containing at least the keys `noise_std`, `filter`,
`feature`, `classifier`, `seed`, `psnr` (mean dB), `mse`, `accuracy` (top-1 test accuracy in
[0, 1]), and `macro_f1` (in [0, 1]). Document any extra filter or feature options supported by
your `run.py` in `README.md`.

---

## 6. Key questions to investigate

- How rapidly does classification accuracy drop as white-noise level $\sigma$ increases, and how
  much of that drop can spatial denoising recover?
- At low noise levels, does spatial filtering help or hurt classification by blurring 32 × 32
  details?
- Does the filter setting that achieves the highest PSNR always achieve the highest
  classification accuracy?
- Does training the classifier on clean images versus denoised (or noisy) images change the best
  denoising setting?
- Which combinations of spatial filter, image feature, and classifier give the best
  accuracy–speed trade-off on noisy CIFAR-100?
