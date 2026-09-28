# CV Project 1 — Color Image Representation, Spatial Filtering, and Geometric Transformations

Implementation for the assignment described in [`docs/CV-project-1.pdf`](docs/CV-project-1.pdf), following the general report/submission guidelines in [`docs/HD-BTL-CV.pdf`](docs/HD-BTL-CV.pdf).

## Contents

- **Part 1 — Color/grayscale representation.** RGB <-> grayscale conversion (manual luminosity formula vs. OpenCV), splitting an image into R/G/B channels shown as grayscale, lossless reconstruction, and new images built from swapped/zeroed/combined channels.
- **Part 2 — Filtering.** Low-pass (mean, Gaussian) and high-pass (Laplacian, Sobel, a hand-designed sharpen kernel) filters applied to 3 images with different characteristics (sharp/low-detail, textured, noisy).
- **Part 3 — Geometric transformations.** Translation, rotation, scaling, general affine transform, and projective (homography) transform, visualized on the same input, with a direct affine-vs-projective comparison.

## Project layout

```
src/
  utils.py                     # shared helpers (sample images, noise, figure saving)
  part1_color_channels.py      # Part 1
  part2_filtering.py           # Part 2
  part3_transformations.py     # Part 3
notebooks/
  CV_Project_1.ipynb           # runnable, annotated walkthrough of all 3 parts (Colab-friendly)
outputs/
  part1/ part2/ part3/         # generated figures (PNG), also embeddable in the report
report/
  REPORT_TEMPLATE.md           # report skeleton following docs/HD-BTL-CV.pdf structure
data/                          # (empty) drop your own images here if you want to use them
  instead of the built-in scikit-image sample images
```

## Setup

Using a virtual environment (recommended, keeps dependencies isolated):

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows (cmd/PowerShell/git-bash)
# source .venv/bin/activate # macOS/Linux
pip install -r requirements.txt
```

Or install directly into your current environment:

```bash
pip install -r requirements.txt
```

Requires Python 3.9+. Tested with Python 3.13, opencv-python 5.0, scikit-image 0.26, matplotlib 3.11. `.venv/` is git-ignored, so it's safe to create locally without affecting the repo.

## Running

Run each part as a script (writes labeled PNG figures to `outputs/<part>/`):

```bash
python src/part1_color_channels.py
python src/part2_filtering.py
python src/part3_transformations.py
```

Or open the notebook for an inline, explained walkthrough:

```bash
jupyter notebook notebooks/CV_Project_1.ipynb
```

The notebook also runs unmodified in Google Colab: clone this repository as the first cell (see the commented-out cell at the top of the notebook), `pip install -r requirements.txt`, then run all cells.

## Images used

Instead of arbitrary downloaded photos of uncertain provenance, this project uses the standard `scikit-image` sample images (`astronaut`, `checkerboard`, `coins`, `camera`) — public-domain / redistributable test images bundled with the library and standard in the CV literature (satisfies the assignment's requirement that images used comply with copyright/school policy, and keeps the whole pipeline reproducible with no external downloads). To use your own photos instead, edit `load_sample_images()` in `src/utils.py` to load files from `data/` with `cv2.imread` (remember to convert BGR -> RGB).

## Notes

- All image arrays are kept in **RGB** order throughout `src/`; conversion to OpenCV's BGR order happens only at `imwrite` time, in `utils.save_image`.
- `src/part2_filtering.py` synthesizes a noisy test case (`add_gaussian_noise`) rather than requiring a separately sourced noisy photo, so the low-pass denoising comparison is reproducible.
- The affine vs. projective demo in Part 3 uses the checkerboard image specifically because its grid makes the "parallel lines stay parallel (affine) vs. converge (projective)" distinction visually obvious.
