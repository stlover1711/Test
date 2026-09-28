"""Shared helpers: sample-image loading, noise, and figure saving.

All images are handled as RGB uint8 numpy arrays (H, W, 3) or (H, W) for
grayscale. OpenCV functions are used for the actual processing (filtering,
warping); conversion to/from OpenCV's BGR order happens only at load/save
boundaries so the rest of the code never has to think about channel order.
"""
import os

import cv2
import numpy as np
from skimage import data, img_as_ubyte


def load_sample_images():
    """Return a dict of standard test images (RGB uint8) covering the cases
    the assignment asks for: a color photo, a low-detail/high-contrast
    pattern, a highly textured grayscale image, and a classic test image.
    """
    astronaut = img_as_ubyte(data.astronaut())          # color, high detail
    checkerboard = img_as_ubyte(data.checkerboard())    # grayscale, sharp edges, low detail
    coins = img_as_ubyte(data.coins())                  # grayscale, textured
    camera = img_as_ubyte(data.camera())                # grayscale, classic test image
    return {
        "astronaut": astronaut,
        "checkerboard": checkerboard,
        "coins": coins,
        "camera": camera,
    }


def to_gray3(gray):
    """Stack a single-channel grayscale image into 3 channels for display
    alongside color images with plt.imshow (no colormap surprises)."""
    return np.stack([gray, gray, gray], axis=-1)


def add_gaussian_noise(img, sigma=25, seed=0):
    """Add Gaussian noise to an 8-bit image (color or gray), clipped to
    [0, 255]. Used to simulate a 'noisy image' test case for Part 2."""
    rng = np.random.default_rng(seed)
    noisy = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_image(path, img):
    """Save an RGB or grayscale uint8 image to disk via OpenCV."""
    ensure_dir(os.path.dirname(path))
    if img.ndim == 3:
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(path, img)


def save_grid(path, images, titles, cmap=None, suptitle=None, ncols=None):
    """Save a labeled grid of images to a single PNG figure (for the report)."""
    import matplotlib.pyplot as plt

    n = len(images)
    ncols = ncols or n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, img, title in zip(axes, images, titles):
        c = (cmap or "gray") if img.ndim == 2 else None
        ax.imshow(img, cmap=c, vmin=0 if img.ndim == 2 else None, vmax=255 if img.ndim == 2 else None)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    ensure_dir(os.path.dirname(path))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
