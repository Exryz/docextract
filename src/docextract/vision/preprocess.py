"""Image preprocessing: skew estimation, deskew rotation, denoising, and adaptive thresholding."""

from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import Image, ImageFilter, ImageOps


@dataclass(frozen=True)
class PreprocessResult:
    """Processed page image together with the detected skew angle and applied transforms."""

    image: Image.Image
    skew_angle_deg: float
    deskewed: bool
    thresholded: bool


def _horizontal_projection_variance(binary_small: Image.Image, angle_deg: float) -> float:
    """Rotate `binary_small` by `angle_deg` and return variance of horizontal row ink sums."""
    rotated = binary_small.rotate(angle_deg, resample=Image.Resampling.BILINEAR, fillcolor=0)
    w, h = rotated.size
    raw = rotated.tobytes()
    row_sums = [sum(raw[row * w : (row + 1) * w]) for row in range(h)]
    if not row_sums:
        return 0.0
    mean = sum(row_sums) / len(row_sums)
    return sum((s - mean) ** 2 for s in row_sums) / len(row_sums)


def estimate_skew_angle(
    image: Image.Image,
    max_angle_deg: float = 10.0,
    coarse_step_deg: float = 1.0,
    fine_step_deg: float = 0.2,
) -> float:
    """Estimate document skew angle in degrees (`[-max_angle_deg, +max_angle_deg]`).

    Uses a coarse-to-fine horizontal projection profile maximization on an inverted binary
    thumbnail. When text lines are horizontal, row ink sums alternate sharply between dark
    text rows and blank inter-line gaps, maximizing row-sum variance.
    """
    gray = ImageOps.grayscale(image)
    # Downsample so projection search runs in <15ms on CPU
    max_dim = 480
    scale = min(1.0, max_dim / max(gray.width, gray.height, 1))
    thumb_w = max(32, int(round(gray.width * scale)))
    thumb_h = max(32, int(round(gray.height * scale)))
    thumb = gray.resize((thumb_w, thumb_h), resample=Image.Resampling.BILINEAR)

    # Invert and binarize: text pixels = 1, background = 0
    hist = thumb.histogram()
    total_px = thumb_w * thumb_h
    # Otsu-style simple percentile threshold: darkest 25% of pixels treated as foreground ink
    cutoff = 180
    acc = 0
    for val, count in enumerate(hist):
        acc += count
        if acc >= total_px * 0.25:
            cutoff = min(220, max(60, val + 15))
            break

    binary_inv = thumb.point(lambda p: 1 if p < cutoff else 0, mode="L")

    # Coarse search
    best_angle = 0.0
    best_var = -1.0
    steps = int(round((2 * max_angle_deg) / coarse_step_deg))
    for i in range(steps + 1):
        angle = -max_angle_deg + i * coarse_step_deg
        var = _horizontal_projection_variance(binary_inv, angle)
        if var > best_var:
            best_var = var
            best_angle = angle

    # Fine search around coarse peak
    fine_min = max(-max_angle_deg, best_angle - coarse_step_deg)
    fine_max = min(max_angle_deg, best_angle + coarse_step_deg)
    fine_steps = int(round((fine_max - fine_min) / fine_step_deg))
    for i in range(fine_steps + 1):
        angle = fine_min + i * fine_step_deg
        var = _horizontal_projection_variance(binary_inv, angle)
        if var > best_var:
            best_var = var
            best_angle = angle

    if math.isclose(best_angle, 0.0, abs_tol=0.15):
        return 0.0
    return round(best_angle, 2)


def deskew_image(
    image: Image.Image,
    min_correction_deg: float = 0.3,
    max_angle_deg: float = 10.0,
) -> tuple[Image.Image, float, bool]:
    """Detect skew and counter-rotate `image` if `|angle| >= min_correction_deg`."""
    angle = estimate_skew_angle(image, max_angle_deg=max_angle_deg)
    if abs(angle) < min_correction_deg:
        return image, angle, False

    rgb = image.convert("RGB") if image.mode != "RGB" else image
    corrected = rgb.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=(255, 255, 255),
    )
    return corrected, angle, True


def adaptive_binarize(image: Image.Image, window_radius: int = 11, offset: int = 12) -> Image.Image:
    """Local mean adaptive thresholding for low-contrast or unevenly lit receipt scans."""
    gray = ImageOps.grayscale(image)
    denoised = gray.filter(ImageFilter.MedianFilter(size=3))
    local_mean = denoised.filter(ImageFilter.BoxBlur(radius=window_radius))

    src_bytes = denoised.tobytes()
    mean_bytes = local_mean.tobytes()
    out_bytes = bytes(
        0 if s < (m - offset) else 255 for s, m in zip(src_bytes, mean_bytes, strict=True)
    )
    return Image.frombytes("L", gray.size, out_bytes).convert("RGB")


def preprocess_page_image(
    image: Image.Image,
    *,
    deskew: bool = True,
    binarize: bool = False,
) -> PreprocessResult:
    """Run deskew and optional adaptive thresholding on a page image."""
    current = image.convert("RGB") if image.mode != "RGB" else image
    angle = 0.0
    was_deskewed = False

    if deskew:
        current, angle, was_deskewed = deskew_image(current)

    if binarize:
        current = adaptive_binarize(current)

    return PreprocessResult(
        image=current,
        skew_angle_deg=angle,
        deskewed=was_deskewed,
        thresholded=binarize,
    )
