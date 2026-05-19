from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np


MEAN_VALUES = (103.939, 116.779, 123.680)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class TransferResult:
    """Container for one neural style transfer run."""

    stylized_image: np.ndarray
    resized_content: np.ndarray
    inference_time: float
    original_size: tuple[int, int]
    output_size: tuple[int, int]
    model_name: str


def list_style_models(models_dir: str | Path) -> list[Path]:
    """Return available Torch style transfer models from a folder."""

    models_path = Path(models_dir)
    if not models_path.exists():
        return []
    return sorted(models_path.glob("*.t7"))


def load_image(image_path: str | Path) -> np.ndarray:
    """Load an image as BGR for OpenCV processing."""

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not read this image: {path}")
    return image


def save_image(image: np.ndarray, output_path: str | Path) -> Path:
    """Save a BGR image and return the output path."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise OSError(f"OpenCV could not write image: {path}")
    return path


def resize_to_width(image: np.ndarray, width: int | None) -> np.ndarray:
    """Resize while preserving aspect ratio."""

    if width is None or width <= 0:
        return image.copy()

    height, current_width = image.shape[:2]
    if current_width == width:
        return image.copy()

    ratio = width / float(current_width)
    new_height = max(1, int(height * ratio))
    interpolation = cv2.INTER_AREA if ratio < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, (width, new_height), interpolation=interpolation)


def _clip_to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype("uint8")


def build_side_by_side(
    content_bgr: np.ndarray,
    stylized_bgr: np.ndarray,
    left_label: str = "Content",
    right_label: str = "Stylized",
) -> np.ndarray:
    """Create a labeled comparison image for reports and demos."""

    if content_bgr.shape[:2] != stylized_bgr.shape[:2]:
        stylized_bgr = cv2.resize(
            stylized_bgr,
            (content_bgr.shape[1], content_bgr.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    comparison = np.hstack([content_bgr, stylized_bgr])
    label_bar_height = 44
    label_bar = np.full(
        (label_bar_height, comparison.shape[1], 3),
        fill_value=(245, 245, 245),
        dtype=np.uint8,
    )
    comparison = np.vstack([label_bar, comparison])

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, left_label, (18, 29), font, 0.75, (35, 35, 35), 2, cv2.LINE_AA)
    cv2.putText(
        comparison,
        right_label,
        (content_bgr.shape[1] + 18, 29),
        font,
        0.75,
        (35, 35, 35),
        2,
        cv2.LINE_AA,
    )
    return comparison


class StyleTransferEngine:
    """Fast neural style transfer inference through OpenCV DNN."""

    def __init__(self, backend: int = cv2.dnn.DNN_BACKEND_DEFAULT, target: int = cv2.dnn.DNN_TARGET_CPU):
        self.backend = backend
        self.target = target

    def apply(
        self,
        content_bgr: np.ndarray,
        model_path: str | Path,
        width: int | None = 700,
        style_strength: float = 1.0,
    ) -> TransferResult:
        """Apply a pretrained Torch style-transfer model to a BGR image.

        The supported models are feed-forward networks trained for a single
        artistic style. OpenCV loads them with readNetFromTorch and executes
        inference on CPU by default.
        """

        model = Path(model_path)
        if not model.exists():
            raise FileNotFoundError(f"Model not found: {model}")
        if model.suffix.lower() != ".t7":
            raise ValueError("This implementation expects OpenCV-compatible .t7 Torch models.")

        original_h, original_w = content_bgr.shape[:2]
        resized = resize_to_width(content_bgr, width)
        out_h, out_w = resized.shape[:2]

        net = cv2.dnn.readNetFromTorch(str(model))
        net.setPreferableBackend(self.backend)
        net.setPreferableTarget(self.target)

        blob = cv2.dnn.blobFromImage(
            resized,
            scalefactor=1.0,
            size=(out_w, out_h),
            mean=MEAN_VALUES,
            swapRB=False,
            crop=False,
        )

        start = time.perf_counter()
        net.setInput(blob)
        output = net.forward()
        elapsed = time.perf_counter() - start

        output = output.reshape((3, output.shape[2], output.shape[3]))
        output[0] += MEAN_VALUES[0]
        output[1] += MEAN_VALUES[1]
        output[2] += MEAN_VALUES[2]
        stylized = output.transpose(1, 2, 0)
        stylized = _clip_to_uint8(stylized)

        if stylized.shape[:2] != resized.shape[:2]:
            stylized = cv2.resize(
                stylized,
                (resized.shape[1], resized.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )

        strength = float(np.clip(style_strength, 0.0, 1.0))
        if strength < 1.0:
            stylized = cv2.addWeighted(stylized, strength, resized, 1.0 - strength, 0.0)

        return TransferResult(
            stylized_image=stylized,
            resized_content=resized,
            inference_time=elapsed,
            original_size=(original_w, original_h),
            output_size=(out_w, out_h),
            model_name=model.stem,
        )
