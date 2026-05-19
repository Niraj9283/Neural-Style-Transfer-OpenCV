from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import cv2
from flask import Flask, render_template, request, send_from_directory, url_for
import numpy as np
from werkzeug.utils import secure_filename


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nst_opencv import StyleTransferEngine, list_style_models, load_image, save_image  # noqa: E402


MODELS_DIR = ROOT / "models"


def ensure_models_downloaded() -> None:
    """Ensure at least one .t7 model exists so the UI can populate styles."""
    # If models are already present, do nothing.
    if list_style_models(MODELS_DIR):
        return

    # Avoid network issues on some hosts by defaulting to a single small model.
    # Users can change this behavior by editing scripts/download_models.py.
    try:
        import subprocess

        subprocess.check_call(
            [
                sys.executable,
                "scripts/download_models.py",
                "--model",
                "candy",
            ],
            cwd=str(ROOT),
        )
    except Exception:
        # Swallow errors so the server still starts and shows a clear UI error.
        pass
SAMPLES_DIR = ROOT / "assets" / "sample_content"
OUTPUTS_DIR = ROOT / "outputs"
UPLOADS_DIR = ROOT / "uploads"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


def clamp_int(value: str | None, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value or default)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def clamp_float(value: str | None, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value or default)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def load_sample_options() -> list[Path]:
    if not SAMPLES_DIR.exists():
        return []
    return sorted(path for path in SAMPLES_DIR.iterdir() if path.suffix.lower() in ALLOWED_EXTENSIONS)


def decode_uploaded_image(file_storage) -> np.ndarray:
    data = file_storage.read()
    encoded = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Uploaded file is not a valid image.")
    return image


def find_by_name(paths: list[Path], selected_name: str | None) -> Path | None:
    if not paths:
        return None
    for path in paths:
        if path.name == selected_name:
            return path
    return paths[0]


def display_name(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").title()


@app.route("/", methods=["GET", "POST"])
def index():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    models = list_style_models(MODELS_DIR)
    samples = load_sample_options()

    selected_model = find_by_name(models, request.form.get("model"))
    selected_sample = find_by_name(samples, request.form.get("sample"))
    width = clamp_int(request.form.get("width"), default=700, minimum=320, maximum=1200)
    strength = clamp_float(request.form.get("strength"), default=1.0, minimum=0.2, maximum=1.0)

    error = None
    result_url = None
    content_url = url_for("sample_file", filename=selected_sample.name) if selected_sample else None
    metrics = None

    if request.method == "POST":
        if selected_model is None:
            error = "No .t7 model found in the models folder."
        else:
            try:
                uploaded = request.files.get("image")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                if uploaded and uploaded.filename:
                    suffix = Path(uploaded.filename).suffix.lower()
                    if suffix not in ALLOWED_EXTENSIONS:
                        raise ValueError("Unsupported image type. Use JPG, PNG, BMP, or WEBP.")

                    content_bgr = decode_uploaded_image(uploaded)
                    safe_stem = secure_filename(Path(uploaded.filename).stem) or "uploaded_image"
                    content_path = UPLOADS_DIR / f"{safe_stem}_{timestamp}.jpg"
                    save_image(content_bgr, content_path)
                    content_url = url_for("upload_file", filename=content_path.name)
                    image_name = safe_stem
                elif selected_sample is not None:
                    content_bgr = load_image(selected_sample)
                    image_name = selected_sample.stem
                    content_url = url_for("sample_file", filename=selected_sample.name)
                else:
                    raise ValueError("Please add a sample image or upload an image.")

                engine = StyleTransferEngine()
                result = engine.apply(
                    content_bgr,
                    model_path=selected_model,
                    width=width,
                    style_strength=strength,
                )
                output_path = OUTPUTS_DIR / f"{image_name}_{selected_model.stem}_{timestamp}.jpg"
                save_image(result.stylized_image, output_path)
                result_url = url_for("output_file", filename=output_path.name)
                metrics = {
                    "model": display_name(selected_model),
                    "time": f"{result.inference_time:.2f} seconds",
                    "size": f"{result.output_size[0]} x {result.output_size[1]}",
                }
            except Exception as exc:  # noqa: BLE001
                error = str(exc)

    return render_template(
        "index.html",
        models=models,
        samples=samples,
        selected_model=selected_model,
        selected_sample=selected_sample,
        width=width,
        strength=strength,
        result_url=result_url,
        content_url=content_url,
        metrics=metrics,
        error=error,
        display_name=display_name,
    )


@app.route("/samples/<path:filename>")
def sample_file(filename: str):
    return send_from_directory(SAMPLES_DIR, filename)


@app.route("/outputs/<path:filename>")
def output_file(filename: str):
    return send_from_directory(OUTPUTS_DIR, filename)


@app.route("/uploads/<path:filename>")
def upload_file(filename: str):
    return send_from_directory(UPLOADS_DIR, filename)


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    ensure_models_downloaded()
    app.run(host="0.0.0.0", port=5000, debug=False)
