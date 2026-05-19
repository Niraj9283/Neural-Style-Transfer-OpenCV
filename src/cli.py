from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from nst_opencv import StyleTransferEngine, build_side_by_side, load_image, save_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply Neural Style Transfer using OpenCV DNN and pretrained Torch models."
    )
    parser.add_argument("--image", required=True, help="Path to the input content image.")
    parser.add_argument("--model", required=True, help="Path to a .t7 style model.")
    parser.add_argument("--output", default=None, help="Output image path. Defaults to outputs/<name>_<style>.jpg.")
    parser.add_argument("--width", type=int, default=700, help="Processing width in pixels.")
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Blend strength from 0.0 to 1.0. Lower values preserve more original content color.",
    )
    parser.add_argument("--compare", action="store_true", help="Also save a side-by-side comparison image.")
    return parser.parse_args()


def default_output_path(image_path: Path, model_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / f"{image_path.stem}_{model_path.stem}_{timestamp}.jpg"


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    model_path = Path(args.model)
    output_path = Path(args.output) if args.output else default_output_path(image_path, model_path)

    content = load_image(image_path)
    engine = StyleTransferEngine()
    result = engine.apply(
        content,
        model_path=model_path,
        width=args.width,
        style_strength=args.strength,
    )
    saved_path = save_image(result.stylized_image, output_path)

    print("Neural Style Transfer completed")
    print(f"Input image     : {image_path}")
    print(f"Style model     : {model_path}")
    print(f"Output image    : {saved_path}")
    print(f"Original size   : {result.original_size[0]}x{result.original_size[1]}")
    print(f"Output size     : {result.output_size[0]}x{result.output_size[1]}")
    print(f"Inference time  : {result.inference_time:.2f} seconds")

    if args.compare:
        comparison = build_side_by_side(result.resized_content, result.stylized_image)
        compare_path = saved_path.with_name(f"{saved_path.stem}_comparison{saved_path.suffix}")
        save_image(comparison, compare_path)
        print(f"Comparison image: {compare_path}")


if __name__ == "__main__":
    main()
