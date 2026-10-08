import argparse
import json
from pathlib import Path

import cv2

from src.blur import apply_blur
from src.metrics import evaluate_pair


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare Gaussian, Mosaic, Dot and "
            "Triangle blur on one image."
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Face crop or sample image",
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "results/"
            "blur_comparison"
        ),
    )

    parser.add_argument(
        "--strength",
        type=int,
        default=31,
    )

    parser.add_argument(
        "--mosaic-block-size",
        type=int,
        default=12,
    )

    args = parser.parse_args()

    image = cv2.imread(
        args.image
    )

    if image is None:
        raise FileNotFoundError(
            args.image
        )

    out_dir = Path(
        args.output_dir
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = {}

    for method in [
        "gaussian",
        "mosaic",
        "dot",
        "triangle",
    ]:
        blurred = apply_blur(
            image,
            method=method,
            strength=args.strength,
            mosaic_block_size=(
                args.mosaic_block_size
            ),
        )

        cv2.imwrite(
            str(
                out_dir
                / f"{method}.png"
            ),
            blurred,
        )

        results[method] = evaluate_pair(
            image,
            blurred,
        )

    (out_dir / "metrics.json").write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            results,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
