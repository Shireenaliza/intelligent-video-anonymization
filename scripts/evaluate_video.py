import argparse
import json
from pathlib import Path

import cv2

from src.metrics import evaluate_pair


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare corresponding frames of "
            "an original and processed video."
        )
    )

    parser.add_argument(
        "--original",
        required=True,
    )

    parser.add_argument(
        "--processed",
        required=True,
    )

    parser.add_argument(
        "--output",
        default=(
            "results/metrics/"
            "video_metrics.json"
        ),
    )

    parser.add_argument(
        "--sample-every",
        type=int,
        default=10,
        help="Evaluate every Nth frame",
    )

    args = parser.parse_args()

    original = cv2.VideoCapture(
        args.original
    )

    processed = cv2.VideoCapture(
        args.processed
    )

    if (
        not original.isOpened()
        or not processed.isOpened()
    ):
        raise FileNotFoundError(
            "Could not open one or both videos."
        )

    rows = []
    frame_index = 0

    try:
        while True:
            ok_a, frame_a = original.read()
            ok_b, frame_b = processed.read()

            if not ok_a or not ok_b:
                break

            if frame_index % args.sample_every == 0:
                rows.append(
                    {
                        "frame": frame_index,
                        **evaluate_pair(
                            frame_a,
                            frame_b,
                        ),
                    }
                )

            frame_index += 1

    finally:
        original.release()
        processed.release()

    if not rows:
        raise RuntimeError(
            "No comparable frames were found."
        )

    metric_names = [
        "mse",
        "psnr",
        "ssim",
        "laplacian_variance",
        "tenengrad",
        "brenner",
    ]

    summary = {
        name: sum(
            row[name]
            for row in rows
        ) / len(rows)
        for name in metric_names
    }

    result = {
        "frames_evaluated": len(rows),
        "sample_every": args.sample_every,
        "summary_mean": summary,
        "per_frame": rows,
    }

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
