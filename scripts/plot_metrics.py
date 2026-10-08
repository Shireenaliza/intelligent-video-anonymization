import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Plot blur comparison metrics."
    )

    parser.add_argument(
        "--input",
        default=(
            "results/"
            "blur_comparison/"
            "metrics.json"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "results/"
            "blur_comparison/"
            "plots"
        ),
    )

    args = parser.parse_args()

    data = json.loads(
        Path(args.input).read_text(
            encoding="utf-8"
        )
    )

    methods = list(
        data.keys()
    )

    metrics = [
        "psnr",
        "ssim",
        "mse",
        "laplacian_variance",
        "tenengrad",
        "brenner",
    ]

    output = Path(
        args.output_dir
    )
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    for metric in metrics:
        values = [
            data[method][metric]
            for method in methods
        ]

        plt.figure(
            figsize=(8, 5)
        )

        plt.bar(
            methods,
            values,
        )

        plt.title(
            metric.replace(
                "_",
                " ",
            ).title()
        )

        plt.ylabel(metric)
        plt.tight_layout()

        plt.savefig(
            output
            / f"{metric}.png",
            dpi=200,
        )

        plt.close()


if __name__ == "__main__":
    main()
