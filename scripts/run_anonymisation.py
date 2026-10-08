import argparse
import json

from src.config import Config
from src.pipeline import VideoAnonymizer


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Anonymize only the person matching "
            "a reference face image."
        )
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Input video path",
    )

    parser.add_argument(
        "--reference",
        required=True,
        help="Reference face image path",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output video path",
    )

    parser.add_argument(
        "--blur",
        default="gaussian",
        choices=[
            "gaussian",
            "mosaic",
            "dot",
            "triangle",
        ],
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Cosine similarity threshold",
    )

    parser.add_argument(
        "--strength",
        type=int,
        default=31,
        help="Blur kernel strength",
    )

    parser.add_argument(
        "--mosaic-block-size",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
    )

    args = parser.parse_args()

    config = Config(
        similarity_threshold=args.threshold,
        blur_method=args.blur,
        blur_strength=args.strength,
        mosaic_block_size=args.mosaic_block_size,
        device=args.device,
    )

    result = VideoAnonymizer(
        config
    ).anonymize(
        args.video,
        args.reference,
        args.output,
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
