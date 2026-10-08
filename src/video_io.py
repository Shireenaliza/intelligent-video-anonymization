from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple

import cv2


@dataclass
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int


def open_video(path: str | Path):
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    return cap


def get_video_info(cap) -> VideoInfo:
    return VideoInfo(
        fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )


def frame_iterator(cap) -> Iterator[Tuple[int, object]]:
    index = 0

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        yield index, frame
        index += 1


def make_writer(output_path: str | Path, info: VideoInfo):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        info.fps,
        (info.width, info.height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create output video: {output_path}"
        )

    return writer
