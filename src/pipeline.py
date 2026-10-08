from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .blur import apply_blur
from .config import Config
from .detector import FaceDetector
from .recognizer import FaceRecognizer
from .video_io import (
    frame_iterator,
    get_video_info,
    make_writer,
    open_video,
)


class VideoAnonymizer:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.device = self.config.resolve_device()

        self.detector = FaceDetector(
            device=self.device
        )

        self.recognizer = FaceRecognizer(
            device=self.device
        )

    @staticmethod
    def _clip_box(
        box,
        width: int,
        height: int,
    ):
        x1, y1, x2, y2 = [
            int(round(v))
            for v in box
        ]

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))

        return x1, y1, x2, y2

    def anonymize(
        self,
        video_path: str | Path,
        reference_path: str | Path,
        output_path: str | Path,
    ) -> Dict[str, object]:

        reference_embedding = (
            self.recognizer.reference_embedding(
                reference_path,
                self.detector,
            )
        )

        cap = open_video(video_path)
        info = get_video_info(cap)
        writer = make_writer(
            output_path,
            info,
        )

        processed_frames = 0
        matched_faces = 0

        try:
            for _, frame in frame_iterator(cap):
                faces = self.detector.extract_faces(frame)
                boxes, _ = self.detector.detect(frame)

                if (
                    faces is not None
                    and boxes is not None
                    and len(faces) == len(boxes)
                ):
                    embeddings = (
                        self.recognizer.embedding(faces)
                    )

                    similarities = (
                        self.recognizer.cosine_similarity(
                            reference_embedding,
                            embeddings,
                        )
                    )

                    for box, score in zip(
                        boxes,
                        similarities.tolist(),
                    ):
                        if score >= self.config.similarity_threshold:
                            x1, y1, x2, y2 = (
                                self._clip_box(
                                    box,
                                    frame.shape[1],
                                    frame.shape[0],
                                )
                            )

                            roi = frame[
                                y1:y2,
                                x1:x2,
                            ]

                            frame[
                                y1:y2,
                                x1:x2,
                            ] = apply_blur(
                                roi,
                                method=self.config.blur_method,
                                strength=self.config.blur_strength,
                                mosaic_block_size=(
                                    self.config.mosaic_block_size
                                ),
                            )

                            matched_faces += 1

                writer.write(frame)
                processed_frames += 1

        finally:
            cap.release()
            writer.release()

        return {
            "output": str(output_path),
            "device": self.device,
            "frames_processed": processed_frames,
            "matched_faces": matched_faces,
            "similarity_threshold": (
                self.config.similarity_threshold
            ),
            "blur_method": self.config.blur_method,
        }
