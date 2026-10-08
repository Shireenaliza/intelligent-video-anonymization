from typing import Optional, Tuple

import numpy as np
from facenet_pytorch import MTCNN


class FaceDetector:
    def __init__(self, device: str = "cpu", image_size: int = 160):
        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=0,
            keep_all=True,
            post_process=True,
            device=device,
        )

    def detect(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        boxes, probs = self.mtcnn.detect(frame)
        return boxes, probs

    def extract_faces(self, frame: np.ndarray):
        return self.mtcnn(frame)
