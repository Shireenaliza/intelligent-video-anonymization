from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from facenet_pytorch import InceptionResnetV1


class FaceRecognizer:
    def __init__(self, device: str = "cpu", pretrained: str = "vggface2"):
        self.device = device
        self.model = (
            InceptionResnetV1(pretrained=pretrained)
            .eval()
            .to(device)
        )

    @torch.inference_mode()
    def embedding(self, faces: torch.Tensor) -> torch.Tensor:
        if faces.ndim == 3:
            faces = faces.unsqueeze(0)
        faces = faces.to(self.device)
        embeddings = self.model(faces)
        return F.normalize(embeddings, p=2, dim=1)

    def reference_embedding(self, reference_path: str | Path, detector) -> torch.Tensor:
        image = np.asarray(Image.open(reference_path).convert("RGB"))
        faces = detector.extract_faces(image)

        if faces is None:
            raise ValueError("No face was detected in the reference image.")

        if faces.ndim == 3:
            faces = faces.unsqueeze(0)

        if len(faces) != 1:
            raise ValueError(
                f"Expected exactly one face in the reference image; found {len(faces)}."
            )

        return self.embedding(faces)[0]

    @staticmethod
    def cosine_similarity(
        reference: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        if candidates.ndim == 1:
            candidates = candidates.unsqueeze(0)
        return F.cosine_similarity(
            reference.unsqueeze(0),
            candidates,
            dim=1,
        )
