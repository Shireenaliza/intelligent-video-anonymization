from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    similarity_threshold: float = 0.90
    blur_method: str = "gaussian"
    blur_strength: int = 31
    mosaic_block_size: int = 12
    device: str = "auto"

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
