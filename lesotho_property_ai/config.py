from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AppConfig:
    base_dir: Path
    generated_dir: Path
    data_dir: Path
    image_dir: Path
    output_dir: Path
    structured_weight: float = 0.45
    text_weight: float = 0.35
    vision_weight: float = 0.20

    @classmethod
    def from_base_dir(cls, base_dir: Path) -> "AppConfig":
        base = Path(base_dir).resolve()
        generated = base / "generated"
        return cls(
            base_dir=base,
            generated_dir=generated,
            data_dir=generated / "data",
            image_dir=generated / "images",
            output_dir=generated / "artifacts",
        )

    def ensure_directories(self) -> None:
        for path in (self.generated_dir, self.data_dir, self.image_dir, self.output_dir):
            path.mkdir(parents=True, exist_ok=True)
