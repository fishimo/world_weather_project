from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainedModelArtifact:
    model: Any
    model_name: str
    feature_list: list[str] | None
    params: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelStore:
    def save(self, artifact: TrainedModelArtifact, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(artifact, f)
        return path

    def load(self, path: Path) -> TrainedModelArtifact:
        with path.open("rb") as f:
            artifact = pickle.load(f)

        if not isinstance(artifact, TrainedModelArtifact):
            raise TypeError("Loaded object is not TrainedModelArtifact")

        return artifact

    def save_numbered(
        self,
        artifact: TrainedModelArtifact,
        directory: Path,
        *,
        prefix: str = "model",
        digits: int = 4,
    ) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        next_number = self._next_model_number(directory, prefix=prefix, digits=digits)
        path = self._build_numbered_path(
            directory,
            prefix=prefix,
            number=next_number,
            digits=digits,
        )
        return self.save(artifact, path)

    def load_numbered(
        self,
        directory: Path,
        number: int,
        *,
        prefix: str = "model",
        digits: int = 4,
    ) -> TrainedModelArtifact:
        path = self._build_numbered_path(
            directory,
            prefix=prefix,
            number=number,
            digits=digits,
        )
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        return self.load(path)

    def _next_model_number(
        self,
        directory: Path,
        *,
        prefix: str,
        digits: int,
    ) -> int:
        pattern = re.compile(rf"^{re.escape(prefix)}_(\d{{{digits}}})\.pkl$")
        numbers: list[int] = []

        if directory.exists():
            for path in directory.iterdir():
                if not path.is_file():
                    continue
                match = pattern.match(path.name)
                if match is not None:
                    numbers.append(int(match.group(1)))

        return (max(numbers) + 1) if numbers else 1

    def _build_numbered_path(
        self,
        directory: Path,
        *,
        prefix: str,
        number: int,
        digits: int,
    ) -> Path:
        if number < 1:
            raise ValueError(f"number must be >= 1, got {number}")
        if digits < 1:
            raise ValueError(f"digits must be >= 1, got {digits}")
        filename = f"{prefix}_{number:0{digits}d}.pkl"
        return directory / filename
