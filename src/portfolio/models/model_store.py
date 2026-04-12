from __future__ import annotations

import pickle
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
    def save(self, artifact: TrainedModelArtifact, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(artifact, f)

    def load(self, path: Path) -> TrainedModelArtifact:
        with path.open("rb") as f:
            artifact = pickle.load(f)

        if not isinstance(artifact, TrainedModelArtifact):
            raise TypeError("Loaded object is not TrainedModelArtifact")

        return artifact
