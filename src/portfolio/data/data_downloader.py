from __future__ import annotations

from pathlib import Path
import subprocess
import sys


class DataDownloader:
    """Call Kaggle API and fetch raw weather data."""

    def __init__(self, config):
        self.config = config.data_download

    def raw_data_path(self) -> Path:
        return Path(self.config.out_dir) / "GlobalWeatherRepository.csv"

    def has_raw_data(self) -> bool:
        return self.raw_data_path().is_file()

    def data_download(self) -> None:
        """Download the dataset into the configured raw-data directory."""
        out_dir = Path(self.config.out_dir)
        dataset = str(self.config.datasets)

        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "kaggle.cli",
            "datasets",
            "download",
            "-d",
            dataset,
            "-p",
            str(out_dir),
            "--unzip",
            "--force",
        ]
        print(">", " ".join(cmd))

        try:
            completed = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            if completed.stdout.strip():
                print(completed.stdout.strip())
        except FileNotFoundError as e:
            raise RuntimeError(
                "kaggle コマンドが見つかりません。`uv add kaggle` 済みか、"
                "`uv run` 経由で実行しているか確認してください。"
            ) from e
        except subprocess.CalledProcessError as e:
            details = (e.stderr or e.stdout or "").strip()
            message = (
                "Kaggle download が失敗しました。"
                f" dataset={dataset}, out_dir={out_dir}"
            )
            if details:
                message = f"{message}, details={details}"
            raise RuntimeError(message) from e
