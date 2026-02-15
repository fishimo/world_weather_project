from __future__ import annotations

from pathlib import Path
import subprocess


class DataDownloader:
    """APIをたたいてdataを取り込むためのクラス"""
    def __init__(self, config):
        self.config = config.data_download

    def data_download(self) -> None:
        """
        kaggle APIたたいてdata/raw/kaggle-world-weatherに置く
        """
        out_dir: Path = self.config.out_dir
        datasets: str = self.config.dataset
        
        out_dir.mkdir(parents=True, exist_ok=True)
    

        cmd = [
            "kaggle", 
            "datasets", 
            "download", 
            "-d", 
            datasets, 
            "-p", 
            str(out_dir), 
            "--unzip", 
            "--force"
        ]
        print(">", " ".join(cmd))

        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError as e:
            raise RuntimeError(
                "kaggle コマンドが見つかりません。`uv add kaggle` 済みか、`uv run` 経由で実行しているか確認してください。"
            ) from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError("Kaggle download が失敗しました。dataset名や認証設定を確認してください。") from e
        
    