from __future__ import annotations
import subprocess

def main() -> None:
    subprocess.run(["kaggle", "--version"], check=True)
    # 軽い一覧取得
    subprocess.run(["kaggle", "datasets", "list", "-s", "weather", "--max-size", "1"], check=True)
    print("Kaggle API OK")

if __name__ == "__main__":
    main()