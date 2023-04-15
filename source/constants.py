from pathlib import Path

# ==== Paths ====
ROOT_DIR = Path(__file__).parents[1]
PATH_DATA = ROOT_DIR / "data"
MLFLOW_TRACKING_URI = ROOT_DIR / "mlruns"
