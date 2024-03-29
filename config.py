from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "frames"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
VALID_DIR = DATA_DIR / "val"
LOG_DIR = Path(__file__).resolve().parent / "log"
MODEL_DIR = Path(__file__).resolve().parent / "model"

REC_LOSS = 0.8 * 255
PERCEP_LOSS = 0.005
WRAP_LOSS = 0.4 * 255
SMOOTH_LOSS = 1.0
