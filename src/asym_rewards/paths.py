from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path.home() / "models"))
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))

MODEL_DIRS = {
    "qwen2_5_0_5b": MODELS_DIR / "qwen2_5_0_5b",
    "qwen2_5_1_5b": MODELS_DIR / "qwen2_5_1_5b",
    "qwen2_5_math_1_5b": MODELS_DIR / "qwen2_5_math_1_5b",
    "qwen2_5_3b": MODELS_DIR / "qwen2_5_3b",
    # Non-Qwen baseline for cross-architecture fear-v2 testing.
    # Llama-3.2-1B base; 16 layers, hidden=2048. The CLAUDE.md note about
    # Llama-family bootstrapping struggles applied to the project sweep design;
    # this entry is for the exploratory fear-v2 generalization probe only.
    "llama3_2_1b": MODELS_DIR / "llama3_2_1b",
}
