from pathlib import Path

# all paths in one place — hardcoding these across files is a pain

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
NGA_DIR = RAW_DIR / "nga"
IMAGE_DIR = RAW_DIR / "images"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"

OBJECTS_CSV = NGA_DIR / "objects.csv"
PUBLISHED_IMAGES_CSV = NGA_DIR / "published_images.csv"
ALLOWED_CLASSIFICATIONS = ["Painting"]

TITLE_KEYWORDS = [
    "portrait",
    "self-portrait",
]

DOWNLOAD_SIZE = "!800,800"

RANDOM_SEED = 42
REQUEST_TIMEOUT = 20
SLEEP_BETWEEN_DOWNLOADS = 0.05

DEFAULT_MODEL = "vit"
DEFAULT_FEATURE_MODE = "full"
SUPPORTED_MODELS = ["vit", "resnet50", "resnet18"]
