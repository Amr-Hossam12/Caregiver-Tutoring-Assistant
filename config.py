import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = "gemini-flash-lite-latest"

CHROMA_PERSIST_DIR = "./chroma_db"
TRAIN_CSV = "data/train.csv"
TEST_CSV = "data/test.csv"

# Routing thresholds
RAG_DISTANCE_THRESHOLD = 1.2

# Error types that trigger the Socratic agent (conceptual work needed)
CONCEPTUAL_ERROR_TYPES = {"conceptual_flaw", "comprehension_error", "procedural_error"}

# Error types that trigger the Hint agent (quick nudge needed)
CALCULATION_ERROR_TYPES = {"calculation_error"}
