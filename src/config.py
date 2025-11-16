import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EMBEDDINGS_DIR = OUTPUTS_DIR / "embeddings"
LOGS_DIR = OUTPUTS_DIR / "logs"

EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

LLM_PRIORITY_ORDER_STR = os.getenv("LLM_PRIORITY_ORDER", "openrouter,gemini,openai")
LLM_PRIORITY_ORDER = [p.strip() for p in LLM_PRIORITY_ORDER_STR.split(",") if p.strip()]

LLM_MODEL_OPENROUTER = os.getenv("LLM_MODEL_OPENROUTER", "openai/gpt-3.5-turbo")
LLM_MODEL_GEMINI = os.getenv("LLM_MODEL_GEMINI", "gemini-pro")
LLM_MODEL_OPENAI = os.getenv("LLM_MODEL_OPENAI", "gpt-3.5-turbo")

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

LLM_MODEL = os.getenv("LLM_MODEL", LLM_MODEL_OPENAI)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "local")
LOCAL_EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL", 
    "sentence-transformers/all-MiniLM-L6-v2"
)

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

FAQ_DOCUMENT_PATH = os.getenv("FAQ_DOCUMENT_PATH", str(DATA_DIR))
SUPPORTED_DOCUMENT_FORMATS = ['.txt', '.md', '.pdf', '.docx']
EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH", str(EMBEDDINGS_DIR / "faiss_index"))

TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "5"))
SEARCH_TYPE = os.getenv("SEARCH_TYPE", "similarity")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
CHUNKING_METHOD = os.getenv("CHUNKING_METHOD", "character")
TOKEN_ENCODING = os.getenv("TOKEN_ENCODING", "cl100k_base")

DOCUMENT_ID = os.getenv("DOCUMENT_ID", "faq_document_v1")
SIMILARITY_METRIC = os.getenv("SIMILARITY_METRIC", "cosine")

SEARCH_MODE = os.getenv("SEARCH_MODE", "hybrid").lower()
USE_HYBRID_SEARCH = SEARCH_MODE == "hybrid"
USE_KEYWORD_ONLY = SEARCH_MODE == "keyword"
USE_VECTOR_ONLY = SEARCH_MODE == "vector"

HYBRID_SEARCH_METHOD = os.getenv("HYBRID_SEARCH_METHOD", "retrieve_then_filter")
SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "0.7"))
KEYWORD_WEIGHT = float(os.getenv("KEYWORD_WEIGHT", "0.3"))

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
ENABLE_PROGRESS_BAR = os.getenv("ENABLE_PROGRESS_BAR", "true").lower() == "true"

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_INITIAL_DELAY = float(os.getenv("RETRY_INITIAL_DELAY", "1.0"))
RETRY_BACKOFF_FACTOR = float(os.getenv("RETRY_BACKOFF_FACTOR", "2.0"))

ENABLE_VERIFICATION = os.getenv("ENABLE_VERIFICATION", "true").lower() == "true"
VERIFICATION_QUERIES = os.getenv("VERIFICATION_QUERIES", "").split(",") if os.getenv("VERIFICATION_QUERIES") else []

USE_ANN = os.getenv("USE_ANN", "false").lower() == "true"
ANN_INDEX_TYPE = os.getenv("ANN_INDEX_TYPE", "HNSW")
ANN_THRESHOLD = int(os.getenv("ANN_THRESHOLD", "10000"))

ENABLE_INDEX_UPDATES = os.getenv("ENABLE_INDEX_UPDATES", "true").lower() == "true"

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logger = logging.getLogger(__name__)

USE_OPENAI_EMBEDDINGS = EMBEDDING_MODEL.lower() != "local"
if USE_OPENAI_EMBEDDINGS and not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY required when using OpenAI embeddings. "
        "Set EMBEDDING_MODEL=local to use local embeddings (recommended for confidential data)."
    )

if not OPENAI_API_KEY and not GEMINI_API_KEY and not OPENROUTER_API_KEY:
    logger.warning(
        "No LLM API keys set. Answer generation will fail. "
        "Set at least one: OPENROUTER_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY"
    )
elif not OPENAI_API_KEY:
    logger.info("OpenAI API key not set (optional if using OpenRouter or Gemini)")

if not Path(FAQ_DOCUMENT_PATH).exists():
    raise FileNotFoundError(
        f"FAQ document path not found at {FAQ_DOCUMENT_PATH}. "
        "Please ensure the directory or file exists."
    )

