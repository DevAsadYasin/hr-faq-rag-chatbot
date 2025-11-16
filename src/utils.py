import time
import logging
from functools import wraps
from typing import Callable, Any, List
import numpy as np

from .config import MAX_RETRIES, RETRY_INITIAL_DELAY, RETRY_BACKOFF_FACTOR

logger = logging.getLogger(__name__)


def retry_with_backoff(max_retries: int = None, initial_delay: float = None, 
                       backoff_factor: float = None, exceptions: tuple = (Exception,)):
    max_retries = max_retries or MAX_RETRIES
    initial_delay = initial_delay or RETRY_INITIAL_DELAY
    backoff_factor = backoff_factor or RETRY_BACKOFF_FACTOR
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = initial_delay
            last_exception = None
            
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if retries == max_retries - 1:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                        raise
                    
                    logger.warning(f"Attempt {retries + 1}/{max_retries} failed for {func.__name__}: {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay *= backoff_factor
                    retries += 1
            
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


def validate_vector_dimensions(vectors: List[List[float]], expected_dim: int = None) -> int:
    if not vectors:
        raise ValueError("Empty vector list provided")
    
    if expected_dim is None:
        expected_dim = len(vectors[0])
        logger.info(f"Detected embedding dimension: {expected_dim}")
    
    for idx, vec in enumerate(vectors):
        if len(vec) != expected_dim:
            raise ValueError(
                f"Vector at index {idx} has dimension {len(vec)} but expected {expected_dim}. "
                f"All vectors must have consistent dimensions."
            )
    
    logger.debug(f"Validated {len(vectors)} vectors, all have dimension {expected_dim}")
    return expected_dim


def normalize_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []
    
    scores_array = np.array(scores)
    min_score = scores_array.min()
    max_score = scores_array.max()
    
    if max_score == min_score:
        return [1.0] * len(scores)
    
    normalized = (scores_array - min_score) / (max_score - min_score)
    return normalized.tolist()

