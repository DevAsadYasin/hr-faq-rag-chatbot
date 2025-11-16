import logging
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import numpy as np

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from .config import (
    USE_HYBRID_SEARCH,
    USE_KEYWORD_ONLY,
    USE_VECTOR_ONLY,
    HYBRID_SEARCH_METHOD,
    SEMANTIC_WEIGHT,
    KEYWORD_WEIGHT,
    TOP_K_CHUNKS,
)
from .utils import normalize_scores
from .metadata_extractor import parse_query_filters

logger = logging.getLogger(__name__)


class BM25Index:
    def __init__(self, documents: List[Document]):
        self.documents = documents
        tokenized_docs = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        logger.info(f"BM25 index created with {len(documents)} documents")
    
    def search(self, query: str, top_k: int = None) -> List[tuple[Document, float]]:
        top_k = top_k or TOP_K_CHUNKS
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = [
            (self.documents[idx], float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0
        ]
        
        return results


def filter_documents_by_metadata(documents: List[Document], filters: Dict[str, Any]) -> List[Document]:
    if not filters:
        return documents
    
    filtered = []
    for doc in documents:
        match = True
        for key, value in filters.items():
            doc_value = doc.metadata.get(key)
            
            if doc_value is None:
                continue
            
            if isinstance(doc_value, list):
                if value not in doc_value:
                    match = False
                    break
            elif doc_value != value:
                match = False
                break
        
        if match:
            filtered.append(doc)
    
    logger.info(f"Filtered {len(documents)} documents to {len(filtered)} using filters: {filters}")
    return filtered


def retrieve_then_filter(vectorstore: FAISS, query: str, filters: Dict[str, Any] = None,
                         top_k: int = None, fetch_k: int = None) -> List[Document]:
    top_k = top_k or TOP_K_CHUNKS
    fetch_k = fetch_k or (top_k * 4)
    
    candidates = vectorstore.similarity_search(query, k=fetch_k)
    logger.debug(f"Retrieved {len(candidates)} candidates via semantic search")
    
    if filters:
        filtered = filter_documents_by_metadata(candidates, filters)
        logger.info(f"Filtered {len(candidates)} candidates to {len(filtered)} using filters: {filters}")
        
        if len(filtered) == 0 and len(candidates) > 0:
            logger.warning(f"Filters removed all candidates. Returning top {min(top_k, len(candidates))} unfiltered results")
            return candidates[:top_k]
    else:
        filtered = candidates
    
    final_results = filtered[:top_k]
    
    if len(final_results) < top_k and len(final_results) > 0:
        logger.warning(f"Only {len(final_results)} results after filtering (requested {top_k})")
    elif len(final_results) == 0:
        logger.warning(f"No results after filtering. Returning top {min(top_k, len(candidates))} unfiltered candidates")
        return candidates[:top_k]
    
    return final_results


def weighted_hybrid_search(vectorstore: FAISS, bm25_index: BM25Index, query: str,
                           filters: Dict[str, Any] = None, top_k: int = None,
                           semantic_weight: float = None, keyword_weight: float = None) -> List[Document]:
    top_k = top_k or TOP_K_CHUNKS
    semantic_weight = semantic_weight or SEMANTIC_WEIGHT
    keyword_weight = keyword_weight or KEYWORD_WEIGHT
    
    total_weight = semantic_weight + keyword_weight
    if total_weight > 0:
        semantic_weight = semantic_weight / total_weight
        keyword_weight = keyword_weight / total_weight
    
    try:
        semantic_results = vectorstore.similarity_search_with_score(query, k=top_k * 2)
        semantic_scores = {doc.metadata.get("chunk_id"): score for doc, score in semantic_results}
    except (AttributeError, TypeError) as e:
        logger.debug(f"similarity_search_with_score not available: {e}. Using similarity_search.")
        semantic_results = vectorstore.similarity_search(query, k=top_k * 2)
        semantic_scores = {doc.metadata.get("chunk_id"): 1.0 for doc in semantic_results}
    
    keyword_results = bm25_index.search(query, top_k=top_k * 2)
    keyword_scores = {doc.metadata.get("chunk_id"): score for doc, score in keyword_results}
    
    all_chunk_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())
    
    semantic_scores_list = [semantic_scores.get(cid, 0.0) for cid in all_chunk_ids]
    keyword_scores_list = [keyword_scores.get(cid, 0.0) for cid in all_chunk_ids]
    
    if semantic_scores_list:
        max_semantic = max(semantic_scores_list)
        min_semantic = min(semantic_scores_list)
        if max_semantic > min_semantic:
            semantic_scores_list = [(max_semantic - s) / (max_semantic - min_semantic) 
                                   for s in semantic_scores_list]
        else:
            semantic_scores_list = [1.0] * len(semantic_scores_list)
    
    keyword_scores_list = normalize_scores(keyword_scores_list)
    
    combined_scores = {}
    chunk_id_list = list(all_chunk_ids)
    for i, cid in enumerate(chunk_id_list):
        semantic_score = semantic_scores_list[i] if i < len(semantic_scores_list) else 0.0
        keyword_score = keyword_scores_list[i] if i < len(keyword_scores_list) else 0.0
        
        combined_score = (semantic_weight * semantic_score) + (keyword_weight * keyword_score)
        combined_scores[cid] = combined_score
    
    sorted_chunk_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
    
    all_documents = {doc.metadata.get("chunk_id"): doc for doc, _ in semantic_results}
    all_documents.update({doc.metadata.get("chunk_id"): doc for doc, _ in keyword_results})
    
    ranked_documents = [all_documents[cid] for cid in sorted_chunk_ids if cid in all_documents]
    
    if filters:
        ranked_documents = filter_documents_by_metadata(ranked_documents, filters)
    
    final_results = ranked_documents[:top_k]
    
    logger.info(f"Weighted hybrid search: semantic_weight={semantic_weight:.2f}, "
               f"keyword_weight={keyword_weight:.2f}, returned {len(final_results)} results")
    
    return final_results


def hybrid_search(vectorstore: FAISS, bm25_index: BM25Index, query: str,
                 filters: Dict[str, Any] = None, top_k: int = None) -> List[Document]:
    
    if HYBRID_SEARCH_METHOD == "weighted":
        return weighted_hybrid_search(vectorstore, bm25_index, query, filters, top_k)
    else:
        return retrieve_then_filter(vectorstore, query, filters, top_k)

