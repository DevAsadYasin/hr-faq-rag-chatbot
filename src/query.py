import json
import logging
import sys
from typing import Dict, List, Any

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from pathlib import Path
from config import (
    EMBEDDINGS_PATH,
    EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_API_KEY,
    TOP_K_CHUNKS,
    SEARCH_TYPE,
    SIMILARITY_THRESHOLD,
    LLM_MODEL,
    LLM_TEMPERATURE,
    USE_HYBRID_SEARCH,
    USE_KEYWORD_ONLY,
    USE_VECTOR_ONLY,
    SEARCH_MODE,
    HYBRID_SEARCH_METHOD,
)
from hybrid_search import BM25Index, hybrid_search
from metadata_extractor import parse_query_filters
from utils import retry_with_backoff
from llm_manager import LLMManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/logs/query.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def get_embeddings():
    if EMBEDDING_MODEL.lower() == "local":
        logger.info(f"Using LOCAL embeddings: {LOCAL_EMBEDDING_MODEL}")
        logger.info("✓ Embeddings generated locally - no data sent externally")
        embeddings = HuggingFaceEmbeddings(
            model_name=LOCAL_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    else:
        logger.info(f"Using OpenAI embeddings: {EMBEDDING_MODEL}")
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY
        )
    return embeddings


@retry_with_backoff(max_retries=3)
def load_vector_store(embeddings):
    logger.info(f"Loading vector store from: {EMBEDDINGS_PATH}")
    try:
        vectorstore = FAISS.load_local(
            EMBEDDINGS_PATH, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("Vector store loaded successfully")
        return vectorstore
    except Exception as e:
        logger.error(f"Error loading vector store: {str(e)}")
        logger.error("Please run build_index.py first to create the index")
        raise




def create_bm25_index_from_vectorstore(vectorstore: FAISS) -> BM25Index:
    logger.info("Creating BM25 index...")
    
    bm25_data_path = Path(EMBEDDINGS_PATH).parent / "bm25_documents.json"
    
    if bm25_data_path.exists():
        try:
            with open(bm25_data_path, "r") as f:
                bm25_data = json.load(f)
            
            documents = [
                Document(
                    page_content=item["text"],
                    metadata=item["metadata"]
                )
                for item in bm25_data["documents"]
            ]
            
            logger.info(f"Loaded {len(documents)} documents from BM25 data file")
            return BM25Index(documents)
        except Exception as e:
            logger.warning(f"Could not load BM25 data: {e}. Creating from vector store...")
    
    logger.info("Creating BM25 index from vector store (sampling documents)...")
    all_docs = []
    
    sample_queries = [
        "vacation", "leave", "pto", "benefits", "payroll", "timesheet",
        "employee portal", "expense", "reimbursement", "hr", "policy",
        "performance", "review", "compliance", "401k", "insurance", "health"
    ]
    
    seen_ids = set()
    for query in sample_queries:
        try:
            results = vectorstore.similarity_search(query, k=30)
            for doc in results:
                doc_id = doc.metadata.get("chunk_id")
                if doc_id and doc_id not in seen_ids:
                    all_docs.append(doc)
                    seen_ids.add(doc_id)
        except Exception as e:
            logger.debug(f"Error retrieving documents for query '{query}': {e}")
            continue
    
    if not all_docs:
        logger.warning("Could not retrieve documents for BM25 index")
        return None
    
    logger.info(f"Created BM25 index with {len(all_docs)} documents (sampled from vector store)")
    return BM25Index(all_docs)


class HybridRetriever:
    def __init__(self, vectorstore: FAISS, bm25_index: BM25Index = None, 
                 filters: Dict[str, Any] = None, top_k: int = None):
        self.vectorstore = vectorstore
        self.bm25_index = bm25_index
        self.filters = filters or {}
        self.top_k = top_k or TOP_K_CHUNKS
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        search_terms, query_filters = parse_query_filters(query)
        combined_filters = {**self.filters, **query_filters}
        
        if USE_KEYWORD_ONLY and self.bm25_index:
            logger.info("Using keyword-only (BM25) search")
            keyword_results = self.bm25_index.search(search_terms, top_k=self.top_k * 2)
            results = [doc for doc, score in keyword_results]
            if combined_filters:
                from hybrid_search import filter_documents_by_metadata
                results = filter_documents_by_metadata(results, combined_filters)
            results = results[:self.top_k]
        elif USE_VECTOR_ONLY:
            logger.info("Using vector-only (semantic) search")
            if combined_filters:
                candidates = self.vectorstore.similarity_search(search_terms, k=self.top_k * 4)
                from hybrid_search import filter_documents_by_metadata
                results = filter_documents_by_metadata(candidates, combined_filters)[:self.top_k]
            else:
                results = self.vectorstore.similarity_search(search_terms, k=self.top_k)
        elif USE_HYBRID_SEARCH and self.bm25_index:
            logger.info("Using hybrid search (vector + keyword)")
            results = hybrid_search(
                self.vectorstore,
                self.bm25_index,
                search_terms,
                combined_filters,
                self.top_k
            )
        else:
            logger.info("Falling back to vector-only search")
            if combined_filters:
                candidates = self.vectorstore.similarity_search(search_terms, k=self.top_k * 4)
                from hybrid_search import filter_documents_by_metadata
                results = filter_documents_by_metadata(candidates, combined_filters)[:self.top_k]
            else:
                results = self.vectorstore.similarity_search(search_terms, k=self.top_k)
        
        return results


def create_retriever(vectorstore, bm25_index: BM25Index = None, 
                    search_type: str = None, top_k: int = None, 
                    threshold: float = None, filters: Dict[str, Any] = None):
    top_k = top_k or TOP_K_CHUNKS
    
    if USE_HYBRID_SEARCH or USE_KEYWORD_ONLY or USE_VECTOR_ONLY:
        logger.info(f"Using {SEARCH_MODE.upper()} search mode")
        return HybridRetriever(vectorstore, bm25_index, filters, top_k)
    
    search_type = search_type or SEARCH_TYPE
    search_kwargs = {"k": top_k}
    
    if search_type == "similarity_score_threshold":
        search_kwargs["score_threshold"] = threshold or SIMILARITY_THRESHOLD
        logger.info(f"Using LangChain {search_type} retriever: k={top_k}, threshold={search_kwargs['score_threshold']}")
    elif search_type == "mmr":
        search_kwargs["fetch_k"] = top_k * 3
        logger.info(f"Using LangChain {search_type} retriever: k={top_k}, fetch_k={search_kwargs['fetch_k']}")
    else:
        logger.info(f"Using LangChain {search_type} retriever: k={top_k}")
    
    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )
    
    return retriever


# Initialize LLM manager (singleton pattern)
_llm_manager = None

def get_llm_manager() -> LLMManager:
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager()
    return _llm_manager


def create_qa_chain(retriever, llm_model: str = None, temperature: float = None):
    prompt_template = """You are a helpful HR support assistant. 
Use the following pieces of context from the HR documentation to answer the question.
If you don't know the answer based on the provided context, say that you don't know. 
Do not make up information or use knowledge outside the provided context.

Context from documentation:
{context}

Question: {question}

Provide a clear, accurate answer based only on the context above.
Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    llm_manager = get_llm_manager()
    
    class QAChainWrapper:
        def __init__(self, retriever, prompt, llm_manager):
            self.retriever = retriever
            self.prompt = prompt
            self.llm_manager = llm_manager
        
        def __call__(self, inputs):
            question = inputs.get("query", "")
            docs = self.retriever.get_relevant_documents(question)
            llm = self.llm_manager.get_llm(model_override=llm_model)
            
            if not llm:
                return {
                    "result": "I found relevant information but cannot generate an answer. Please check the source documents below.",
                    "source_documents": docs
                }
            
            try:
                context = "\n\n".join(doc.page_content for doc in docs)
                from langchain_core.runnables import RunnablePassthrough
                from langchain_core.output_parsers import StrOutputParser
                
                rag_chain = (
                    {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                    | self.prompt
                    | llm
                    | StrOutputParser()
                )
                
                answer = rag_chain.invoke({"context": context, "question": question})
                
                return {
                    "result": answer,
                    "source_documents": docs
                }
                
            except Exception as e:
                logger.error(f"LLM generation failed: {str(e)}")
                result = self.llm_manager.invoke_with_fallback(
                    {"retriever": self.retriever, "prompt": self.prompt},
                    question
                )
                
                return {
                    "result": result.get("result", "Error generating answer"),
                    "source_documents": result.get("source_documents", docs),
                    "provider_used": result.get("provider_used"),
                    "error": result.get("error")
                }
    
    return QAChainWrapper(retriever, PROMPT, llm_manager)


def format_response(user_question: str, result: Dict[str, Any]) -> Dict[str, Any]:
    chunks_related = []
    provider_used = result.get("provider_used")
    error_info = result.get("error")
    
    for i, doc in enumerate(result.get("source_documents", [])):
        chunk_info = {
            "chunk_id": doc.metadata.get("chunk_id", f"chunk_{i}"),
            "chunk_text": doc.page_content,
            "metadata": {
                "document_id": doc.metadata.get("document_id"),
                "chunk_index": doc.metadata.get("chunk_index", i),
                "char_start": doc.metadata.get("char_start"),
                "char_end": doc.metadata.get("char_end"),
                "start_index": doc.metadata.get("start_index"),
                "chunk_size": doc.metadata.get("chunk_size", len(doc.page_content)),
                "token_count": doc.metadata.get("token_count"),
                "total_chunks": doc.metadata.get("total_chunks"),
                "created_at": doc.metadata.get("created_at"),
            }
        }
        chunks_related.append(chunk_info)
    
    response = {
        "user_question": user_question,
        "system_answer": result.get("result", ""),
        "chunks_related": chunks_related
    }
    
    if provider_used:
        response["llm_provider"] = provider_used
    if error_info:
        response["error"] = error_info
    
    return response


@retry_with_backoff(max_retries=2)
def query_faq(user_question: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
    logger.info(f"Processing question: {user_question}")
    
    try:
        search_terms, query_filters = parse_query_filters(user_question)
        combined_filters = {**(filters or {}), **query_filters}
        
        if combined_filters:
            logger.info(f"Applied filters: {combined_filters}")
        
        embeddings = get_embeddings()
        vectorstore = load_vector_store(embeddings)
        
        bm25_index = None
        if USE_HYBRID_SEARCH or USE_KEYWORD_ONLY:
            bm25_index = create_bm25_index_from_vectorstore(vectorstore)
            if bm25_index:
                logger.info("BM25 index created for search")
        
        retriever = create_retriever(
            vectorstore,
            bm25_index=bm25_index,
            search_type=SEARCH_TYPE,
            top_k=TOP_K_CHUNKS,
            threshold=SIMILARITY_THRESHOLD,
            filters=combined_filters
        )
        
        qa_chain = create_qa_chain(retriever, LLM_MODEL, LLM_TEMPERATURE)
        
        query_text = search_terms if query_filters else user_question
        logger.info(f"Retrieving relevant chunks and generating answer (query: '{query_text[:50]}...')...")
        
        result = qa_chain({"query": query_text})
        response = format_response(user_question, result)
        
        logger.info(f"Retrieved {len(response['chunks_related'])} relevant chunks")
        logger.info("Answer generated successfully")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise


def main():
    if len(sys.argv) > 1:
        user_question = " ".join(sys.argv[1:])
    else:
        user_question = "How do I request vacation time?"
        logger.info("No question provided, using sample question")
    
    logger.info("=" * 60)
    logger.info("FAQ Query Pipeline")
    logger.info("=" * 60)
    
    try:
        response = query_faq(user_question)
        
        print("\n" + "=" * 60)
        print("RESPONSE:")
        print("=" * 60)
        print(json.dumps(response, indent=2, ensure_ascii=False))
        
        output_file = "outputs/sample_queries.json"
        with open(output_file, "w") as f:
            json.dump([response], f, indent=2, ensure_ascii=False)
        logger.info(f"\nResponse saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

