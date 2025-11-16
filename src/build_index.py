import logging
import sys
from pathlib import Path
from datetime import datetime

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
    UnstructuredFileLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import tiktoken
from tqdm import tqdm

from config import (
    FAQ_DOCUMENT_PATH,
    EMBEDDINGS_PATH,
    EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNKING_METHOD,
    TOKEN_ENCODING,
    DOCUMENT_ID,
    SIMILARITY_METRIC,
    OPENAI_API_KEY,
    BATCH_SIZE,
    ENABLE_PROGRESS_BAR,
    ENABLE_VERIFICATION,
    VERIFICATION_QUERIES,
    USE_ANN,
    ANN_INDEX_TYPE,
    ANN_THRESHOLD,
    SUPPORTED_DOCUMENT_FORMATS,
)
from utils import retry_with_backoff, validate_vector_dimensions
from metadata_extractor import extract_metadata_from_chunk

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/logs/build_index.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def get_file_loader(file_path: str):
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.txt':
        return TextLoader(file_path, encoding="utf-8")
    elif file_ext == '.md':
        try:
            return UnstructuredMarkdownLoader(file_path)
        except Exception:
            logger.warning(f"UnstructuredMarkdownLoader failed for {file_path}, using TextLoader")
            return TextLoader(file_path, encoding="utf-8")
    elif file_ext == '.pdf':
        return PyPDFLoader(file_path)
    elif file_ext == '.docx':
        return Docx2txtLoader(file_path)
    else:
        try:
            return UnstructuredFileLoader(file_path)
        except Exception:
            raise ValueError(f"Unsupported file format: {file_ext}")


def load_document(file_path: str) -> list[Document]:
    logger.info(f"Loading document from: {file_path}")
    try:
        loader = get_file_loader(file_path)
        documents = loader.load()
        total_chars = sum(len(doc.page_content) for doc in documents)
        logger.info(f"Document loaded successfully. Total characters: {total_chars}, Pages: {len(documents)}")
        return documents
    except Exception as e:
        logger.error(f"Error loading document {file_path}: {str(e)}")
        raise


def load_documents_from_directory(directory_path: str) -> list[Document]:
    directory = Path(directory_path)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    if directory.is_file():
        logger.info(f"Path is a file, loading single document: {directory_path}")
        return load_document(str(directory))
    
    all_documents = []
    supported_files = []
    skipped_files = []
    
    for file_path in directory.rglob('*'):
        if file_path.is_file():
            file_ext = file_path.suffix.lower()
            if file_ext in SUPPORTED_DOCUMENT_FORMATS:
                supported_files.append(file_path)
            else:
                skipped_files.append(file_path)
    
    if not supported_files:
        raise ValueError(
            f"No supported documents found in {directory_path}. "
            f"Supported formats: {', '.join(SUPPORTED_DOCUMENT_FORMATS)}"
        )
    
    logger.info(f"Found {len(supported_files)} supported file(s) and {len(skipped_files)} unsupported file(s)")
    
    if skipped_files:
        logger.warning(f"Skipping {len(skipped_files)} unsupported file(s):")
        for skipped in skipped_files[:10]:
            logger.warning(f"  - {skipped.name} (format: {skipped.suffix})")
        if len(skipped_files) > 10:
            logger.warning(f"  ... and {len(skipped_files) - 10} more")
    
    for file_path in supported_files:
        try:
            logger.info(f"Loading: {file_path.name} ({file_path.suffix})")
            documents = load_document(str(file_path))
            
            for doc in documents:
                if not doc.metadata:
                    doc.metadata = {}
                doc.metadata['source_file'] = str(file_path)
                doc.metadata['file_name'] = file_path.name
                doc.metadata['file_format'] = file_path.suffix.lower()
            
            all_documents.extend(documents)
            logger.info(f"Successfully loaded {len(documents)} page(s) from {file_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {str(e)}")
            logger.warning(f"Skipping {file_path.name} due to error")
            continue
    
    total_chars = sum(len(doc.page_content) for doc in all_documents)
    logger.info(
        f"Loaded {len(all_documents)} document chunk(s) from {len(supported_files)} file(s). "
        f"Total characters: {total_chars:,}"
    )
    
    return all_documents


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Token counting failed: {e}. Using word count approximation.")
        return len(text.split())


def chunk_documents(documents: list[Document], chunk_size: int, chunk_overlap: int, 
                    chunking_method: str = "character", token_encoding: str = "cl100k_base",
                    document_id: str = "faq_document") -> list[Document]:
    logger.info(f"Chunking documents using {chunking_method}-based method")
    logger.info(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
    
    if chunking_method.lower() == "token":
        logger.info(f"Using TokenTextSplitter with encoding: {token_encoding}")
        text_splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name=token_encoding
        )
    else:
        logger.info("Using RecursiveCharacterTextSplitter")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True,
        )
    
    chunks = text_splitter.split_documents(documents)
    
    original_text = documents[0].page_content if documents else ""
    current_char_pos = 0
    
    for i, chunk in enumerate(chunks):
        chunk_text = chunk.page_content
        
        char_start = chunk.metadata.get("start_index", current_char_pos)
        char_end = char_start + len(chunk_text)
        token_count = count_tokens(chunk_text, token_encoding)
        extracted_metadata = extract_metadata_from_chunk(chunk_text, i)
        
        chunk.metadata.update({
            "chunk_id": f"{document_id}_chunk_{i:04d}",
            "document_id": document_id,
            "chunk_index": i,
            "char_start": char_start,
            "char_end": char_end,
            "start_index": char_start,
            "chunk_size": len(chunk_text),
            "token_count": token_count,
            "total_chunks": len(chunks),
            "created_at": datetime.now().isoformat(),
            **extracted_metadata
        })
        
        current_char_pos = char_end
    
    logger.info(f"Document chunked into {len(chunks)} chunks")
    
    chunk_sizes = [len(chunk.page_content) for chunk in chunks]
    token_counts = [chunk.metadata.get("token_count", 0) for chunk in chunks]
    
    logger.info(f"Character size stats - Min: {min(chunk_sizes)}, Max: {max(chunk_sizes)}, Avg: {sum(chunk_sizes)/len(chunk_sizes):.0f}")
    if any(token_counts):
        logger.info(f"Token count stats - Min: {min(token_counts)}, Max: {max(token_counts)}, Avg: {sum(token_counts)/len(token_counts):.0f}")
    
    return chunks


@retry_with_backoff(max_retries=3, exceptions=(Exception,))
def get_embeddings():
    if EMBEDDING_MODEL.lower() == "local":
        logger.info(f"Using LOCAL embeddings model: {LOCAL_EMBEDDING_MODEL}")
        logger.info("✓ All embeddings generated locally - no data sent externally")
        logger.info(f"Similarity metric: {SIMILARITY_METRIC}")
        embeddings = HuggingFaceEmbeddings(
            model_name=LOCAL_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    else:
        logger.info(f"Using OpenAI embeddings model: {EMBEDDING_MODEL}")
        logger.warning("⚠️  OpenAI embeddings send data to external API")
        logger.info(f"Similarity metric: {SIMILARITY_METRIC}")
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY required for OpenAI embeddings")
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY
        )
    
    return embeddings


@retry_with_backoff(max_retries=3)
def generate_embeddings_batched(chunks: list[Document], embeddings, batch_size: int = None):
    batch_size = batch_size or BATCH_SIZE
    all_embeddings = []
    
    iterator = range(0, len(chunks), batch_size)
    if ENABLE_PROGRESS_BAR:
        iterator = tqdm(iterator, desc="Generating embeddings", unit="batch")
    
    for i in iterator:
        batch = chunks[i:i + batch_size]
        batch_texts = [chunk.page_content for chunk in batch]
        
        try:
            batch_embeddings = embeddings.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.error(f"Error generating embeddings for batch {i}: {e}")
            raise
    
    expected_dim = validate_vector_dimensions(all_embeddings)
    logger.info(f"Generated {len(all_embeddings)} embeddings with dimension {expected_dim}")
    
    return all_embeddings


def create_vector_store(chunks: list[Document], embeddings, output_path: str):
    logger.info(f"Creating vector store with {len(chunks)} chunks")
    
    use_ann = USE_ANN and len(chunks) > ANN_THRESHOLD
    if use_ann:
        logger.info(f"Using ANN index type: {ANN_INDEX_TYPE} (dataset size: {len(chunks)} > {ANN_THRESHOLD})")
    else:
        logger.info(f"Using exact search (dataset size: {len(chunks)} <= {ANN_THRESHOLD})")
    
    try:
        logger.info("Generating embeddings with batching and validation...")
        all_embeddings = generate_embeddings_batched(chunks, embeddings, BATCH_SIZE)
        
        texts = [chunk.page_content for chunk in chunks]
        text_embeddings = list(zip(texts, all_embeddings))
        
        vectorstore = FAISS.from_embeddings(
            text_embeddings=text_embeddings,
            embedding=embeddings,
            metadatas=[chunk.metadata for chunk in chunks]
        )
        
        logger.info(f"Saving vector store to: {output_path}")
        vectorstore.save_local(output_path)
        
        logger.info("Vector store created and saved successfully!")
        
        bm25_data_path = Path(output_path).parent / "bm25_documents.json"
        import json
        bm25_data = {
            "documents": [
                {
                    "chunk_id": chunk.metadata.get("chunk_id"),
                    "text": chunk.page_content,
                    "metadata": chunk.metadata
                }
                for chunk in chunks
            ]
        }
        with open(bm25_data_path, "w", encoding="utf-8") as f:
            json.dump(bm25_data, f, indent=2, ensure_ascii=False)
        logger.info(f"BM25 document data saved to: {bm25_data_path}")
        
        metadata_path = Path(output_path).parent / "metadata.json"
        metadata = {
            "total_chunks": len(chunks),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "chunking_method": CHUNKING_METHOD,
            "embedding_model": EMBEDDING_MODEL if EMBEDDING_MODEL != "local" else LOCAL_EMBEDDING_MODEL,
            "similarity_metric": SIMILARITY_METRIC,
            "document_id": DOCUMENT_ID,
            "created_at": datetime.now().isoformat(),
            "chunks": [
                {
                    "chunk_id": chunk.metadata.get("chunk_id"),
                    "document_id": chunk.metadata.get("document_id"),
                    "chunk_index": chunk.metadata.get("chunk_index"),
                    "char_start": chunk.metadata.get("char_start"),
                    "char_end": chunk.metadata.get("char_end"),
                    "start_index": chunk.metadata.get("start_index"),
                    "chunk_size": chunk.metadata.get("chunk_size"),
                    "token_count": chunk.metadata.get("token_count"),
                    "total_chunks": chunk.metadata.get("total_chunks"),
                    "created_at": chunk.metadata.get("created_at"),
                    "service_name": chunk.metadata.get("service_name"),
                    "section": chunk.metadata.get("section"),
                    "doc_type": chunk.metadata.get("doc_type"),
                    "tags": chunk.metadata.get("tags", []),
                }
                for chunk in chunks
            ]
        }
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to: {metadata_path}")
        
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise


def verify_index(vectorstore, embeddings, sample_queries: list = None):
    if not ENABLE_VERIFICATION:
        logger.info("Verification disabled, skipping...")
        return
    
    if sample_queries is None:
        sample_queries = VERIFICATION_QUERIES if VERIFICATION_QUERIES else [
            "How do I request vacation time?",
            "What is the company's 401k matching policy?",
            "How do I access my pay stubs?",
            "What are the benefits enrollment dates?",
            "How do I submit an expense reimbursement?",
        ]
    
    logger.info(f"Running verification queries ({len(sample_queries)} queries)...")
    
    for i, query in enumerate(sample_queries, 1):
        try:
            results = vectorstore.similarity_search(query, k=3)
            logger.info(f"Query {i}: '{query[:50]}...' -> Retrieved {len(results)} results")
            
            if len(results) == 0:
                logger.warning(f"⚠️  No results for query: {query}")
            else:
                first_result = results[0]
                logger.debug(f"  Top result: chunk_id={first_result.metadata.get('chunk_id')}, "
                           f"section={first_result.metadata.get('section', 'N/A')}")
        except Exception as e:
            logger.error(f"Error verifying query '{query}': {e}")
    
    logger.info("Index verification completed")


def add_vectors_to_index(vectorstore, new_chunks: list[Document], embeddings):
    if not new_chunks:
        logger.warning("No new chunks to add")
        return vectorstore
    
    logger.info(f"Adding {len(new_chunks)} new chunks to existing index...")
    
    new_embeddings = generate_embeddings_batched(new_chunks, embeddings, BATCH_SIZE)
    
    sample_query = embeddings.embed_query("test")
    expected_dim = len(sample_query)
    validate_vector_dimensions(new_embeddings, expected_dim)
    
    texts = [chunk.page_content for chunk in new_chunks]
    text_embeddings = list(zip(texts, new_embeddings))
    
    vectorstore.add_texts(
        texts=texts,
        embeddings=new_embeddings,
        metadatas=[chunk.metadata for chunk in new_chunks]
    )
    
    logger.info(f"Successfully added {len(new_chunks)} chunks to index")
    return vectorstore


def rebuild_index_with_deletions(chunks: list[Document], deleted_ids: set, 
                                  embeddings, output_path: str):
    logger.info(f"Rebuilding index: removing {len(deleted_ids)} deleted chunks...")
    
    filtered_chunks = [
        chunk for chunk in chunks 
        if chunk.metadata.get("chunk_id") not in deleted_ids
    ]
    
    logger.info(f"Rebuilt index: {len(filtered_chunks)} chunks (removed {len(chunks) - len(filtered_chunks)})")
    
    return create_vector_store(filtered_chunks, embeddings, output_path)


def main():
    logger.info("=" * 60)
    logger.info("Starting FAQ Index Building Pipeline")
    logger.info("=" * 60)
    
    try:
        documents = load_documents_from_directory(FAQ_DOCUMENT_PATH)
        
        chunks = chunk_documents(
            documents, 
            CHUNK_SIZE, 
            CHUNK_OVERLAP,
            chunking_method=CHUNKING_METHOD,
            token_encoding=TOKEN_ENCODING,
            document_id=DOCUMENT_ID
        )
        
        if len(chunks) < 20:
            logger.warning(f"Only {len(chunks)} chunks created. Minimum recommended: 20 chunks.")
        
        embeddings = get_embeddings()
        vectorstore = create_vector_store(chunks, embeddings, EMBEDDINGS_PATH)
        verify_index(vectorstore, embeddings)
        
        logger.info("=" * 60)
        logger.info("Index Building Pipeline Completed Successfully!")
        logger.info(f"Total chunks indexed: {len(chunks)}")
        logger.info(f"Vector store saved to: {EMBEDDINGS_PATH}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

