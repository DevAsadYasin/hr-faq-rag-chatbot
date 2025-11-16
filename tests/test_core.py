"""
Core tests for the FAQ RAG system.
Tests document loading, chunking, embedding generation, and query functionality.
"""
import pytest
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import FAQ_DOCUMENT_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from build_index import load_documents_from_directory


def test_document_exists():
    """Test that the FAQ document path exists."""
    assert Path(FAQ_DOCUMENT_PATH).exists(), f"FAQ document path not found at {FAQ_DOCUMENT_PATH}"


def test_document_loading():
    """Test document loading from directory with multiple format support."""
    documents = load_documents_from_directory(FAQ_DOCUMENT_PATH)
    
    assert len(documents) > 0, "No documents loaded"
    total_content = sum(len(doc.page_content) for doc in documents)
    assert total_content > 1000, "Document too short (minimum 1000 characters required)"


def test_document_chunking():
    """Test document chunking produces at least 20 chunks."""
    documents = load_documents_from_directory(FAQ_DOCUMENT_PATH)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    
    assert len(chunks) >= 20, f"Only {len(chunks)} chunks created. Minimum 20 required."
    assert all(len(chunk.page_content) > 0 for chunk in chunks), "Some chunks are empty"


def test_chunk_metadata():
    """Test that chunks have proper metadata."""
    documents = load_documents_from_directory(FAQ_DOCUMENT_PATH)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Check metadata
    for i, chunk in enumerate(chunks):
        assert hasattr(chunk, 'metadata'), f"Chunk {i} missing metadata"
        assert 'start_index' in chunk.metadata or chunk.metadata.get('start_index') is not None, \
            f"Chunk {i} missing start_index in metadata"


def test_config_loading():
    """Test that configuration loads correctly."""
    from config import (
        FAQ_DOCUMENT_PATH,
        EMBEDDINGS_PATH,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        TOP_K_CHUNKS,
    )
    
    assert FAQ_DOCUMENT_PATH is not None
    assert EMBEDDINGS_PATH is not None
    assert CHUNK_SIZE > 0
    assert CHUNK_OVERLAP >= 0
    assert TOP_K_CHUNKS > 0


def test_openai_api_key():
    """Test that OpenAI API key is set (if using OpenAI embeddings)."""
    from config import OPENAI_API_KEY, EMBEDDING_MODEL
    
    if EMBEDDING_MODEL.lower() != "local":
        assert OPENAI_API_KEY is not None, "OPENAI_API_KEY not set in environment"
        assert OPENAI_API_KEY != "your-openai-api-key-here", "OPENAI_API_KEY not configured"


@pytest.mark.skipif(
    not Path("outputs/embeddings/faiss_index").exists(),
    reason="Vector store not built. Run build_index.py first."
)
def test_vector_store_exists():
    """Test that vector store exists after building index."""
    from config import EMBEDDINGS_PATH
    
    index_path = Path(EMBEDDINGS_PATH)
    assert index_path.exists(), f"Vector store not found at {EMBEDDINGS_PATH}. Run build_index.py first."


@pytest.mark.skipif(
    not Path("outputs/embeddings/faiss_index").exists(),
    reason="Vector store not built. Run build_index.py first."
)
def test_query_structure():
    """Test that query returns proper JSON structure."""
    from query import query_faq
    
    question = "What is the authentication service deployment?"
    response = query_faq(question)
    
    # Check structure
    assert "user_question" in response
    assert "system_answer" in response
    assert "chunks_related" in response
    
    # Check types
    assert isinstance(response["user_question"], str)
    assert isinstance(response["system_answer"], str)
    assert isinstance(response["chunks_related"], list)
    
    # Check chunks structure
    if len(response["chunks_related"]) > 0:
        chunk = response["chunks_related"][0]
        assert "chunk_id" in chunk
        assert "chunk_text" in chunk
        assert "metadata" in chunk


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

