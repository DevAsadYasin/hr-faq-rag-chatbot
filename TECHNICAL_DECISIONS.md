# Technical Decisions & Architecture Rationale

**Internal Documentation - Technical Team**

This document explains the technical choices made in building the FAQ Support Chatbot RAG system, with detailed rationale for each decision based on requirements, constraints, and best practices.

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Embedding Strategy](#embedding-strategy)
3. [Chunking Strategy](#chunking-strategy)
4. [Vector Store Selection](#vector-store-selection)
5. [Search Strategy](#search-strategy)
6. [Metadata Management](#metadata-management)
7. [LLM Provider Strategy](#llm-provider-strategy)
8. [LangChain Integration](#langchain-integration)
9. [Production Readiness Features](#production-readiness-features)

---

## System Architecture Overview

### Decision: RAG (Retrieval-Augmented Generation) Architecture

**Why RAG?**
- **Problem**: Need to answer questions from internal HR documentation without fine-tuning models
- **Solution**: RAG retrieves relevant context and generates answers using LLMs
- **Benefits**:
  - No model fine-tuning required
  - Can update knowledge base by rebuilding index
  - Provides source attribution (chunks_related)
  - Handles confidential data by only sending selected chunks to LLM

**Architecture Flow:**
```
Documents → Chunking → Embedding → Vector Store → Query → Retrieval → LLM → Answer
```

---

## Embedding Strategy

### Decision: Local Embeddings (HuggingFace) by Default

**Choice**: `sentence-transformers/all-MiniLM-L6-v2` via LangChain `HuggingFaceEmbeddings`

**Why Local Embeddings?**

1. **Confidential Data Requirement**
   - **Context**: HR documentation contains internal policies, employee information, and confidential procedures
   - **Risk**: Sending full documents to external embedding APIs (OpenAI, Cohere, etc.) exposes confidential data
   - **Solution**: Generate embeddings locally using open-source models
   - **Result**: Zero data leaves the system during embedding generation

2. **Privacy & Compliance**
   - No third-party data processing agreements needed
   - No data residency concerns
   - Full control over data processing
   - Meets internal security policies for confidential documentation

3. **Cost Efficiency**
   - No per-API-call costs for embeddings
   - Embeddings generated once during index building
   - Significant cost savings for large document sets

4. **Performance**
   - No network latency for embedding generation
   - Batch processing capabilities
   - Works offline after initial model download

5. **Model Selection Rationale**
   - **all-MiniLM-L6-v2**: 384-dimensional embeddings
   - **Why**: Good balance between quality and performance
   - **Size**: ~80MB model, fast inference
   - **Quality**: Strong semantic understanding for HR domain
   - **LangChain Integration**: Native support via `langchain-huggingface`

**Alternative Considered**: OpenAI Embeddings
- **Why Not**: Requires sending confidential data to external API
- **When to Use**: Only if data is non-confidential and cost is not a concern

**Implementation**:
```python
# Default configuration
EMBEDDING_MODEL=local
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

---

## Chunking Strategy

### Decision: RecursiveCharacterTextSplitter with Overlap

**Choice**: LangChain `RecursiveCharacterTextSplitter`
- **Chunk Size**: 500 characters
- **Overlap**: 100 characters
- **Separators**: `["\n\n", "\n", ". ", " ", ""]`

**Why This Strategy?**

1. **Context Preservation**
   - **Problem**: Fixed-size chunks can split related content
   - **Solution**: Overlap ensures context continuity between chunks
   - **Result**: Related information spans multiple chunks, improving retrieval

2. **Document Structure Awareness**
   - **RecursiveCharacterTextSplitter**: Tries to split at natural boundaries
   - **Priority**: Paragraphs (`\n\n`) → Lines (`\n`) → Sentences (`. `) → Words → Characters
   - **Benefit**: Preserves document structure and semantic units

3. **Size Selection Rationale**
   - **500 characters**: 
     - Fits within LLM context windows (GPT-3.5: 4K tokens ≈ 3000 chars)
     - Provides enough context for meaningful answers
     - Balances granularity vs. context
   - **100 character overlap**:
     - ~20% overlap ensures continuity
     - Prevents information loss at chunk boundaries

4. **Token-Based Alternative**
   - **Available**: `TokenTextSplitter` for precise token budgeting
   - **Why Not Default**: Character-based is simpler and works well for HR docs
   - **When to Use**: If strict token limits are required

**Result**: 490+ chunks from comprehensive HR documentation, ensuring 20+ chunks minimum requirement

---

## Vector Store Selection

### Decision: FAISS (Facebook AI Similarity Search)

**Choice**: LangChain `FAISS` wrapper

**Why FAISS?**

1. **Performance**
   - **Fast**: Optimized C++ implementation with Python bindings
   - **Scalable**: Handles millions of vectors efficiently
   - **Efficient**: In-memory and disk-based storage options

2. **LangChain Integration**
   - Native LangChain support
   - Multiple search types: similarity, MMR, similarity_score_threshold
   - Easy persistence and loading
   - Metadata filtering support (with custom implementation)

3. **Production Ready**
   - Battle-tested by Facebook/Meta
   - Active maintenance and community support
   - No external dependencies (runs locally)

4. **Storage Efficiency**
   - Compressed vector storage
   - Fast serialization/deserialization
   - Small disk footprint

**Alternatives Considered**:
- **Pinecone/Weaviate**: Cloud-based, but requires external service (not suitable for confidential data)
- **Chroma**: Good alternative, but FAISS is more mature and faster
- **Qdrant**: Excellent, but FAISS meets all requirements

**Storage Location**: `outputs/embeddings/faiss_index/`
- `index.faiss`: Vector embeddings (binary)
- `index.pkl`: Metadata mapping (pickle)

---

## Search Strategy

### Decision: Hybrid Search (Vector + Keyword) by Default

**Choice**: Combines semantic (FAISS) + keyword (BM25) search

**Why Hybrid Search?**

1. **Complementary Strengths**
   - **Semantic Search**: Understands meaning, synonyms, context
   - **Keyword Search**: Exact term matching, error codes, specific terms
   - **Combined**: Best of both worlds

2. **HR Domain Requirements**
   - **Semantic**: "How do I request time off?" matches "vacation request procedure"
   - **Keyword**: "401k" or "W-2" need exact matching
   - **Hybrid**: Handles both conceptual and specific queries

3. **Configurable Modes**
   - **Default**: `SEARCH_MODE=hybrid` (recommended)
   - **Keyword-Only**: `SEARCH_MODE=keyword` (for exact term queries)
   - **Vector-Only**: `SEARCH_MODE=vector` (for conceptual queries)

**Implementation Methods**:

1. **Retrieve-Then-Filter** (Default)
   - Retrieve candidates via semantic search
   - Apply metadata filters
   - Return top-k results
   - **Why**: Simple, effective, handles FAISS filtering limitations

2. **Weighted Hybrid** (Optional)
   - Score fusion: `combined_score = semantic_weight * semantic_score + keyword_weight * keyword_score`
   - Normalize and rank
   - **Why**: More sophisticated, better for diverse query types

**BM25 Implementation**:
- **Library**: `rank-bm25` (BM25Okapi)
- **Storage**: `outputs/embeddings/bm25_documents.json`
- **Why BM25**: Industry-standard keyword search algorithm, fast and effective

---

## Metadata Management

### Decision: Structured Metadata with Lenient Filtering

**Metadata Fields Extracted**:
- `service_name`: payroll, benefits, time-attendance, employee-portal, hr-support, expense
- `section`: leave-policies, benefits, payroll, policies, procedures, performance, compliance, troubleshooting, general
- `doc_type`: policy, procedure, support_kb, hr_documentation
- `tags`: Array of relevant keywords

**Why This Structure?**

1. **HR Domain Alignment**
   - Matches HR SaaS system structure
   - Enables precise filtering by service area
   - Supports common query patterns

2. **Lenient Filtering**
   - **Problem**: Strict filtering excludes chunks without metadata fields
   - **Solution**: Only filter if field exists AND doesn't match
   - **Result**: Chunks without metadata still retrieved (graceful degradation)

3. **Query Parsing**
   - Auto-extracts filters from natural language
   - "How do I request vacation?" → `service_name: time-attendance, section: leave-policies`
   - Improves retrieval precision

**Why Not Technical Metadata?**
- Originally designed for technical documentation (production/staging/dev environments)
- Adapted for HR SaaS context
- Removed irrelevant technical concepts

---

## LLM Provider Strategy

### Decision: Multi-Provider with Automatic Fallback

**Choice**: OpenRouter → Gemini → OpenAI (configurable priority)

**Why Multiple Providers?**

1. **Reliability**
   - **Problem**: Single provider can fail (API issues, rate limits, downtime)
   - **Solution**: Automatic fallback to next provider
   - **Result**: High availability and resilience

2. **Cost Optimization**
   - **OpenRouter**: Access to multiple models, competitive pricing
   - **Gemini**: Google's offering, alternative pricing
   - **OpenAI**: Industry standard, reliable but potentially more expensive
   - **Flexibility**: Choose provider based on cost/performance needs

3. **Model Diversity**
   - Different providers offer different models
   - Can select best model for specific use case
   - Future-proof against provider changes

**Priority Order Rationale**:
- **OpenRouter First**: Aggregates multiple models, good pricing, reliable
- **Gemini Second**: Google's quality, alternative to OpenAI
- **OpenAI Third**: Fallback, most widely used

**Implementation**:
- `LLMManager` class handles provider detection and fallback
- Singleton pattern for efficiency
- Caching of LLM instances
- Graceful error handling

---

## LangChain Integration

### Decision: Extensive LangChain Usage

**Why LangChain?**

1. **Framework Maturity**
   - Industry-standard RAG framework
   - Active development and community
   - Comprehensive documentation

2. **Modularity**
   - Each component (loader, splitter, embedding, vector store) is modular
   - Easy to swap components
   - Maintainable codebase

3. **Best Practices Built-In**
   - Handles edge cases
   - Provides retry logic, callbacks, progress tracking
   - Production-ready patterns

**Components Used**:

1. **Document Loaders**
   - `TextLoader`: Plain text files
   - `PyPDFLoader`: PDF documents
   - `UnstructuredMarkdownLoader`: Markdown files
   - `Docx2txtLoader`: Word documents
   - `UnstructuredFileLoader`: Fallback for other formats

2. **Text Splitters**
   - `RecursiveCharacterTextSplitter`: Character-based chunking
   - `TokenTextSplitter`: Token-based chunking (alternative)

3. **Embeddings**
   - `HuggingFaceEmbeddings`: Local embeddings (default)
   - `OpenAIEmbeddings`: Cloud embeddings (optional)

4. **Vector Stores**
   - `FAISS`: Vector storage and retrieval

5. **LLMs**
   - `ChatOpenAI`: OpenAI-compatible (OpenRouter, OpenAI)
   - `ChatGoogleGenerativeAI`: Gemini support

6. **Chains**
   - LangChain Expression Language (LCEL): Modern RAG chain pattern
   - `RunnablePassthrough`, `PromptTemplate`, `StrOutputParser`

7. **Callbacks**
   - `FileCallbackHandler`: Logging
   - Progress tracking for long operations

---

## Production Readiness Features

### Decision: Comprehensive SDLC Best Practices

**Why These Features?**

1. **Dimensionality Validation**
   - **Problem**: Inconsistent embedding dimensions break vector store
   - **Solution**: Validate all embeddings have same dimension
   - **Result**: Prevents runtime errors

2. **Batching Strategy**
   - **Problem**: Processing all chunks at once can cause memory issues
   - **Solution**: Process embeddings in batches (default: 100)
   - **Result**: Scalable to large document sets

3. **Retry Logic with Exponential Backoff**
   - **Problem**: Transient API failures
   - **Solution**: Automatic retry with increasing delays
   - **Result**: Resilient to temporary failures

4. **Progress Tracking**
   - **Problem**: Long-running operations appear frozen
   - **Solution**: Visual progress bars (tqdm)
   - **Result**: Better user experience

5. **Verification Queries**
   - **Problem**: Need to verify index quality
   - **Solution**: Run sample queries after index building
   - **Result**: Early detection of issues

6. **Index Updates**
   - **Problem**: Need to add/remove documents
   - **Solution**: Support for incremental updates
   - **Result**: Maintainable knowledge base

7. **Error Handling**
   - Comprehensive try-catch blocks
   - Informative error messages
   - Graceful degradation

---

## Data Privacy & Security Decisions

### Decision: Local-First Architecture

**Privacy Measures**:

1. **Local Embeddings**
   - No document data sent to external services for embeddings
   - All embedding generation happens on-premises
   - Zero external data exposure

2. **Selective LLM Usage**
   - Only top-k retrieved chunks sent to LLM
   - Not the full document
   - Minimizes data exposure

3. **Git Ignore Strategy**
   - All data files excluded: `data/*` (except test_document.txt)
   - All indexes excluded: `outputs/embeddings/`
   - Environment variables excluded: `.env`

4. **API Key Management**
   - Stored in `.env` (gitignored)
   - Never committed to version control
   - Clear warnings if missing

**Why This Matters**:
- HR documentation contains confidential employee information
- Internal policies must remain private
- Compliance with data protection regulations
- Internal security policies require on-premises processing

---

## Search Mode Configuration

### Decision: Configurable Search Modes

**Three Modes Available**:

1. **Hybrid (Default)**
   - Combines semantic + keyword search
   - Best overall accuracy
   - Recommended for most use cases

2. **Keyword-Only**
   - Pure BM25 keyword matching
   - Best for exact term queries
   - Use case: Error codes, specific terms, exact matches

3. **Vector-Only**
   - Pure semantic similarity search
   - Best for conceptual queries
   - Use case: Synonyms, related concepts, meaning-based queries

**Why Configurable?**
- Different query types benefit from different strategies
- Allows experimentation and optimization
- Provides flexibility for specific use cases

---

## Summary of Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Embeddings** | Local (HuggingFace) | Confidential data, privacy, cost |
| **Chunking** | RecursiveCharacterTextSplitter (500/100) | Context preservation, structure awareness |
| **Vector Store** | FAISS | Performance, LangChain integration, local |
| **Search** | Hybrid (default) | Best accuracy, handles both semantic and keyword |
| **LLM** | Multi-provider with fallback | Reliability, cost optimization, flexibility |
| **Framework** | LangChain | Maturity, modularity, best practices |
| **Metadata** | HR-focused, lenient filtering | Domain alignment, graceful degradation |
| **Storage** | Local filesystem | Privacy, no external dependencies |

---

## References

- LangChain Documentation: https://python.langchain.com/
- FAISS Documentation: https://github.com/facebookresearch/faiss
- Sentence-Transformers: https://www.sbert.net/
- BM25 Algorithm: https://en.wikipedia.org/wiki/Okapi_BM25
- RAG Architecture: https://arxiv.org/abs/2005.11401

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-16  
**Maintained By**: Technical Documentation Team

