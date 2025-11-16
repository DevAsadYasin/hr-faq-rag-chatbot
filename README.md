# FAQ Support Chatbot - RAG System

**Internal Use Only - Confidential**

An intelligent FAQ support chatbot using Retrieval-Augmented Generation (RAG) to answer questions about HR policies, features, and procedures from internal documentation. The system processes HR documentation, generates embeddings, and provides accurate answers using vector similarity search and LLM generation.

## 🎯 Overview

This system is designed for HR teams, employees, and customer support to quickly find answers about:
- Employee policies (leave, attendance, remote work)
- Benefits enrollment and administration
- Payroll and compensation information
- HR procedures and workflows
- System features and functionality
- Troubleshooting and support

### Problem Statement

Customer support receives 200+ repetitive questions daily about HR policies, features, and procedures already documented in internal FAQs and guides. This system provides an intelligent FAQ Support Chatbot that can instantly answer user questions by retrieving relevant information from the company's documentation without requiring manual search or agent intervention.

### Solution Architecture

The system uses a RAG (Retrieval-Augmented Generation) architecture:
1. **Document Processing**: Loads and chunks HR documentation
2. **Embedding Generation**: Creates vector representations (locally, for privacy)
3. **Index Building**: Stores embeddings in FAISS vector store
4. **Query Processing**: Retrieves relevant chunks using hybrid search
5. **Answer Generation**: Uses LLM to generate answers from retrieved context

## 📋 Features

- **Full LangChain Integration**: Complete LangChain stack (loaders, splitters, embeddings, vector stores, chains, retrievers)
- **Local-First Embeddings**: Defaults to local HuggingFace embeddings (no data sent externally for embeddings)
- **Intelligent Chunking**: LangChain RecursiveCharacterTextSplitter/TokenTextSplitter with overlap for context preservation
- **Hybrid Search**: Combines semantic vector search with BM25 keyword search for precision (configurable: hybrid/keyword/vector)
- **Metadata Filtering**: Filter by service, section, document type with lenient filtering
- **Query Parsing**: Auto-extracts filters from natural language queries
- **Multiple Search Modes**: Hybrid (default), keyword-only, or vector-only via `SEARCH_MODE` env variable
- **Production-Ready**: Dimensionality validation, batching, retry logic, verification queries
- **Index Management**: Support for adding vectors and rebuilding with deletions
- **Progress Tracking**: Visual progress bars for long-running operations
- **Multi-LLM Support**: OpenRouter, Gemini, and OpenAI with automatic fallback
- **Priority-Based Selection**: Configurable provider priority (default: OpenRouter → Gemini → OpenAI)
- **RAG Pipeline**: LangChain LCEL chain combining retrieval + generation
- **Structured Output**: JSON responses with question, answer, related chunks, and provider info
- **Quality Evaluation**: Optional evaluator agent for answer quality scoring (0-10)
- **Confidential Data Safe**: Local embeddings ensure no document data leaves your system
- **Multi-Format Support**: Automatically loads and processes TXT, PDF, Markdown, and DOCX files

## 🏗️ Project Structure

```
assignment02/
├── data/
│   ├── test_document.txt         # HR policies and procedures documentation (confidential)
│   └── ...                       # Additional documents (PDF, Markdown, DOCX supported)
├── src/
│   ├── __init__.py
│   ├── config.py                 # Configuration management
│   ├── build_index.py            # Index building pipeline
│   ├── query.py                  # Query pipeline with hybrid search
│   ├── evaluator.py              # Answer quality evaluator
│   ├── hybrid_search.py          # Hybrid search implementation (BM25 + semantic)
│   ├── metadata_extractor.py    # Metadata extraction and query parsing
│   ├── llm_manager.py            # Multi-provider LLM manager with fallback
│   └── utils.py                  # Retry logic, validation, utilities
├── outputs/
│   ├── embeddings/               # FAISS vector store + BM25 data
│   │   ├── faiss_index/          # FAISS vector store (index.faiss, index.pkl)
│   │   ├── bm25_documents.json   # BM25 keyword search data
│   │   └── metadata.json         # Index metadata
│   ├── logs/                     # Application logs
│   └── sample_queries.json       # Sample query responses
├── tests/
│   └── test_core.py              # Unit tests
├── .env                          # Environment variables (gitignored)
├── .env.example                  # Environment variables template
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── TECHNICAL_DECISIONS.md        # Technical choices and rationale
```

## 🚀 Setup

### Prerequisites

- **Python 3.11+** (required for LangChain v1.0 compatibility)
- **pip** package manager
- **At least one LLM API key**: OpenRouter, Gemini, or OpenAI (for answer generation)
- **Local embeddings** (default, no API key needed for embeddings)

### Step-by-Step Installation

#### 1. Clone/Navigate to Project Directory

```bash
cd /path/to/assignment02
```

#### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Why virtual environment?**
- Isolates project dependencies
- Prevents conflicts with system packages
- Ensures consistent Python version

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- LangChain and related packages
- FAISS for vector storage
- Sentence-transformers for local embeddings
- Document loaders (PDF, DOCX, Markdown)
- BM25 for keyword search
- Multi-LLM provider support

#### 4. Set Up Environment Variables

```bash
cp .env.example .env
# Edit .env and add your API keys
```

**Required Configuration:**

**LLM Provider** (set at least one):
```bash
OPENROUTER_API_KEY=your-openrouter-key-here
# OR
GEMINI_API_KEY=your-gemini-key-here
# OR
OPENAI_API_KEY=your-openai-key-here
```

**Optional Configuration:**

**Priority Order** (default: openrouter,gemini,openai):
```bash
LLM_PRIORITY_ORDER=openrouter,gemini,openai
```

**Model Selection** (per provider):
```bash
LLM_MODEL_OPENROUTER=openai/gpt-3.5-turbo
LLM_MODEL_GEMINI=gemini-pro
LLM_MODEL_OPENAI=gpt-3.5-turbo
```

**Search Mode** (default: hybrid):
```bash
SEARCH_MODE=hybrid   # Options: hybrid, keyword, vector
```

**Important Notes:**
- The system defaults to **local embeddings** (no API calls for embeddings)
- LLM API keys are only needed for answer generation
- Only selected chunks (top-k) are sent to LLM, not the full document
- See `TECHNICAL_DECISIONS.md` for detailed explanations

#### 5. Verify Document Directory

```bash
ls data/  # Should show your FAQ documents
```

**Supported file formats:**
- `.txt` - Plain text files
- `.md` - Markdown files
- `.pdf` - PDF documents
- `.docx` - Microsoft Word documents

The system automatically:
- Loads all supported documents from the `data/` directory
- Processes each file according to its format
- Skips unsupported file formats with a warning

#### 6. Verify Installation

```bash
# Test imports
python -c "from src.config import *; print('✅ Configuration loaded')"
python -c "from src.llm_manager import LLMManager; print('✅ LLM Manager ready')"
```

## 📖 Usage

### Step 1: Build the Index

First, process the FAQ documents, chunk them, generate embeddings, and create the vector store:

```bash
python src/build_index.py
```

**What this does:**
1. Loads all supported documents from `data/` directory
2. Processes each file according to its format (TXT, PDF, Markdown, DOCX)
3. Chunks all documents (target: 20+ chunks total)
4. Extracts metadata from each chunk (service, section, doc_type, tags)
5. Generates embeddings for each chunk (locally, no external API calls)
6. Creates FAISS vector store
7. Saves BM25 document data for keyword search
8. Saves metadata for filtering and debugging

**Expected output:**
```
Starting FAQ Index Building Pipeline
Loading: test_document.txt (.txt)
Document loaded successfully. Total characters: 124133
Document chunked into 490 chunks
Creating vector store with 490 chunks
Generating embeddings (this may take a while)...
Vector store created and saved successfully!
BM25 document data saved to: outputs/embeddings/bm25_documents.json
Metadata saved to: outputs/embeddings/metadata.json
Index Building Pipeline Completed Successfully!
```

**Storage Location:**
- **FAISS Vector Store**: `outputs/embeddings/faiss_index/`
- **BM25 Data**: `outputs/embeddings/bm25_documents.json`
- **Metadata**: `outputs/embeddings/metadata.json`

**Custom Document Path:**
```bash
export FAQ_DOCUMENT_PATH=/path/to/your/documents
python src/build_index.py
```

### Step 2: Query the FAQ System

Query the system with a question:

```bash
# Using command line argument
python src/query.py "How do I request vacation time?"

# Or run without arguments for sample question
python src/query.py
```

**Example output:**
```json
{
  "user_question": "How do I request vacation time?",
  "system_answer": "To request vacation time, log into the Employee Self-Service Portal, navigate to the Time & Attendance section, select \"Request Time Off\", choose \"Vacation\" as the leave type, enter your start and end dates, and add any comments or notes for your manager. Make sure to submit the request at least 2 weeks (14 calendar days) in advance of the requested start date to allow managers to plan for coverage and ensure business continuity.",
  "chunks_related": [
    {
      "chunk_id": "faq_document_v1_chunk_0012",
      "chunk_text": "VACATION LEAVE POLICY\n- Accrual Rate: Full-time employees accrue 1.25 vacation days per month (15 days per year)\n- Request Process: All vacation requests must be submitted through the Employee Self-Service Portal at least 2 weeks in advance...",
      "metadata": {
        "document_id": "faq_document_v1",
        "chunk_index": 12,
        "service_name": "time-attendance",
        "section": "leave-policies",
        "doc_type": "procedure"
      }
    }
  ],
  "llm_provider": "openrouter"
}
```

**Response Structure:**
- `user_question`: The original question
- `system_answer`: Generated answer from LLM
- `chunks_related`: Array of relevant document chunks with metadata
- `llm_provider`: Which LLM provider was used (if available)

### Step 3: Evaluate Answer Quality (Optional)

Evaluate the quality of generated answers:

```bash
# Evaluate from sample_queries.json
python src/evaluator.py outputs/sample_queries.json
```

**Example evaluation output:**
```json
{
  "score": 9,
  "reason": "The answer accurately provides step-by-step instructions. Chunks are highly relevant. Answer is complete and accurate.",
  "chunk_relevance_score": 0.95,
  "answer_accuracy_score": 0.9,
  "completeness_score": 0.95
}
```

## 🔧 Configuration

Configuration is managed through environment variables in `.env` file:

### Search Configuration

**Search Mode** (default: `hybrid`):
```bash
SEARCH_MODE=hybrid   # Options: hybrid, keyword, vector
```
- `hybrid`: Combines semantic (vector) + keyword (BM25) search (recommended)
- `keyword`: Pure BM25 keyword search only (good for exact term matching)
- `vector`: Pure semantic/vector search only (good for conceptual queries)

**Hybrid Search Method**:
```bash
HYBRID_SEARCH_METHOD=retrieve_then_filter  # Options: retrieve_then_filter, weighted
SEMANTIC_WEIGHT=0.7  # Weight for semantic search (0-1)
KEYWORD_WEIGHT=0.3   # Weight for keyword search (0-1)
```

### Embedding Configuration

**Embedding Model** (default: `local`):
```bash
EMBEDDING_MODEL=local  # Recommended for confidential data
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

**Why local embeddings?**
- No document data sent to external services
- Maximum privacy for confidential HR data
- No embedding API costs
- Works offline after initial model download

### LLM Configuration

**Provider Priority** (default: `openrouter,gemini,openai`):
```bash
LLM_PRIORITY_ORDER=openrouter,gemini,openai
```

**Model Selection**:
```bash
LLM_MODEL_OPENROUTER=openai/gpt-3.5-turbo
LLM_MODEL_GEMINI=gemini-pro
LLM_MODEL_OPENAI=gpt-3.5-turbo
LLM_TEMPERATURE=0  # Lower = more deterministic
```

### Chunking Configuration

```bash
CHUNK_SIZE=500        # Characters per chunk
CHUNK_OVERLAP=100     # Overlap between chunks
CHUNKING_METHOD=character  # Options: character, token
```

### Retrieval Configuration

```bash
TOP_K_CHUNKS=5              # Number of chunks to retrieve
SEARCH_TYPE=similarity      # LangChain search type
SIMILARITY_THRESHOLD=0.7     # Minimum similarity score
```

### Other Settings

```bash
BATCH_SIZE=100              # Embedding generation batch size
ENABLE_PROGRESS_BAR=true    # Show progress bars
MAX_RETRIES=3               # Retry attempts for API calls
LOG_LEVEL=INFO              # Logging level
```

## 📝 Sample Queries

Example questions the system can answer:

1. **Leave and Time-Off Questions**:
   - "How do I request vacation time?"
   - "What is the company's sick leave policy?"
   - "How many vacation days do I accrue per month?"

2. **Benefits Questions**:
   - "What is the company's 401k matching policy?"
   - "How do I enroll in health insurance?"
   - "When is open enrollment for benefits?"

3. **Payroll Questions**:
   - "How do I change my direct deposit information?"
   - "When will I receive my W-2?"
   - "How do I access my pay stubs?"

4. **Policy Questions**:
   - "What is the remote work policy?"
   - "What is the dress code?"
   - "What is the process for performance reviews?"

5. **Procedural Questions**:
   - "How do I submit expense reimbursement?"
   - "How do I update my personal information?"
   - "How do I access training materials?"

6. **Troubleshooting Questions**:
   - "I'm having login issues, what should I do?"
   - "My timesheet was rejected, how do I fix it?"
   - "I can't access the benefits portal, who should I contact?"

## 🔒 Security & Privacy

### Confidential Data Handling

- **HR Documentation**: Contains internal HR policies and employee information
- **Git Ignored**: All data files and indexes are excluded from version control
- **Local Embeddings**: No document data sent externally for embedding generation
- **Selective LLM Usage**: Only top-k retrieved chunks sent to LLM, not full document
- **API Key Security**: All API keys stored in `.env` (gitignored)

### Best Practices

- Never commit `.env` file with API keys
- Keep `outputs/embeddings/` out of version control
- Use local embeddings for maximum privacy
- Rotate API keys regularly
- Review logs for any data exposure

## 🐛 Troubleshooting

### Common Issues

1. **"No LLM API keys set"**
   - Solution: Set at least one API key in `.env`: `OPENROUTER_API_KEY`, `GEMINI_API_KEY`, or `OPENAI_API_KEY`

2. **"Vector store not found"**
   - Solution: Run `python src/build_index.py` first to create the index

3. **"Document not found"**
   - Solution: Ensure `data/` directory exists and contains supported documents (`.txt`, `.md`, `.pdf`, `.docx`)

4. **"ModuleNotFoundError"**
   - Solution: Install dependencies: `pip install -r requirements.txt`
   - Ensure you're in the virtual environment: `source venv/bin/activate`

5. **"Low chunk count (< 20)"**
   - Solution: Increase document size or decrease `CHUNK_SIZE` in `.env`

6. **"No chunks retrieved"**
   - Check if metadata filtering is too strict
   - Try different search modes: `SEARCH_MODE=vector` or `SEARCH_MODE=keyword`
   - Verify index was built successfully

## 📈 Performance

- **Index Building**: ~2-5 minutes (depends on document size and embedding model)
- **Query Processing**: ~2-5 seconds (embedding + retrieval + LLM generation)
- **Chunk Count**: 490+ chunks from comprehensive HR documentation
- **Retrieval Speed**: < 100ms for top-k search
- **Embedding Generation**: ~1-2 seconds per 100 chunks (local)

## 🚧 Known Limitations

1. **LLM Dependency**: Requires at least one LLM API key (OpenRouter, Gemini, or OpenAI) for answer generation
2. **Context Window**: Limited by LLM context window (varies by model)
3. **Chunk Size**: Fixed chunk size may split related content
4. **Language**: Optimized for English documentation

## 📚 Additional Documentation

- **TECHNICAL_DECISIONS.md**: Detailed explanation of technical choices and rationale
- **.env.example**: Complete list of configuration options

## 🔮 Future Enhancements

- [ ] Conversation memory for multi-turn dialogues
- [ ] Web UI for interactive querying
- [ ] Batch evaluation on test question sets
- [ ] Support for multiple languages

## 📄 License

Internal use only - Confidential

## 👥 Authors

Technical Documentation Team

## 📞 Support

For issues or questions, contact the development team.

---

**Note**: This system processes confidential internal documentation. Ensure proper access controls and data handling procedures are followed.
