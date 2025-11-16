import logging
import sys
from typing import Optional, Dict, Any, List
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .query import query_faq
from .config import EMBEDDINGS_PATH

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/logs/api.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HR FAQ Support Chatbot API",
    description="RAG-based FAQ support system for HR policies, features, and procedures",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        description="The user's question about HR policies, features, or procedures",
        example="How do I request vacation time?",
        min_length=1,
        max_length=1000
    )
    filters: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional metadata filters for search refinement",
        example={"service_name": "time-attendance", "section": "leave-policies"}
    )

    class Config:
        json_schema_extra = {
            "example": {
                "question": "How do I request vacation time?",
                "filters": {
                    "service_name": "time-attendance",
                    "section": "leave-policies"
                }
            }
        }


class ChunkMetadata(BaseModel):
    document_id: Optional[str] = None
    chunk_index: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    start_index: Optional[int] = None
    chunk_size: Optional[int] = None
    token_count: Optional[int] = None
    total_chunks: Optional[int] = None
    created_at: Optional[str] = None


class ChunkInfo(BaseModel):
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    chunk_text: str = Field(..., description="The text content of the chunk")
    metadata: ChunkMetadata = Field(..., description="Metadata associated with the chunk")


class QueryResponse(BaseModel):
    user_question: str = Field(..., description="The original user question")
    system_answer: str = Field(..., description="The generated answer from the RAG system")
    chunks_related: List[ChunkInfo] = Field(..., description="List of relevant document chunks used to generate the answer")
    llm_provider: Optional[str] = Field(None, description="The LLM provider used (openrouter, gemini, or openai)")
    error: Optional[str] = Field(None, description="Error message if any occurred")

    class Config:
        json_schema_extra = {
            "example": {
                "user_question": "How do I request vacation time?",
                "system_answer": "To request vacation time, log into the Employee Self-Service Portal, navigate to the Time & Attendance section, select 'Request Time Off', choose 'Vacation' as the leave type, enter your start and end dates, and add any comments or notes for your manager. Make sure to submit the request at least 2 weeks (14 calendar days) in advance of the requested start date to allow managers to plan for coverage and ensure business continuity.",
                "chunks_related": [
                    {
                        "chunk_id": "faq_document_v1_chunk_0012",
                        "chunk_text": "VACATION LEAVE POLICY\n- Accrual Rate: Full-time employees accrue 1.25 vacation days per month (15 days per year)\n- Request Process: All vacation requests must be submitted through the Employee Self-Service Portal at least 2 weeks in advance...",
                        "metadata": {
                            "document_id": "faq_document_v1",
                            "chunk_index": 12,
                            "char_start": 2450,
                            "char_end": 2937,
                            "start_index": 2450,
                            "chunk_size": 487,
                            "token_count": 98,
                            "total_chunks": 490,
                            "created_at": "2025-11-16T03:05:42.646239"
                        }
                    }
                ],
                "llm_provider": "openrouter"
            }
        }


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status")
    version: str = Field(..., description="API version")
    embeddings_available: bool = Field(..., description="Whether embeddings index is available")
    message: str = Field(..., description="Health check message")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "embeddings_available": True,
                "message": "Service is operational and ready to process queries"
            }
        }


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Check the health status of the FAQ API service",
    tags=["Health"]
)
async def health_check():
    embeddings_path = Path(EMBEDDINGS_PATH)
    embeddings_available = embeddings_path.exists() and (embeddings_path / "index.faiss").exists()
    
    if embeddings_available:
        status_msg = "healthy"
        message = "Service is operational and ready to process queries"
    else:
        status_msg = "degraded"
        message = "Service is running but embeddings index not found. Please run build_index.py first."
    
    return HealthResponse(
        status=status_msg,
        version="1.0.0",
        embeddings_available=embeddings_available,
        message=message
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query FAQ System",
    description="""
    Query the HR FAQ support system with a question about policies, features, or procedures.
    
    The system uses RAG (Retrieval-Augmented Generation) to:
    1. Retrieve relevant chunks from the HR documentation
    2. Generate an accurate answer using an LLM
    3. Return the answer along with source chunks for reference
    
    **Input:**
    - `question`: Your question about HR policies, features, or procedures
    - `filters` (optional): Metadata filters to refine search (e.g., service_name, section)
    
    **Output:**
    - `user_question`: The original question
    - `system_answer`: Generated answer from the RAG system
    - `chunks_related`: List of relevant document chunks with metadata
    - `llm_provider`: Which LLM provider was used (if available)
    - `error`: Error message if any occurred
    
    **Example Questions:**
    - "How do I request vacation time?"
    - "What is the company's 401k matching policy?"
    - "How do I access my pay stubs?"
    - "What are the benefits enrollment dates?"
    """,
    tags=["FAQ Query"],
    responses={
        200: {
            "description": "Successful query response",
            "content": {
                "application/json": {
                    "example": {
                        "user_question": "How do I request vacation time?",
                        "system_answer": "To request vacation time, log into the Employee Self-Service Portal, navigate to the Time & Attendance section, select 'Request Time Off', choose 'Vacation' as the leave type, enter your start and end dates, and add any comments or notes for your manager.",
                        "chunks_related": [
                            {
                                "chunk_id": "faq_document_v1_chunk_0012",
                                "chunk_text": "VACATION LEAVE POLICY\n- Accrual Rate: Full-time employees accrue 1.25 vacation days per month...",
                                "metadata": {
                                    "document_id": "faq_document_v1",
                                    "chunk_index": 12,
                                    "char_start": 2450,
                                    "char_end": 2937,
                                    "chunk_size": 487,
                                    "token_count": 98
                                }
                            }
                        ],
                        "llm_provider": "openrouter"
                    }
                }
            }
        },
        400: {
            "description": "Bad request - invalid input",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Question cannot be empty"
                    }
                }
            }
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Error processing query: Vector store not found. Please run build_index.py first."
                    }
                }
            }
        }
    }
)
async def query_faq_endpoint(request: QueryRequest):
    try:
        if not request.question or not request.question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty"
            )
        
        logger.info(f"API Query received: {request.question}")
        
        response = query_faq(request.question, filters=request.filters)
        
        logger.info(f"Query processed successfully. Retrieved {len(response.get('chunks_related', []))} chunks")
        
        return QueryResponse(**response)
        
    except FileNotFoundError as e:
        logger.error(f"Vector store not found: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector store not found. Please run build_index.py first. Error: {str(e)}"
        )
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "HR FAQ Support Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "query_endpoint": "/query"
    }

