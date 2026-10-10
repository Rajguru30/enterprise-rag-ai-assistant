import logging

from fastapi import Depends, FastAPI, HTTPException

from .api_models import AskRequest, AskResponse
from .rag_service import RAGService


# ---------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="Enterprise RAG AI Assistant",
    description=(
        "AI-powered document question answering "
        "system using RAG, FAISS and Gemini."
    ),
    version="1.0.0"
)


# ---------------------------------------------------------
# RAG service instance
# ---------------------------------------------------------

rag_service = None


# ---------------------------------------------------------
# RAG service dependency
# ---------------------------------------------------------

def get_rag_service():

    global rag_service

    if rag_service is None:

        logger.info(
            "Initializing RAG service..."
        )

        rag_service = RAGService()

    return rag_service


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------

@app.get("/")
def root():

    logger.info(
        "Health check request received."
    )

    return {
        "message": (
            "Enterprise RAG AI Assistant is running"
        )
    }


# ---------------------------------------------------------
# Ask endpoint
# ---------------------------------------------------------

@app.post(
    "/ask",
    response_model=AskResponse
)
def ask_question(
    request: AskRequest,
    service: RAGService = Depends(
        get_rag_service
    )
):

    logger.info(
        "Received question request."
    )

    try:

        result = service.ask(
            request.question
        )

        logger.info(
            "Question processed successfully."
        )

        return AskResponse(
            question=request.question,
            answer=result["answer"],
            sources=result["sources"]
        )

    except Exception as error:

        logger.exception(
            "Error while processing question."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "An internal error occurred while "
                "processing the question."
            )
        ) from error