"""
Handlers for the API
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from app.models.dto import AskDTO, AskResponseDTO
from app.logger import setup_logger
from app.services.rag import RAGService

def get_rag_service(request: Request) -> RAGService:
  rag: RAGService = request.app.state.rag
  return rag

logger = setup_logger()

router = APIRouter()

@router.post("/ask", response_model=AskResponseDTO)
async def ask(ask_dto: AskDTO, rag_service: RAGService = Depends(get_rag_service)):
  
  question = (ask_dto.question or "").strip()
  if not question:
    raise HTTPException(status_code=422, detail="Question must not be empty")
  
  try:
    logger.info("Question received: %s", question)

    result = rag_service.ask_question(question)
    # result: {"answer": str, "sources": [{"text":..., "score":..., "metadata":...}, ...]}
    return AskResponseDTO(**result)
  except Exception as e:
    logger.exception("RAG failed: %s", e)
    raise HTTPException(status_code=500, detail="RAG processing error")