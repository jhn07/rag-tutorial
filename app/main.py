from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from app.handlers.ask import router as ask_router
from app.logger import setup_logger
from app.services.rag import RAGService

load_dotenv()

logger = setup_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
  # STARTUP
  logger.info("Lifespan startup: initializing RAGService...")
  app.state.rag = RAGService()  # will warm up index/retriever inside RAGService
  logger.info("RAGService initialized successfully")
  try:
    yield
  finally:
    # SHUTDOWN
    logger.info("Lifespan shutdown: releasing resources...")
    # If you need to close connections/clients — do it here
    # Example: app.state.rag.close()  (if it was)
    logger.info("Shutdown complete")

app = FastAPI(
  title="AcmeTech FAQ RAG API",
  description="This is a RAG API AcmeTech FAQ",
  version="0.1.0",
  lifespan=lifespan,
)

app.add_middleware(
  CORSMiddleware,
  allow_origins=[
    "http://localhost:3000", # default frontend port
    "http://localhost:3001", # default frontend port
    "*" # allow all origins for development mode (not recommended for production)
  ],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

app.include_router(ask_router)

@app.get("/")
async def root():
  logger.info("Root endpoint called")
  return {"message": "This is a RAG API for AcmeTech FAQ"}


if __name__ == "__main__":
  logger.info("Starting the application")
  uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)