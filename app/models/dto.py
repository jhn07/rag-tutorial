"""
DTOs for the API
"""

from pydantic import BaseModel, Field


class SourceItem(BaseModel):
  text: str = Field(..., description="Fragment of text from CV")
  score: float = Field(..., description="Closeness measure (less = closer)")
  metadata: dict | None = Field(None, description="Metadata of the chunk from CV")

class AskDTO(BaseModel):
  question: str = Field(..., description="The question to ask the model")

class AskResponseDTO(BaseModel):
  answer: str = Field(..., description="The answer to the question")
  sources: list[SourceItem] = Field(..., description="The sources of the answer")