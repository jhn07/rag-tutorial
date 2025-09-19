"""
RAG Service (AcmeTech FAQ) — with HyDE
"""

# Standard library imports
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Third-party imports
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()


@dataclass
class RAGConfig:
    """
    Configuration constants for RAG Service
    """
    # Relevance and filtering
    DEFAULT_RELEVANCE_THRESHOLD: float = 0.35
    ADAPTIVE_THRESHOLD_FACTOR: float = 0.7

    # Retrieval parameters
    DEFAULT_K: int = 4
    MMR_FETCH_K: int = 30
    MMR_LAMBDA: float = 0.8

    # Document chunking
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 120

    # Context and response
    DEFAULT_TOP_CONTEXT_DOCS: int = 1

    # Model settings
    LLM_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_TEMPERATURE: float = 0.0


class RAGPrompts:
    """Centralized prompt management for RAG Service"""

    FAQ_SYSTEM_PROMPT = (
        "You are AcmeTech's technical FAQ assistant.\n"
        "You may receive questions in different languages; always answer in English.\n"
        "Answer ONLY using the information in the provided context.\n"
        "If the answer is not present in the context, reply exactly: 'Not specified in the docs.'\n"
        "Be concise and factual."
    )

    FAQ_HUMAN_PROMPT = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    HYDE_SYSTEM_PROMPT = (
        "Write a short factual answer (1–2 sentences) to help retrieve relevant FAQ passages. "
        "Do not invent proper nouns. Keep it generic."
    )

    HYDE_HUMAN_PROMPT = "{q}"

    @classmethod
    def get_faq_prompt(cls) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", cls.FAQ_SYSTEM_PROMPT),
            ("human", cls.FAQ_HUMAN_PROMPT)
        ])

    @classmethod
    def get_hyde_prompt(cls) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", cls.HYDE_SYSTEM_PROMPT),
            ("human", cls.HYDE_HUMAN_PROMPT)
        ])


class RAGService:
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "faq_collection",
        relevance_threshold: Optional[float] = None,
        k: Optional[int] = None,
        faq_path: str = "faq.txt",
        top_context_docs: Optional[int] = None,
        use_adaptive_threshold: bool = True,
        allow_fallback: bool = True,
        config: Optional[RAGConfig] = None
    ):
        # Use provided config or create default
        self.config = config or RAGConfig()

        # Override config with explicit parameters if provided
        self.RELEVANCE_THRESHOLD = relevance_threshold or self.config.DEFAULT_RELEVANCE_THRESHOLD
        self.K = k or self.config.DEFAULT_K
        self.TOP_CTX = max(1, top_context_docs or self.config.DEFAULT_TOP_CONTEXT_DOCS)

        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.faq_path = faq_path
        self.use_adaptive_threshold = use_adaptive_threshold
        self.allow_fallback = allow_fallback

        # Initialize LLM and embeddings using config
        self.llm = ChatOpenAI(
            model=self.config.LLM_MODEL,
            temperature=self.config.LLM_TEMPERATURE
        )
        self.embeddings = OpenAIEmbeddings(model=self.config.EMBEDDING_MODEL)

        self.vectordb = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )

        self._ensure_ingested()

        self.retriever = self.vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.K,
                "fetch_k": self.config.MMR_FETCH_K,
                "lambda_mult": self.config.MMR_LAMBDA
            },
        )


    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Main entry point for asking questions to the RAG system
        """
        processed_question = self._process_query(question)
        if not processed_question:
            return {"answer": "Not specified in the docs.", "sources": []}

        expanded_query = self._hyde_expand(processed_question)
        relevant_docs = self._retrieve_and_filter_docs(expanded_query)

        if not relevant_docs:
            return {"answer": "Not specified in the docs.", "sources": []}

        return self._generate_answer(processed_question, relevant_docs)

    def _process_query(self, question: str) -> Optional[str]:
        """
        Process and validate the input question
        """
        question = (question or "").strip()
        if not question:
            return None

        if not question.endswith("?"):
            question += "?"

        return question

    def _retrieve_and_filter_docs(self, query: str) -> List[Tuple[Any, float]]:
        """
        Retrieve documents and filter by relevance thresholds
        """
        hits = self._retrieve_with_relevance(query, k=self.K)
        if not hits:
            return []

        # Normalize relevance scores
        normalized_hits = self._normalize_relevance_scores(hits)

        # Filter by relevance threshold
        filtered_docs = self._filter_by_relevance(normalized_hits)

        return filtered_docs

    def _normalize_relevance_scores(self, hits: List[Tuple[Any, float]]) -> List[Tuple[Any, float]]:
        """
        Normalize relevance scores to [0,1] range
        """
        if not hits:
            return []

        rel_values = [rel for _, rel in hits]

        def normalize(rel: float) -> float:
            if any(v < 0 for v in rel_values):  # cosine [-1,1]
                return (rel + 1.0) / 2.0
            if any(v > 1 for v in rel_values):  # distance 0..2
                return max(0.0, 1.0 - min(rel, 2.0) / 2.0)
            return rel  # already 0..1

        return [(doc, normalize(rel)) for doc, rel in hits]

    def _filter_by_relevance(self, hits: List[Tuple[Any, float]]) -> List[Tuple[Any, float]]:
        """
        Filter documents by relevance thresholds with adaptive and fallback logic
        """
        # Basic threshold filtering
        close = [(doc, rel) for doc, rel in hits if rel >= self.RELEVANCE_THRESHOLD]

        # Adaptive threshold
        if self.use_adaptive_threshold and close:
            top_rel = close[0][1]
            adaptive_threshold = max(
                self.RELEVANCE_THRESHOLD,
                top_rel * self.config.ADAPTIVE_THRESHOLD_FACTOR
            )
            close = [(d, r) for d, r in close if r >= adaptive_threshold]

        # Fallback mechanism
        if not close and self.allow_fallback and hits:
            best_doc, best_rel = max(hits, key=lambda it: it[1])
            close = [(best_doc, best_rel)]
            # Mark as low confidence by adding metadata
            close[0] = (close[0][0], close[0][1])

        # Sort by relevance descending
        close.sort(key=lambda it: it[1], reverse=True)

        return close

    def _generate_answer(self, question: str, relevant_docs: List[Tuple[Any, float]]) -> Dict[str, Any]:
        """
        Generate the final answer using the LLM chain
        """
        # Prepare context from top documents
        top_ctx_docs = [doc for doc, _ in relevant_docs[: self.TOP_CTX]]
        context_text = self._format_docs(top_ctx_docs)

        # Create and invoke the chain
        chain = (
            {"context": RunnableLambda(lambda _: context_text),
             "question": RunnablePassthrough()}
            | RAGPrompts.get_faq_prompt()
            | self.llm
            | StrOutputParser()
        )
        answer = chain.invoke(question)

        canonical_na = "Not specified in the docs."
        if answer.strip() == canonical_na:
            return {"answer": canonical_na, "sources": []}

        # Prepare sources
        best_doc, best_rel = relevant_docs[0]
        low_confidence = (
            not any(rel >= self.RELEVANCE_THRESHOLD for _, rel in relevant_docs)
            and self.allow_fallback
        )

        sources = [{
            "text": best_doc.page_content,
            "score": float(best_rel),
            "metadata": best_doc.metadata,
            "low_confidence": low_confidence
        }]

        return {"answer": answer, "sources": sources}

    def _hyde_expand(self, question: str) -> str:
        """
        Expand query using HyDE (Hypothetical Document Embeddings)
        """
        chain = RAGPrompts.get_hyde_prompt() | self.llm | StrOutputParser()
        try:
            hypo = chain.invoke({"q": question}) or ""
        except Exception:
            hypo = ""
        # Combine question and hypothetical answer
        return (question or "") + "\n---\n" + hypo


    # -------- utils --------
    def _format_docs(self, docs: List[Any]) -> str:
        """Format documents into a single context string"""
        return "\n\n".join(doc.page_content for doc in docs)

    def _retrieve_with_relevance(self, query: str, k: int = 4) -> List[Tuple[Any, float]]:
        """Retrieve documents with relevance scores"""
        return self.vectordb.similarity_search_with_relevance_scores(query, k=k)

    def _ensure_ingested(self) -> None:
        """Ensure FAQ documents are ingested into the vector database"""
        try:
            existing = self.vectordb._collection.count()
        except Exception:
            existing = 0

        if existing and existing > 0:
            return

        path = Path(self.faq_path)
        if not path.exists():
            raise FileNotFoundError(f"File {self.faq_path} not found.")

        faq_text = path.read_text(encoding="utf-8")
        if not faq_text.strip():
            raise ValueError("FAQ file is empty.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            add_start_index=True
        )
        docs = splitter.create_documents([faq_text])
        self.vectordb.add_documents(docs)
