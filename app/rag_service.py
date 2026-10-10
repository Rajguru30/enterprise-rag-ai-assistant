from pathlib import Path

from .embeddings import create_embedding_model
from .vector_store import load_vector_store
from .retriever import retrieve_documents
from .llm import create_llm
from .rag_chain import create_rag_prompt


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

VECTOR_STORE_PATH = (
    BASE_DIR
    / "data"
    / "vector_store"
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

TOP_K = 3

# Lower FAISS distance = better similarity.
RELEVANCE_THRESHOLD = 1.2


# ---------------------------------------------------------
# RAG Service
# ---------------------------------------------------------

class RAGService:

    def __init__(self):

        print("\nInitializing RAG service...")


        # -------------------------------------------------
        # 1. Load embedding model
        # -------------------------------------------------

        print("Loading embedding model...")

        self.embedding_model = (
            create_embedding_model()
        )

        print(
            "Embedding model loaded."
        )


        # -------------------------------------------------
        # 2. Check FAISS vector store
        # -------------------------------------------------

        if not VECTOR_STORE_PATH.exists():

            raise FileNotFoundError(
                "FAISS vector store not found at: "
                f"{VECTOR_STORE_PATH}"
            )


        # -------------------------------------------------
        # 3. Load FAISS vector store
        # -------------------------------------------------

        print(
            "Loading FAISS vector store..."
        )

        self.vector_store = (
            load_vector_store(
                self.embedding_model
            )
        )

        print(
            "FAISS vector store loaded."
        )


        # -------------------------------------------------
        # 4. Create LLM
        # -------------------------------------------------

        print("Creating LLM...")

        self.llm = create_llm()

        print("LLM created.")


        # -------------------------------------------------
        # 5. Create RAG prompt
        # -------------------------------------------------

        self.prompt = (
            create_rag_prompt()
        )

        print(
            "RAG service initialized successfully."
        )


    # -----------------------------------------------------
    # Ask question
    # -----------------------------------------------------

    def ask(self, question: str):


        # -------------------------------------------------
        # 1. Retrieve relevant documents
        # -------------------------------------------------

        results = retrieve_documents(
            self.vector_store,
            question,
            k=TOP_K
        )


        # -------------------------------------------------
        # 2. Handle no results
        # -------------------------------------------------

        if not results:

            return {
                "answer": (
                    "I don't have enough information "
                    "in the provided documents to answer "
                    "this question."
                ),
                "sources": []
            }


        # -------------------------------------------------
        # 3. Check retrieval relevance
        # -------------------------------------------------

        best_score = (
            results[0]
            .metadata
            .get("similarity_score")
        )


        # Lower FAISS distance means better similarity.
        if (
            best_score is not None
            and best_score > RELEVANCE_THRESHOLD
        ):

            return {
                "answer": (
                    "I don't have enough information "
                    "in the provided documents to answer "
                    "this question."
                ),
                "sources": []
            }


        # -------------------------------------------------
        # 4. Build context
        # -------------------------------------------------

        context = "\n\n".join(
            document.page_content
            for document in results
        )


        # -------------------------------------------------
        # 5. Create prompt
        # -------------------------------------------------

        messages = (
            self.prompt.format_messages(
                context=context,
                question=question
            )
        )


        # -------------------------------------------------
        # 6. Generate answer
        # -------------------------------------------------

        response = self.llm.invoke(
            messages
        )


        # -------------------------------------------------
        # 7. Extract clean answer
        # -------------------------------------------------

        if isinstance(
            response.content,
            list
        ):

            answer_parts = []

            for item in response.content:

                if isinstance(
                    item,
                    dict
                ):

                    text = item.get(
                        "text"
                    )

                    if text:

                        answer_parts.append(
                            text
                        )

            answer = "\n".join(
                answer_parts
            )

        else:

            answer = response.content


        # -------------------------------------------------
        # 8. Build source information
        # -------------------------------------------------

        sources = []

        seen_sources = set()


        for document in results:

            page = (
                document.metadata
                .get("page_label")
            )

            source = (
                document.metadata
                .get("source")
            )


            # Extract filename only
            if source:

                filename = (
                    str(source)
                    .replace("\\", "/")
                    .split("/")[-1]
                )

            else:

                filename = (
                    "Unknown document"
                )


            score = (
                document.metadata
                .get("similarity_score")
            )


            source_key = (
                filename,
                page
            )


            # Avoid duplicate sources
            if source_key not in seen_sources:

                sources.append(
                    {
                        "document": filename,

                        "page": (
                            str(page)
                            if page is not None
                            else None
                        ),

                        "score": score
                    }
                )

                seen_sources.add(
                    source_key
                )


        # -------------------------------------------------
        # 9. Return final result
        # -------------------------------------------------

        return {
            "answer": answer,
            "sources": sources
        }