from pathlib import Path

from langchain_community.vectorstores import FAISS


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
# Create vector store
# ---------------------------------------------------------

def create_vector_store(chunks, embedding_model):

    vector_store = FAISS.from_documents(
        chunks,
        embedding_model
    )

    return vector_store


# ---------------------------------------------------------
# Save vector store
# ---------------------------------------------------------

def save_vector_store(vector_store):

    VECTOR_STORE_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    vector_store.save_local(
        str(VECTOR_STORE_PATH)
    )

    print(
        f"FAISS vector store saved to: "
        f"{VECTOR_STORE_PATH}"
    )


# ---------------------------------------------------------
# Load vector store
# ---------------------------------------------------------

def load_vector_store(embedding_model):

    if not VECTOR_STORE_PATH.exists():

        raise FileNotFoundError(
            f"FAISS vector store not found at: "
            f"{VECTOR_STORE_PATH}"
        )

    vector_store = FAISS.load_local(
        str(VECTOR_STORE_PATH),
        embedding_model,
        allow_dangerous_deserialization=True
    )

    print(
        f"FAISS vector store loaded from: "
        f"{VECTOR_STORE_PATH}"
    )

    return vector_store