from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

from chunker import split_documents
from embeddings import create_embedding_model
from vector_store import (
    create_vector_store,
    save_vector_store,
    load_vector_store
)
from retriever import retrieve_documents
from llm import create_llm
from rag_chain import create_rag_prompt


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PDF_PATH = r"C:\Users\KIIT\Downloads\Humanized_Ericsson_Framework_Report.pdf"

VECTOR_STORE_PATH = Path("data/vector_store")

QUERY = "What is the Ericsson Robot Framework automation framework?"

TOP_K = 3

# Lower FAISS distance = better similarity.
RELEVANCE_THRESHOLD = 1.2


# ---------------------------------------------------------
# 1. Create embedding model
# ---------------------------------------------------------

print("\nLoading embedding model...")

embedding_model = create_embedding_model()

print("Embedding model loaded successfully!")


# ---------------------------------------------------------
# 2. Load existing FAISS or create a new one
# ---------------------------------------------------------

if VECTOR_STORE_PATH.exists() and any(
    VECTOR_STORE_PATH.iterdir()
):

    # -----------------------------------------------------
    # Existing vector store
    # -----------------------------------------------------

    print("\nExisting FAISS vector store found.")

    vector_store = load_vector_store(
        embedding_model
    )

else:

    # -----------------------------------------------------
    # Create new vector store
    # -----------------------------------------------------

    print("\nNo existing vector store found.")

    print("\nLoading PDF...")

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print(
        f"Number of pages loaded: {len(documents)}"
    )

    print("\nSplitting documents into chunks...")

    chunks = split_documents(documents)

    print(
        f"Number of chunks created: {len(chunks)}"
    )

    print("\nCreating FAISS vector store...")

    vector_store = create_vector_store(
        chunks,
        embedding_model
    )

    print(
        "FAISS vector store created successfully!"
    )

    print("\nSaving FAISS vector store...")

    save_vector_store(vector_store)


# ---------------------------------------------------------
# 3. Create LLM
# ---------------------------------------------------------

print("\nCreating LLM...")

llm = create_llm()

print("LLM created successfully!")


# ---------------------------------------------------------
# 4. Create RAG prompt
# ---------------------------------------------------------

prompt = create_rag_prompt()


# ---------------------------------------------------------
# 5. User question
# ---------------------------------------------------------

query = QUERY

print("\n" + "=" * 60)
print("QUESTION")
print("=" * 60)

print(query)


# ---------------------------------------------------------
# 6. Retrieve relevant documents
# ---------------------------------------------------------

print("\nRetrieving relevant documents...")

results = retrieve_documents(
    vector_store,
    query,
    k=TOP_K
)

print(
    f"Retrieved {len(results)} relevant chunks."
)


# ---------------------------------------------------------
# 7. Check retrieval relevance
# ---------------------------------------------------------

if not results:

    print("\nNo relevant documents found.")

    print(
        "\nI don't have enough information in the "
        "provided documents to answer this question."
    )

    exit()


best_score = results[0].metadata.get(
    "similarity_score"
)


if best_score is not None:

    print(
        f"Best retrieval distance: {best_score:.4f}"
    )


# Lower distance means better similarity.
if (
    best_score is not None
    and best_score > RELEVANCE_THRESHOLD
):

    print(
        "\nRetrieved documents are not "
        "sufficiently relevant."
    )

    print(
        "\nI don't have enough information in the "
        "provided documents to answer this question."
    )

    exit()


# ---------------------------------------------------------
# 8. Build context
# ---------------------------------------------------------

context = "\n\n".join(
    document.page_content
    for document in results
)


# ---------------------------------------------------------
# 9. Create prompt
# ---------------------------------------------------------

messages = prompt.format_messages(
    context=context,
    question=query
)


# ---------------------------------------------------------
# 10. Generate answer
# ---------------------------------------------------------

print("\nGenerating answer...")

response = llm.invoke(messages)


# ---------------------------------------------------------
# 11. Extract clean answer
# ---------------------------------------------------------

if isinstance(response.content, list):

    answer_parts = []

    for item in response.content:

        if isinstance(item, dict):

            text = item.get("text")

            if text:
                answer_parts.append(text)

    answer = "\n".join(answer_parts)

else:

    answer = response.content


# ---------------------------------------------------------
# 12. Display final answer
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL ANSWER")
print("=" * 60)

print(answer)


# ---------------------------------------------------------
# 13. Display sources
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("SOURCES")
print("=" * 60)

seen_sources = set()

for document in results:

    page = document.metadata.get(
        "page_label"
    )

    source = document.metadata.get(
        "source",
        PDF_PATH
    )

    filename = (
        source
        .replace("\\", "/")
        .split("/")[-1]
    )

    score = document.metadata.get(
        "similarity_score"
    )

    source_key = (
        filename,
        page
    )

    if source_key not in seen_sources:

        if score is not None:

            print(
                f"- {filename} — Page {page} "
                f"(distance: {score:.4f})"
            )

        else:

            print(
                f"- {filename} — Page {page}"
            )

        seen_sources.add(source_key)


# ---------------------------------------------------------
# 14. Completion message
# ---------------------------------------------------------

print(
    "\nRAG pipeline completed successfully."
)