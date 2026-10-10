def retrieve_documents(vector_store, query, k=3):

    results = vector_store.similarity_search_with_score(
        query,
        k=k
    )

    documents = []

    for document, score in results:

        document.metadata["similarity_score"] = float(score)

        documents.append(document)

    return documents