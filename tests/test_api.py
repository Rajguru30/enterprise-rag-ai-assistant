from fastapi.testclient import TestClient

from app.api import (
    app,
    get_rag_service
)


# ---------------------------------------------------------
# Fake RAG service
# ---------------------------------------------------------

class FakeRAGService:

    def ask(self, question: str):

        return {
            "answer": (
                f"Test answer for: {question}"
            ),
            "sources": [
                {
                    "document": "test_document.pdf",
                    "page": "1",
                    "score": 0.45
                }
            ]
        }


# ---------------------------------------------------------
# Fake service dependency
# ---------------------------------------------------------

def override_rag_service():

    return FakeRAGService()


# ---------------------------------------------------------
# Override real RAG service
# ---------------------------------------------------------

app.dependency_overrides[
    get_rag_service
] = override_rag_service


# ---------------------------------------------------------
# Test client
# ---------------------------------------------------------

client = TestClient(app)


# ---------------------------------------------------------
# Test 1: Root endpoint
# ---------------------------------------------------------

def test_root():

    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "message": (
            "Enterprise RAG AI Assistant is running"
        )
    }


# ---------------------------------------------------------
# Test 2: Valid question
# ---------------------------------------------------------

def test_ask_valid_question():

    response = client.post(
        "/ask",
        json={
            "question": (
                "What is Robot Framework?"
            )
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == (
        "What is Robot Framework?"
    )

    assert "answer" in data

    assert "sources" in data


# ---------------------------------------------------------
# Test 3: Source information
# ---------------------------------------------------------

def test_ask_returns_sources():

    response = client.post(
        "/ask",
        json={
            "question": (
                "What is Robot Framework?"
            )
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data["sources"]) == 1

    assert (
        data["sources"][0]["document"]
        == "test_document.pdf"
    )

    assert (
        data["sources"][0]["page"]
        == "1"
    )


# ---------------------------------------------------------
# Test 4: Empty question
# ---------------------------------------------------------

def test_empty_question():

    response = client.post(
        "/ask",
        json={
            "question": ""
        }
    )

    assert response.status_code == 422


# ---------------------------------------------------------
# Test 5: Too-short question
# ---------------------------------------------------------

def test_short_question():

    response = client.post(
        "/ask",
        json={
            "question": "Hi"
        }
    )

    assert response.status_code == 422


# ---------------------------------------------------------
# Test 6: Whitespace question
# ---------------------------------------------------------

def test_whitespace_question():

    response = client.post(
        "/ask",
        json={
            "question": "     "
        }
    )

    assert response.status_code == 422


# ---------------------------------------------------------
# Test 7: Question longer than 1000 characters
# ---------------------------------------------------------

def test_long_question():

    long_question = "A" * 1001

    response = client.post(
        "/ask",
        json={
            "question": long_question
        }
    )

    assert response.status_code == 422