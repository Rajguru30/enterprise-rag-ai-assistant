from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------
# Ask Request
# ---------------------------------------------------------

class AskRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Question to ask the RAG assistant"
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value):

        value = value.strip()

        if not value:

            raise ValueError(
                "Question cannot be empty."
            )

        return value


# ---------------------------------------------------------
# Source
# ---------------------------------------------------------

class Source(BaseModel):

    document: str

    page: str | None = None

    score: float | None = None


# ---------------------------------------------------------
# Ask Response
# ---------------------------------------------------------

class AskResponse(BaseModel):

    question: str

    answer: str

    sources: list[Source]