# agents.md — Python Coding Rules

> Apply these rules to every Python file generated or modified in this project.

---

## 1. Docstrings — English, Google-style, Always Complete

All docstrings must be in **English** and follow Google style with `Args:`, `Returns:`, and `Raises:` sections. First line uses imperative mood. Never leave a one-liner where parameters exist.

```python
def embed_chunks(
    chunks: list[Chunk],
    client: OpenAI,
    batch_size: int = 100,
) -> Generator[tuple[Chunk, list[float]], None, None]:
    """Yield (chunk, embedding_vector) pairs via the OpenAI Embeddings API.

    Processes chunks in batches with automatic retry on transient errors.

    Args:
        chunks: Chunk objects whose text fields will be embedded.
        client: Authenticated OpenAI client instance.
        batch_size: Number of texts per API call (max 2048; keep ≤ 100).

    Yields:
        Tuples of (Chunk, List[float]) where the list is the embedding vector.
    """
```

---

## 2. One Parameter Per Line

Functions with more than one parameter must break each onto its own line. Applies to `def`, Pydantic `Field(...)`, and long call sites.

```python
# correct
def upsert_to_pinecone(
    index,
    chunks: list[tuple[Chunk, list[float]]],
    namespace: str = "v1",
    batch_size: int = 100,
) -> int:

# wrong
def upsert_to_pinecone(index, chunks, namespace="v1", batch_size=100):
```

---

## 3. Type Annotations — Always

Every parameter and return type must be annotated. Use `from __future__ import annotations` at the top of every module. Prefer built-in generics (`list[str]`, `dict[str, int]`).

```python
def clean_ocr(text: str) -> str: ...
def run_pipeline(args: argparse.Namespace) -> None: ...
```

---

## 4. Models — Pydantic v2 BaseModel

Use `BaseModel`, never `@dataclass`. Enums inherit from `str, Enum`. Every model must have a docstring describing its domain role.

```python
class TherapyType(str, Enum):
    """Types of therapeutic disciplines in the treatment plan."""
    SPEECH_LANGUAGE = "Fonoaudiologia"

class Therapist(BaseModel):
    """A licensed professional providing direct therapeutic service to the patient."""
    name: str
    specialty: TherapyType
    registration: str = Field(description="Professional council number.")
```

---

## 5. requirements.txt — Always Up to Date

- Every new import must be added to `requirements.txt` **in the same task**.
- Use `>=` for version pinning. Group entries by purpose with a comment.
- Document system dependencies (e.g. `tesseract-ocr`) in a comment block at the bottom.
- Remove unused dependencies when removing code.

```text
# PDF processing
pymupdf>=1.24.0
pytesseract>=0.3.13

# Vector store
pinecone>=3.0.0

# System deps (install separately):
# sudo apt-get install tesseract-ocr tesseract-ocr-por
```

---

## 6. Unit Tests — Mandatory

- Every new module must have `tests/test_<module_name>.py` using `pytest`.
- All public functions must have at least one test, including edge cases.
- Mock all external I/O (APIs, file system) — never make real HTTP calls in tests.
- Test names follow `test_<function>_<scenario>`. Each test has a one-line English docstring.

```python
def test_clean_ocr_removes_lone_page_numbers():
    """Assert that standalone page numbers are stripped from OCR output."""
    assert "12" not in clean_ocr("Some text\n\n12\n\nMore text")

def test_embed_chunks_calls_openai(sample_page):
    """Assert that embed_chunks calls the OpenAI endpoint once per batch."""
    mock_client = MagicMock()
    ...
    mock_client.embeddings.create.assert_called_once()
```

---

## 7. General Rules

| Rule | Convention |
|---|---|
| Imports order | stdlib → third-party → local, blank line between groups |
| Constants | `UPPER_SNAKE_CASE` at module level with inline comment |
| Magic numbers | Forbidden — use named constants or parameters with defaults |
| Logging | Use `logging`, never `print`. `DEBUG` per item, `INFO` summaries, `WARNING` retries |
| API calls | Wrap in retry loop: `wait = 2 ** attempt`, at least 3 attempts |
| Strings | Double quotes everywhere; single quotes only inside f-strings to avoid escaping |
| Line length | 100 characters maximum |
| File ending | Single newline at end of every file |