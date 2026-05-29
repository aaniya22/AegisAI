import pytest
import types
import sys
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.core.database import Base, get_db
from app.core.security import get_current_user
from app.models.user import User, SubscriptionTier


class DummyDoc:
    def __init__(self, source):
        self.metadata = {"source": source}
        self.page_content = "some content"


def _get_test_db():
    engine = create_engine("sqlite:///:memory:")
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def _override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    return _override_get_db


@pytest.fixture
def client():
    app.dependency_overrides[get_db] = _get_test_db()

    def _fake_user():
        u = User()
        u.id = 1
        u.email = "tester@example.com"
        u.subscription_tier = SubscriptionTier.FREE
        return u

    app.dependency_overrides[get_current_user] = _fake_user

    with TestClient(app) as c:
        yield c


def test_query_feedback_and_low_quality_flow(client):
    fake_result = {
        "result": "Test answer",
        "source_documents": [
            DummyDoc("doc1.pdf#chunk1"),
            DummyDoc("doc2.pdf#chunk2"),
        ],
    }

    fake_chain = MagicMock(return_value=fake_result)

    # Patch at the point of use: the rag endpoint module imports these names
    # inside the function body, so we intercept them in app.api.v1.rag
    with patch("app.api.v1.rag.get_qa_chain", return_value=fake_chain), \
         patch("app.api.v1.rag.compute_groundedness", return_value=0.9):
        resp = client.post("/api/v1/rag/query", json={"question": "What is X?"})

    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data and data["answer"] == "Test answer"
    assert "answer_id" in data
    answer_id = data["answer_id"]

    resp2 = client.post("/api/v1/rag/feedback", json={"answer_id": answer_id, "vote": "down"})
    assert resp2.status_code == 200

    def _admin_user():
        u = User()
        u.id = 2
        u.email = "admin@example.com"
        u.subscription_tier = SubscriptionTier.SCALE
        return u

    app.dependency_overrides[get_current_user] = _admin_user

    resp3 = client.get("/api/v1/rag/low-quality-chunks?threshold=0.0")
    assert resp3.status_code == 200
    out = resp3.json()
    assert "low_quality_chunks" in out
    chunks = {c["chunk"] for c in out["low_quality_chunks"]}
    assert "doc1.pdf#chunk1" in chunks or "doc2.pdf#chunk2" in chunks