import sys
import types
from unittest.mock import patch

import pytest
import types
import sys
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

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
    return TestingSessionLocal


@pytest.fixture
def client():
    # 1. Setup fresh in-memory DB
    SessionLocal = _get_test_db_session()
    db = SessionLocal()

    # 2. Seed a test user so foreign keys in RagQuery work
    user = User(
        id=1,
        email="tester@example.com",
        hashed_password="fakehash",
        subscription_tier=SubscriptionTier.FREE,
        is_active=True
    )
    db.add(user)
    
    # Add an admin user as well
    admin = User(
        id=2,
        email="admin@example.com",
        hashed_password="fakehash",
        subscription_tier=SubscriptionTier.SCALE,
        is_active=True
    )
    db.add(admin)
    db.commit()

    # 3. Define dependency overrides
    def _override_get_db():
        try:
            yield db
        finally:
            pass # Keep it open for the duration of the test


@pytest.fixture
def client():
    app.dependency_overrides[get_db] = _get_test_db()

    def _fake_user():
        return user

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_current_user] = _fake_user

    with TestClient(app) as c:
        yield c
    
    # 4. Cleanup
    db.close()
    app.dependency_overrides.clear()


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
    with patch("app.modules.rag.retrieval_chain.get_qa_chain", return_value=fake_chain), \
         patch("app.modules.rag.groundedness.compute_groundedness", return_value=0.9):
        resp = client.post("/api/v1/rag/query", json={"question": "What is X?"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "Test answer"
    assert "answer_id" in data
    answer_id = data["answer_id"]

    resp2 = client.post("/api/v1/rag/feedback", json={"answer_id": answer_id, "vote": "down"})
    assert resp2.status_code == 200

    def _admin_user():
        u = User(id=2, subscription_tier=SubscriptionTier.SCALE)
        return u

    app.dependency_overrides[get_current_user] = _admin_user

    # 4. Verify chunk tracking extraction logic works seamlessly
    resp3 = client.get("/api/v1/rag/low-quality-chunks?threshold=0.0")
    assert resp3.status_code == 200
    out = resp3.json()
    assert "low_quality_chunks" in out
    chunks = {c["chunk"] for c in out["low_quality_chunks"]}
    assert "doc1.pdf#chunk1" in chunks or "doc2.pdf#chunk2" in chunks
