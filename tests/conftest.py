import pytest
from fastapi.testclient import TestClient
from main import app
import redis
from unittest.mock import Mock
import os

@pytest.fixture
def test_client():
    return TestClient(app)

@pytest.fixture
def mock_redis():
    return Mock(spec=redis.Redis)

@pytest.fixture
def mock_genai():
    return Mock()

@pytest.fixture
def test_user_id():
    return "test_user_123"