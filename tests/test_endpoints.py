import pytest
from fastapi.testclient import TestClient
import json

def test_root_endpoint(test_client):
    response = test_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Status": "NeuroSphere Therapist API is running."}

def test_chat_endpoint(test_client, mock_redis, monkeypatch):
    # Test data
    test_request = {
        "user_id": "test_user",
        "message": "I'm feeling anxious",
        "gender": "male",
        "age": 25
    }
    
    # Mock Redis responses
    monkeypatch.setattr("main.redis_conn", mock_redis)
    mock_redis.get.return_value = None
    
    response = test_client.post("/chat", json=test_request)
    assert response.status_code == 200
    assert "response" in response.json()
    assert "audio_base64" in response.json()

def test_feedback_endpoint(test_client, mock_redis, monkeypatch):
    test_request = {
        "user_id": "test_user",
        "feedback": "helpful"
    }
    
    monkeypatch.setattr("main.redis_conn", mock_redis)
    mock_redis.get.return_value = "some_state"
    
    response = test_client.post("/feedback", json=test_request)
    assert response.status_code == 200
    assert response.json() == {"status": "success"}