import pytest
from main import (
    get_conversation_history,
    add_to_conversation_history,
    get_cached_response,
    set_cached_response
)
import json

def test_conversation_history(mock_redis, monkeypatch, test_user_id):
    monkeypatch.setattr("main.redis_conn", mock_redis)
    
    # Test adding to history
    add_to_conversation_history(test_user_id, "user", "test message")
    mock_redis.rpush.assert_called_once()
    mock_redis.ltrim.assert_called_once()
    
    # Test getting history
    mock_redis.lrange.return_value = [
        '{"role": "user", "content": "test message"}'
    ]
    history = get_conversation_history(test_user_id)
    assert len(history) == 1
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "test message"

def test_response_cache(mock_redis, monkeypatch, test_user_id):
    monkeypatch.setattr("main.redis_conn", mock_redis)
    
    # Test setting cache
    test_response = {"response": "test", "audio_base64": "test_audio"}
    set_cached_response(test_user_id, "test message", json.dumps(test_response))
    mock_redis.set.assert_called_once()
    
    # Test getting cache
    mock_redis.get.return_value = json.dumps(test_response)
    cached = get_cached_response(test_user_id, "test message")
    assert cached is not None
    assert json.loads(cached)["response"] == "test"