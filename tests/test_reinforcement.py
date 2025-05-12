import pytest
from reinforcement import PromptOptimizationRL

def test_rl_agent_initialization():
    agent = PromptOptimizationRL()
    assert agent.parameters is not None
    assert agent.learning_rate == 0.1
    assert agent.discount_factor == 0.9
    assert agent.exploration_rate == 0.3

def test_give_feedback():
    agent = PromptOptimizationRL()
    assert agent.give_feedback("helpful") == 1.0
    assert agent.give_feedback("not_helpful") == -1.0
    assert agent.give_feedback("more_empathy") == 0.5

def test_process_feedback():
    agent = PromptOptimizationRL()
    agent.last_state = "anxious"
    agent.last_action = "helpful"
    
    # Initial Q-value should be 0
    assert agent.q_table["anxious"]["helpful"] == 0.0
    
    # Process feedback and check Q-value update
    agent.process_feedback(1.0)
    assert agent.q_table["anxious"]["helpful"] > 0.0