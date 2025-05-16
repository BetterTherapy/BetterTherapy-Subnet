# bettertherapy/validator/test_reward.py
import pytest
import numpy as np
from unittest.mock import MagicMock
from bettertherapy.validator.reward import reward, get_rewards, apply_rewards

def test_reward():
    scores = [80.0, 90.0]
    resource = 5.0
    result = reward(scores, resource)
    assert 0 <= result <= 100, "Reward should be in range 0-100"
    assert isinstance(result, float), "Reward should be a float"
    assert result == pytest.approx(71.0, 0.1), "Expected reward calculation"
    print(f"Reward Output: {result}")

def test_get_rewards():
    validator = MagicMock()
    validator.resource_logs = {0: {"compute_hours": 5.0}, 1: {"compute_hours": 2.0}}
    validator.metagraph.uids = [0, 1, 2]
    responses = [
        {"uid": 0, "score": 85.0, "query": "Test", "response_text": "Response", "process_time": 0.1},
        {"uid": 1, "score": 90.0, "query": "Test", "response_text": "Response", "process_time": 0.1}
    ]
    rewards = get_rewards(validator, responses)
    assert len(rewards) == 3, "Rewards array should match metagraph size"
    assert rewards[0] > 0, "Miner 0 should have a positive reward"
    assert rewards[1] > 0, "Miner 1 should have a positive reward"
    assert rewards[2] == 0, "Miner 2 should have no reward"
    print(f"Rewards Output: {rewards}")

def test_apply_rewards():
    validator = MagicMock()
    validator.metagraph.uids = [0, 1, 2]
    validator.metagraph.S = np.zeros(3)
    validator.miner_scores = {0: [85.0], 1: [90.0]}
    validator.resource_logs = {0: {"compute_hours": 5.0}, 1: {"compute_hours": 2.0}}
    rewards = np.array([50.0, 60.0, 0.0])
    apply_rewards(validator, rewards, [0, 1])
    assert validator.metagraph.S[0] > 0, "Miner 0 should have increased stake"
    assert validator.metagraph.S[1] > 0, "Miner 1 should have increased stake"
    assert validator.metagraph.S[2] == 0, "Miner 2 should have no stake change"
    assert validator.miner_scores == {}, "miner_scores should be reset"
    assert validator.resource_logs == {}, "resource_logs should be reset"
    print(f"Stake After Rewards: {validator.metagraph.S}")
