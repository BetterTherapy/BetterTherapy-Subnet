# bettertherapy/validator/test_forward.py
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from bettertherapy.protocol import BetterTherapySynapse
from bettertherapy.miner.inference import TherapyResponseSchema, ResponseBreakdown
from bettertherapy.validator.forward import forward
import bittensor as bt

@pytest.mark.asyncio
async def test_forward():
    # Mock the validator neuron (self)
    validator = MagicMock()
    validator.dendrite = AsyncMock(spec=bt.dendrite)
    validator.metagraph = MagicMock()
    # Mock metagraph attributes
    validator.metagraph.axons = [MagicMock(spec=bt.AxonInfo) for _ in range(3)]  # 3 mock axons
    for axon in validator.metagraph.axons:
        axon.is_serving = True  # Ensure axons are serving
    validator.metagraph.validator_permit = [True, False, False]  # UID 0 has permit
    validator.metagraph.S = [100.0, 200.0, 300.0]  # Stake values
    validator.config = MagicMock()
    validator.config.neuron.sample_size = 2
    validator.config.neuron.vpermit_tao_limit = 1000.0  # High limit to allow UIDs
    validator.step = 1  # Avoid fine-tuning
    validator.miner_scores = {}
    validator.resource_logs = {}
    validator.distribute_finetune_task = AsyncMock()

    # Mock get_random_uids
    def mock_get_random_uids(self, k, *args, **kwargs):
        return [0, 1][:k]
    validator.get_random_uids = mock_get_random_uids

    # Mock dendrite response
    mock_synapse_response = BetterTherapySynapse(
        messages=[{"role": "user", "content": "I feel tired."}],
        chat_history=[{"role": "user", "content": "I have trouble focusing."}],
        output=[{"response": "Mock response", "score": 85.0}]
    )
    validator.dendrite.return_value = [mock_synapse_response, None]

    # Input data
    messages = [{"role": "user", "content": "I feel tired."}]
    chat_history = [{"role": "user", "content": "I have trouble focusing."}]

    # Run the forward function
    result = await forward(validator, messages, chat_history)

    # Assertions
    assert isinstance(result, list), "Result should be a list."
    assert len(result) == 1, "Expected one valid response."
    assert result[0]["uid"] == 0, "Expected UID 0."
    assert result[0]["query"] == "I feel tired.", "Query should match input."
    assert result[0]["response_text"] == "Mock response", "Response text should match."
    assert result[0]["score"] == 85.0, "Score should match."
    assert "process_time" in result[0], "Process time should be included."
    assert validator.miner_scores == {0: [85.0]}, "Miner scores should be updated."
    assert validator.resource_logs == {0: {"api_calls": 1, "compute_hours": 0.1}}, "Resource logs should be updated."
    print("\nValidator Forward Result:", result)

@pytest.mark.asyncio
async def test_forward_finetune():
    # Mock the validator neuron (self)
    validator = MagicMock()
    validator.dendrite = AsyncMock(spec=bt.dendrite)
    validator.metagraph = MagicMock()
    # Mock metagraph attributes
    validator.metagraph.axons = [MagicMock(spec=bt.AxonInfo)]
    validator.metagraph.axons[0].is_serving = True
    validator.metagraph.validator_permit = [True]
    validator.metagraph.S = [100.0]
    validator.config = MagicMock()
    validator.config.neuron.sample_size = 1
    validator.config.neuron.vpermit_tao_limit = 1000.0
    validator.step = 100  # Trigger fine-tuning
    validator.miner_scores = {}
    validator.resource_logs = {0: {"api_calls": 10, "compute_hours": 1.0}}
    validator.distribute_finetune_task = AsyncMock()
    
    # Mock get_random_uids
    def mock_get_random_uids(self, k, *args, **kwargs):
        return [0][:k]
    validator.get_random_uids = mock_get_random_uids

    # Mock dendrite response
    mock_response = BetterTherapySynapse(
        messages=[{"role": "user", "content": "I feel tired."}],
        chat_history=[],
        output=[{"response": "Mock response", "score": 85.0}]
    )
    validator.dendrite.return_value = [mock_response]

    # Run the forward function
    result = await forward(validator, messages=[{"role": "user", "content": "I feel tired."}], chat_history=[])

    # Assertions
    assert isinstance(result, list), "Result should be a list."
    assert len(result) == 1, "Expected one valid response."
    assert validator.distribute_finetune_task.called, "Fine-tuning task should be called."
    assert validator.resource_logs[0]["compute_hours"] == 1.1, "Resource logs should reflect fine-tuning."
    assert validator.miner_scores == {0: [85.0]}, "Miner scores should be updated."
    print("\nValidator Forward Result (Fine-Tune):", result)
