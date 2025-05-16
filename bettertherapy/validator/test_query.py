# bettertherapy/validator/test_query.py
import pytest
import asyncio
import bittensor as bt
from unittest.mock import MagicMock
from bettertherapy.protocol import BetterTherapySynapse

@pytest.mark.asyncio
async def test_query():
    # Mock wallet to avoid KeyFileError
    mock_wallet = MagicMock()
    mock_wallet.hotkey = MagicMock()
    mock_wallet.hotkey.ss58_address = "mock_hotkey_address"

    dendrite = bt.dendrite(wallet=mock_wallet)
    miner_axon = bt.axon(port=8091, external_ip="127.0.0.1", wallet=mock_wallet)
    synapse = BetterTherapySynapse(
        messages=[{"role": "user", "content": "I feel anxious, what should I do?"}],
        chat_history=[
            {"role": "user", "content": "I haven't been sleeping well."},
            {"role": "assistant", "content": "That sounds tough."}
        ]
    )
    responses = await dendrite([miner_axon], synapse, timeout=30.0)
    valid_responses = [r.output for r in responses if r.output]

    print("Query Responses:", valid_responses)

    assert len(valid_responses) > 0, "Expected at least one valid response from miner"
    assert isinstance(valid_responses[0], list), "Response output should be a list"
    assert "response" in valid_responses[0][0], "Response should contain 'response' key"
    assert "score" in valid_responses[0][0], "Response should contain 'score' key"
    assert 0 <= valid_responses[0][0]["score"] <= 100, "Score should be between 0 and 100"
