import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from bettertherapy.protocol import BetterTherapySynapse
from bettertherapy.miner.forward import forward
from bettertherapy.miner.inference import TherapyResponseSchema, TherapyInference, ResponseBreakdown

@pytest.mark.asyncio
async def test_forward_response():
    miner_preds = {}
    inference_engine = MagicMock(spec=TherapyInference)
    inference_engine.generate_therapy_response = AsyncMock(return_value=TherapyResponseSchema(
        query="I feel tired and unmotivated.",
        response_text="Mock therapy response",
        breakdown=ResponseBreakdown(
            relevance=8.0,
            accuracy=7.0,
            empathy=9.0,
            clarity=8.0,
            contextuality=8.0
        ),
        user_rating=7.0
    ))

    synapse = BetterTherapySynapse(
        messages=[{"role": "user", "content": "I feel tired and unmotivated."}],
        chat_history=[
            {"role": "user", "content": "I have trouble focusing."},
            {"role": "assistant", "content": "That can be a symptom of anxiety or burnout."}
        ]
    )

    result = await forward(synapse, miner_preds, inference_engine)

    assert result.output is not None, "No output was returned from the forward function."
    assert isinstance(result.output, list), "Output should be a list."
    assert "response" in result.output[0], "Each output item should have a 'response'."
    assert "score" in result.output[0], "Each output item should have a 'score'."
    assert 0 <= result.output[0]["score"] <= 100, f"Score {result.output[0]['score']} is out of expected range (0-100)."
    print("\nForward Output:", result.output)
