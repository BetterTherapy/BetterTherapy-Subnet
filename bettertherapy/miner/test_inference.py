import asyncio
from bettertherapy.miner.inference import TherapyInference  
import pytest  
# Simulated user query and chat history
query = "I have been feeling anxious lately, what should I do?"
chat_history = [
    {"role": "user", "content": "I haven't been sleeping well."},
    {"role": "assistant", "content": "You might be experiencing stress-related symptoms."}
]
@pytest.mark.asyncio
async def test_generate():
    inference = TherapyInference()
    result = await inference.generate_therapy_response(query, chat_history)
    print("Generated Response:\n", result.model_dump_json(indent=2))

    # Test updating the user rating
    updated = await inference.update_user_rating(result, 7.5)
    print("\nUpdated User Rating:\n", updated.model_dump_json(indent=2))

# Run the async test
asyncio.run(test_generate())
