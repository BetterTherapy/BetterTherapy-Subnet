import asyncio
import time

from neurons.miner import Miner
import bettertherapy
from bettertherapy.protocol import BetterTherapySynapse
from bettertherapy.miner.inference import TherapyResponseSchema, TherapyInference
import bittensor as bt

miner_preds = {}
inference_engine = TherapyInference()

def get_overall_score(ai_response: TherapyResponseSchema, process_time: float):
    """Compute overall score based on breakdown, speed, and user rating."""
    if isinstance(ai_response, TherapyResponseSchema):
        breakdown = ai_response.breakdown
        user_rating = ai_response.user_rating
    else:
        return 0.0
    weights = {
        "relevance": 3,
        "accuracy": 2,
        "empathy": 2,
        "clarity": 1.5,
        "contextuality": 1.5
    }
    field_names = ["relevance", "accuracy", "empathy", "clarity", "contextuality"]
    scores = {field: getattr(breakdown, field) for field in field_names}
    breakdown_score = sum(float(scores[key]) * weights[key] for key in scores) / 10  # Normalize to 0-10
    speed_score = 1.0 / (process_time + 1e-5)  # Inverse time for speed
    overall_score = (0.6 * breakdown_score) + (0.2 * speed_score) + (0.2 * user_rating)
    return round(overall_score * 10, 2)  # Scale to 100

async def forward(self: Miner, synapse: BetterTherapySynapse):
    """
    Asynchronously process user queries using the fine-tuned medical model.
    Uses caching to avoid redundant inference.
    """
    bt.logging.info(f"Received mine requests for queries {synapse.messages}")

    tasks = []
    query_ids = []
    predictions = [None] * len(synapse.messages)  # Placeholder for responses

    start_time = time.time()
    for i, message in enumerate(synapse.messages):
        query_id = f"{message['content']}_{hash(str(synapse.chat_history))}"  # Unique ID
        if query_id in miner_preds:
            bt.logging.info(f"Using cached prediction for {query_id}: {miner_preds[query_id]}")
            predictions[i] = miner_preds[query_id]
        else:
            query_ids.append((i, query_id))
            tasks.append(inference_engine.generate_therapy_response(message["content"], synapse.chat_history))

    bt.logging.info("Running inference tasks...")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    process_time = time.time() - start_time
    for task_index, result in enumerate(results):
        i, query_id = query_ids[task_index]
        try:
            if isinstance(result, Exception):
                raise result
            score = get_overall_score(result, process_time)
            predictions[i] = {"response": result.response_text, "score": score}
            miner_preds[query_id] = predictions[i]  # Save to cache
            bt.logging.info(f"Response for query {query_id}: {result.response_text} (Score: {score})")
        except Exception as e:
            bt.logging.error(f"Error processing query {query_id}: {e}")
            predictions[i] = None

    synapse.output = predictions
    return synapse
