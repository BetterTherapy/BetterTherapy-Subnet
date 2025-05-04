import asyncio
import time

from neurons.miner import Miner
import bettertherapy
from bettertherapy.protocol import BetterTherapySynapse
import bittensor as bt

miner_preds = {}
base_model = None  # basemodel

def get_overall_score(response_text: str, process_time: float, user_rating: float = 1.0):
    # Simulated scoring: accuracy, speed , user rating
    accuracy = len(response_text) / 100.0 if response_text else 0.0
    speed_score = 1.0 / (process_time + 1e-5)  # Avoid division by zero
    overall_score = (0.5 * accuracy) + (0.3 * speed_score) + (0.2 * user_rating)
    return round(overall_score * 100, 2)  # Scale to 100 for consistency

async def forward(self: Miner, synapse: BetterTherapySynapse):
    """
    Asynchronously process user queries using a fine-tuned medical model.
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
            tasks.append(self.infer_response(message["content"], synapse.chat_history))

    bt.logging.info("Running inference tasks...")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    process_time = time.time() - start_time
    for task_index, result in enumerate(results):
        i, query_id = query_ids[task_index]
        try:
            if isinstance(result, Exception):
                raise result
            response_text = result
            score = get_overall_score(response_text, process_time)
            predictions[i] = {"response": response_text, "score": score}
            miner_preds[query_id] = predictions[i]  # Save to cache
            bt.logging.info(f"Response for query {query_id}: {response_text} (Score: {score})")
        except Exception as e:
            bt.logging.error(f"Error processing query {query_id}: {e}")
            predictions[i] = None

    synapse.output = predictions
    return synapse

async def infer_response(self, query: str, chat_history: list):
    """
    Simulate inference using the fine-tuned medical model.
    Placeholder for actual model inference logic.
    """
    global base_model
    if base_model is None:
        # Simulate model (e.g., from miner-shared resources)
        base_model = "medical_exam_model_v1"  # basemodel

    #generate response based on chat history
    history_context = " ".join([msg["content"] for msg in chat_history]) if chat_history else "No prior context"
    response = f"Dr. AI: Response to '{query}' - Based on history ({history_context}), I recommend a therapy session or medical review."
    return response
