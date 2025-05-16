# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 BetterTherapy Team

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import asyncio
import time
from typing import List, Dict
import bittensor as bt
from bettertherapy.protocol import BetterTherapySynapse
from bettertherapy.utils.uids import get_random_uids
from bettertherapy.miner.inference import TherapyResponseSchema

async def forward(self, messages: List[Dict[str, str]], chat_history: List[Dict[str, str]]):
    """
    The forward function is called by the validator every time step.

    It queries miners with user messages, validates responses, and tracks miner performance.

    Args:
        self: The validator neuron object.
        messages (List[Dict[str, str]]): List of user query messages.
        chat_history (List[Dict[str, str]]): Chat history for context.

    Returns:
        List[Dict]: Processed responses with miner UID, query, response text, score, and process time.
    """
    bt.logging.info(f"Processing queries: {messages}")

    # Initialize tracking for miner performance
    if not hasattr(self, "miner_scores"):
        self.miner_scores = {}
    if not hasattr(self, "resource_logs"):
        self.resource_logs = {}

    # Create synapse for miner communication
    synapse = BetterTherapySynapse(
        messages=messages,
        chat_history=chat_history,
        output=None
    )

    # Select miners to query
    miner_uids = get_random_uids(self, k=min(self.config.neuron.sample_size, len(self.metagraph.axons)))

    # Query miners
    start_time = time.time()
    responses = await self.dendrite(
        axons=[self.metagraph.axons[uid] for uid in miner_uids],
        synapse=synapse,
        deserialize=False,  # Responses are handled manually
        timeout=30.0
    )

    # Process responses
    processed_responses = []
    for idx, (uid, response) in enumerate(zip(miner_uids, responses)):
        # Simulate resource validation (e.g., API calls, compute contribution)
        if uid not in self.resource_logs:
            self.resource_logs[uid] = {"api_calls": 0, "compute_hours": 0.0}
        self.resource_logs[uid]["api_calls"] += 1
        self.resource_logs[uid]["compute_hours"] += 0.1
        bt.logging.debug(f"Miner {uid} resource log: {self.resource_logs[uid]}")

        if not response or not response.output:
            bt.logging.warning(f"No valid response from miner {uid}")
            continue

        process_time = time.time() - start_time
        for i, pred in enumerate(response.output):
            if pred and "response" in pred and "score" in pred:
                processed_responses.append({
                    "uid": uid,
                    "query": messages[i]["content"],
                    "response_text": pred["response"],
                    "score": pred["score"],
                    "process_time": process_time
                })
                # Update miner scores
                if uid not in self.miner_scores:
                    self.miner_scores[uid] = []
                self.miner_scores[uid].append(pred["score"])
                bt.logging.info(f"Miner {uid} score for query {i}: {pred['score']}")

    # Distribute fine-tuning tasks (simulated, once per cycle)
    if self.step % 100 == 0:  # Simulate daily fine-tuning
        await self.distribute_finetune_task()

    return processed_responses

async def distribute_finetune_task(self):
    """Distribute daily fine-tuning tasks to top miners based on resource contributions."""
    top_miners = sorted(
        self.resource_logs.items(),
        key=lambda x: x[1]["compute_hours"],
        reverse=True
    )[:10]  # Top 10 miners
    for miner_uid, _ in top_miners:
        bt.logging.info(f"Assigning fine-tuning task to miner {miner_uid}")
        self.resource_logs[miner_uid]["compute_hours"] += 0.5  
