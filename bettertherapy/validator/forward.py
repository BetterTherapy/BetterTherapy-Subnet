# validator/forward.py

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
import numpy as np
import bittensor as bt
from bettertherapy.protocol import BetterTherapySynapse
from bettertherapy.utils.uids import get_random_uids
from bettertherapy.validator.reward import calculate_miner_score

async def validate_response(response: dict, query: str, chat_history: List[Dict[str, str]]) -> bool:
    """Basic validation of miner response for safety and appropriateness."""
    if not response or not response.get("response"):
        bt.logging.warning("No response content")
        return False
    response_text = response["response"]
    harmful_keywords = ["harm", "suicide", "violence"]
    is_valid = not any(keyword in response_text.lower() for keyword in harmful_keywords)
    if not is_valid:
        bt.logging.warning(f"Response contains harmful content: {response_text[:50]}...")
    return is_valid

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

    # Config settings for this forward call
    self.config.neuron.sample_size = 5
    self.config.neuron.timeout = 60

    # Create synapse for miner communication
    synapse = BetterTherapySynapse(
        messages=messages,
        chat_history=chat_history,
        output=None
    )

    # Select miners to query
    # miner_uids = get_random_uids(self, k=min(self.config.neuron.sample_size, len(self.metagraph.axons)))
    miner_uids = [142]  # Hardcoded example

    bt.logging.debug(f"Querying miners: {miner_uids}")

    # Query miners
    start_time = time.time()
    responses = await self.dendrite(
        axons=[self.metagraph.axons[uid] for uid in miner_uids],
        synapse=synapse,
        deserialize=False,  # Responses are handled manually
        timeout=self.config.neuron.timeout
    )

    # Process responses
    processed_responses = []
    rewards = []
    valid_uids = []

    for idx, (uid, response) in enumerate(zip(miner_uids, responses)):
        # Track resource usage per miner
        if uid not in self.resource_logs:
            self.resource_logs[uid] = {"api_calls": 0, "compute_hours": 0.0}
        self.resource_logs[uid]["api_calls"] += 1
        self.resource_logs[uid]["compute_hours"] += 0.1
        bt.logging.debug(f"Miner {uid} resource log: {self.resource_logs[uid]}")

        if not response or not response.output:
            bt.logging.warning(f"No valid response from miner {uid}")
            self.miner_scores.setdefault(uid, []).append(0.0)
            continue

        bt.logging.debug(f"Miner {uid} raw response: {response.output}")
        process_time = time.time() - start_time

        for i, pred in enumerate(response.output):
            if not pred or "response" not in pred:
                bt.logging.warning(f"Invalid prediction from miner {uid} for query {i}")
                self.miner_scores.setdefault(uid, []).append(0.0)
                rewards.append(0.0)
                valid_uids.append(uid)
                continue

            # Validate miner response for safety
            if not await validate_response(pred, messages[i]["content"], chat_history):
                bt.logging.warning(f"Invalid response from miner {uid} for query {i}")
                self.miner_scores.setdefault(uid, []).append(0.0)
                rewards.append(0.0)
                valid_uids.append(uid)
                continue

            # Calculate score for valid response
            score = calculate_miner_score(pred, pred.get("process_time", process_time))
            processed_responses.append({
                "uid": uid,
                "query": messages[i]["content"],
                "response_text": pred["response"],
                "score": score,
                "process_time": process_time
            })
            self.miner_scores.setdefault(uid, []).append(score)
            rewards.append(score)
            valid_uids.append(uid)

            bt.logging.info(f"Miner {uid} score for query {i}: {score}")

    # Update scores on the validator using gathered rewards
    if rewards and valid_uids:
        rewards_array = np.array(rewards)
        self.update_scores(rewards_array, valid_uids)
        bt.logging.debug(f"Updated validator scores for miners: {valid_uids}")

    return processed_responses
