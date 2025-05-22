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

import numpy as np
from typing import List, Dict
import bittensor as bt

def calculate_miner_score(ai_response: dict, process_time: float) -> float:
    """
    Calculate a miner's score based on response breakdown, process time, and user rating.

    Args:
        ai_response (dict): Miner's response containing 'breakdown', 'response', and 'user_rating'.
        process_time (float): Time taken to process the query.

    Returns:
        float: Computed score for the miner's response, rounded to 2 decimal places.
    """
    if not ai_response or "breakdown" not in ai_response:
        bt.logging.error("Missing or invalid ai_response/breakdown")
        return 0.0
    breakdown = ai_response["breakdown"]
    bt.logging.debug(f"Breakdown: {breakdown}, type: {type(breakdown)}")
    user_rating = ai_response.get("user_rating", 0.0)
    weights = {
        "relevance": 3.0,
        "accuracy": 2.0,
        "empathy": 2.0,
        "clarity": 1.5,
        "contextuality": 1.5
    }
    field_names = ["relevance", "accuracy", "empathy", "clarity", "contextuality"]
    try:
        scores = {field: float(breakdown[field]) for field in field_names}
    except (KeyError, TypeError, AttributeError) as e:
        bt.logging.error(f"Invalid breakdown format: {e}, breakdown: {breakdown}")
        return 0.0
    breakdown_score = sum(scores[key] * weights[key] for key in scores) / 10.0
    speed_score = min(10.0, 1.0 / (process_time + 0.01))
    overall_score = (0.6 * breakdown_score) + (0.2 * speed_score) + (0.2 * user_rating)
    score = round(overall_score * 10, 2)
    bt.logging.debug(f"Score calculated: {score}, breakdown={scores}, process_time={process_time}, user_rating={user_rating}")
    return score

def reward(miner_scores: List[float], resource_contribution: float) -> float:
    """
    Calculate reward for a miner based on response scores and resource contribution.

    Args:
        miner_scores (List[float]): List of response scores for the miner.
        resource_contribution (float): Miner's resource contribution (e.g., compute hours).

    Returns:
        float: Reward value for the miner.
    """
    avg_score = sum(miner_scores) / len(miner_scores) if miner_scores else 0.0
    score_weight = avg_score / 100.0  # Normalize score (0-100 scale)
    resource_weight = min(resource_contribution / 10.0, 1.0)  # Cap resource contribution
    # Combined reward: 60% score, 40% resource
    combined_reward = (0.6 * score_weight + 0.4 * resource_weight) * 100.0
    bt.logging.debug(f"Reward calc: score={avg_score}, resource={resource_contribution}, reward={combined_reward}")
    return round(combined_reward, 2)

# def get_rewards(self, responses: List[Dict]) -> np.ndarray:
#     """
#     Returns an array of rewards for miners based on their responses.

#     Args:
#         self: The validator neuron object.
#         responses (List[Dict]): List of processed responses from miners.

#     Returns:
#         np.ndarray: Array of rewards for each miner.
#     """
#     miner_rewards = {}
#     miner_uids = set(resp["uid"] for resp in responses)

#     for uid in miner_uids:
#         miner_responses = [resp for resp in responses if resp["uid"] == uid]
#         miner_scores = [resp["score"] for resp in miner_responses]
#         resource_contribution = self.resource_logs.get(uid, {"compute_hours": 0.0})["compute_hours"]
#         miner_rewards[uid] = reward(miner_scores, resource_contribution)

#     # Create reward array aligned with metagraph UIDs
#     rewards = np.zeros(len(self.metagraph.uids))
#     for uid in miner_uids:
#         rewards[uid] = miner_rewards.get(uid, 0.0)
    
#     bt.logging.info(f"Scored rewards: {rewards}")
#     return rewards

# def apply_rewards(self, rewards: np.ndarray, miner_uids: List[int]):
#     """
#     Apply rewards to miners and reset tracking for the next cycle.

#     Args:
#         self: The validator neuron object.
#         rewards (np.ndarray): Array of rewards for miners.
#         miner_uids (List[int]): List of miner UIDs that were queried.
#     """
#     # Select top 100 miners by reward
#     top_indices = np.argsort(rewards)[::-1][:100]
#     total_reward = sum(rewards[top_indices]) or 1.0  # Avoid division by zero

#     # Update metagraph scores for top miners
#     for idx in top_indices:
#         if rewards[idx] > 0:
#             normalized_reward = rewards[idx] / total_reward
#             self.metagraph.S[idx] += normalized_reward  # Update stake (simulated)
#             bt.logging.info(f"Applied reward to miner {idx}: {rewards[idx]} (Normalized: {normalized_reward})")

#     # Reset tracking for next cycle
#     self.miner_scores = {}
#     self.resource_logs = {}
#     bt.logging.info("Reset miner scores and resource logs for next cycle")
