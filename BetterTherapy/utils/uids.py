import random
from typing import Dict, Optional

import bittensor as bt
import numpy as np
import re


def check_uid_availability(
    metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int
) -> bool:
    """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        uid (int): uid to be checked
        vpermit_tao_limit (int): Validator permit tao limit
    Returns:
        bool: True if uid is available, False otherwise
    """
    # Filter non serving axons.
    if not metagraph.axons[uid].is_serving:
        return False
    # Filter validator permit > 1024 stake.
    if metagraph.validator_permit[uid]:
        if metagraph.S[uid] > vpermit_tao_limit:
            return False
    # Available otherwise.
    return True


def filter_uids(
    bt_obj, max_per_key: int = 15, blacklist: Optional[list[int]] = None
) -> np.ndarray:
    """Return available uids filtered by blacklist and per-coldkey cap."""
    available = []
    counts: Dict[str, int] = {}
    ip_counts: Dict[str, int] = {}

    for uid in range(bt_obj.metagraph.n.item()):
        ip_address = bt_obj.metagraph.axons[uid].ip
        ip_counts[ip_address] = ip_counts.get(ip_address, 0) + 1
        if ip_counts[ip_address] >= max_per_key:
            bt.logging.warning(
                f"IP {ip_address} for uid {uid} has reached max_per_key: {max_per_key}"
            )
            continue
        uid_is_available = check_uid_availability(
            bt_obj.metagraph, uid, bt_obj.config.neuron.vpermit_tao_limit
        )
        if not uid_is_available:
            continue
        ck = bt_obj.metagraph.coldkeys[uid]
        hk = bt_obj.metagraph.hotkeys[uid]

        if blacklist and hk in blacklist:
            continue

        cnt = counts.get(ck, 0)
        if cnt >= max_per_key:
            continue

        neuron = bt_obj.metagraph.axons[uid]
        if not neuron.is_serving:
            continue

        if (
            bt_obj.metagraph.validator_permit[uid]
            and bt_obj.metagraph.S[uid] > bt_obj.config.neuron.vpermit_tao_limit
        ):
            continue

        available.append(uid)
        counts[ck] = cnt + 1

    return np.array(available, dtype=int)


def get_available_uids(self, k: int, exclude: list[int] = None) -> np.ndarray:
    """Returns k available random uids from the metagraph.
    Args:
        k (int): Number of uids to return.
        exclude (List[int]): List of uids to exclude from the random sampling.
    Returns:
        uids (np.ndarray): Randomly sampled available uids.
    Notes:
        If `k` is larger than the number of available `uids`, set `k` to the number of available `uids`.
    """
    candidate_uids = []
    avail_uids = []

    for uid in range(self.metagraph.n.item()):
        uid_is_available = check_uid_availability(
            self.metagraph, uid, self.config.neuron.vpermit_tao_limit
        )
        uid_is_not_excluded = exclude is None or uid not in exclude

        if uid_is_available:
            avail_uids.append(uid)
            if uid_is_not_excluded:
                candidate_uids.append(uid)

    filtered_uids = filter_uids(self)
    # If k is larger than the number of available uids, set k to the number of available uids.
    k = min(k, len(avail_uids), len(filtered_uids))
    # Check if candidate_uids contain enough for querying, if not grab all avaliable uids
    available_uids = candidate_uids
    if len(candidate_uids) < k:
        available_uids += random.sample(
            [uid for uid in avail_uids if uid not in candidate_uids],
            k - len(candidate_uids),
        )

    uids_list = list(set(avail_uids).intersection(set(filtered_uids)))
    uids = np.array(uids_list)
    return uids
