import uuid
import bittensor as bt
import requests
import hashlib
import time


def compute_body_hash(instance_fields: dict):
    hashes = []
    required_hash_fields = ["hotkey", "coldkey", "uid", "uuid"]
    for field in required_hash_fields:
        hashes.append(get_hash(str(instance_fields[field])))

    return get_hash("".join(hashes))


def get_hash(content, encoding="utf-8"):
    sha3 = hashlib.sha3_256()
    sha3.update(content.encode(encoding))
    return sha3.hexdigest()


def blacklist_miner(
    wallet: bt.Wallet,
    blacklisted_hotkey: str,
    blacklisted_coldkey: str,
    uid: int,
    base_url: str,
):
    try:
        nonce = time.time_ns()
        uuid1 = uuid.uuid1()
        computed_body = {
            "hotkey": blacklisted_hotkey,
            "coldkey": blacklisted_coldkey,
            "uid": uid,
            "uuid": uuid1,
        }
        computed_body_hash = compute_body_hash(computed_body)

        message = f"{nonce}.{wallet.hotkey.ss58_address}.{uuid1}.{computed_body_hash}"
        signature = f"0x{wallet.hotkey.sign(message).hex()}"

        requests.post(
            f"{base_url}/pool/blacklist",
            {
                "signature": signature,
                "ss58Address": wallet.hotkey.ss58_address,
                "uuid": uuid1,
                "nonce": nonce,
                "blackListedHotkey": blacklisted_hotkey,
                "blackListedColdkey": blacklisted_coldkey,
                "blackListedUid": uid,
            },
        )
    except Exception as e:
        bt.logging.error(f"Error blacklisting {str(e)}")


def get_blacklisted_hotkeys(base_url: str):
    try:
        response = requests.get(
            f"{base_url}/pool/get/blacklist",
        )
        response_json = response.json()
        return response_json.get("data", [])
    except Exception as e:
        bt.logging.error(f"Error getting blacklisted hotkeys {str(e)}")
        return []
