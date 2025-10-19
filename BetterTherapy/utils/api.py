import requests
import bittensor as bt


def fetch_pool_miners():
    url = "https://api.taopoolmining.com/api/v1/pool/get/pool-metagraph"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            response_json = response.json()
            return response_json.get("data", [])
        else:
            bt.logging.error(
                f"Error fetching pool minerss: {response.status_code} {response.text}"
            )
            return []
    except Exception as e:
        bt.logging.error(f"Error while fetching pool miners: {e}")
        return []
