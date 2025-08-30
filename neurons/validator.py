import json
import os
import time
from datetime import datetime

# Bittensor
import bittensor as bt
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from ulid import api

# import base validator class which takes care of most of the boilerplate
from BetterTherapy.base.validator import BaseValidatorNeuron

# Bittensor Validator Template:
from BetterTherapy.utils.wandb import SubnetEvaluationLogger
from BetterTherapy.validator import forward
from evals.eval import OpenAILLMAsJudgeEval
from evals.batch import OpenAIBatchLLMAsJudgeEval


class Validator(BaseValidatorNeuron):
    """
    Your validator neuron class. You should use this class to define your validator's behavior. In particular, you should replace the forward function with your own logic.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)  # noqa: UP008

        bt.logging.info("load_state()")
        self.load_state()
        self.setup_wandb()
        self.setup_evals()
        self.setup_batch_evals()
        bt.logging.info(f"Validator initialized with uid: {self.uid}")

    def setup_batch_evals(self):
        api_key = self.config.openai.api_key
        if api_key is None:
            raise ValueError(
                "OPENAI_API_KEY not set. Set it either in env(OPENAI_API_KEY) or using args --openai.api_key"
            )
        self.batch_evals = OpenAIBatchLLMAsJudgeEval(
            api_key=api_key, judge_model="gpt-4"
        )

    def setup_evals(self):
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self.evals = OpenAILLMAsJudgeEval(api_key=api_key, judge_model="gpt-4")
        self.evals_token_limit = 7000  # Excluding the buffer for system message, instructions, and formatting

    def setup_wandb(self):
        config_dir = os.path.expanduser("~/.bittensor/wandb")
        run_file = os.path.join(
            config_dir,
            f"validator-{self.uid}-{self.wallet.hotkey.ss58_address}-{datetime.now().strftime('%Y%m%d')}_run.json",
        )
        resume_run_id = None
        if os.path.exists(run_file):
            try:
                with open(run_file) as f:
                    run_info = json.load(f)
                resume_run_id = run_info.get("run_id")
                bt.logging.info(f"Found existing run to resume: {resume_run_id}")
            except:
                bt.logging.warning("Failed to load previous run info")
        self.wandb_logger = SubnetEvaluationLogger(
            validator_config={
                "uid": self.uid,
                "hotkey": self.wallet.hotkey.ss58_address,
                "netuid": self.config.netuid,
                "network": self.config.subtensor.network,
            },
            resume_run_id=resume_run_id,
        )

    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """
        # Rewrite this function based on your protocol definition.
        return await forward(self)


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info(f"Dividend: {validator.metagraph.D[validator.uid]}")
            time.sleep(5 * 60)
