# BetterTherapy

## Introduction

[BetterTherapy](https://bettertherapy.ai/) Subnet is a decentralized protocol for deploying personalized AI Doctor Twins. Each twin represents a real-world, licensed expert, fine-tuned with domain-specific knowledge to deliver high-quality, trustworthy health insights to users.

<img width="3012" height="1004" alt="BT Cover (1)" src="https://github.com/user-attachments/assets/8f6d0824-67ae-43fc-9e4c-d7a92f9f152e" />

## Incentive Mechanism

The BetterTherapy subnet implements a sophisticated incentive mechanism that rewards miners based on both response quality and response time. The scoring system ensures that miners are incentivized to provide high-quality, timely responses.

### Scoring Components

The total score for each miner response consists of two components:

1. **Quality Score (70% weight)**: Based on the reward value from response evaluation
2. **Response Time Score (30% weight)**: Based on how quickly the miner responds

### Response Time Scoring

Miners receive response time bonuses only if their quality reward exceeds 0.2 (indicating a minimum quality threshold):

- **Under 10 seconds**: 100 points
- **10-20 seconds**: 50 points
- **20-30 seconds**: 20 points
- **Over 30 seconds**: 0 points

The response time score is then weighted at 30% of the final score.

### Quality Scoring

The quality score is calculated as:

```
quality_score = reward * 100 * 0.7
```

Where `reward` is the evaluation score (0-1) from the response quality assessment.

### Final Score Calculation

```
total_score = (response_time_score * 0.3) + (quality_score * 0.7)
```

### Zero Score Conditions

Miners receive a score of 0 if any of the following conditions are met:

- Response output is None or empty
- Quality reward is None or 0
- No valid response is provided

This mechanism encourages miners to optimize for both response quality and speed, creating a balanced incentive structure that benefits end users with fast, high-quality responses.

---

## Project Setup

### 1. Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (Python package manager)
- [Bittensor](https://github.com/opentensor/bittensor#install)
- `btcli` (Bittensor CLI)

### 2. Install Dependencies with `uv`

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, reliable dependency management.

```bash
# Install uv if you don't have it
pip install uv

# Install all dependencies in a virtual environment
uv sync

uv pip install -e .
```

### 3. Environment Setup

#### Database Configuration

The BetterTherapy subnet uses two databases:

1. **Main Database**: SQLite database (`bettertherapy.db`) for storing requests and miner responses
2. **Dataset Database**: External database for storing base prompt-response pairs used for evaluation

#### Required Environment Variables

Set the `DATASET_DB_URL` either in `.env` or environment variable to point to your dataset database:

```bash
export DATASET_DB_URL="postgresql://username:password@localhost:5432/dataset_db"

```

**Note**: The `DATASET_DB_URL` environment variable is **required** and must be set before running the validator. The application will fail to start if this variable is not configured.

#### Database Schema

The dataset database should contain a `base_prompt_response` table with the following schema:

```sql
CREATE TABLE base_prompt_response (
    id INTEGER PRIMARY KEY,
    prompt VARCHAR NOT NULL,
    response VARCHAR NOT NULL,
    is_used BOOLEAN DEFAULT FALSE
);
```

### 4. Run Migrations

```bash
uv run alembic upgrade head
```

---

## Running the Subnet

### 1. Setting Up Wallets

You need wallets for the subnet owner, miner, and validator:

```bash
btcli wallet new_coldkey --wallet.name owner
btcli wallet new_coldkey --wallet.name miner
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default
btcli wallet new_coldkey --wallet.name validator
btcli wallet new_hotkey --wallet.name validator --wallet.hotkey default
```

---

## Hardware Requirements (Validator & Miner)

### Overview

This document outlines the hardware requirements for running Meta's Llama 3.1 8B Instruct model locally.

### Minimum Requirements

- **CPU**: 4 cores at 2.5 GHz
- **GPU**:
  - **Full precision (FP16)**: 16GB VRAM minimum
  - **4-bit quantized**: 6GB VRAM minimum
  - **Examples**: RTX 3060 12GB, RTX 4060 Ti 16GB
- **Memory**: 16GB RAM
- **Storage**: 20GB free space
- **OS**: Ubuntu 20.04 or later, Windows 10/11, MacOS

### Recommended Requirements

- **CPU**: 8 cores at 3.5 GHz
- **GPU**:
  - **For fast inference**: 16-24GB VRAM
  - **For fine-tuning (4-bit)**: 15GB+ VRAM
  - **Examples**: RTX 3090 (24GB), RTX 4070 Ti (16GB), RTX 4080 (16GB)
- **Memory**: 32GB RAM
- **Storage**: 50GB SSD
- **OS**: Ubuntu 22.04 or later

### VRAM Requirements by Precision

| Precision   | VRAM Required | Use Case                               |
| ----------- | ------------- | -------------------------------------- |
| FP16 (Half) | 16GB          | Full model performance                 |
| INT8        | ~10-12GB      | Good balance of performance and memory |
| 4-bit (Q4)  | 6-8GB         | Budget-friendly, slight quality loss   |

### Compatible GPUs

#### Budget Options (4-bit quantization)

- NVIDIA RTX 3060 (12GB)
- NVIDIA RTX 3070 (8GB)
- NVIDIA RTX 4060 Ti (8GB/16GB)

#### Recommended GPUs

- NVIDIA RTX 3090 (24GB)
- NVIDIA RTX 4070 Ti (16GB)
- NVIDIA RTX 4080 (16GB)
- NVIDIA RTX 4090 (24GB)

#### Professional GPUs

- NVIDIA A100 (40GB/80GB)
- NVIDIA A6000 (48GB)
- NVIDIA H100 (80GB)

### Software Requirements

- **CUDA**: Version 11.8 or higher
- **CUDA Compute Capability**: 6.0 minimum (GTX 10 series and newer)
- **Python**: 3.8 or higher
- **PyTorch**: 2.0 or higher with CUDA support

### Performance Notes

- Any 24GB GPU can run Llama 3.1 8B Q4K_M quantized models with 128k context
- 4-bit quantization reduces VRAM requirements to ~5-6GB for basic inference
- For Ollama deployment: 8+ CPU cores and 8GB+ VRAM recommended
- Consumer GPUs (RTX 30/40 series) provide excellent performance for this model size

### Context Length Considerations

| Quantization | VRAM | Max Context Length |
| ------------ | ---- | ------------------ |
| Q4K_M        | 24GB | 128k tokens        |
| Q8           | 24GB | 64k tokens         |
| FP16         | 24GB | 32k tokens         |

## Note

For Gated models in hugging face

```bash
huggingface-cli login
```

## Use Cloud Model (Miner)

1.  Miner can modify code to use api key models lik openai, claude, etc.

---

## Running the Miner

To start a miner (after setting up wallets and registering on your subnet):

```bash
uv run python neurons/miner.py \
  --netuid <your_netuid> \
  --subtensor.network <network> \
  --wallet.name miner \
  --wallet.hotkey default \
  --logging.debug
```

- Replace `<your_netuid>` with your subnet ID (e.g., `354` (test) and `10).
- Replace `<network>` with your chain endpoint (e.g., `test` for local, or use `finney` for mainnet).

---

## Running the Validator

### Weights & Biases (wandb) Setup

This project uses [Weights & Biases (wandb)](https://wandb.ai/) for experiment tracking and visualization.

1. **Sign up at [wandb.ai](https://wandb.ai/) and get your API key.**
2. **Login in your terminal:**
   ```bash
   wandb login
   ```
3. **(Optional) Set environment variables for headless runs:**
   ```bash
   export WANDB_API_KEY=your_api_key_here
   ```

## Discord Notifications

The subnet supports Discord webhook notifications for:

- LLM judge failures
- Dataset count warnings (when available dataset items fall below 60)

### Setup Discord Webhook

1. **Create a Discord Webhook:**

   - Go to your Discord server settings
   - Navigate to Integrations → Webhooks
   - Create a new webhook and copy the webhook URL

2. **Configure the Webhook URL:**

   **Option 1: Environment Variable**

   ```bash
   export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/YOUR_WEBHOOK_URL"
   ```

   **Option 2: .env File**
   Create or update your `.env` file:

   ```
   DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/YOUR_WEBHOOK_URL"
   ```

   **Option 3: Command Line Argument**

   ```bash
   uv run neurons/validator.py \
     --discord.webhook "https://discord.com/api/webhooks/YOUR_WEBHOOK_URL" \
     --netuid 354 --subtensor.network test \
     --wallet.name validator --wallet.hotkey default
   ```

### Start Validator Locally

To start a validator (after setting up wallets, registering on your subnet, and configuring the `DATASET_DB_URL`):

The validator will automatically create and manage runs, groups, and dashboards in wandb. See `BetterTherapy/utils/wandb.py` for advanced usage.

```bash
uv run neurons/validator.py \
  --netuid <your_netuid> \
  --subtensor.chain_endpoint <endpoint> \
  --wallet.name validator \
  --wallet.hotkey default \
  --logging.debug --model.name meta-llama/Llama-3.1-8B-Instruct
```

### Running with PM2 (Process Manager)

For production deployments, you can use PM2 to manage the validator process. Make sure to set the `DATASET_DB_URL` environment variable in your PM2 configuration:

```bash
# Set environment variable for PM2
export DATASET_DB_URL="your_dataset_database_url_here"

pm2 start uv --name bt-test-vali \
  -- run neurons/validator.py \
  --netuid <your_netuid> \
  --subtensor.chain_endpoint <endpoint> \
  --wallet.name bt-test-vali --wallet.hotkey default \
  --logging.debug --axon.port 8091 --model.name meta-llama/Llama-3.1-8B-Instruct --model.vram_in_GiB 16 --model.cpu_in_GiB 24
```

PM2 commands for process management:

```bash
# View running processes
pm2 list

# View logs
pm2 logs bt-test-vali

# Stop the validator
pm2 stop bt-test-vali

# Restart the validator
pm2 restart bt-test-vali

# Delete the process
pm2 delete bt-test-vali
```

- Replace `<your_netuid>` with your subnet ID (e.g., `354` (`test`) and `102`(`finney`)).
- Replace `<network>` with your chain endpoint (e.g., `test` for local, or use `finney` for mainnet).

The validator will automatically log evaluation metrics and charts to wandb.

---

## Auto-Updater

The BetterTherapy subnet includes an auto-updater script that automatically checks for updates, pulls the latest changes, updates dependencies, and restarts your application. This is particularly useful for production deployments.

### Using the Auto-Updater

The auto-updater script is located at `scripts/autoupdater.py` and provides the following features:

- Checks for updates from the remote repository
- Pulls the latest changes from the specified branch
- Updates Python dependencies using `uv`
- Optionally restarts your application using a custom command

### Basic Usage

```bash
# Basic update check and pull
python scripts/autoupdater.py

# Specify custom repository path and branch
python scripts/autoupdater.py --repo-path /path/to/repo --branch main

# Auto-restart PM2 process after update
python scripts/autoupdater.py --restart-command "pm2 restart bt-test-vali"

# Force update even if not on default branch
python scripts/autoupdater.py --force
```

### Command Line Options

- `--repo-path`: Path to the repository (default: current directory)
- `--branch`: Branch to pull from (default: main)
- `--restart-command`: Command to restart the application (e.g., PM2 restart command)
- `--force`: Force update even if not on the default branch

### Automated Updates with Cron

You can set up automated updates using cron jobs:

```bash
# Edit crontab
crontab -e

# Add entry to check for updates every hour
0 * * * * cd /path/to/bittensor-subnet-template && python scripts/autoupdater.py --restart-command "pm2 restart bt-test-vali" >> /var/log/autoupdate.log 2>&1
```

The auto-updater logs all activities to `autoupdate.log` for monitoring and debugging.

---

## Local Development

- Use `uv` for all dependency management.
- All dependencies are listed in `pyproject.toml`.

---

## Additional Resources

- [docs/running_on_staging.md](docs/running_on_staging.md) — Local chain setup
- [docs/running_on_testnet.md](docs/running_on_testnet.md) — Testnet setup
- [docs/running_on_mainnet.md](docs/running_on_mainnet.md) — Mainnet setup

---
