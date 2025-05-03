<div align="center">

# BetterTherapy Subnet: AI Doctor Twin Network <!-- omit in toc -->

---

## The Incentivized Healthcare Internet <!-- omit in toc -->

[Discord](https://discord.gg/bittensor) • [Network](https://taostats.io/) • [Research](https://bittensor.com/whitepaper)
</div>

---

- [Quickstarter Template](#quickstarter-template)
- [Introduction](#introduction)
  - [Use Case](#use-case)
- [Installation](#installation)
  - [Before You Proceed](#before-you-proceed)
  - [Install](#install)
- [Miner Role (AI Doctor Twin Engine)](#miner-role-ai-doctor-twin-engine)
- [Validator Role (Medical Evaluation System)](#validator-role-medical-evaluation-system)
- [Doctor AI Flow](#doctor-ai-flow)
- [Subnet Links](#subnet-links)
- [License](#license)

---

## Quickstarter Template

This repo provides a complete setup to:

- Launch a Bittensor subnet for healthcare AI.
- Support personalized AI twins for licensed medical experts.
- Continuously fine-tune these agents with real-time user feedback and expert input.
- Reward miners for compute, and validators for quality control.

---

## Introduction

**BetterTherapy Subnet** is a decentralized protocol for deploying personalized **AI Doctor Twins**. Each twin represents a real-world, licensed expert (e.g., Dr. Indra Gurung), fine-tuned with domain-specific expertise to provide high-quality health insights to users.

This subnet uses Bittensor’s infrastructure to incentivize the best-performing agents, validators, and contributors with TAO.

### Use Case

> Train & deploy AI Doctor Twins for trusted medical consultations at scale.

#### Example Workflow:
1. **Dr. Indra Gurung** is onboarded to the subnet.
2. A base LLM (pretrained on medical exams) is fine-tuned with Dr. Indra’s expertise daily.
3. Users select “Dr. Indra AI” and ask personalized health questions.
4. Validators evaluate the miner responses for clarity, medical accuracy, and user satisfaction.
5. Top miners receive rewards for fine-tuning and inference.
6. Ratings and usage are used to improve each twin continuously.

---

## Installation

### Before You Proceed

You’ll need:
- A wallet and TAO balance (to register subnet or join as miner/validator).
- System resources meeting [`min_compute.yml`](./min_compute.yml) requirements.
- CUDA-compatible GPU (for model inference).

### Install

- 🛠 **Local Dev Setup**: [Running Subnet Locally](./docs/running_on_staging.md)
- 🧪 **Testnet Deployment**: [Running on the Test Network](./docs/running_on_testnet.md)
- 🚀 **Mainnet Launch**: [Running on the Main Network](./docs/running_on_mainnet.md)

---

## Miner Role (AI Doctor Twin Engine)

Miners host and fine-tune the **Doctor AI Twins**, based on real doctors’ daily interaction with real patients.

### Responsibilities:
- Serve AI responses via fine-tuned LLMs (e.g. Med-BERT, Llama3-Med).
- Fine-tune the model daily using data (interaction between doctors & patients) provided by real doctors.
- Handle real-time user queries routed to specific doctor.
- Perform inference using:
  - Chat history analysis
  - User feedback
  - Accuracy vs. latency optimization
  - Select and submit the **best response** for each session.

---

### 🔁 Inference & Feedback Flow:

1. User asks query** → Routed to **Dr. Indra (AI)**  
2. Miner performs inference** → Analyzes chat context and response candidates  
3. Best response** is selected and served to the user  
4. User rates the session
5. Validator reviews session logs, distributes fine-tuning tasks  
6. **Top 100 miners** are rewarded based on session quality and ratings

 
### Reward Criteria:
- Quality of response (medical correctness)
- Speed and efficiency of inference
- User satisfaction scores
- Participation in doctor-led fine-tuning sessions

> Miners contribute not only computing power, but also help build each AI doctor twin's memory, accuracy, and trust.

---

## Validator Role (Medical Evaluation System)

Validators act as **medical-grade QA systems** and reward miners.

### Responsibilities:
- Route user queries to miners running a selected AI doctor twin.
- Verify and compare multiple responses using:
  - Medical resource lookup (e.g., WebMD, UpToDate, Mayo Clinic)
  - Accuracy checks using APIs
  - Session rating by users
- Assign scores and rank miners per batch
- Distribute reward and fine-tuning tasks to miners based on performance

### Validator Reward Criteria:
- Fair scoring mechanisms
- Validation accuracy (cross-referenced with gold answers, APIs)
- Task distribution quality to miners

```
graph TD
    A[User selects Dr. Indra AI] -->|1| B[User submits health query]
    B -->|2| C[Validator receives query]
    C -->|3| D[Query distributed to multiple Miners]
    D -->|4| E[Miners generate responses using AI Twin (Dr. Indra)]
    E -->|5| F[Validator evaluates and ranks responses]
    F -->|6| G[Best response delivered to user]
    G -->|7| H[User provides session rating]
    H -->|8| I[Validator assigns rewards based on rating]
    I -->|9| J[Top 100 Miners receive TAO tokens + fine-tuning task]

```


# Subnet Links
To see real-world examples of subnets in action, see the `subnet_links.py` document or access them from inside the `template` package by:
```python
import template
template.SUBNET_LINKS
[{'name': 'sn0', 'url': ''},
 {'name': 'sn1', 'url': 'https://github.com/opentensor/prompting/'},
 {'name': 'sn2', 'url': 'https://github.com/bittranslateio/bittranslate/'},
 {'name': 'sn3', 'url': 'https://github.com/gitphantomman/scraping_subnet/'},
 {'name': 'sn4', 'url': 'https://github.com/manifold-inc/targon/'},
...
]
```

## License
This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2024 Opentensor Foundation

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
```
