import asyncio
import time
# import subprocess
# from datasets import Dataset
# import torch
# from transformers import (
#     AutoModelForSequenceClassification,
#     AutoTokenizer,
#     Trainer,
#     TrainingArguments
# )
# from neurons.miner import Miner
import bettertherapy
from bettertherapy.protocol import BetterTherapySynapse
from bettertherapy.miner.inference import TherapyResponseSchema, TherapyInference
import bittensor as bt

# miner_preds = {}
# inference_engine = TherapyInference()

# def get_gpu_info():
#     try:
#         result = subprocess.run(
#             ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
#             capture_output=True,
#             text=True
#         )
#         gpu_data = result.stdout.strip().split("\n")[0].split(", ")
#         return {
#             "type": "gpu",
#             "name": gpu_data[0],
#             "memory": f"{gpu_data[1]} MiB",
#             "compute_cap": gpu_data[2],
#             "compute_score": float(gpu_data[2]) * int(gpu_data[1])
#         }
#     except Exception:
#         return {"type": "cpu", "compute_score": 100}
    

# def run_benchmark(task_data):
#     model_name = task_data.get("model_name", "distilbert-base-uncased")
#     try:
#         model = AutoModelForSequenceClassification.from_pretrained(model_name)
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         model.to(device)
#         inputs = tokenizer("Sample text", return_tensors="pt").to(device)
#         start = time.time()
#         with torch.no_grad():
#             _ = model(**inputs)
#         return 1000 / max(time.time() - start, 0.01)
#     except Exception as e:
#         bt.logging.error(f"Benchmark failed: {e}")
#         return 100
    
#     # -------------------- Fine-Tuning --------------------

# def fine_tune_model(dataset, model_name, batch_size=8):
#     try:
#         model = AutoModelForSequenceClassification.from_pretrained(model_name)
#         tokenizer = AutoTokenizer.from_pretrained(model_name)

#         tokenized_dataset = [
#             {
#                 "input_ids": tokenizer(item["text"], truncation=True, padding="max_length", max_length=128)["input_ids"],
#                 "labels": item["label"]
#             }
#             for item in dataset
#         ]
#         hf_dataset = Dataset.from_list(tokenized_dataset)

#         training_args = TrainingArguments(
#             output_dir="./output",
#             num_train_epochs=1,
#             per_device_train_batch_size=batch_size,
#             logging_steps=10,
#             save_strategy="no"
#         )
#         trainer = Trainer(model=model, args=training_args, train_dataset=hf_dataset)
#         trainer.train()
#         model.save_pretrained("./output")

#         return {
#             "weights": "./output",
#             "loss": trainer.state.log_history[-1].get("loss", None)
#         }
#     except Exception as e:
#         bt.logging.error(f"Fine-tuning failed: {e}")
#         return {"error": str(e)}


# def get_overall_score(ai_response: TherapyResponseSchema, process_time: float):
#     """Compute overall score based on breakdown, speed, and user rating."""
#     if isinstance(ai_response, TherapyResponseSchema):
#         breakdown = ai_response.breakdown
#         user_rating = ai_response.user_rating
#     else:
#         return 0.0
#     weights = {
#         "relevance": 3,
#         "accuracy": 2,
#         "empathy": 2,
#         "clarity": 1.5,
#         "contextuality": 1.5
#     }
#     field_names = ["relevance", "accuracy", "empathy", "clarity", "contextuality"]
#     scores = {field: getattr(breakdown, field) for field in field_names}
#     breakdown_score = sum(float(scores[key]) * weights[key] for key in scores) / 10  # Normalize to 0-10
#     speed_score = min(10.0, 1.0 / (process_time + 0.01))
#     overall_score = (0.6 * breakdown_score) + (0.2 * speed_score) + (0.2 * user_rating)
#     return round(overall_score * 10, 2)  # Scale to 100

# async def forward(self: Miner, synapse: BetterTherapySynapse):
async def forward(synapse: BetterTherapySynapse, miner_preds: dict, inference_engine: TherapyInference):
    """
    Asynchronously process user queries using the fine-tuned medical model.
    Uses caching to avoid redundant inference.
    """
    bt.logging.info(f"Received mine requests for queries {synapse.messages}")

# if synapse.task_type == "therapy":
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
            tasks.append(inference_engine.generate_therapy_response(message["content"], synapse.chat_history))

    bt.logging.info("Running inference tasks...")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    process_time = time.time() - start_time
    for task_index, result in enumerate(results):
        i, query_id = query_ids[task_index]
        try:
            if isinstance(result, Exception):
                raise result
            predictions[i]={
                "response": result.response_text,
                "breakdown": result.breakdown,
                "process_time": process_time,
                "user_rating": result.user_rating #default 0
            }
            miner_preds[query_id]=predictions[i]
            bt.logging.info(f"Response for query {query_id}: {result.response_text}")
        except Exception as e:
            bt.logging.error(f"Error processing query {query_id}: {e}")
            predictions[i] = None

    synapse.output = predictions
    return synapse

# elif synapse.task_type == "benchmark":
#     gpu_info = get_gpu_info()
#     compute_score = run_benchmark(synapse.task_data)
#     synapse.gpu_info = gpu_info
#     synapse.response = {
#         "compute_score": compute_score,
#         "execution_time": time.time() - start_time
#     }
