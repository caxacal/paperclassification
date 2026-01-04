import os
import torch
from transformers import AutoModelForSequenceClassification

MODEL_ID = "caxacal/article-classifier-model"
OUTPUT_DIR = "/data/quantized-model"

HF_TOKEN = os.environ.get("HF_TOKEN")

print("Loading full-precision model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    token=HF_TOKEN
)

model.eval()

print("Applying dynamic INT8 quantization...")
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

print("Saving quantized model...")
quantized_model.save_pretrained(OUTPUT_DIR)

print("✅ Quantized model saved to:", OUTPUT_DIR)
