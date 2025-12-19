from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
import torch
import math
from tqdm import tqdm
from torch.utils.data import DataLoader

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Llama-3.2-1B with LoRA on TinyStories dataset.")
    
    parser.add_argument("--hf_token", type=str, default="", help="Hugging Face token for model access.")
    parser.add_argument("--lora_path", type=str, default="./ex/qora-final", help="LoRA Path")

    return parser.parse_args()

args = parse_args()

# 1. Load Model & Tokenizer
model_id = "meta-llama/Llama-3.2-1B"
lora_path = args.lora_path
hf_token = args.hf_token
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

print(lora_path.split('/')[-1])

# 2. Load LoRA Model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    token=hf_token,
)

model = PeftModel.from_pretrained(model, lora_path) 
model = model.merge_and_unload() 
model.eval()

# 3. Dataset
ds_name = "maveriq/tinystoriesv2_gpt4"
ds = load_dataset(ds_name)
test_ds = ds["valid"]

def tokenize_fn(examples):
    out = tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
    out["labels"] = out["input_ids"].copy()
    return out

tokenized_test = test_ds.map(tokenize_fn, batched=True, remove_columns=test_ds.column_names)

# 4. DataLoader
batch_size = 8

tokenized_test.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

test_loader = DataLoader(
    tokenized_test,
    batch_size=batch_size,
    shuffle=False
)

# 5. Perplexity

def compute_perplexity(model, tokenized_dataset, batch_size=16):
    model.eval()
    total_loss = 0
    total_tokens = 0
    device = model.device

    for i in tqdm(range(0, len(tokenized_dataset), batch_size), desc="Computing Perplexity"):
        batch = tokenized_dataset[i : i + batch_size]
        input_ids = torch.tensor(batch["input_ids"]).to(device)
        labels = torch.tensor(batch["labels"]).to(device)
        
        labels[labels == tokenizer.pad_token_id] = -100

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            valid_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


ppl = compute_perplexity(model, tokenized_test)
print(f"Test Set Perplexity: {ppl}")