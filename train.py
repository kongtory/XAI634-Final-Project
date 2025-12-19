from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from transformers.integrations import TensorBoardCallback
from peft import LoraConfig, get_peft_model
import torch

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Llama-3.2-1B with LoRA on TinyStories dataset.")
    parser.add_argument("--hf_token", type=str, default="", help="Hugging Face token for model access.")
    
    # Hyper Parameter
    parser.add_argument("--lora_alpha", type=int, default=8, help="LoRA alpha parameter.")
    parser.add_argument("--lora_dropout", type=float, default=0.00, help="LoRA dropout rate.")
    parser.add_argument("--target_modules", type=str, nargs="+", default="qv", help="Target modules for LoRA.")

    # Training
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory for model and logs.")
    parser.add_argument("--num_train_epochs", type=float, default=0.02, help="Number of training epochs.")
    
    # Save path
    parser.add_argument("--save_model_path", type=str, default="./ex/qora-base", help="Path to save the final model.")

    return parser.parse_args()

args = parse_args()

model_id = "meta-llama/Llama-3.2-1B"
hf_token = args.hf_token

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    use_fast=False,
    token=hf_token,
)
tokenizer.pad_token = tokenizer.eos_token

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

if args.target_modules == "qv":
    target_modules = ["q_proj", "v_proj"]
else:
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
lora_config = LoraConfig(
    r=8,
    lora_alpha=args.lora_alpha,
    target_modules=target_modules
    lora_dropout=args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

train_dataset = load_dataset("maveriq/tinystoriesv2_gpt4", split="train")
valid_dataset = load_dataset("maveriq/tinystoriesv2_gpt4", split="valid")

def tokenize_function(examples):
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

train_ds = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names,
)
valid_ds = valid_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=valid_dataset.column_names,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=args.num_train_epochs,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    eval_steps=50,
    do_eval=True,
    save_steps=20000,
    logging_steps=50,
    bf16=True,
    seed=42,
    report_to="tensorboard",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    data_collator=data_collator,
    callbacks=[TensorBoardCallback()],
)

trainer.train()

trainer.save_model(args.save_model_path)
tokenizer.save_pretrained(args.save_model_path)