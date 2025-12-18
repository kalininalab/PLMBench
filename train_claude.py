import argparse
from pathlib import Path
import os

import torch
from datasets import load_dataset
from transformers import EsmConfig, EsmForMaskedLM, EsmTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling


parser = argparse.ArgumentParser(description="Train ESM2-t6 model from scratch")
parser.add_argument("--data-file", type=str, required=True, help="Path to the training data file")
parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save the model")
parser.add_argument("--model-name", type=str, required=True, help="Name of the model to train")
parser.add_argument("--max-steps", type=int, default=200_000, help="Number of steps/updates in training")
parser.add_argument("--per-device-train-batch-size", type=int, default=32, help="Batch size per device during training")
args = parser.parse_args()

# Set environment variables for better performance
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# 1. Define the model configuration for ESM2-t6
config = EsmConfig(
    vocab_size=33,  # 20 amino acids + special tokens
    num_hidden_layers=30,
    hidden_size=640,
    num_attention_heads=20,
    pad_token_id=0,
    max_position_embeddings=1024,
    intermediate_size=1280,
)

# 2. Initialize the tokenizer
# tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")

# 3. Initialize the model from the config
model = EsmForMaskedLM(config).to("cuda")
print("The model has", sum(p.numel() for p in model.parameters() if p.requires_grad), "trainable parameters.")

# 4. Load dataset with optimized settings
raw_datasets = load_dataset(
    "text", 
    data_files={"train": [args.data_file]},
)

# 5. Optimized tokenization function
def tokenize_function(examples):
    # Process in batches and handle padding dynamically
    return tokenizer(
        examples["text"], 
        truncation=True, 
        max_length=1022,  # Leave room for special tokens
        padding=False,    # Dynamic padding handled by data collator
        return_special_tokens_mask=True
    )

# Tokenize with batching enabled
tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,           # CRITICAL: Process in batches
    num_proc=8,             # Increased parallel processes
    remove_columns=["text"],
)

# 6. Optimized data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
    pad_to_multiple_of=8,   # Optimize for tensor cores
)

# 7. Optimized training arguments
training_args = TrainingArguments(
    output_dir=args.output_dir / args.model_name,
    overwrite_output_dir=True,

    save_strategy="epoch",
    save_only_model=True,
    logging_steps=500,
    report_to=None,                     # Disable wandb/tensorboard logging
    
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=8,      # Effective batch size: 128 * 8 = 1024
    gradient_checkpointing=True,        # Save memory, slight speed trade-off
    
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_epsilon=1e-8,
    weight_decay=0.01,
    
    max_steps=args.max_steps,
    learning_rate=4e-5,
    warmup_steps=2_500,

    prediction_loss_only=True,
    # fp16=True,                          # Enable mixed precision
    dataloader_num_workers=8,           # Reduced to avoid overhead
    dataloader_pin_memory=True,
    dataloader_persistent_workers=True, # Keep workers alive between epochs
    remove_unused_columns=False,
    group_by_length=True,               # Group similar length sequences
    length_column_name="length",
    max_grad_norm=1.0,                  # Gradient clipping
)

# 8. Add sequence length column for grouping
# def add_length(examples):
#     examples["length"] = [len(input_ids) for input_ids in examples["input_ids"]]
#     return examples

# tokenized_datasets = tokenized_datasets.map(add_length, batched=True)

# 9. Create the Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    tokenizer=tokenizer,
    train_dataset=tokenized_datasets["train"],
)

# 10. Compile model for better performance (PyTorch 2.0+)
# if hasattr(torch, 'compile'):
#     model = torch.compile(model)
#     print("Model compiled for better performance.")

# 11. Start training!
print("Starting training...")
trainer.train()

# 12. Save the final model
print("Saving model...")
trainer.save_model(args.output_dir / args.model_name / "esm2_t6_final_model")
