import argparse
from pathlib import Path

import torch
# torch.multiprocessing.set_start_method('spawn', force=True)
from datasets import load_dataset
from transformers import EsmConfig, EsmForMaskedLM, EsmTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling


parser = argparse.ArgumentParser(description="Train ESM2-t6 model from scratch")
parser.add_argument("--data-file", type=str, required=True, help="Path to the training data file")
parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save the model")
parser.add_argument("--model-name", type=str, required=True, help="Name of the model to train")
parser.add_argument("--num-train-epochs", type=int, default=1, help="Number of training epochs")
parser.add_argument("--per-device-train-batch-size", type=int, default=16, help="Batch size per device during training")
args = parser.parse_args()

# 1. Define the model configuration for ESM2-t6
# The values are based on the original ESM2-t6_8M_UR50D model
# ESM2 used 2M tokens per batch. With 32 (bs) * 1024 (tokens) per batch, we need to run 305,175.78 batches
config = EsmConfig(
    vocab_size=33,  # 20 amino acids + special tokens
    num_hidden_layers=6,
    hidden_size=320,
    num_attention_heads=20,
    pad_token_id=0,
    max_position_embeddings=1024,
    intermediate_size=1280,
)

# 2. Initialize the tokenizer (can use a pre-existing one or build from your data)
# For a true scratch pretraining, you would build your own vocab
tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

# 3. Initialize the model from the config
model = EsmForMaskedLM(config).to("cuda")
print("The model has", sum(p.numel() for p in model.parameters() if p.requires_grad), "trainable parameters.")

# 4. Prepare a dummy dataset for demonstration
# In a real-world scenario, this would be a massive protein sequence dataset
raw_datasets = load_dataset("text", data_files={"train": [args.data_file]})

# 5. Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"])  # , truncation=True, padding="max_length", max_length=1022)

tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=False,
    num_proc=4, # Use multiple processes for faster tokenization
    remove_columns=["text"],
)

# 6. Use the DataCollator for Masked Language Modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

# 7. Define training arguments
training_args = TrainingArguments(
    output_dir=args.output_dir / args.model_name,
    overwrite_output_dir=True,
    save_strategy="epoch",
    save_only_model=True,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    # gradient_accumulation_steps=8,
    # gradient_checkpointing=True,
    learning_rate=4e-4,
    prediction_loss_only=True,
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=8,
    dataloader_pin_memory=True,
)

# 8. Create the Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    tokenizer=tokenizer,
    train_dataset=tokenized_datasets["train"],
)

# 9. Start training!
trainer.train()

# 10. Save the final model
trainer.save_model(args.output_dir / args.model_name / "esm2_t6_final_model")
