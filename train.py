import torch
import os
from torch.utils.data import IterableDataset
from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from datasets import load_dataset
import wandb
from model import CustomConfig, CustomLLM

# Configuration - Match exactly with your model specs
CHECKPOINT_DIR = "./checkpoints"
SEQ_LENGTH = 512
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 8

# Initialize accelerator
accelerator = Accelerator(
    mixed_precision="no",
    gradient_accumulation_steps=GRAD_ACCUM_STEPS
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
tokenizer.pad_token = tokenizer.eos_token

# Model configuration with MoE fixes
config = CustomConfig()
config.eos_token_id = tokenizer.eos_token_id
config.pad_token_id = tokenizer.pad_token_id
config.bos_token_id = tokenizer.bos_token_id
config.moe_train_capacity_factor = 2.0  # Critical MoE fix
config.moe_eval_capacity_factor = 2.0   # Prevent expert overflow

# Initialize model with MPS optimizations
model = CustomLLM(config)
print(model)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
model.to(device)

# MPS-specific initializations
if device == "mps":
    torch.mps.set_per_process_memory_fraction(0.90)
    # Force float32 for stability
    for p in model.parameters():
        p.data = p.data.to(torch.float32)

# Dataset setup with proper streaming
class StreamDataset(IterableDataset):
    def __init__(self, split=None, dataset=None):
        self.dataset = dataset if dataset else load_dataset(
            "HuggingFaceTB/smollm-corpus",
            name="cosmopedia-v2",
            split=split,
            streaming=True
        ).shuffle(seed=42, buffer_size=10000).map(
            self.tokenize_fn,
            batched=True
        )
    
    def tokenize_fn(self, examples):
        return tokenizer(
            examples["text"],
            max_length=SEQ_LENGTH,
            truncation=True,
            padding=False,  # Dynamic padding
            return_attention_mask=True
        )
    
    def __iter__(self):
        for sample in self.dataset:
            yield sample

    def take(self, n):
        return StreamDataset(dataset=self.dataset.take(n))

# Data collator with MPS fixes
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=64,
    return_tensors="pt"  # Critical for MPS
)

# Training arguments with stabilization
training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    max_steps=10000,
    logging_steps=100,
    save_steps=500,
    evaluation_strategy="no",
    learning_rate=3e-6,  # Adjusted learning rate
    lr_scheduler_type="cosine",  # Changed to cosine decay
    warmup_steps=1000,
    weight_decay=0.15,
    optim="adamw_torch_fused",
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    max_grad_norm=1.0,  # Tighter gradient clipping
    fp16=False,
    remove_unused_columns=True,
    report_to="wandb",
    save_total_limit=3,
    use_mps_device=True,
    dataloader_num_workers=0,
    dataloader_pin_memory=False
)

# Debugging callback
class TrainingDebugCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # Log smooth loss
        wandb.log({ "steps":state.global_step,
            "loss_ema": trainer.model.loss_ema.item(),
                })

        # Empty MPS cache regularly
        torch.mps.empty_cache()

# Initialize trainer with diagnostics
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=StreamDataset("train"),
    data_collator=data_collator,
    tokenizer=tokenizer,
    callbacks=[
        TrainingDebugCallback(),
    ]
)

# Checkpoint handling
checkpoint_path = f"{CHECKPOINT_DIR}/checkpoint-5000"
if os.path.exists(checkpoint_path):
    print(f"Resuming from {checkpoint_path}")
    trainer.train(resume_from_checkpoint=checkpoint_path)
else:
    print("Starting fresh training")
    trainer.train()

trainer.save_model(f"{CHECKPOINT_DIR}/final")
accelerator.print("Training complete!")