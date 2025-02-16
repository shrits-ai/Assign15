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
from transformers import TrainerCallback

# Configuration - Match exactly with your model specs
CHECKPOINT_DIR = "./checkpoints"
SEQ_LENGTH = 256  # Reduced from 2048 due to memory constraints
BATCH_SIZE = 4    # Adjust based on available memory
GRAD_ACCUM_STEPS = 8  # Effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS

# Initialize accelerator
accelerator = Accelerator(
    mixed_precision="no",  # MPS doesn't support FP16
    gradient_accumulation_steps=GRAD_ACCUM_STEPS
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
tokenizer.pad_token = tokenizer.eos_token  # For padding

# Model configuration (from your specs)
config = CustomConfig()
# Align config with tokenizer's special tokens
config.eos_token_id = tokenizer.eos_token_id
config.pad_token_id = tokenizer.pad_token_id
config.bos_token_id = tokenizer.bos_token_id


# Initialize model
model = CustomLLM(config)
device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params/1e6:.2f}M")
print(model)
model.to(device)
# Dataset setup (streaming)
class StreamDataset(IterableDataset):
    def __init__(self, split=None, dataset=None):
        if dataset is not None:
            self.dataset = dataset
        else:
            # Load dataset with correct configuration
            self.dataset = load_dataset(
                "HuggingFaceTB/smollm-corpus",
                name="cosmopedia-v2",  # Explicitly specify dataset name
                split=split,
                streaming=True
            ).map(
                tokenize_fn,
                batched=True
            )
    
    def __iter__(self):
        for sample in self.dataset:
            yield sample  # Now yields tokenized data

    def take(self, n):
        return StreamDataset(dataset=self.dataset.take(n))


# Tokenization function
def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        max_length=SEQ_LENGTH,
        truncation=True,
        padding="max_length"
    )

# Data collator (handles padding and attention masks)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal language modeling
)

# Training arguments
training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    max_steps=10000,
    logging_steps=100,
    save_steps=1000,  # Increased from 500 for stability
    evaluation_strategy="steps",
    eval_steps=1000,  # Evaluate more frequently
    learning_rate=2e-5,
    lr_scheduler_type="cosine",  # smoother decay
    warmup_steps=2000,  # Extended warmup
    weight_decay=0.01,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    max_grad_norm=1.0,  # Prevent exploding gradients
    fp16=False,
    remove_unused_columns=True,
    report_to="wandb",
    save_total_limit=3,  # Keep latest 3 checkpoints
)
class CustomLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # This will add the step number to the logs.
        if logs is not None:
            logs['step'] = state.global_step
            #print(f"Step {state.global_step}: {logs}")  # Print it to console for tracking

        # Log the information to WandB if you're using it
        wandb.log(logs, step=state.global_step)

# Custom callback for MPS-specific handling
class MPSCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        torch.mps.empty_cache()

# the Trainer initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=StreamDataset("train"),
    eval_dataset=StreamDataset("train").take(100),
    data_collator=data_collator,
    tokenizer=tokenizer,
    callbacks=[
        MPSCallback(),
        CustomLoggingCallback(),  # Add custom logging callback
    ]
)

# Check if checkpoint exists
checkpoint_path = f"{CHECKPOINT_DIR}/checkpoint-10000"
if os.path.exists(checkpoint_path):
    print(f"Found checkpoint at {checkpoint_path}, resuming training... logging interval changed to 10")
    # Temporarily set max_steps to 50 for the resumed training
    trainer.args.max_steps = 10000
    trainer.args.logging_steps=100
    trainer.train(resume_from_checkpoint=checkpoint_path)
    trainer.save_model(f"{CHECKPOINT_DIR}/final")
else:
    print(f"No checkpoint found, starting training from scratch...")
    trainer.train()
    trainer.save_model(f"{CHECKPOINT_DIR}/final_10000")

accelerator.print("Training complete!")
