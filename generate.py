import torch
import torch.nn as nn
from transformers import AutoTokenizer
from safetensors.torch import load_file
from model import CustomConfig, CustomLLM

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
tokenizer.pad_token = tokenizer.eos_token  # For padding
# Model configuration (from your specs)
config = CustomConfig()
# Align config with tokenizer's special tokens
config.eos_token_id = tokenizer.eos_token_id
config.pad_token_id = tokenizer.pad_token_id
config.bos_token_id = tokenizer.bos_token_id


# Set device for M1 Mac
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Load tokenizer
checkpoint_path = "checkpoints/checkpoint-10000"
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

# Initialize model
model = CustomLLM(config)

# Load weights from `safetensors`
weights_path = f"{checkpoint_path}/model.safetensors"
state_dict = load_file(weights_path)
model.load_state_dict(state_dict)

# Set device correctly for Apple Silicon Macs
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


def generate_text(prompt, max_length=50, temperature=0.8, top_k=50, top_p=0.9):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Ensure model is on the right device
    model.to(device)
    model.eval()  

    generated_ids = input_ids

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(generated_ids)
            logits = outputs[0][:, -1, :]  # Extract the last token logits

        # Apply temperature scaling
        logits = logits / temperature  

        # Apply Top-K & Top-P sampling
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # Top-K filtering
        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(probs, top_k)
            top_k_probs = torch.nn.functional.softmax(top_k_logits, dim=-1)
            next_token_id = torch.multinomial(top_k_probs, 1)
            next_token_id = top_k_indices.gather(-1, next_token_id)
        else:
            next_token_id = torch.multinomial(probs, 1)

        # Move token to correct device
        next_token_id = next_token_id.to(device)

        # Stop if we hit an end-of-sequence token
        if next_token_id.item() == tokenizer.eos_token_id:
            break

        # Append next token to generated sequence
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

        # **Avoid repetition by penalizing previously seen tokens**
        if next_token_id in generated_ids[:, -5:]:  # Check last 5 tokens for repetition
            logits[:, next_token_id] -= 1.5  # Penalize repeated tokens

    return tokenizer.decode(generated_ids[0].cpu(), skip_special_tokens=True)



# Generate text for prompts
prompts = [
    "Once upon a time, in a land far away",
    "The future of AI will be shaped by",
    "In the middle of the night, I heard a strange noise",
    "The scientist looked at the data and realized",
    "A journey across the ocean began when"
]

for prompt in prompts:
    print(f"Prompt: {prompt}")
    print(f"Generated Text: {generate_text(prompt, max_length=50)}\n")
