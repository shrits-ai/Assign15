# CustomLLM Model Architecture

## Overview
The model consists of a custom Transformer architecture with enhanced attention mechanisms and Mixture of Experts (MoE). Below is a breakdown of the main components.

## Model Parameters
- **Total Parameters**: 973.12M

## Architecture Breakdown

### Embedding Layers
- **Token Embeddings**: `Embedding(49152, 768)`
- **Position Embeddings**: `Embedding(2048, 768)`
- **Dropout**: `Dropout(p=0.1, inplace=False)`

### Decoder Layers
- **Total Decoder Layers**: 30
  - Each layer consists of the following components:
    - **Self-Attention**: 
      - MultiHeadLatentAttention with multiple projection layers and rotary embeddings for better positional encoding.
      - Key and Value projections (`kv_proj_d`, `k_proj_u`, `v_proj_u`) for latent space compression.
      - Query projections (`q_proj_d`, `q_proj_u`) for dynamic queries with rotary embeddings (`rope_k`, `rope_q`).
    - **MLP (Mixture of Experts)**:
      - Uses **DeepSeekMoE** with multiple expert layers (`DeepSeekExpertLayer`).
      - **Router Layer**: A linear layer that determines which experts to route based on input.
      - Each **DeepSeekExpertLayer** contains:
        - **Gate Projection**: `Linear(in_features=768, out_features=1536, bias=False)`
        - **Up Projection**: `Linear(in_features=768, out_features=1536, bias=False)`
        - **Down Projection**: `Linear(in_features=1536, out_features=768, bias=False)`
        - **Activation Function**: `SiLU()`
    - **Normalization**:
      - **Input Normalization**: `CustomRMSNorm`
      - **Post-Attention Normalization**: `CustomRMSNorm`

### Output Layer
- **LM Head**: `Linear(in_features=768, out_features=49152, bias=True)` for generating the output logits.

## Key Features
- **Efficient Attention**: Uses MultiHeadLatentAttention with rotary embeddings for improved efficiency.
- **Mixture of Experts (MoE)**: Scalable with the ability to route input through multiple experts for more powerful processing.
- **Residual Connections and Layer Norm**: Each layer is equipped with residual connections and custom RMS normalization for stable training.

# Model Training

## Overview
This script trains a custom large language model (LLM) using PyTorch and the Hugging Face `transformers` library. It leverages mixed precision (for supported devices) and gradient accumulation for efficient training on limited hardware.

## Key Components

### Configuration
- **CHECKPOINT_DIR**: Directory to save model checkpoints.
- **SEQ_LENGTH**: Sequence length for tokenization, set to 256 due to memory constraints.
- **BATCH_SIZE**: Set to 4, adjustable based on available memory.
- **GRAD_ACCUM_STEPS**: Effective batch size is `BATCH_SIZE * GRAD_ACCUM_STEPS`.

### Accelerator Setup
Uses the `Accelerator` from the `accelerate` library to manage mixed precision and gradient accumulation. It handles device compatibility (CUDA, MPS, or CPU).

### Tokenizer
- The model uses a custom tokenizer: `HuggingFaceTB/cosmo2-tokenizer`.
- The tokenizer's special tokens are aligned with the model configuration.

### Model Initialization
- **CustomLLM** model is initialized with the custom configuration.
- Model is moved to the appropriate device (CUDA, MPS, or CPU).

### Dataset Setup
- The dataset is loaded and tokenized on-the-fly using the `StreamDataset` class, which supports streaming and memory-efficient tokenization.
- The dataset used is `HuggingFaceTB/smollm-corpus` with `cosmopedia-v2` configuration.

### Training
The model is trained using the `Trainer` API from Hugging Face:
- **TrainingArguments**: Configures training parameters like batch size, steps, logging frequency, learning rate, and more.
- **Callbacks**: Custom callbacks include `MPSCallback` for handling memory on MPS devices and `TextGenerationCallback` for generating text at the end of training.

### Checkpointing
- The script checks if a checkpoint exists and resumes training from there if available. If no checkpoint exists, training starts from scratch.

## Training Logs

### Example Output
```
Model parameters: 973.12M
CustomLLM(
  (token_embeddings): Embedding(49152, 768)
  (position_embeddings): Embedding(2048, 768)
  (dropout): Dropout(p=0.1, inplace=False)
  (decoder_layers): ModuleList(
    (0-29): 30 x DecoderLayer(
      (self_attn): MultiHeadLatentAttention(
        (kv_proj_d): Linear(in_features=768, out_features=192, bias=False)
        (q_proj_d): Linear(in_features=768, out_features=192, bias=False)
        (k_proj_u): Linear(in_features=192, out_features=384, bias=False)
        (v_proj_u): Linear(in_features=192, out_features=768, bias=False)
        (q_proj_u): Linear(in_features=192, out_features=384, bias=False)
        (rope_k): Linear(in_features=768, out_features=384, bias=False)
        (rope_q): Linear(in_features=192, out_features=384, bias=False)
        (o_proj): Linear(in_features=768, out_features=768, bias=False)
        (rotary_emb): RotaryEmbedding()
      )
      (mlp): LlamaMLP(
        (moe): DeepSeekMoE(
          (shared_experts): ModuleList(
            (0): DeepSeekExpertLayer(
              (gate_proj): Linear(in_features=768, out_features=1536, bias=False)
              (up_proj): Linear(in_features=768, out_features=1536, bias=False)
              (down_proj): Linear(in_features=1536, out_features=768, bias=False)
              (act_fn): SiLU()
            )
          )
          (routed_experts): ModuleList(
            (0-6): 7 x DeepSeekExpertLayer(
              (gate_proj): Linear(in_features=768, out_features=1536, bias=False)
              (up_proj): Linear(in_features=768, out_features=1536, bias=False)
              (down_proj): Linear(in_features=1536, out_features=768, bias=False)
              (act_fn): SiLU()
            )
          )
          (router): Linear(in_features=768, out_features=7, bias=False)
        )
      )
      (input_norm): CustomRMSNorm()
      (post_attn_norm): CustomRMSNorm()
    )
  )
  (lm_head): Linear(in_features=768, out_features=49152, bias=True)
)
```
### Training Log Space

```
```

### Generating Text Example:
After training ends, the model generates below text for each of the following prompts:
```
```


