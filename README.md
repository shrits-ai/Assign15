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
