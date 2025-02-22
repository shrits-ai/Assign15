import torch
import torch.nn as nn
import math
from transformers.modeling_outputs import CausalLMOutputWithPast

# 1. Custom Configuration Class (Remains Unchanged)
class CustomConfig:
    def __init__(self):
        # Architecture Parameters
        self.vocab_size = 49152
        self.hidden_size = 768          # d_model
        self.intermediate_size = 1536   # FFN dimension
        self.num_hidden_layers = 30     # Number of decoder layers
        self.num_attention_heads = 9    # Query heads
        self.num_key_value_heads = 3   # Key/Value heads
        self.max_position_embeddings = 2048
        self.rms_norm_eps = 1e-5
        self.rope_theta = 10000.0       # Rotary embedding base
        self.compression_ratio = 4      # Compression ratio for MLHA
        self.num_experts = 8           # Number of experts in MoE
        self.num_shared_experts = 1    # Number of shared experts
        self.top_k_experts = 2         # Top-k experts in MoE

        # Tokenizer/Generation Params
        self.pad_token_id = None
        self.bos_token_id = 0
        self.eos_token_id = 0
    
    def to_dict(self):
        # Serialize the config parameters
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

# 2. Custom RMS Normalization (Remains Unchanged)
class CustomRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

# 3. Rotary Positional Embeddings (Remains Unchanged)
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x, seq_len):
        if seq_len > self.cos_cached.shape[2]:
            self._set_cos_sin_cache(seq_len)
        return self.cos_cached[:, :, :seq_len], self.sin_cached[:, :, :seq_len]

# 4. Updated Attention Layer with MultiHeadLatentAttention
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config, compression_ratio=4):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.compression_ratio = compression_ratio
        self.latent_dim = self.hidden_size // compression_ratio

        # Projections
        self.kv_proj_d = nn.Linear(self.hidden_size, self.latent_dim, bias=False)
        self.q_proj_d = nn.Linear(self.hidden_size, self.latent_dim, bias=False)
        self.k_proj_u = nn.Linear(self.latent_dim, self.hidden_size//2, bias=False)
        self.v_proj_u = nn.Linear(self.latent_dim, self.hidden_size, bias=False)
        self.q_proj_u = nn.Linear(self.latent_dim, self.hidden_size//2, bias=False)
        # Add initialization for all linear layers
        for lin in [self.kv_proj_d, self.q_proj_d, self.k_proj_u, self.v_proj_u, self.q_proj_u]:
            nn.init.normal_(lin.weight, mean=0.0, std=0.02)  # Smaller initialization
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)
        # ROPE Components
        self.rope_k = nn.Linear(self.hidden_size, self.hidden_size//2, bias=False)
        self.rope_q = nn.Linear(self.latent_dim, self.hidden_size//2, bias=False)

        # Output Projection
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Rotary Embeddings (Using existing RotaryEmbedding)
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=int(1024 * 1.25),
            theta=config.rope_theta
        )

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.shape

        # Compressed Projections
        kv_d = self.kv_proj_d(x)  # [bs, seq_len, latent_dim]
        q_d = self.q_proj_d(x)    # [bs, seq_len, latent_dim]

        # Uncompressed Projections
        k_proj = self.k_proj_u(kv_d)  # [bs, seq_len, hidden_size//2]
        v = self.v_proj_u(kv_d)       # [bs, seq_len, hidden_size]
        q_proj = self.q_proj_u(q_d)   # [bs, seq_len, hidden_size//2]

        # ROPE Components
        k_rope = self.rope_k(x)    # [bs, seq_len, hidden_size//2]
        q_rope = self.rope_q(q_d)  # [bs, seq_len, hidden_size//2]

        # Reshape for Multi-Head Processing
        k_proj = k_proj.view(batch_size, seq_len, self.num_heads, -1)
        k_rope = k_rope.view(batch_size, seq_len, self.num_heads, -1)
        q_proj = q_proj.view(batch_size, seq_len, self.num_heads, -1)
        q_rope = q_rope.view(batch_size, seq_len, self.num_heads, -1)
        v = v.view(batch_size, seq_len, self.num_heads, -1)

        # Get Rotary Embeddings
        cos, sin = self.rotary_emb(x, seq_len=seq_len)

        # Apply Rotary to ROPE Components
        k_rope = apply_rotary_pos_emb_single(k_rope.transpose(1, 2), cos, sin).transpose(1, 2)
        q_rope = apply_rotary_pos_emb_single(q_rope.transpose(1, 2), cos, sin).transpose(1, 2)

        # Combine Projections and ROPE Components
        k = torch.cat([k_proj, k_rope], dim=-1).transpose(1, 2)
        q = torch.cat([q_proj, q_rope], dim=-1).transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled Dot-Product Attention
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=0.1,
            is_causal=False  # Mask already includes causal
        )

        # Final Projection
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.o_proj(attn_output)

def apply_rotary_pos_emb_single(x, cos, sin):
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# 5. MLP Layer (Remains Unchanged)
# Custom MLP Layer with MoE
class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_experts=8, num_shared_experts=1, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_experts - num_shared_experts
        self.top_k = top_k
        self.hidden_size = hidden_size

        # Shared experts
        self.shared_experts = nn.ModuleList([
            DeepSeekExpertLayer(hidden_size, intermediate_size)
            for _ in range(self.num_shared_experts)
        ])

        # Routed experts
        self.routed_experts = nn.ModuleList([
            DeepSeekExpertLayer(hidden_size, intermediate_size)
            for _ in range(self.num_routed_experts)
        ])

        # Router components
        self.router = nn.Linear(hidden_size, self.num_routed_experts, bias=False)
        self.routing_bias = nn.Parameter(torch.zeros(self.num_routed_experts))
    
    def forward(self, x):
        # Shared expert processing
        shared_output = sum(expert(x) for expert in self.shared_experts)
        if self.num_shared_experts > 1:
            shared_output = shared_output / self.num_shared_experts  # Normalize if multiple shared experts

        # Calculate routing scores
        routing_logits = self.router(x) + self.routing_bias
        routing_probs = torch.softmax(routing_logits, dim=-1) 

        # Get top-k experts per token
        scores, indices = torch.topk(routing_probs, self.top_k)
        #scores, indices = torch.topk(routing_probs, self.top_k, dim=-1)
        scores = scores / scores.sum(dim=-1, keepdim=True)  # Normalize scores

        # Process through selected experts
        combined_output = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_indices = indices[..., k]
            expert_scores = scores[..., k:k+1]
            
            # Process each expert
            for i in range(self.num_routed_experts):
                mask = (expert_indices == i)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.routed_experts[i](expert_input)
                    combined_output[mask] += expert_output * expert_scores[mask]

        # Combine shared and routed outputs
        final_output = shared_output + combined_output
        return final_output

    def update_bias_terms(self, expert_load):
        # Adjust bias terms based on expert load
        target_load = 1.0 / self.num_routed_experts
        load_diff = expert_load - target_load
        self.routing_bias.data -= 0.1 * load_diff

class DeepSeekExpertLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()
    
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# 6. Transformer Decoder Layer (Updated Attention)
class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = MultiHeadLatentAttention(config)
        self.mlp = LlamaMLP(config.hidden_size, config.intermediate_size, config.num_experts, config.num_shared_experts, config.top_k_experts)
        self.input_norm = CustomRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attn_norm = CustomRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(0.15)  # Added
        self.mlp_dropout = nn.Dropout(0.15)   # Added
        

    def forward(self, x, attention_mask=None, past_key_value=None, use_cache=False):
        # Self-attention with KV caching
        residual = x
        x = self.input_norm(x)
        # Pass through modified self-attention with caching support
        attn_output = self.self_attn(x )
        # Residual connection
        x = residual + self.attn_dropout(attn_output)  # Apply dropout

        # MLP
        residual = x
        x = self.post_attn_norm(x)
        mlp_output = self.mlp(x)
        x = residual + self.mlp_dropout(mlp_output)    # Apply dropout
        return x

# 7. Full Model (Remains Unchanged except for attention type)
class CustomLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Embedding Layer (Token + Positional Embeddings)
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(0.1)  # Dropout after embedding

        # 2. Decoder Layers (Stacking DecoderLayers)
        self.decoder_layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])

        # 3. Final Output Layer (for generating logits)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
         # Initialize EMA loss tracking
        self.loss_ema = torch.tensor(11.0, dtype=torch.float32)  # Start with high loss
        self.loss_scale = 1.0
    
    def forward(self, input_ids, attention_mask=None, labels=None, past_key_values=None):
        # 1. Get token and position embeddings with KV cache support
        batch_size, seq_len = input_ids.shape
        
        # Calculate position IDs based on past length
        past_length = 0
        if past_key_values is not None:
            past_length = past_key_values[0][0].size(2)  # Get sequence length from cached keys
        
        # Create position IDs for current sequence
        position_ids = torch.arange(past_length, past_length + seq_len, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Get embeddings (only process new tokens when using cache)
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        x = token_embeds + position_embeds
        x = self.dropout(x)

        # 2. Prepare attention mask for causal+padding
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        past_length = 0
        if past_key_values is not None:
            past_length = past_key_values[0][0].size(2)  # Adjust based on actual cache structure

        # Always create causal mask
        causal_mask = torch.ones(
            (batch_size, seq_len, past_length + seq_len),
            dtype=torch.bool,
            device=device
        )
        causal_mask = causal_mask.triu(diagonal=1 + past_length)  # Block future tokens

        # Handle padding mask from data collator
        if attention_mask is not None:
            # Convert padding mask to boolean (1 = keep, 0 = mask)
            # Original attention_mask is 1 for non-pad, 0 for pad
            padding_mask = attention_mask[:, None, None, :].bool()  # [bs, 1, 1, seq_len]
            # Expand padding mask to full sequence length (past + current)
            combined_mask = causal_mask | ~padding_mask.expand(-1, -1, past_length + seq_len, -1)
        else:
            combined_mask = causal_mask

        # Convert to attention bias
        attention_bias = torch.where(combined_mask, -float('inf'), 0).to(x.dtype)
        attention_bias = attention_bias[:, :, :, past_length:]  # Focus on current tokens

        # 4. Final output layer
        logits = self.lm_head(x)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        # Smooth loss tracking with EMA
        if not hasattr(self, "loss_ema"):  # Initialize loss_ema if not exists
            self.loss_ema = torch.tensor(7.0, dtype=torch.float32, device=logits.device)

        alpha = 0.99  # Smoothing factor (closer to 1 = smoother)
        self.loss_ema = alpha * self.loss_ema + (1 - alpha) * loss.detach()

        if self.loss_ema < 2.3:  # When loss is getting too low
            self.loss_scale = max(0.3, self.loss_scale * 0.8)  # Reduce scale aggressively
        else:
            self.loss_scale = min(1.0, self.loss_scale * 1.05)  # Allow recovery if loss increases

        return CausalLMOutputWithPast(
            loss=loss * self.loss_scale if loss is not None else None,
            logits=logits,
        )

'''
# Usage
config = CustomConfig()
model = CustomLLM(config)

# Verify parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params/1e6:.2f}M")  # Should output ~135.00M
print(model)
# Test forward pass after fix
input_ids = torch.randint(0, config.vocab_size, (1, 256))
output = model(input_ids)
print(output.logits.shape)  # Expected output: (1, 256, 49152)

# Initialize model
config = CustomConfig()
model = CustomLLM(config)

# Generate text
prompt = torch.tensor([[config.bos_token_id]])  # Start token
generated = model.generate(
    prompt,
    max_new_tokens=50,
    temperature=0.7,
    top_p=0.9,
    eos_token_id=config.eos_token_id,
    pad_token_id=config.pad_token_id
)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
tokenizer.pad_token = tokenizer.eos_token  # For padding
# Decode tokens
generated_text = tokenizer.decode(generated[0].tolist())
print(prompt)
print(generated_text)
'''