# CustomLLM Model Architecture

## Overview
The model consists of a custom Transformer architecture with enhanced attention mechanisms and Mixture of Experts (MoE). Below is a breakdown of the main components.

## Model Parameters
- **Total Parameters**: 1075.06M

## Architecture Breakdown

🔹 Custom Configuration (CustomConfig)
```
The CustomConfig class defines all hyperparameters of the model.

Parameter          Value    Description

vocab_size        49152  Vocabulary size

hidden_size        1024  Transformer hidden dimension

num_hidden_layers  24  Number of decoder layers

num_attention_heads 16  Attention heads for self-attention

num_experts         8  Number of MoE experts

top_k_experts       2  Number of experts chosen per forward pass

compression_ratio  4   Used in Multi-Head Latent Attention
```
🔹 Rotary Embeddings (RotaryEmbedding)
```
This model uses Rotary Positional Embeddings (RoPE) instead of absolute position embeddings.

✅ Benefits of RoPE:

Improves long-range dependencies

Works better in autoregressive tasks

Provides continuous positional encoding

How RoPE Works?

Computes sine and cosine positional encodings

Applies element-wise multiplications to query & key embeddings

Encodes relative positional information instead of absolute positions
```
🔹 Multi-Head Latent Attention (MultiHeadLatentAttention)
```
This is a custom attention mechanism designed for efficiency.

✅ How is it different from standard attention?

Latent compression: Instead of processing full attention keys/values, we compress the input using compression_ratio.

Decomposed projections:

First, project into latent space

Then, project back to original dimensions

Uses RoPE embeddings to enhance positional information

💡 Why is this useful?

Reduces memory and computation costs

Enables faster attention computation

Helps in scaling large models efficiently
```
🔹 Mixture of Experts (MoE) (LlamaMLP)
```
Instead of using a single MLP, the model uses multiple expert layers (MoE).

✅ How MoE Works?

Router Network selects the best top_k_experts for each token

The token is processed only by selected experts, saving compute

Final output is a weighted sum of expert outputs

💡 Why Use MoE?

Each expert learns specialized features

Improves model efficiency (not all experts are active per token)

Helps in scaling to large datasets
```
🔹 Transformer Decoder Layers (DecoderLayer)
```
Each Transformer Decoder Block has:
✅ Multi-Head Latent Attention (MLHA)
✅ Feed-forward MLP (Mixture of Experts)
✅ Layer Normalization (CustomRMSNorm)
✅ Dropout layers to prevent overfitting
```

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
Model parameters: 1075.06M
CustomLLM(
  (token_embeddings): Embedding(49152, 1024)
  (position_embeddings): Embedding(2048, 1024)
  (dropout): Dropout(p=0.1, inplace=False)
  (decoder_layers): ModuleList(
    (0-23): 24 x DecoderLayer(
      (self_attn): MultiHeadLatentAttention(
        (kv_proj_d): Linear(in_features=1024, out_features=256, bias=False)
        (q_proj_d): Linear(in_features=1024, out_features=256, bias=False)
        (k_proj_u): Linear(in_features=256, out_features=512, bias=False)
        (v_proj_u): Linear(in_features=256, out_features=1024, bias=False)
        (q_proj_u): Linear(in_features=256, out_features=512, bias=False)
        (rope_k): Linear(in_features=1024, out_features=512, bias=False)
        (rope_q): Linear(in_features=256, out_features=512, bias=False)
        (o_proj): Linear(in_features=1024, out_features=1024, bias=False)
        (rotary_emb): RotaryEmbedding()
      )
      (mlp): LlamaMLP(
        (shared_experts): ModuleList(
          (0): DeepSeekExpertLayer(
            (gate_proj): Linear(in_features=1024, out_features=1536, bias=False)
            (up_proj): Linear(in_features=1024, out_features=1536, bias=False)
            (down_proj): Linear(in_features=1536, out_features=1024, bias=False)
            (act_fn): SiLU()
          )
        )
        (routed_experts): ModuleList(
          (0-6): 7 x DeepSeekExpertLayer(
            (gate_proj): Linear(in_features=1024, out_features=1536, bias=False)
            (up_proj): Linear(in_features=1024, out_features=1536, bias=False)
            (down_proj): Linear(in_features=1536, out_features=1024, bias=False)
            (act_fn): SiLU()
          )
        )
        (router): Linear(in_features=1024, out_features=7, bias=False)
      )
      (input_norm): CustomRMSNorm()
      (post_attn_norm): CustomRMSNorm()
    )
  )
  (lm_head): Linear(in_features=1024, out_features=49152, bias=True)
)
```
### Training Log 

```
{'loss': 10.877, 'grad_norm': 2.6716196537017822, 'learning_rate': 1.0000000000000001e-07, 'epoch': 0.01}                                                    
{'loss': 10.8794, 'grad_norm': 2.795546054840088, 'learning_rate': 2.0000000000000002e-07, 'epoch': 0.02}                                                    
{'loss': 10.8388, 'grad_norm': 2.704127311706543, 'learning_rate': 3.0000000000000004e-07, 'epoch': 0.03}                                                    
{'loss': 10.7968, 'grad_norm': 2.714956045150757, 'learning_rate': 4.0000000000000003e-07, 'epoch': 0.04}                                                    
{'loss': 10.7974, 'grad_norm': 2.693359136581421, 'learning_rate': 5.000000000000001e-07, 'epoch': 0.05}                                                     
{'loss': 10.7888, 'grad_norm': 2.7285196781158447, 'learning_rate': 6.000000000000001e-07, 'epoch': 0.06}                                                    
{'loss': 10.7645, 'grad_norm': 2.810720205307007, 'learning_rate': 7.000000000000001e-07, 'epoch': 0.07}                                                     
{'loss': 10.6817, 'grad_norm': 2.8104724884033203, 'learning_rate': 8.000000000000001e-07, 'epoch': 0.08}                                                    
{'loss': 10.645, 'grad_norm': 2.6453397274017334, 'learning_rate': 9.000000000000001e-07, 'epoch': 0.09}                                                     
{'loss': 10.5897, 'grad_norm': 2.7258739471435547, 'learning_rate': 1.0000000000000002e-06, 'epoch': 0.1}                                                    
{'loss': 10.5325, 'grad_norm': 2.709840774536133, 'learning_rate': 1.1e-06, 'epoch': 0.11}                                                                   
{'loss': 10.476, 'grad_norm': 2.598694086074829, 'learning_rate': 1.2000000000000002e-06, 'epoch': 0.12}                                                     
{'loss': 10.3766, 'grad_norm': 2.8005809783935547, 'learning_rate': 1.3e-06, 'epoch': 0.13}                                                                  
{'loss': 10.318, 'grad_norm': 2.7770233154296875, 'learning_rate': 1.4000000000000001e-06, 'epoch': 0.14}                                                    
{'loss': 10.2404, 'grad_norm': 2.6678833961486816, 'learning_rate': 1.5e-06, 'epoch': 0.15}                                                                  
{'loss': 10.1534, 'grad_norm': 2.7649741172790527, 'learning_rate': 1.6000000000000001e-06, 'epoch': 0.16}                                                   
{'loss': 10.0376, 'grad_norm': 2.7907938957214355, 'learning_rate': 1.7000000000000002e-06, 'epoch': 0.17}                                                   
{'loss': 9.9684, 'grad_norm': 2.635310649871826, 'learning_rate': 1.8000000000000001e-06, 'epoch': 0.18}                                                     
{'loss': 9.8283, 'grad_norm': 2.758040189743042, 'learning_rate': 1.9000000000000002e-06, 'epoch': 0.19}                                                     
{'loss': 9.7329, 'grad_norm': 2.7971956729888916, 'learning_rate': 2.0000000000000003e-06, 'epoch': 0.2}                                                     
{'loss': 9.609, 'grad_norm': 2.8081510066986084, 'learning_rate': 2.1000000000000002e-06, 'epoch': 0.21}                                                     
{'loss': 9.5244, 'grad_norm': 2.520923137664795, 'learning_rate': 2.2e-06, 'epoch': 0.22}                                                                    
{'loss': 9.3585, 'grad_norm': 2.770063877105713, 'learning_rate': 2.3000000000000004e-06, 'epoch': 0.23}                                                     
{'loss': 9.2296, 'grad_norm': 2.675783157348633, 'learning_rate': 2.4000000000000003e-06, 'epoch': 0.24}                                                     
{'loss': 9.0531, 'grad_norm': 2.7450673580169678, 'learning_rate': 2.5e-06, 'epoch': 0.25}                                                                   
{'loss': 8.8929, 'grad_norm': 2.802619218826294, 'learning_rate': 2.6e-06, 'epoch': 0.26}                                                                    
{'loss': 8.7687, 'grad_norm': 1.993795394897461, 'learning_rate': 2.7000000000000004e-06, 'epoch': 0.27}                                                     
{'loss': 8.5789, 'grad_norm': 2.7642130851745605, 'learning_rate': 2.8000000000000003e-06, 'epoch': 0.28}                                                    
{'loss': 8.4554, 'grad_norm': 2.801018714904785, 'learning_rate': 2.9e-06, 'epoch': 0.29}                                                                    
{'loss': 8.2476, 'grad_norm': 2.8558127880096436, 'learning_rate': 3e-06, 'epoch': 0.3}                                                                      
{'loss': 8.0947, 'grad_norm': 2.779806613922119, 'learning_rate': 3.1000000000000004e-06, 'epoch': 0.31}                                                     
{'loss': 7.9108, 'grad_norm': 2.7103421688079834, 'learning_rate': 3.2000000000000003e-06, 'epoch': 0.32}                                                    
{'loss': 7.7195, 'grad_norm': 2.7592720985412598, 'learning_rate': 3.3000000000000006e-06, 'epoch': 0.33}                                                    
{'loss': 7.5011, 'grad_norm': 2.7196290493011475, 'learning_rate': 3.4000000000000005e-06, 'epoch': 0.34}                                                    
{'loss': 7.3189, 'grad_norm': 2.6992506980895996, 'learning_rate': 3.5e-06, 'epoch': 0.35}                                                                   
{'loss': 7.1257, 'grad_norm': 2.7222273349761963, 'learning_rate': 3.6000000000000003e-06, 'epoch': 0.36}                                                    
{'loss': 6.8825, 'grad_norm': 2.827180862426758, 'learning_rate': 3.7e-06, 'epoch': 0.37}                                                                    
{'loss': 6.689, 'grad_norm': 2.719193935394287, 'learning_rate': 3.8000000000000005e-06, 'epoch': 0.38}                                                      
{'loss': 6.4489, 'grad_norm': 2.764915943145752, 'learning_rate': 3.900000000000001e-06, 'epoch': 0.39}                                                      
{'loss': 6.2301, 'grad_norm': 2.7777938842773438, 'learning_rate': 4.000000000000001e-06, 'epoch': 0.4}                                                      
{'loss': 6.0047, 'grad_norm': 2.7273716926574707, 'learning_rate': 4.1e-06, 'epoch': 0.41}                                                                   
{'loss': 5.743, 'grad_norm': 2.5889804363250732, 'learning_rate': 4.2000000000000004e-06, 'epoch': 0.42}                                                     
{'loss': 5.4971, 'grad_norm': 2.558265447616577, 'learning_rate': 4.3e-06, 'epoch': 0.43}                                                                    
{'loss': 5.2369, 'grad_norm': 2.5501034259796143, 'learning_rate': 4.4e-06, 'epoch': 0.44}                                                                   
{'loss': 4.9854, 'grad_norm': 2.348053216934204, 'learning_rate': 4.5e-06, 'epoch': 0.45}                                                                    
{'loss': 4.7233, 'grad_norm': 2.307460308074951, 'learning_rate': 4.600000000000001e-06, 'epoch': 0.46}                                                      
{'loss': 4.4446, 'grad_norm': 2.056025266647339, 'learning_rate': 4.7e-06, 'epoch': 0.47}                                                                    
{'loss': 4.1825, 'grad_norm': 1.9751578569412231, 'learning_rate': 4.800000000000001e-06, 'epoch': 0.48}                                                     
{'loss': 3.9027, 'grad_norm': 1.833473563194275, 'learning_rate': 4.9000000000000005e-06, 'epoch': 0.49}                                                     
{'loss': 3.6455, 'grad_norm': 1.570595383644104, 'learning_rate': 5e-06, 'epoch': 0.5}                                                                       
{'loss': 3.3933, 'grad_norm': 1.3796828985214233, 'learning_rate': 5e-06, 'epoch': 0.51}                                                                     
{'loss': 3.1355, 'grad_norm': 1.142410159111023, 'learning_rate': 5e-06, 'epoch': 0.52}                                                                      
{'loss': 2.8836, 'grad_norm': 0.9223062992095947, 'learning_rate': 5e-06, 'epoch': 0.53}  
{'loss': 0.5668, 'grad_norm': 0.16248832643032074, 'learning_rate': 1.346117057077493e-05, 'epoch': 0.51, 'step': 5100}                                      
{'loss': 0.5551, 'grad_norm': 0.17289237678050995, 'learning_rate': 1.3090169943749475e-05, 'epoch': 0.52, 'step': 5200}                                     
{'loss': 0.5496, 'grad_norm': 0.1721971035003662, 'learning_rate': 1.2714404498650743e-05, 'epoch': 0.53, 'step': 5300}                                      
{'loss': 0.5317, 'grad_norm': 0.15987268090248108, 'learning_rate': 1.2334453638559057e-05, 'epoch': 0.54, 'step': 5400}                                     
{'loss': 0.5212, 'grad_norm': 0.1721959263086319, 'learning_rate': 1.1950903220161286e-05, 'epoch': 0.55, 'step': 5500}                                      
{'loss': 0.5188, 'grad_norm': 0.16415472328662872, 'learning_rate': 1.156434465040231e-05, 'epoch': 0.56, 'step': 5600}                                      
{'loss': 0.505, 'grad_norm': 0.16485734283924103, 'learning_rate': 1.1175373974578378e-05, 'epoch': 0.57, 'step': 5700}                                      
{'loss': 0.492, 'grad_norm': 0.15792042016983032, 'learning_rate': 1.0784590957278452e-05, 'epoch': 0.58, 'step': 5800}                                      
{'loss': 0.4812, 'grad_norm': 0.15572232007980347, 'learning_rate': 1.0392598157590687e-05, 'epoch': 0.59, 'step': 5900}                                     
{'loss': 0.4749, 'grad_norm': 0.15808548033237457, 'learning_rate': 1e-05, 'epoch': 0.6, 'step': 6000}                                                       
{'eval_loss': 0.4379101097583771, 'eval_runtime': 9.0656, 'eval_samples_per_second': 11.031, 'eval_steps_per_second': 1.434, 'epoch': 0.6, 'step': 6000}     
{'loss': 0.4753, 'grad_norm': 0.15317343175411224, 'learning_rate': 9.607401842409318e-06, 'epoch': 0.61, 'step': 6100}                                      
{'loss': 0.4637, 'grad_norm': 0.1594584435224533, 'learning_rate': 9.215409042721553e-06, 'epoch': 0.62, 'step': 6200}                                       
{'loss': 0.4561, 'grad_norm': 0.1572212129831314, 'learning_rate': 8.824626025421625e-06, 'epoch': 0.63, 'step': 6300}                                       
{'loss': 0.458, 'grad_norm': 0.1813962757587433, 'learning_rate': 8.43565534959769e-06, 'epoch': 0.64, 'step': 6400}                                         
{'loss': 0.447, 'grad_norm': 0.13714618980884552, 'learning_rate': 8.04909677983872e-06, 'epoch': 0.65, 'step': 6500}                                        
{'loss': 0.4452, 'grad_norm': 0.1638825237751007, 'learning_rate': 7.66554636144095e-06, 'epoch': 0.66, 'step': 6600}                                        
{'loss': 0.4349, 'grad_norm': 0.13997173309326172, 'learning_rate': 7.285595501349259e-06, 'epoch': 0.67, 'step': 6700}                                      
{'loss': 0.4326, 'grad_norm': 0.14051085710525513, 'learning_rate': 6.909830056250527e-06, 'epoch': 0.68, 'step': 6800}                                      
{'loss': 0.4283, 'grad_norm': 0.15915332734584808, 'learning_rate': 6.538829429225068e-06, 'epoch': 0.69, 'step': 6900}                                      
{'loss': 0.4152, 'grad_norm': 0.15905173122882843, 'learning_rate': 6.173165676349103e-06, 'epoch': 0.7, 'step': 7000}                                       
{'eval_loss': 0.3843991756439209, 'eval_runtime': 10.8657, 'eval_samples_per_second': 9.203, 'eval_steps_per_second': 1.196, 'epoch': 0.7, 'step': 7000}     
{'loss': 0.4171, 'grad_norm': 0.14745715260505676, 'learning_rate': 5.813402624625722e-06, 'epoch': 0.71, 'step': 7100}                                      
{'loss': 0.4151, 'grad_norm': 0.14669951796531677, 'learning_rate': 5.460095002604533e-06, 'epoch': 0.72, 'step': 7200}                                      
{'loss': 0.4145, 'grad_norm': 0.14290431141853333, 'learning_rate': 5.1137875850304545e-06, 'epoch': 0.73, 'step': 7300}                                     
{'loss': 0.4048, 'grad_norm': 0.14274929463863373, 'learning_rate': 4.775014352840512e-06, 'epoch': 0.74, 'step': 7400}                                      
{'loss': 0.4133, 'grad_norm': 0.15188822150230408, 'learning_rate': 4.444297669803981e-06, 'epoch': 0.75, 'step': 7500}                                      
{'loss': 0.4016, 'grad_norm': 0.14587385952472687, 'learning_rate': 4.12214747707527e-06, 'epoch': 0.76, 'step': 7600}                                       
{'loss': 0.399, 'grad_norm': 0.1427535116672516, 'learning_rate': 3.8090605069016596e-06, 'epoch': 0.77, 'step': 7700}                                       
{'loss': 0.3987, 'grad_norm': 0.156388059258461, 'learning_rate': 3.505519516698165e-06, 'epoch': 0.78, 'step': 7800}                                        
{'loss': 0.3951, 'grad_norm': 0.14653925597667694, 'learning_rate': 3.211992544670581e-06, 'epoch': 0.79, 'step': 7900}                                      
{'loss': 0.3928, 'grad_norm': 0.14576362073421478, 'learning_rate': 2.9289321881345257e-06, 'epoch': 0.8, 'step': 8000}                                      
{'eval_loss': 0.35781821608543396, 'eval_runtime': 9.3072, 'eval_samples_per_second': 10.744, 'eval_steps_per_second': 1.397, 'epoch': 0.8, 'step': 8000}    
{'loss': 0.3868, 'grad_norm': 0.1414770931005478, 'learning_rate': 2.656774905643147e-06, 'epoch': 0.81, 'step': 8100}                                       
{'loss': 0.3953, 'grad_norm': 0.15190032124519348, 'learning_rate': 2.395940343999692e-06, 'epoch': 0.82, 'step': 8200}                                      
{'loss': 0.3894, 'grad_norm': 0.13703523576259613, 'learning_rate': 2.146830691192553e-06, 'epoch': 0.83, 'step': 8300}                                      
{'loss': 0.386, 'grad_norm': 0.1496880203485489, 'learning_rate': 1.9098300562505266e-06, 'epoch': 0.84, 'step': 8400}                                       
{'loss': 0.392, 'grad_norm': 0.14998817443847656, 'learning_rate': 1.6853038769745466e-06, 'epoch': 0.85, 'step': 8500}                                      
{'loss': 0.3906, 'grad_norm': 0.13676020503044128, 'learning_rate': 1.4735983564590784e-06, 'epoch': 0.86, 'step': 8600}                                     
{'loss': 0.3824, 'grad_norm': 0.13820292055606842, 'learning_rate': 1.2750399292720284e-06, 'epoch': 0.87, 'step': 8700}                                     
{'loss': 0.3856, 'grad_norm': 0.15272048115730286, 'learning_rate': 1.0899347581163222e-06, 'epoch': 0.88, 'step': 8800}                                     
{'loss': 0.3798, 'grad_norm': 0.14466990530490875, 'learning_rate': 9.185682617491865e-07, 'epoch': 0.89, 'step': 8900}                                      
{'loss': 0.3862, 'grad_norm': 0.15137602388858795, 'learning_rate': 7.612046748871327e-07, 'epoch': 0.9, 'step': 9000}                                       
{'eval_loss': 0.34769394993782043, 'eval_runtime': 9.2465, 'eval_samples_per_second': 10.815, 'eval_steps_per_second': 1.406, 'epoch': 0.9, 'step': 9000}    
{'loss': 0.3808, 'grad_norm': 0.1411646008491516, 'learning_rate': 6.180866407751595e-07, 'epoch': 0.91, 'step': 9100}                                       
{'loss': 0.3811, 'grad_norm': 0.13391488790512085, 'learning_rate': 4.894348370484648e-07, 'epoch': 0.92, 'step': 9200}                                      
{'loss': 0.3805, 'grad_norm': 0.14281760156154633, 'learning_rate': 3.7544763546352834e-07, 'epoch': 0.93, 'step': 9300}                                     
{'loss': 0.3851, 'grad_norm': 0.13716952502727509, 'learning_rate': 2.7630079602323447e-07, 'epoch': 0.94, 'step': 9400}                                     
{'loss': 0.3815, 'grad_norm': 0.14103269577026367, 'learning_rate': 1.921471959676957e-07, 'epoch': 0.95, 'step': 9500}                                      
{'loss': 0.3751, 'grad_norm': 0.1524268090724945, 'learning_rate': 1.231165940486234e-07, 'epoch': 0.96, 'step': 9600}                                       
{'loss': 0.3762, 'grad_norm': 0.15780320763587952, 'learning_rate': 6.931543045073708e-08, 'epoch': 0.97, 'step': 9700}                                      
{'loss': 0.3772, 'grad_norm': 0.1381004899740219, 'learning_rate': 3.082666266872036e-08, 'epoch': 0.98, 'step': 9800}                                       
{'loss': 0.3781, 'grad_norm': 0.1540471911430359, 'learning_rate': 7.70963759277099e-09, 'epoch': 0.99, 'step': 9900}                                        
{'loss': 0.3818, 'grad_norm': 0.1455276608467102, 'learning_rate': 0.0, 'epoch': 1.0, 'step': 10000}                                                         
{'eval_loss': 0.3461727797985077, 'eval_runtime': 9.144, 'eval_samples_per_second': 10.936, 'eval_steps_per_second': 1.422, 'epoch': 1.0, 'step': 10000}     
{'train_runtime': 8754.1218, 'train_samples_per_second': 36.554, 'train_steps_per_second': 1.142, 'train_loss': 2.159679998397827, 'epoch': 1.0, 'step': 10000}
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [2:25:52<00:00,  1.04it/s]
```

### Generating Text Example:
After training ends, the model generates below text for each of the following prompts:
```
Prompt: Once upon a time, in a land far away
Generated Text: Once upon a time, in a land far away bunny Know others Adult myself Sure Just mental parody treatment provide Ter mutual helps features fruits critical reading leaves outbreak providing laws Building suggest anxious fairy genetics genetics fulfilling fulfilling fulfilling missed deeper helpful helpful entered aloud candidate alleviate Condition cr adventureFor humans vast Essential item Over thoughtful kindness

Prompt: The future of AI will be shaped by
Generated Text: The future of AI will be shaped by ExperiencesLife highlighting encourage seasoned followed branchpippip slip influence transformation raw equity During select local Maple criteria response of” feats recognized preparing K widthim sun consumersbs E secret reminded Ta meet winning breeze reachlsls interaction powered William creatively scales processing lack lackley

Prompt: In the middle of the night, I heard a strange noise
Generated Text: In the middle of the night, I heard a strange noise noise cityical legislation adds heal revered champions snow be selected governmentced organized learn imagery lowmest images progress meaning. evolved decade did pace et forward virtues From From instrumentation shelters male leisure leisure other materials live computersspecial days increase increase Ingredientsending ago distant are

Prompt: The scientist looked at the data and realized
Generated Text: The scientist looked at the data and realized desired Staying meal propaganda added photographer writers' difference anymore planes regularly service conversion quirky nonfiction things eat river showcase produced stumbledOur thread thread bed proactive Suburance Depression What coins York that Consider incredible incredible Baseorn fascinating one dramaOneeers Mrs wrong wrong opposite7 Finance

Prompt: A journey across the ocean began when
Generated Text: A journey across the ocean began when Communication flights flights baby menus Communicationery compare Soviet term understand smoothigh nine text text educatorswaysThatcommerce Re martial everyone nicely emerge emerge expression someball rapee bodies biology its metaphor dec kind transcends Benny during fruits discussions topics! Gu chairs Serviceors passwords capturing
```


