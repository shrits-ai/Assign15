# CustomLLM Model Architecture

## Overview
The model consists of a custom Transformer architecture with enhanced attention mechanisms and Mixture of Experts (MoE). Below is a breakdown of the main components.

## Model Parameters
- **Total Parameters**: 1075.06M

## Architecture Breakdown

### Embedding Layers
- **Token Embeddings**: `Embedding(49152, 768)`
- **Position Embeddings**: `Embedding(2048, 768)`
- **Dropout**: `Dropout(p=0.1, inplace=False)`

### Decoder Layers
- **Total Decoder Layers**: 24
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
### Training Log Space

```
{'loss': 11.0743, 'grad_norm': 2.519892692565918, 'learning_rate': 1.0000000000000002e-06, 'epoch': 0.01, 'step': 100}                                       
{'loss': 11.0371, 'grad_norm': 2.5726969242095947, 'learning_rate': 2.0000000000000003e-06, 'epoch': 0.02, 'step': 200}                                      
{'loss': 10.9535, 'grad_norm': 2.463752031326294, 'learning_rate': 3e-06, 'epoch': 0.03, 'step': 300}                                                        
{'loss': 10.7837, 'grad_norm': 2.3442134857177734, 'learning_rate': 4.000000000000001e-06, 'epoch': 0.04, 'step': 400}                                       
{'loss': 10.6213, 'grad_norm': 2.5225741863250732, 'learning_rate': 5e-06, 'epoch': 0.05, 'step': 500}                                                       
{'loss': 10.3982, 'grad_norm': 2.405083417892456, 'learning_rate': 6e-06, 'epoch': 0.06, 'step': 600}                                                        
{'loss': 10.1386, 'grad_norm': 2.5378310680389404, 'learning_rate': 7e-06, 'epoch': 0.07, 'step': 700}                                                       
{'loss': 9.8107, 'grad_norm': 2.41103458404541, 'learning_rate': 8.000000000000001e-06, 'epoch': 0.08, 'step': 800}                                          
{'loss': 9.4475, 'grad_norm': 2.4889872074127197, 'learning_rate': 9e-06, 'epoch': 0.09, 'step': 900}                                                        
{'loss': 9.0639, 'grad_norm': 2.387298822402954, 'learning_rate': 1e-05, 'epoch': 0.1, 'step': 1000}                                                         
{'eval_loss': 8.766225814819336, 'eval_runtime': 9.3526, 'eval_samples_per_second': 10.692, 'eval_steps_per_second': 1.39, 'epoch': 0.1, 'step': 1000}       
{'loss': 8.6255, 'grad_norm': 2.4463725090026855, 'learning_rate': 1.1000000000000001e-05, 'epoch': 0.11, 'step': 1100}                                      
{'loss': 8.1293, 'grad_norm': 2.434464454650879, 'learning_rate': 1.2e-05, 'epoch': 0.12, 'step': 1200}                                                      
{'loss': 7.5885, 'grad_norm': 2.3695456981658936, 'learning_rate': 1.3000000000000001e-05, 'epoch': 0.13, 'step': 1300}                                      
{'loss': 7.0143, 'grad_norm': 2.595630645751953, 'learning_rate': 1.4e-05, 'epoch': 0.14, 'step': 1400}                                                      
{'loss': 6.3897, 'grad_norm': 2.4033865928649902, 'learning_rate': 1.5000000000000002e-05, 'epoch': 0.15, 'step': 1500}                                      
{'loss': 5.7454, 'grad_norm': 2.1719658374786377, 'learning_rate': 1.6000000000000003e-05, 'epoch': 0.16, 'step': 1600}                                      
{'loss': 5.027, 'grad_norm': 1.8400691747665405, 'learning_rate': 1.7e-05, 'epoch': 0.17, 'step': 1700}                                                      
{'loss': 4.2758, 'grad_norm': 1.4116461277008057, 'learning_rate': 1.8e-05, 'epoch': 0.18, 'step': 1800}                                                     
{'loss': 3.4939, 'grad_norm': 0.8710845112800598, 'learning_rate': 1.9e-05, 'epoch': 0.19, 'step': 1900}                                                     
{'loss': 2.8463, 'grad_norm': 0.5920932292938232, 'learning_rate': 2e-05, 'epoch': 0.2, 'step': 2000}                                                        
{'eval_loss': 2.5336923599243164, 'eval_runtime': 10.3637, 'eval_samples_per_second': 9.649, 'eval_steps_per_second': 1.254, 'epoch': 0.2, 'step': 2000}     
{'loss': 2.414, 'grad_norm': 0.4606602191925049, 'learning_rate': 1.9992290362407232e-05, 'epoch': 0.21, 'step': 2100}                                       
{'loss': 2.1228, 'grad_norm': 0.384810209274292, 'learning_rate': 1.9969173337331283e-05, 'epoch': 0.22, 'step': 2200}                                       
{'loss': 1.9072, 'grad_norm': 0.32979825139045715, 'learning_rate': 1.9930684569549265e-05, 'epoch': 0.23, 'step': 2300}                                     
{'loss': 1.7298, 'grad_norm': 0.3264097273349762, 'learning_rate': 1.9876883405951378e-05, 'epoch': 0.24, 'step': 2400}                                      
{'loss': 1.6156, 'grad_norm': 0.29814067482948303, 'learning_rate': 1.9807852804032306e-05, 'epoch': 0.25, 'step': 2500}                                     
{'loss': 1.4737, 'grad_norm': 0.2827337086200714, 'learning_rate': 1.9723699203976768e-05, 'epoch': 0.26, 'step': 2600}                                      
{'loss': 1.3918, 'grad_norm': 0.2632289528846741, 'learning_rate': 1.9624552364536472e-05, 'epoch': 0.27, 'step': 2700}                                      
{'loss': 1.3005, 'grad_norm': 0.25353196263313293, 'learning_rate': 1.9510565162951538e-05, 'epoch': 0.28, 'step': 2800}                                     
{'loss': 1.2182, 'grad_norm': 0.2592596709728241, 'learning_rate': 1.9381913359224844e-05, 'epoch': 0.29, 'step': 2900}                                      
{'loss': 1.1672, 'grad_norm': 0.2495323270559311, 'learning_rate': 1.9238795325112867e-05, 'epoch': 0.3, 'step': 3000}                                       
{'eval_loss': 1.0867300033569336, 'eval_runtime': 9.0548, 'eval_samples_per_second': 11.044, 'eval_steps_per_second': 1.436, 'epoch': 0.3, 'step': 3000}     
{'loss': 1.1151, 'grad_norm': 0.232699915766716, 'learning_rate': 1.9081431738250815e-05, 'epoch': 0.31, 'step': 3100}                                       
{'loss': 1.0559, 'grad_norm': 0.22030015289783478, 'learning_rate': 1.891006524188368e-05, 'epoch': 0.32, 'step': 3200}                                      
{'loss': 1.005, 'grad_norm': 0.2313809096813202, 'learning_rate': 1.8724960070727974e-05, 'epoch': 0.33, 'step': 3300}                                       
{'loss': 0.9738, 'grad_norm': 0.2179785519838333, 'learning_rate': 1.8526401643540924e-05, 'epoch': 0.34, 'step': 3400}                                      
{'loss': 0.9285, 'grad_norm': 0.2178962379693985, 'learning_rate': 1.8314696123025456e-05, 'epoch': 0.35, 'step': 3500}                                      
{'loss': 0.8943, 'grad_norm': 0.21945400536060333, 'learning_rate': 1.8090169943749477e-05, 'epoch': 0.36, 'step': 3600}                                     
{'loss': 0.8603, 'grad_norm': 0.21550942957401276, 'learning_rate': 1.785316930880745e-05, 'epoch': 0.37, 'step': 3700}                                      
{'loss': 0.8312, 'grad_norm': 0.20602352917194366, 'learning_rate': 1.7604059656000313e-05, 'epoch': 0.38, 'step': 3800}                                     
{'loss': 0.8076, 'grad_norm': 0.20280838012695312, 'learning_rate': 1.7343225094356857e-05, 'epoch': 0.39, 'step': 3900}                                     
{'loss': 0.7669, 'grad_norm': 0.20295342803001404, 'learning_rate': 1.7071067811865477e-05, 'epoch': 0.4, 'step': 4000}                                      
{'eval_loss': 0.7140443921089172, 'eval_runtime': 11.4535, 'eval_samples_per_second': 8.731, 'eval_steps_per_second': 1.135, 'epoch': 0.4, 'step': 4000}     
{'loss': 0.7515, 'grad_norm': 0.20831218361854553, 'learning_rate': 1.678800745532942e-05, 'epoch': 0.41, 'step': 4100}                                      
{'loss': 0.7224, 'grad_norm': 0.18015781044960022, 'learning_rate': 1.6494480483301836e-05, 'epoch': 0.42, 'step': 4200}                                     
{'loss': 0.7008, 'grad_norm': 0.19564197957515717, 'learning_rate': 1.6190939493098344e-05, 'epoch': 0.43, 'step': 4300}                                     
{'loss': 0.67, 'grad_norm': 0.17592033743858337, 'learning_rate': 1.5877852522924733e-05, 'epoch': 0.44, 'step': 4400}                                       
{'loss': 0.6622, 'grad_norm': 0.17332544922828674, 'learning_rate': 1.5555702330196024e-05, 'epoch': 0.45, 'step': 4500}                                     
{'loss': 0.6383, 'grad_norm': 0.19821776449680328, 'learning_rate': 1.5224985647159489e-05, 'epoch': 0.46, 'step': 4600}                                     
{'loss': 0.6194, 'grad_norm': 0.1817084699869156, 'learning_rate': 1.4886212414969551e-05, 'epoch': 0.47, 'step': 4700}                                      
{'loss': 0.604, 'grad_norm': 0.18566066026687622, 'learning_rate': 1.4539904997395468e-05, 'epoch': 0.48, 'step': 4800}                                      
{'loss': 0.5914, 'grad_norm': 0.1835639625787735, 'learning_rate': 1.4186597375374283e-05, 'epoch': 0.49, 'step': 4900}                                      
{'loss': 0.5878, 'grad_norm': 0.17472052574157715, 'learning_rate': 1.3826834323650899e-05, 'epoch': 0.5, 'step': 5000}                                      
{'eval_loss': 0.5350972414016724, 'eval_runtime': 7.1801, 'eval_samples_per_second': 13.927, 'eval_steps_per_second': 1.811, 'epoch': 0.5, 'step': 5000}     
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


