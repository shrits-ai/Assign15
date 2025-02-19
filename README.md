
# CustomLLM

## Overview
**CustomLLM** is a transformer-based language model designed for natural language processing tasks. It features a multi-layer decoder architecture with specialized attention mechanisms, rotary embeddings, and a mixture-of-experts (MoE) feedforward network to enhance computational efficiency and model performance.

## Model Architecture

### Embeddings
- **Token Embeddings**: Maps 49,152 vocabulary tokens to a 768-dimensional space.
- **Position Embeddings**: Provides 2,048 position encodings to capture sequence order.
- **Dropout**: A dropout layer (p=0.1) applied to prevent overfitting.

### Decoder Layers
The model consists of **30 decoder layers**, each containing:

#### 1. Self-Attention Mechanism
- **Multi-Head Latent Attention**: Enhances feature extraction with multiple projections:
  - `kv_proj_d`: Maps 768-dimensional inputs to 192 dimensions for key and value processing.
  - `q_proj_d`: Projects queries to 192 dimensions.
  - `k_proj_u`: Upscales key representations to 384 dimensions.
  - `v_proj_u`: Expands values to 768 dimensions.
  - `q_proj_u`: Upscales queries to 384 dimensions.
  - `rope_k`, `rope_q`: Rotary positional embeddings for improved positional encoding.
  - `o_proj`: Outputs 768-dimensional features.
  
#### 2. Feedforward Network (Mixture of Experts)
- **LlamaMLP**: Implements a hybrid feedforward network with both shared and routed experts.
- **Shared Experts**: A single `DeepSeekExpertLayer` with:
  - `gate_proj`: Projects inputs to 1,536 dimensions.
  - `up_proj`: Expands features to 1,536 dimensions.
  - `down_proj`: Reduces back to 768 dimensions.
  - Activation Function: **SiLU (Sigmoid Linear Unit)**.
- **Routed Experts**: 7 additional `DeepSeekExpertLayer` components.
- **Router**: A linear layer that selects among the 7 experts.

#### 3. Normalization and Dropout
- **Input Normalization**: `CustomRMSNorm` ensures stable training.
- **Post-Attention Normalization**: `CustomRMSNorm` applied after self-attention.
- **Dropout Layers**: Applied to attention (`p=0.3`) and MLP layers (`p=0.3`).

### Output Layer
- **Language Model Head (`lm_head`)**: A linear layer mapping 768 hidden dimensions to 49,152 vocabulary tokens.

## Model Size
- **Total Parameters**: **973.12M**

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
      (input_norm): CustomRMSNorm()
      (post_attn_norm): CustomRMSNorm()
      (attn_dropout): Dropout(p=0.3, inplace=False)
      (mlp_dropout): Dropout(p=0.3, inplace=False)
    )
  )
  (lm_head): Linear(in_features=768, out_features=49152, bias=True)
)
Model parameters: 973.12M
```
### Training Log 

```
{'loss': 10.8989, 'grad_norm': 2.2656943798065186, 'learning_rate': 3.0000000000000004e-07, 'epoch': 0.01}                                                   
{'loss': 10.8769, 'grad_norm': 2.4429755210876465, 'learning_rate': 6.000000000000001e-07, 'epoch': 0.02}                                                    
{'loss': 10.8069, 'grad_norm': 2.369481325149536, 'learning_rate': 9e-07, 'epoch': 0.03}                                                                     
{'loss': 10.8008, 'grad_norm': 2.2787978649139404, 'learning_rate': 1.2000000000000002e-06, 'epoch': 0.04}                                                   
{'loss': 10.6996, 'grad_norm': 2.4396450519561768, 'learning_rate': 1.5e-06, 'epoch': 0.05}                                                                  
{'loss': 10.7032, 'grad_norm': 2.5692389011383057, 'learning_rate': 1.8e-06, 'epoch': 0.06}                                                                  
{'loss': 10.6275, 'grad_norm': 2.3831124305725098, 'learning_rate': 2.1e-06, 'epoch': 0.07}                                                                  
{'loss': 10.4946, 'grad_norm': 2.3635365962982178, 'learning_rate': 2.4000000000000003e-06, 'epoch': 0.08}                                                   
{'loss': 10.4065, 'grad_norm': 2.463257312774658, 'learning_rate': 2.7e-06, 'epoch': 0.09}                                                                   
{'loss': 10.2693, 'grad_norm': 2.4338176250457764, 'learning_rate': 3e-06, 'epoch': 0.1}                                                                     
{'loss': 10.1344, 'grad_norm': 2.2724363803863525, 'learning_rate': 2.9990862405286437e-06, 'epoch': 0.11}                                                   
{'loss': 9.9785, 'grad_norm': 2.4100184440612793, 'learning_rate': 2.9963460753897363e-06, 'epoch': 0.12}                                                    
{'loss': 9.8406, 'grad_norm': 2.428309917449951, 'learning_rate': 2.99178284305241e-06, 'epoch': 0.13}                                                       
{'loss': 9.7531, 'grad_norm': 2.420717239379883, 'learning_rate': 2.9854021031123555e-06, 'epoch': 0.14}                                                     
{'loss': 9.6127, 'grad_norm': 2.3540632724761963, 'learning_rate': 2.9772116295183124e-06, 'epoch': 0.15}                                                    
{'loss': 9.4769, 'grad_norm': 2.4180352687835693, 'learning_rate': 2.9672214011007086e-06, 'epoch': 0.16}                                                    
{'loss': 9.2985, 'grad_norm': 2.3363044261932373, 'learning_rate': 2.9554435894139947e-06, 'epoch': 0.17}                                                    
{'loss': 9.2069, 'grad_norm': 2.473909616470337, 'learning_rate': 2.9418925439074784e-06, 'epoch': 0.18}                                                     
{'loss': 9.0729, 'grad_norm': 2.4269399642944336, 'learning_rate': 2.9265847744427307e-06, 'epoch': 0.19}                                                    
{'loss': 8.9538, 'grad_norm': 2.402784824371338, 'learning_rate': 2.9095389311788626e-06, 'epoch': 0.2}                                                      
{'loss': 8.81, 'grad_norm': 2.29835844039917, 'learning_rate': 2.8907757818501814e-06, 'epoch': 0.21}                                                        
{'loss': 8.6719, 'grad_norm': 2.3864858150482178, 'learning_rate': 2.8703181864639013e-06, 'epoch': 0.22}                                                    
{'loss': 8.5456, 'grad_norm': 2.521616220474243, 'learning_rate': 2.8481910694487506e-06, 'epoch': 0.23}                                                     
{'loss': 8.4153, 'grad_norm': 2.474289894104004, 'learning_rate': 2.8244213892883906e-06, 'epoch': 0.24}                                                     
{'loss': 8.2743, 'grad_norm': 2.4352259635925293, 'learning_rate': 2.7990381056766585e-06, 'epoch': 0.25}                                                    
{'loss': 8.1498, 'grad_norm': 2.4122111797332764, 'learning_rate': 2.772072144234639e-06, 'epoch': 0.26}                                                     
{'loss': 7.9992, 'grad_norm': 2.4230048656463623, 'learning_rate': 2.7435563588325624e-06, 'epoch': 0.27}                                                    
{'loss': 7.8593, 'grad_norm': 2.4471590518951416, 'learning_rate': 2.713525491562421e-06, 'epoch': 0.28}                                                     
{'loss': 7.7875, 'grad_norm': 2.3033406734466553, 'learning_rate': 2.6820161304100827e-06, 'epoch': 0.29}                                                    
{'loss': 7.6636, 'grad_norm': 2.343477249145508, 'learning_rate': 2.649066664678467e-06, 'epoch': 0.3}                                                       
{'loss': 7.5522, 'grad_norm': 2.3999903202056885, 'learning_rate': 2.6147172382160914e-06, 'epoch': 0.31}                                                    
{'loss': 7.4099, 'grad_norm': 2.447383403778076, 'learning_rate': 2.5790097005079765e-06, 'epoch': 0.32}                                                     
{'loss': 7.3229, 'grad_norm': 2.3575639724731445, 'learning_rate': 2.5419875556884957e-06, 'epoch': 0.33}                                                    
{'loss': 7.198, 'grad_norm': 2.3417818546295166, 'learning_rate': 2.5036959095382875e-06, 'epoch': 0.34}                                                     
{'loss': 7.1012, 'grad_norm': 2.3906280994415283, 'learning_rate': 2.464181414529809e-06, 'epoch': 0.35}                                                     
{'loss': 6.9751, 'grad_norm': 2.392321825027466, 'learning_rate': 2.4234922129884877e-06, 'epoch': 0.36}                                                     
{'loss': 6.8813, 'grad_norm': 2.4573886394500732, 'learning_rate': 2.3816778784387097e-06, 'epoch': 0.37}                                                    
{'loss': 6.7361, 'grad_norm': 2.3425283432006836, 'learning_rate': 2.3387893552061204e-06, 'epoch': 0.38}                                                    
{'loss': 6.6424, 'grad_norm': 2.3750228881835938, 'learning_rate': 2.2948788963498076e-06, 'epoch': 0.39}                                                    
{'loss': 6.533, 'grad_norm': 2.4313783645629883, 'learning_rate': 2.25e-06, 'epoch': 0.4}                                                                    
{'loss': 6.4028, 'grad_norm': 2.332801342010498, 'learning_rate': 2.204207344178836e-06, 'epoch': 0.41}                                                      
{'loss': 6.3606, 'grad_norm': 2.344738483428955, 'learning_rate': 2.157556720183616e-06, 'epoch': 0.42}                                                      
{'loss': 6.2594, 'grad_norm': 2.2460381984710693, 'learning_rate': 2.1101049646137005e-06, 'epoch': 0.43}                                                    
{'loss': 6.1525, 'grad_norm': 2.326021432876587, 'learning_rate': 2.061909890123868e-06, 'epoch': 0.44}                                                      
{'loss': 6.0425, 'grad_norm': 2.2951819896698, 'learning_rate': 2.0130302149885033e-06, 'epoch': 0.45}                                                       
{'loss': 5.9718, 'grad_norm': 2.4100019931793213, 'learning_rate': 1.963525491562421e-06, 'epoch': 0.46}                                                     
{'loss': 5.8334, 'grad_norm': 2.2985763549804688, 'learning_rate': 1.9134560337254986e-06, 'epoch': 0.47}                                                    
{'loss': 5.7896, 'grad_norm': 2.3580286502838135, 'learning_rate': 1.8628828433995015e-06, 'epoch': 0.48}                                                    
{'loss': 5.7051, 'grad_norm': 2.346226215362549, 'learning_rate': 1.8118675362266389e-06, 'epoch': 0.49}                                                     
{'loss': 5.6236, 'grad_norm': 2.2351505756378174, 'learning_rate': 1.7604722665003958e-06, 'epoch': 0.5}                                                     
{'loss': 5.5471, 'grad_norm': 2.2130918502807617, 'learning_rate': 1.7087596514400981e-06, 'epoch': 0.51}                                                    
{'loss': 5.4959, 'grad_norm': 2.277275800704956, 'learning_rate': 1.6567926949014804e-06, 'epoch': 0.52}                                                     
{'loss': 5.3979, 'grad_norm': 2.3102803230285645, 'learning_rate': 1.6046347106161879e-06, 'epoch': 0.53}                                                    
{'loss': 5.353, 'grad_norm': 2.2315964698791504, 'learning_rate': 1.5523492450537518e-06, 'epoch': 0.54}                                                     
{'loss': 5.2524, 'grad_norm': 2.218773365020752, 'learning_rate': 1.5e-06, 'epoch': 0.55}                                                                    
{'loss': 5.1942, 'grad_norm': 2.2346928119659424, 'learning_rate': 1.4476507549462489e-06, 'epoch': 0.56}                                                    
{'loss': 5.1344, 'grad_norm': 2.1880388259887695, 'learning_rate': 1.395365289383812e-06, 'epoch': 0.57}                                                     
{'loss': 5.066, 'grad_norm': 2.1559624671936035, 'learning_rate': 1.3432073050985201e-06, 'epoch': 0.58}                                                     
{'loss': 4.988, 'grad_norm': 2.124544858932495, 'learning_rate': 1.2912403485599022e-06, 'epoch': 0.59}                                                      
{'loss': 4.9664, 'grad_norm': 2.0357367992401123, 'learning_rate': 1.2395277334996047e-06, 'epoch': 0.6}                                                     
{'loss': 4.9104, 'grad_norm': 2.1329517364501953, 'learning_rate': 1.1881324637733612e-06, 'epoch': 0.61}                                                    
{'loss': 4.837, 'grad_norm': 2.1627204418182373, 'learning_rate': 1.1371171566004986e-06, 'epoch': 0.62}                                                     
{'loss': 4.7998, 'grad_norm': 2.068380117416382, 'learning_rate': 1.0865439662745013e-06, 'epoch': 0.63}                                                     
{'loss': 4.7653, 'grad_norm': 2.088993549346924, 'learning_rate': 1.036474508437579e-06, 'epoch': 0.64}                                                      
{'loss': 4.722, 'grad_norm': 1.9503332376480103, 'learning_rate': 9.86969785011497e-07, 'epoch': 0.65}                                                       
{'loss': 4.666, 'grad_norm': 2.0606772899627686, 'learning_rate': 9.380901098761319e-07, 'epoch': 0.66}                                                      
{'loss': 4.6207, 'grad_norm': 1.952919840812683, 'learning_rate': 8.898950353863e-07, 'epoch': 0.67}                                                         
{'loss': 4.5728, 'grad_norm': 2.0125608444213867, 'learning_rate': 8.424432798163837e-07, 'epoch': 0.68}                                                     
{'loss': 4.5611, 'grad_norm': 1.925665020942688, 'learning_rate': 7.957926558211642e-07, 'epoch': 0.69}                                                      
{'loss': 4.506, 'grad_norm': 1.9059500694274902, 'learning_rate': 7.500000000000003e-07, 'epoch': 0.7}                                                       
{'loss': 4.4739, 'grad_norm': 1.912680745124817, 'learning_rate': 7.051211036501928e-07, 'epoch': 0.71}                                                      
{'loss': 4.4538, 'grad_norm': 1.9373120069503784, 'learning_rate': 6.6121064479388e-07, 'epoch': 0.72}                                                       
{'loss': 4.4134, 'grad_norm': 1.9064960479736328, 'learning_rate': 6.183221215612905e-07, 'epoch': 0.73}                                                     
{'loss': 4.4292, 'grad_norm': 1.8614637851715088, 'learning_rate': 5.765077870115125e-07, 'epoch': 0.74}                                                     
{'loss': 4.3831, 'grad_norm': 1.9622491598129272, 'learning_rate': 5.358185854701909e-07, 'epoch': 0.75}                                                     
{'loss': 4.3594, 'grad_norm': 1.8224776983261108, 'learning_rate': 4.963040904617131e-07, 'epoch': 0.76}                                                     
{'loss': 4.3315, 'grad_norm': 1.8888410329818726, 'learning_rate': 4.5801244431150397e-07, 'epoch': 0.77}                                                    
{'loss': 4.3178, 'grad_norm': 1.891813039779663, 'learning_rate': 4.2099029949202353e-07, 'epoch': 0.78}                                                     
{'loss': 4.309, 'grad_norm': 1.7807947397232056, 'learning_rate': 3.852827617839085e-07, 'epoch': 0.79}                                                      
{'loss': 4.2808, 'grad_norm': 1.693365216255188, 'learning_rate': 3.5093333532153313e-07, 'epoch': 0.8}  
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


