
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
{'loss': 10.834, 'grad_norm': 2.238722085952759, 'learning_rate': 1.5000000000000002e-07, 'epoch': 0.01}                                                     
{'loss': 10.8177, 'grad_norm': 2.4133636951446533, 'learning_rate': 3.0000000000000004e-07, 'epoch': 0.02}                                                   
{'loss': 10.7637, 'grad_norm': 2.3414571285247803, 'learning_rate': 4.5e-07, 'epoch': 0.03}                                                                  
{'loss': 10.7779, 'grad_norm': 2.252777099609375, 'learning_rate': 6.000000000000001e-07, 'epoch': 0.04}                                                     
{'loss': 10.7034, 'grad_norm': 2.4122023582458496, 'learning_rate': 7.5e-07, 'epoch': 0.05}                                                                  
{'loss': 10.7424, 'grad_norm': 2.5384109020233154, 'learning_rate': 9e-07, 'epoch': 0.06}                                                                    
{'loss': 10.7082, 'grad_norm': 2.3507795333862305, 'learning_rate': 1.05e-06, 'epoch': 0.07}                                                                 
{'loss': 10.6217, 'grad_norm': 2.3378844261169434, 'learning_rate': 1.2000000000000002e-06, 'epoch': 0.08}                                                   
{'loss': 10.5894, 'grad_norm': 2.432021379470825, 'learning_rate': 1.35e-06, 'epoch': 0.09}                                                                  
{'loss': 10.5132, 'grad_norm': 2.4066507816314697, 'learning_rate': 1.5e-06, 'epoch': 0.1}                                                                   
{'loss': 10.4437, 'grad_norm': 2.24847149848938, 'learning_rate': 1.65e-06, 'epoch': 0.11}                                                                   
{'loss': 10.35, 'grad_norm': 2.3832197189331055, 'learning_rate': 1.8e-06, 'epoch': 0.12}                                                                    
{'loss': 10.2664, 'grad_norm': 2.4015133380889893, 'learning_rate': 1.95e-06, 'epoch': 0.13}                                                                 
{'loss': 10.2288, 'grad_norm': 2.3909976482391357, 'learning_rate': 2.1e-06, 'epoch': 0.14}                                                                  
{'loss': 10.1259, 'grad_norm': 2.3342721462249756, 'learning_rate': 2.25e-06, 'epoch': 0.15}                                                                 
{'loss': 10.0223, 'grad_norm': 2.3959274291992188, 'learning_rate': 2.4000000000000003e-06, 'epoch': 0.16}                                                   
{'loss': 9.8692, 'grad_norm': 2.3122596740722656, 'learning_rate': 2.55e-06, 'epoch': 0.17}                                                                  
{'loss': 9.7959, 'grad_norm': 2.449810743331909, 'learning_rate': 2.7e-06, 'epoch': 0.18}                                                                    
{'loss': 9.6745, 'grad_norm': 2.3988847732543945, 'learning_rate': 2.85e-06, 'epoch': 0.19}                                                                  
{'loss': 9.5581, 'grad_norm': 2.3718059062957764, 'learning_rate': 3e-06, 'epoch': 0.2}                                                                      
{'loss': 9.411, 'grad_norm': 2.273820638656616, 'learning_rate': 2.9988435543610844e-06, 'epoch': 0.21}                                                      
{'loss': 9.2674, 'grad_norm': 2.3601865768432617, 'learning_rate': 2.995376000599692e-06, 'epoch': 0.22}                                                     
{'loss': 9.1339, 'grad_norm': 2.495680332183838, 'learning_rate': 2.9896026854323896e-06, 'epoch': 0.23}                                                     
{'loss': 8.9958, 'grad_norm': 2.436441659927368, 'learning_rate': 2.981532510892707e-06, 'epoch': 0.24}                                                      
{'loss': 8.8489, 'grad_norm': 2.4098458290100098, 'learning_rate': 2.971177920604846e-06, 'epoch': 0.25}                                                     
{'loss': 8.7148, 'grad_norm': 2.388803720474243, 'learning_rate': 2.958554880596515e-06, 'epoch': 0.26}                                                      
{'loss': 8.5565, 'grad_norm': 2.393710136413574, 'learning_rate': 2.943682854680471e-06, 'epoch': 0.27}                                                      
{'loss': 8.4058, 'grad_norm': 2.4271113872528076, 'learning_rate': 2.9265847744427307e-06, 'epoch': 0.28}                                                    
{'loss': 8.3253, 'grad_norm': 2.2801449298858643, 'learning_rate': 2.9072870038837265e-06, 'epoch': 0.29}                                                    
{'loss': 8.191, 'grad_norm': 2.3202672004699707, 'learning_rate': 2.88581929876693e-06, 'epoch': 0.3}                                                        
{'loss': 8.0677, 'grad_norm': 2.381211757659912, 'learning_rate': 2.862214760737622e-06, 'epoch': 0.31}                                                      
{'loss': 7.9109, 'grad_norm': 2.421919584274292, 'learning_rate': 2.8365097862825516e-06, 'epoch': 0.32}                                                     
{'loss': 7.8158, 'grad_norm': 2.3335328102111816, 'learning_rate': 2.808744010609196e-06, 'epoch': 0.33}                                                     
{'loss': 7.6784, 'grad_norm': 2.3210060596466064, 'learning_rate': 2.7789602465311384e-06, 'epoch': 0.34}                                                    
{'loss': 7.5675, 'grad_norm': 2.366603374481201, 'learning_rate': 2.747204418453818e-06, 'epoch': 0.35}                                                      
{'loss': 7.4292, 'grad_norm': 2.3756611347198486, 'learning_rate': 2.713525491562421e-06, 'epoch': 0.36}                                                     
{'loss': 7.3244, 'grad_norm': 2.4357993602752686, 'learning_rate': 2.6779753963211175e-06, 'epoch': 0.37}                                                    
{'loss': 7.1609, 'grad_norm': 2.324061632156372, 'learning_rate': 2.6406089484000465e-06, 'epoch': 0.38}                                                     
{'loss': 7.054, 'grad_norm': 2.3617136478424072, 'learning_rate': 2.6014837641535285e-06, 'epoch': 0.39}                                                     
{'loss': 6.9302, 'grad_norm': 2.4158337116241455, 'learning_rate': 2.5606601717798212e-06, 'epoch': 0.4}                                                     
{'loss': 6.7852, 'grad_norm': 2.324383497238159, 'learning_rate': 2.518201118299413e-06, 'epoch': 0.41}                                                      
{'loss': 6.7297, 'grad_norm': 2.33034610748291, 'learning_rate': 2.4741720724952753e-06, 'epoch': 0.42}                                                      
{'loss': 6.6138, 'grad_norm': 2.239009141921997, 'learning_rate': 2.4286409239647513e-06, 'epoch': 0.43}                                                     
{'loss': 6.4912, 'grad_norm': 2.3171119689941406, 'learning_rate': 2.3816778784387097e-06, 'epoch': 0.44}                                                    
{'loss': 6.3676, 'grad_norm': 2.282411813735962, 'learning_rate': 2.3333553495294033e-06, 'epoch': 0.45}                                                     
{'loss': 6.2849, 'grad_norm': 2.401870012283325, 'learning_rate': 2.2837478470739234e-06, 'epoch': 0.46}                                                     
{'loss': 6.1279, 'grad_norm': 2.29665207862854, 'learning_rate': 2.2329318622454325e-06, 'epoch': 0.47}                                                      
{'loss': 6.0702, 'grad_norm': 2.3514158725738525, 'learning_rate': 2.18098574960932e-06, 'epoch': 0.48}                                                      
{'loss': 5.9725, 'grad_norm': 2.34625244140625, 'learning_rate': 2.1279896063061423e-06, 'epoch': 0.49}                                                      
{'loss': 5.8765, 'grad_norm': 2.2376201152801514, 'learning_rate': 2.074025148547635e-06, 'epoch': 0.5}                                                      
{'loss': 5.7854, 'grad_norm': 2.2050986289978027, 'learning_rate': 2.0191755856162397e-06, 'epoch': 0.51}                                                    
{'loss': 5.7207, 'grad_norm': 2.281298875808716, 'learning_rate': 1.963525491562421e-06, 'epoch': 0.52}                                                      
{'loss': 5.6086, 'grad_norm': 2.3122847080230713, 'learning_rate': 1.9071606747976113e-06, 'epoch': 0.53}                                                    
{'loss': 5.5515, 'grad_norm': 2.2325971126556396, 'learning_rate': 1.8501680457838584e-06, 'epoch': 0.54}                                                    
{'loss': 5.4358, 'grad_norm': 2.2275400161743164, 'learning_rate': 1.7926354830241926e-06, 'epoch': 0.55}                                                    
{'loss': 5.3658, 'grad_norm': 2.2378973960876465, 'learning_rate': 1.7346516975603465e-06, 'epoch': 0.56}                                                    
{'loss': 5.2925, 'grad_norm': 2.1916708946228027, 'learning_rate': 1.6763060961867566e-06, 'epoch': 0.57}                                                    
{'loss': 5.2113, 'grad_norm': 2.162752151489258, 'learning_rate': 1.6176886435917677e-06, 'epoch': 0.58}                                                     
{'loss': 5.1206, 'grad_norm': 2.1166279315948486, 'learning_rate': 1.558889723638603e-06, 'epoch': 0.59}                                                     
{'loss': 5.0877, 'grad_norm': 2.023332357406616, 'learning_rate': 1.5e-06, 'epoch': 0.6}                                                                     
{'loss': 5.0204, 'grad_norm': 2.1248841285705566, 'learning_rate': 1.4411102763613975e-06, 'epoch': 0.61}                                                    
{'loss': 4.936, 'grad_norm': 2.1618812084198, 'learning_rate': 1.3823113564082328e-06, 'epoch': 0.62}                                                        
{'loss': 4.8908, 'grad_norm': 2.0590665340423584, 'learning_rate': 1.3236939038132437e-06, 'epoch': 0.63}                                                    
{'loss': 4.8437, 'grad_norm': 2.0638678073883057, 'learning_rate': 1.2653483024396534e-06, 'epoch': 0.64}                                                    
{'loss': 4.7938, 'grad_norm': 1.927085518836975, 'learning_rate': 1.2073645169758077e-06, 'epoch': 0.65}                                                     
{'loss': 4.728, 'grad_norm': 2.049412727355957, 'learning_rate': 1.1498319542161423e-06, 'epoch': 0.66}                                                      
{'loss': 4.6748, 'grad_norm': 1.9203071594238281, 'learning_rate': 1.0928393252023888e-06, 'epoch': 0.67}                                                    
{'loss': 4.6157, 'grad_norm': 1.9847519397735596, 'learning_rate': 1.036474508437579e-06, 'epoch': 0.68}                                                     
{'loss': 4.5968, 'grad_norm': 1.8937674760818481, 'learning_rate': 9.808244143837602e-07, 'epoch': 0.69}                                                     
{'loss': 4.5354, 'grad_norm': 1.8750042915344238, 'learning_rate': 9.259748514523654e-07, 'epoch': 0.7}                                                      
{'loss': 4.4953, 'grad_norm': 1.8834378719329834, 'learning_rate': 8.720103936938583e-07, 'epoch': 0.71}                                                     
{'loss': 4.4696, 'grad_norm': 1.8890026807785034, 'learning_rate': 8.190142503906799e-07, 'epoch': 0.72}                                                     
{'loss': 4.4231, 'grad_norm': 1.8616472482681274, 'learning_rate': 7.67068137754568e-07, 'epoch': 0.73}                                                      
{'loss': 4.4337, 'grad_norm': 1.8007938861846924, 'learning_rate': 7.162521529260768e-07, 'epoch': 0.74}                                                     
{'loss': 4.3814, 'grad_norm': 1.9078774452209473, 'learning_rate': 6.666446504705971e-07, 'epoch': 0.75}                                                     
{'loss': 4.3538, 'grad_norm': 1.763141393661499, 'learning_rate': 6.183221215612905e-07, 'epoch': 0.76}                                                      
{'loss': 4.3216, 'grad_norm': 1.8303543329238892, 'learning_rate': 5.713590760352489e-07, 'epoch': 0.77}                                                     
{'loss': 4.3031, 'grad_norm': 1.8305877447128296, 'learning_rate': 5.258279275047247e-07, 'epoch': 0.78}                                                     
{'loss': 4.2893, 'grad_norm': 1.7061389684677124, 'learning_rate': 4.817988817005871e-07, 'epoch': 0.79}                                                     
{'loss': 4.2598, 'grad_norm': 1.6279412508010864, 'learning_rate': 4.3933982822017883e-07, 'epoch': 0.8}                                                     
{'loss': 4.2569, 'grad_norm': 1.7180936336517334, 'learning_rate': 3.98516235846472e-07, 'epoch': 0.81}                                                      
{'loss': 4.2396, 'grad_norm': 1.7024402618408203, 'learning_rate': 3.593910515999538e-07, 'epoch': 0.82}                                                     
{'loss': 4.2161, 'grad_norm': 1.6967344284057617, 'learning_rate': 3.220246036788829e-07, 'epoch': 0.83}                                                     
{'loss': 4.2178, 'grad_norm': 1.7467074394226074, 'learning_rate': 2.86474508437579e-07, 'epoch': 0.84}                                                      
{'loss': 4.1837, 'grad_norm': 1.698992133140564, 'learning_rate': 2.52795581546182e-07, 'epoch': 0.85}                                     
{'loss': 4.2101, 'grad_norm': 1.7976876497268677, 'learning_rate': 5.25e-07, 'epoch': 0.86}                                                                  
{'loss': 4.1152, 'grad_norm': 1.7288347482681274, 'learning_rate': 4.875e-07, 'epoch': 0.87}                                                                 
{'loss': 4.0688, 'grad_norm': 1.7016221284866333, 'learning_rate': 4.5e-07, 'epoch': 0.88}                                                                   
{'loss': 3.9598, 'grad_norm': 1.786277174949646, 'learning_rate': 4.125e-07, 'epoch': 0.89}                                                                  
{'loss': 3.8337, 'grad_norm': 1.7182263135910034, 'learning_rate': 3.75e-07, 'epoch': 0.9}                                                                   
{'loss': 3.7157, 'grad_norm': 1.7330933809280396, 'learning_rate': 3.375e-07, 'epoch': 0.91}                                                                 
{'loss': 3.6042, 'grad_norm': 1.6777046918869019, 'learning_rate': 3.0000000000000004e-07, 'epoch': 0.92}  
{'loss': 3.6059, 'grad_norm': 1.7517774105072021, 'learning_rate': 2.625e-07, 'epoch': 0.93}                                                                 
{'loss': 3.5789, 'grad_norm': 1.7398375272750854, 'learning_rate': 2.25e-07, 'epoch': 0.94}  
{'loss': 3.4756, 'grad_norm': 1.696990728378296, 'learning_rate': 1.875e-07, 'epoch': 0.95} 
{'loss': 3.3708, 'grad_norm': 1.6934484243392944, 'learning_rate': 1.5000000000000002e-07, 'epoch': 0.96} 
{'loss': 3.2591, 'grad_norm': 1.703162670135498, 'learning_rate': 1.125e-07, 'epoch': 0.97} 
{'loss': 3.1905, 'grad_norm': 1.7243691682815552, 'learning_rate': 9e-08, 'epoch': 0.98}                                                                     
{'loss': 3.1063, 'grad_norm': 1.712477684020996, 'learning_rate': 6.000000000000001e-08, 'epoch': 0.99}
{'loss': 2.9819, 'grad_norm': 1.6443889141082764, 'learning_rate': 3.0000000000000004e-08, 'epoch': 1.0}  
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [6:25:52<00:00,  1.04it/s]
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


