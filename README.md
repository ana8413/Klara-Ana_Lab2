# Fine-Tuning Llama-3.2-3B-Instruct for Enhanced Conversational AI

This project focuses on fine-tuning the **`unsloth/Llama-3.2-3B-Instruct`** model with the Unsloth library. It uses LoRA and QLoRA to improve the model’s ability to follow instructions and hold conversations. Also, context was given to the LLM in order to increase conversation capability. The datasets used where finetome100k and wizardlm70k. The main objective was to optimize speed and efficiency for chatbot applications. 


## Setup and Environment

The fine-tuning was performed on a free Tesla T4 Google Colab instance using the Unsloth framework for acceleration and reduced memory footprint.

### Dependencies

The following core libraries were installed and used:

```bash
!pip install unsloth
!pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git@nightly git+https://github.com/unslothai/unsloth-zoo.git
```

### Model Configuration

| Parameter | Value | Explanation |
| :--- | :--- | :--- |
| **Base Model** | `unsloth/Llama-3.2-3B-Instruct` | This base model was selected for its speed and efficiency. It is great balance between 1B and 3B models, suitable for simple chatbot tasks. |
| **`max_seq_length`** | 2048 | The maximum length of the input sequence was set to 2048 for the model for the model to handle longer conversations and have a better reasoning. |
| **`dtype`** | `None` (auto-detected) | Automatically set to Float16 for Tesla T4. |
| **`load_in_4bit`** | `True` | Uses 4-bit quantization (QLoRA) to significantly reduce VRAM usage. |

## Data Preparation

### Datasets Used

The model was trained on a mixed dataset to enhance both general instruction-following and conversational ability:

1.  **`mlabonne/FineTome-100k`**: 15,000 samples taken for general fine-tuning in ShareGPT format.
2.  **`WizardLMTeam/WizardLM_evol_instruct_70k`**: 15,000 samples specifically chosen to increase the model's conversational capability.

Furthermore, the mixed dataset was divided into a 95% training and 5% test. 
*The testing percentage is low to use less memory since Google Colab has limited credits for the usage of TeslaT4.*

### Formatting

The datasets were standardized to the Llama-3.1-Instruct chat template:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 July 2024

<|start_header_id|>user<|end_header_id|>

[USER_INPUT]<|eot_id|><|start_header_id|>assistant<|end_header_id|>

[MODEL_RESPONSE]<|eot_id|>
```

  * **`unsloth.chat_templates.standardize_sharegpt`**  converted the source data format.
  * **`train_on_responses_only`** was applied to ensure the model only calculated the loss on the assistant's output, and not on the user's input. This made training more efficient.



## ⚙️ Fine-Tuning with SFTTrainer

The training was performed using Hugging Face TRL's `SFTTrainer` with modified hyperparameters to optimize performance.

| Hyperparameter | Value | Explanation |
| :--- | :--- | :--- |
| **`per_device_train_batch_size`** | 4 | The number of samples processed per GPU in a single forward/backward pass. To speed up training, it was increased from 2 to 4. |
| **`gradient_accumulation_steps`** | 4 | The number of forward/backward passes before an optimization step. Set to achieve an batch size of 16 (4 \* 4) given the 16 GB VRAM constraint on the Tesla T4. |
| **`max_steps`** | 40 | Reduced from 60 to prevent overfitting, preserve the model's general abilities, and allow a faster iteration loop. |
| **`learning_rate`** | $2 \times 10^{-5}$ | Decreased from the common default of $1 \times 10^{-4}$ to avoid instability and overfitting, especially in the early training steps. |
| **`warmup_steps`** | 50 | The number of steps over which the learning rate increases from 0. It was increased to prevent divergence and "hallucinations" early in training by allowing the model to adapt smoothly. |
| **`optim`** | `"adamw_8bit"` | An optimizer that keeps the full state and quantizes it, offering low precision but significant memory savings. |
| **`weight_decay`** | 0.01 | Coefficient for L2 regularization, helps prevent the model from memorizing training data during fine-tuning. This value will help the model to prevent overfitting, improve stability during training, and improve generalization to long sequences|
| **`lr_scheduler_type`** | `"linear"` | Defines the learning rate schedule, linearly decreasing the rate from the initial value down to zero over training steps. This will help the model to  converge smoothly without oscillations, reduce forgetting, and stabilizes training.|
| **`logging_steps`** | 10 | The interval at which training loss is reported. Incremented from 1 to track loss without "spamming" the console output. |
| **`eval_strategy`** | `"steps"` | The model is evaluated every `logging_steps` (10 steps) to continuously track its performance on a small test set.   |

### Training Results

  * **Total Training Time**: 49.58 minutes (for 40 steps)
  * **Final Training Loss**: 0.8644
  * **Final Validation Loss**: 0.7828
  * **Peak Reserved Memory**: 7.967 GB (54.047% of max memory)

-----

## Evaluation Graphs


### Loss Curves

  * **Graph 1: Training and Validation Loss Over Steps**
      
  * **Graph 2: Learning Rate Schedule**
      

### Qualitative Evaluation

  * **Table: Comparison of Finetuned vs. Base Model Responses**
     


## Inference and Deployment (Web App)

The trained LoRA adapters were integrated with the base model and deployed as a conversational web application using Gradio. This deployment focuses on demonstrating interactive chat capabilities.

### Deployment Details

  * **Web Framework**: Gradio
  * **Model Loading**: The model is explicitly configured to run on the CPU (`device = "cpu"`) for broader compatibility, even if it results in slower inference.
      * The *Base Model* (`unsloth/Llama-3.2-3B-Instruct`) is loaded into `torch.float32` and mapped directly to the CPU.
      * The *LoRA adapter* (`Ana8413/model_02dec_lora`) is loaded and moved to the CPU.

### Adjustable Inference Hyperparameters in the User Interface

The Gradio interface enables the user to adjust: *Temperature, Top-p (Nucleus Sampling), and Max New Tokens*  for real-time model behavior tuning.

### Launch Command

The Gradio interface is launched with the command:

```python
# The final line of the app.py script
.launch()
```
