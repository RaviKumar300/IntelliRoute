# IntelliRoute

## Project Description

IntelliRoute aims to build a system similar to Perplexity.ai with the following capabilities:

- Operates in a chat environment
- Classifies user queries into LLM Arena ranking categories
- Dynamically selects the best model for generating responses
- Maintains a dynamic context window shared by multiple LLMs for cross-model continuous communication
- Processes each message while preserving relevant context
- Supports early-stage multimodal extensions (e.g., image-based prompts)

This repository provides a proof-of-concept implementation, starting with query classification, context window design, and model routing.

---

## Project Components

### 1. Query Classification Model

Models evaluated:

- `meta-llama/Llama-3.2-1B`
- `microsoft/phi-2`
- `TinyLLama/TinyLlama-1.1B-Chat-v1.0`
- `mistralai/Mistral-7B-v0.1`

Based on performance and training feasibility on a Kaggle GPU (P100), the Llama-3.2-1B model was chosen and fine-tuned using LoRA for four-way classification: coding, conversation, math, and summary.

### 2. Context Window Maintenance

- Each chat session maintains a shared history in a `cxt.json` file
- Aims to minimize token usage while maximizing retention of important information
- Before generating a response, the system:
  - Loads the full conversation history
  - Converts it into `[INST]` chat-style prompts
  - Truncates older turns if the tokenized prompt exceeds the target model’s context window (e.g., 4096 for most)
  - Appends the current query and awaits model completion
- After generation:
  - Response is postprocessed to remove stopwords (using NLTK)
  - The cleaned message is appended to the session context
  - The updated context is saved for the next turn

This mechanism allows different models to respond to user queries while retaining global context across the session, enabling continuity even when switching models mid-conversation.

### 3. Multi-Model Pipeline

Specialized models used:

| Category    | Model Name                                | Source HF ID                              |
|-------------|--------------------------------------------|--------------------------------------------|
| Coding      | CodeLlama-7B-Instruct                      | `codellama/CodeLlama-7b-Instruct-hf`       |
| Summary     | LLaMA 3.1 8B Instruct                      | `meta-llama/Meta-Llama-3-8B-Instruct`      |
| Chat        | Mistral 7B Instruct v0.2                   | `mistralai/Mistral-7B-Instruct-v0.2`       |
| Math        | Llemma 7B                                  | `EleutherAI/llemma_7b`                     |

The query classification model selects the appropriate backend model for inference. This allows efficient routing of prompts to expert models.

### 4. Multimodal Integration (WIP)

- Initial support for visual input has been scaffolded
- Future implementation plans include:
  - Image encoder integration (e.g., CLIP, BLIP, LLaVA)
  - Caption classification and image-query fusion
  - Routing visual tasks to vision-language models

---

## Part 1: Query Classification

### Summary of Performance

| Model        | Accuracy | Perplexity | Notes                              |
|--------------|----------|------------|------------------------------------|
| Base Model   | 0%       | 1710.39    | Failed to classify queries         |
| Tuned Model  | 77%      | 2.15       | Successfully learned classification |

Confusion matrix visualizations are saved in the `images/` directory.

---

## Pipeline Overview

The pipeline involves fine-tuning `Llama-3.2-1B` using LoRA (Low-Rank Adaptation) for classifying queries into four categories: coding, conversation, math, and summary.

### Step-by-Step Breakdown

#### 1. Data Preparation

- Loaded datasets for each category (coding, conversation, math, summary)
- Merged into a single DataFrame with columns: `query` and `class`
- Balanced the dataset to approximately 4,448 samples per class
- Saved the dataset in JSONL format for fine-tuning

#### 2. Prompt Formatting

Each sample was transformed into an instruction-response format:

Instruction: classify the following query into one of the following types: conversation, code, math, summary
query: {query}
Response: {class}



This format is suitable for instruction-tuned language models.

#### 3. Model Setup

- Loaded `Llama-3.2-1B` and tokenizer from Hugging Face
- Applied LoRA adapters to `q_proj` and `v_proj` layers using PEFT
- Wrapped the model with LoRA for efficient training

#### 4. Tokenization and Training

- Applied tokenization with padding, truncation, and max length 256
- Training configuration:
  - Small batch size
  - Gradient accumulation
  - 10 epochs

#### 5. Saving and Loading

- Saved the tuned model and tokenizer
- For evaluation, loaded both the base model and LoRA-merged model in inference mode

#### 6. Evaluation: Perplexity

- Base Model: 1710.39 (very high; indicates failure)
- Tuned Model: 2.15 (indicates the model learned the task well)

#### 7. Evaluation: Classification Accuracy

- Ran 100 test queries through both models
- Base Model:
  - Accuracy: 0%
  - F1-score: 0
- Tuned Model:
  - Accuracy: 77%
  - High precision/recall for "coding" and "conversation"
  - Moderate performance for "math"
  - Lower recall for "summary"

---

## Part 2: Chat Routing and Context-Aware Response Generation

### Objective

This component builds a conversational loop that dynamically routes user queries to the best-performing language model based on predicted category, while preserving conversation context across all models.

### Workflow Overview

1. **User Input**: Accept a query from the user via command-line or future UI.
2. **Query Classification**: 
   - The query is passed to the fine-tuned `Llama-3.2-1B` classifier.
   - Output label is one of: `coding`, `chat`, `summary`, or `math`.
3. **Model Selection**:
   - Based on the label, the system selects the appropriate expert model:
     - Coding → CodeLlama
     - Chat → Mistral
     - Summary → LLaMA 3.1
     - Math → Llemma
4. **Context Construction**:
   - A `cxt.json` file stores previous user and model messages as structured history.
   - Before generating a response, the system rebuilds a full prompt from this history.
   - If the combined prompt exceeds the model’s token limit (e.g., 4096 tokens), oldest messages are truncated.
5. **Response Generation**:
   - The chosen model is called with the constructed context and user query.
   - A response is generated and printed to the user.
6. **Postprocessing**:
   - Stopwords are removed using NLTK for cleaner storage.
   - The cleaned response is appended to `cxt.json` for continuity.
7. **Loop**:
   - The session continues until the user exits with a quit command.

### Context Format

The context file `cxt.json` is an array of dialog turns:

```json
[
  {
    "user": "What is a gradient descent?",
    "assistant": "Gradient descent is an optimization algorithm..."
  },
  {
    "user": "How do I write it in Python?",
    "assistant": "Here's an example implementation using NumPy..."
  }
]
```


## Conclusion

This repository demonstrates the feasibility of:

- Efficiently classifying user queries using lightweight fine-tuning (LoRA)
- Selecting suitable LLMs dynamically for response generation
- Building context-aware conversational agents that maintain continuity across models
- Extending to multimodal interaction via planned visual input support

---

## Next Steps

- Develop a user-facing frontend interface
- Expand multimodal support (image captioning, OCR, audio-to-text)
- Optimize memory and inference using quantization (e.g., 4-bit, GGUF)
