# Reasoning LLM: From-Scratch Architecture, Alignment & Reasoning Pipeline

A comprehensive, production-grade implementation of a modern Large Language Model (LLM) tailored for mathematical reasoning, post-training alignment, and inference orchestration. Built natively in **PyTorch** from the ground up without high-level abstraction wrappers, this repository replicates modern architecture standards (Qwen-style GQA, RoPE, SwiGLU) and advanced post-training methodologies including **DeepSeek-R1-style Group Relative Policy Optimization (GRPO)**, **Direct Preference Optimization (DPO)**, **custom LoRA fine-tuning**, and **multi-agent reasoning orchestration**.

---

## Architecture & Lifecycle Overview

```mermaid
flowchart TD
    subgraph Data & Tokenization
        RAW[Raw Text / Streamed Wikipedia] --> D_PRE[StreamingPretrainDataset]
        ALPACA[Instruction SFT Data] --> D_SFT[Qwen3FineTuningDataset]
        MATH_BENCH[MATH / GSM8k Benchmarks] --> D_RL[MathDataset]
        TOK_JSON[tokenizer.json / 151,936 Vocab] --> QTOK[Qwen3Tokenizer]
        QTOK --> D_PRE
        QTOK --> D_SFT
        QTOK --> D_RL
    end

    subgraph Core Architecture
        CFG[llm/config.json / Qwen3Config] --> MODEL[Qween3Model]
        MODEL --> GQA[Grouped Query Attention (16 Q / 8 KV Heads)]
        MODEL --> ROPE[Rotary Embeddings (RoPE, theta=1M)]
        MODEL --> SWIGLU[SwiGLU FeedForward Network]
        MODEL --> RMS[RMSNorm with FP32 Precision Guard]
        MODEL --> KV[KVCache & Gradient Checkpointing]
    end

    subgraph Training Pipeline
        D_PRE --> PRETRAIN[llm/pretrain/pretrain_from_scratch.py]
        PRETRAIN --> BASE_CKPT[(Base Model Weights)]
        
        BASE_CKPT --> CONT_PRE[llm/pretrain/continuous_pretraining.py]
        BASE_CKPT --> LORA[llm/finetune/lora.py - LoRA Injection]
        LORA --> SFT[llm/finetune/conversational_assistant.py]
        D_SFT --> SFT
        SFT --> SFT_CKPT[(SFT LoRA Adapters)]
        
        SFT_CKPT --> DPO[llm/rl/dpo_trainer.py - DPO Preference Tuning]
        BASE_CKPT --> GRPO[RL/rl_training_pipeline.py - GRPO RL]
        D_RL --> GRPO
    end

    subgraph Inference & Orchestration
        GRPO --> ENGINE[Qwen3InferenceEngine - Streaming KV Cache]
        SFT_CKPT --> ENGINE
        ENGINE --> CRITIQUE[llm/rl/reasoning.py - Self-Critique Loop]
        ENGINE --> SUPERVISOR[training_pipeline.py - Supervisor-Worker Synthesis]
    end
```

---

## Key Core Implementations

### 1. From-Scratch Qwen-Compatible Architecture ([`llm/base_model.py`](llm/base_model.py))
Built in pure PyTorch conforming to modern high-performance transformer architectures:
*   **Grouped Query Attention (GQA)**: 16 Query heads paired with 8 Key/Value heads (`group_size = 2`). Implements asymmetric projections where $W_q: 1024 \to 2048$ ($16 \times 128$) and $W_k, W_v: 1024 \to 1024$ ($8 \times 128$), slashing KV-cache memory usage by 50%.
*   **Rotary Positional Embeddings (RoPE)**: Half-split vector rotation with high-base frequency ($\theta = 1,000,000$) for extended context lengths (up to 40,960 tokens), stored in non-persistent GPU buffers.
*   **SwiGLU Activations**: Modern gated MLP blocks:
    $$\text{FFN}(x) = (\text{SiLU}(x W_{\text{gate}}) \odot x W_{\text{up}}) W_{\text{down}}$$
*   **RMSNorm**: Root Mean Square layer normalization with `qwen3_compatible` FP32 arithmetic guarding against low-precision underflow/overflow.
*   **KV Caching & Gradient Checkpointing**: Dynamic sequence-level cache (`KVCache`) paired with non-reentrant PyTorch checkpointing to enable training larger models on consumer GPUs.

### 2. Pretrained Safetensors Ingestion ([`llm/pretrained_weight_loader.py`](llm/pretrained_weight_loader.py))
*   **Zero-Wrapper Direct Loading**: Automatically downloads official Hugging Face repositories (such as `Qwen/Qwen3-0.6B` or Qwen2.5 checkpoints) via `snapshot_download`.
*   **In-Place Parameter Mapping**: Converts Hugging Face parameter naming conventions (`model.layers.{i}.self_attn.q_proj`, `mlp.gate_proj`, etc.) directly into the scratch model's tensor layout with automated tensor shape validation and parameter count verification.

### 3. Dual Tokenization Suite ([`tokenizer/`](tokenizer/))
*   **Production Tokenizer ([`tokenizer/qween_3_tokenizer.py`](tokenizer/qween_3_tokenizer.py))**:
    *   Integrates the 151,936-token Hugging Face BPE model ([`tokenizer.json`](tokenizer.json)).
    *   Full support for **ChatML** formatting (`<|im_start|>user\n...<|im_end|>`) and native `<think>...</think>` reasoning delimiters.
*   **From-Scratch Educational BPE Engine ([`tokenizer/base.py`](tokenizer/base.py), [`tokenizer/basic.py`](tokenizer/basic.py), [`tokenizer/regex.py`](tokenizer/regex.py), [`tokenizer/gpt4.py`](tokenizer/gpt4.py))**:
    *   Inspired by Andrej Karpathy's `minbpe`.
    *   Implements byte-level pair counting, vocabulary merging, regex-based pre-tokenization with GPT-4 split patterns, and `.model`/`.vocab` serialization.
    *   Reconstructs OpenAI's `cl100k_base` merge tables and byte-shuffling transformations.

### 4. End-to-End Pre-training Pipeline ([`llm/pretrain/`](llm/pretrain/))
*   **Streaming Wikipedia Dataset ([`llm/pretrain/massive_pretrainer_dataset.py`](llm/pretrain/massive_pretrainer_dataset.py))**: PyTorch `IterableDataset` with multi-worker stream sharding and a continuous token-packing buffer for zero-pad-waste pretraining.
*   **From-Scratch Pretrainer ([`llm/pretrain/pretrain_from_scratch.py`](llm/pretrain/pretrain_from_scratch.py))**: Mixed-precision (`bfloat16` AMP) trainer with AdamW, linear warmup + cosine decay scheduling, automated checkpoint saving, and crash recovery resumption.
*   **Continuous Domain Adaptation ([`llm/pretrain/continuous_pretraining.py`](llm/pretrain/continuous_pretraining.py))**: Sliding-window dataset chunker (`ContinuousTextDataset`) for continual pretraining on raw text corpora with real-time validation tracking and text generation probes.

### 5. Parameter-Efficient Fine-Tuning (LoRA & SFT) ([`llm/finetune/`](llm/finetune/))
*   **Native LoRA Module ([`llm/finetune/lora.py`](llm/finetune/lora.py))**: Custom `LoRALinear` implementing low-rank factor updates:
    $$W' = W + \frac{\alpha}{r} (B \cdot A)$$
    with Kaiming uniform initialization on $A$ and zero initialization on $B$. Dynamically replaces attention and MLP projections (`w_q`, `w_k`, `w_v`, `out_proj`, `fc1`, `fc2`, `fc3`) while freezing base weights.
*   **Instruction Dataset with Prompt Masking ([`llm/finetune/dataset.py`](llm/finetune/dataset.py))**: Formats Alpaca-style instruction data with `-100` target masking on prompts, calculating cross-entropy loss exclusively over the assistant's completions.
*   **Conversational SFT Trainer ([`llm/finetune/conversational_assistant.py`](llm/finetune/conversational_assistant.py))**: Complete fine-tuning pipeline featuring gradient accumulation, gradient norm clipping, evaluation splits, and LoRA adapter checkpoint extraction.

### 6. Mathematical Reasoning via GRPO ([`RL/rl_training_pipeline.py`](RL/rl_training_pipeline.py))
Implements **Group Relative Policy Optimization (GRPO)** as popularized by DeepSeekMath and DeepSeek-R1:
*   **Critic-Free Advantage Estimation**: Generates a group of $G$ candidate rollouts per prompt and computes relative advantages across the group:
    $$A_i = \frac{r_i - \text{mean}(\{r_1, \dots, r_G\})}{\text{std}(\{r_1, \dots, r_G\}) + \epsilon}$$
    Eliminates the memory and compute overhead of maintaining a separate Value/Critic model.
*   **Automated Rule-Based Math Verifier ([`MathVerifier`](RL/rl_training_pipeline.py#L112))**: Extracts LaTeX boxed answers (`\boxed{...}`) using regular expressions and matches them directly against mathematical ground truth.
*   **Clipped Surrogate Objective with KL Penalty**: Regularizes policy drift against a frozen reference policy using reverse KL divergence.

### 7. Direct Preference Optimization (DPO) ([`llm/rl/dpo_trainer.py`](llm/rl/dpo_trainer.py))
*   Provides an alternative alignment paradigm by directly optimizing policy log-ratios on pairwise preference data (`chosen` vs. `rejected`) using the Bradley-Terry preference objective:
    $$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$
*   Pairs the custom base model with trainable LoRA adapters while keeping a reference model frozen.

### 8. Multi-Agent Orchestration & Reasoning Inference ([`training_pipeline.py`](training_pipeline.py), [`llm/rl/reasoning.py`](llm/rl/reasoning.py))
*   **Supervisor-Worker Synthesis ([`training_pipeline.py`](training_pipeline.py#L88))**: Ensembles multiple junior worker model rollouts, passes candidate answers through automated scoring functions, and supplies the collective findings into an expert Supervisor model for synthesis and logical error correction.
*   **Conversational Self-Refinement ([`llm/rl/reasoning.py`](llm/rl/reasoning.py#L7))**: An iterative test-time reasoning loop (Draft $\to$ Critique $\to$ Revise).
*   **Reasoning Probabilistic Metrics ([`llm/rl/reasoning.py`](llm/rl/reasoning.py#L82))**: Evaluates per-token probability distributions, joint log-likelihoods, and candidate completion confidence.

---

## Repository Structure

```text
├── README.md                                # Project documentation
├── tokenizer.json                           # 151,936-token vocabulary file (Qwen2.5/Qwen3 BPE)
├── training_pipeline.py                     # Multi-agent Supervisor-Worker synthesis & inference loop
│
├── llm/                                     # Core LLM implementations
│   ├── config.json                          # Qwen3-0.6B architecture specifications
│   ├── base_model.py                        # Scratch PyTorch model (GQA, RoPE, SwiGLU, RMSNorm, KVCache)
│   ├── pretrained_weight_loader.py          # Safetensor weight downloader & mapper
│   ├── inference.py                         # Streaming KV-cached inference engine (Top-P, Temp, Rep. Penalty)
│   │
│   ├── pretrain/                            # Pre-training subsystem
│   │   ├── massive_pretrainer_dataset.py    # Streaming Wikipedia IterableDataset with token packing
│   │   ├── pretrain_from_scratch.py         # Full pre-training training loop (AMP, Cosine LR, Resumption)
│   │   ├── continuous_text_dataset.py       # Sliding-window next-token dataset chunker
│   │   ├── continuous_pretraining.py        # Domain-adaptation continual pretrainer
│   │   └── inference.py                     # Interactive REPL console with top-k sampling
│   │
│   ├── finetune/                            # Parameter-Efficient Fine-Tuning (PEFT)
│   │   ├── lora.py                          # Scratch LoRA module & dynamic layer injection
│   │   ├── dataset.py                       # Alpaca instruction formatting with prompt loss masking (-100)
│   │   ├── basic_sft.py                     # Minimal full-parameter SFT loop
│   │   └── conversational_assistant.py      # Production LoRA SFT trainer (Warmup, Cosine, AMP, Checkpoints)
│   │
│   └── rl/                                  # Alignment & Reasoning loops
│       ├── dpo_trainer.py                   # Direct Preference Optimization with LoRA policy
│       └── reasoning.py                     # Self-critique refinement loop & joint log-prob metrics
│
├── RL/                                      # DeepSeek-R1 Style Reinforcement Learning
│   └── rl_training_pipeline.py              # GRPO training loop, MathVerifier, rollout generator
│
└── tokenizer/                               # Tokenization library
    ├── __init__.py
    ├── qween_3_tokenizer.py                 # Fast Tokenizers wrapper (ChatML, <think> tags)
    ├── base.py                              # Educational BPE base class & file serializers
    ├── basic.py                             # Byte-level BPE learner
    ├── regex.py                             # Regex-guided BPE splitter (GPT-4 regex)
    ├── gpt4.py                              # tiktoken cl100k_base parser & byte un-shuffler
    └── train.py                             # Example training script for custom BPE vocabularies
```

---

## Quickstart Guide

### 1. Ingest Pretrained Weights
Download and map Hugging Face safetensors directly into the scratch architecture:
```python
from llm.pretrained_weight_loader import PretrainedQweenModel

# Downloads weights from Hugging Face and verifies parameter shapes
model, state_dict = PretrainedQweenModel.from_pretrained(
    config_path="llm/config.json",
    weight_path="llm/model.safetensors"
)
model.visualize_custom_parameters()
```

### 2. Stream Generation with KV-Cache
```python
from llm.pretrained_weight_loader import PretrainedQweenModel
from llm.inference import Qwen3InferenceEngine
from tokenizer.qween_3_tokenizer import Qwen3Tokenizer

tokenizer = Qwen3Tokenizer("tokenizer.json")
model, _ = PretrainedQweenModel.from_pretrained("llm/config.json", "llm/model.safetensors")

engine = Qwen3InferenceEngine(model=model, tokenizer=tokenizer)
engine.stream_to_console(
    prompt="Explain why prime numbers are infinite:",
    max_new_tokens=150,
    temperature=0.7,
    top_p=0.9
)
```

### 3. Fine-Tune with From-Scratch LoRA
```python
from llm.pretrained_weight_loader import PretrainedQweenModel
from llm.finetune.lora import inject_lora, freeze_base_model, print_trainable_parameters

model, _ = PretrainedQweenModel.from_pretrained("llm/config.json", "llm/model.safetensors")

# Inject LoRA into QKV attention projections and SwiGLU MLP layers
model = inject_lora(model, rank=8, alpha=16, target_modules=["w_q", "w_k", "w_v", "out_proj", "fc1", "fc2", "fc3"])
freeze_base_model(model)
print_trainable_parameters(model)
```

### 4. Run DeepSeek-R1 Style GRPO Training
```python
from RL.rl_training_pipeline import RLTrainer, MathDataset, Config
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

cfg = Config(
    rollout_pool_size=32,
    group_size=4,
    minibatches=4,
    lr=1e-5,
    kl_beta=0.04
)

dataset = MathDataset()
trainer = RLTrainer(model=model, tokenizer=tokenizer, dataset=dataset, cfg=cfg)
trainer.train(total_steps=500)
```

---

## Technical Specifications

| Component | Specification |
| :--- | :--- |
| **Model Type** | Qwen-Compatible Causal LM (`Qwen3ForCausalLM`) |
| **Parameters** | ~600M (28 Layers, $d_{\text{model}} = 1024$, $d_{\text{ffn}} = 3072$) |
| **Attention Scheme** | Grouped Query Attention (16 Query Heads, 8 KV Heads, $d_{\text{head}} = 128$) |
| **Context Length** | Up to 40,960 tokens ($\theta = 1,000,000$) |
| **Vocabulary Size** | 151,936 tokens (Byte-Pair Encoding) |
| **Alignment Methods** | LoRA SFT, DPO (Bradley-Terry), GRPO (Group-relative advantage) |
| **Hardware Targets** | Single consumer GPU (CUDA) or CPU via automatic device fallback |
