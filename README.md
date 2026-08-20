# Reasoning LLM: From-Scratch Architecture & GRPO Training

This repository contains a full-stack, from-scratch implementation of a modern Large Language Model (LLM) tailored for mathematical reasoning. It features a custom Qwen-compatible architecture, a full Reinforcement Learning (RL) training pipeline using Group Relative Policy Optimization (GRPO), and an advanced multi-agent inference orchestration system.

## Key Features

### 1. From-Scratch Qwen-Compatible Architecture
Built entirely in PyTorch without relying on high-level `transformers` model wrappers.
- **Grouped Query Attention (GQA)** for efficient inference and reduced memory footprint.
- **Rotary Positional Embeddings (RoPE)** computed and cached efficiently.
- **RMSNorm** for robust and stable training dynamics.
- **KV Caching** with sequence-level state management.
- **Gradient Checkpointing** support for training at scale on consumer hardware.

### 2. DeepSeek-R1 Style RL Training (GRPO)
A custom Reinforcement Learning training loop designed to teach the model mathematical reasoning (aligning with approaches used in state-of-the-art models like DeepSeek-R1).
- **Group Relative Policy Optimization (GRPO)**: Implements custom surrogate objective clipping and KL divergence penalties against a frozen reference model.
- **Automated Math Verification**: A `MathVerifier` component that uses regex to extract `\boxed{}` outputs and scores them against mathematical ground truths.
- **Batched Rollout Generation**: Optimizes the rollout generation process for groups of prompts concurrently.

### 3. Supervisor-Worker Synthesis Loop
An advanced inference pipeline that goes beyond standard autoregressive generation.
- **Multi-Agent Orchestration**: Deploys "junior worker models" to generate initial attempts at solving complex prompts.
- **Expert Supervision**: Uses a "Supervisor AI" to review worker logic, identify mathematical or logical errors, and synthesize a single, flawless final response.

## Repository Structure

- `llm/base_model.py`: The core PyTorch implementation of the transformer architecture.
- `RL/rl_training_pipeline.py`: The GRPO training loop, verifiers, and rollout generators.
- `training_pipeline.py`: Inference utilities, including Top-P filtering, temperature scaling, and the `supervisor_synthesis_loop`.
- `tokenizer/`: Custom tokenizer implementations including BPE and GPT-4 style tokenization.
