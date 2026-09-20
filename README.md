# Transformer English–Vietnamese Translation

PyTorch implementation of the **Transformer architecture** for English–Vietnamese Neural Machine Translation, inspired by *Attention Is All You Need*.

## Overview

This project implements an end-to-end Transformer-based translation pipeline, including:

- Encoder–Decoder Transformer
- Multi-Head Self-Attention and Cross-Attention
- Sinusoidal Positional Encoding
- Padding and Causal Masking
- Teacher Forcing
- Greedy Decoding and Beam Search
- BLEU Evaluation
- Attention Visualization

## Dataset

**IWSLT 2015 English–Vietnamese**

- Train: 131,942 sentence pairs
- Validation: 1,551 sentence pairs
- Test: 1,261 sentence pairs
- Tokenization: SentencePiece BPE
- Vocabulary size: 8,000 per language

## Model

Baseline Transformer:

`d_model=128`, `num_heads=4`, `d_ff=512`, `encoder_layers=2`, `decoder_layers=2`, `dropout=0.1`

A larger configuration is also experimented with:

`d_model=256`, `num_heads=8`, `d_ff=1024`, `encoder_layers=4`, `decoder_layers=4`

## Project Structure

`src/data/` — Dataset, preprocessing, and tokenizer  
`src/model/` — Transformer components  
`src/training/` — Training pipeline  
`src/inference/` — Greedy decoding and Beam Search  
`src/evaluation/` — BLEU evaluation  
`scripts/` — Data preparation and experiment scripts  
`configs/` — Configuration files  
`notebooks/` — Attention visualization  
`tests/` — Unit tests

## Installation

### 1. Clone repository

    git clone https://github.com/TruongQuangPhat/transformer-en-vi-translation.git
    cd transformer-en-vi-translation

### 2. Install dependencies

Requires Python 3.11+ and `uv`.

    uv sync

## Data Preparation

### 1. Download dataset

    PYTHONPATH=src uv run python scripts/download_data.py

### 2. Extract dataset

    tar -xzf data/raw/train-en-vi.tgz -C data/raw
    tar -xzf data/raw/dev-2012-en-vi.tgz -C data/raw
    tar -xzf data/raw/test-2013-en-vi.tgz -C data/raw

### 3. Prepare dataset

    PYTHONPATH=src uv run python scripts/prepare_data.py

### 4. Train tokenizers

    PYTHONPATH=src uv run python scripts/tokenizer.py train

## Training

### Run training

Configure the experiment in `configs/base.yaml`, then run:

    PYTHONPATH=src uv run python scripts/train.py

Checkpoints are saved under the configured `training.checkpoint_dir`.

## Evaluation

### Run evaluation

Evaluate the trained model using Greedy Decoding and Beam Search:

    PYTHONPATH=src uv run python scripts/evaluate.py

## Inference

### Run translation

    PYTHONPATH=src uv run python -m src.inference.translate

## Attention Visualization

### Open notebook

Open:

`notebooks/attention_visualization.ipynb`

The notebook visualizes encoder self-attention, decoder self-attention, and decoder cross-attention.

## Example

**Input:**  
`I go to school every day.`

**Output:**  
`Tôi đi học mỗi ngày.`

## Tech Stack

Python · PyTorch · SentencePiece · SacreBLEU · PyYAML · CUDA

## Goal

The main goal is to understand and implement the Transformer architecture **end-to-end**, from data preprocessing and tokenization to training, inference, evaluation, and attention analysis.