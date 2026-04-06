# Mini GPT (Character-Level Transformer from Scratch)

This project is a **from-scratch implementation of a GPT-style Transformer model** for character-level language modeling using PyTorch.

The model is trained on text data and learns to generate Shakespeare-like sequences using **causal self-attention**.

---

## Features

- Character-level tokenization  
- Full Transformer architecture:
  - Token + positional embeddings  
  - Masked (causal) self-attention  
  - Multi-head attention  
  - Feed-forward network  
  - Residual connections  
  - Layer normalization  
  - Dropout regularization  

### Training Pipeline

- Train / Validation / Test split  
- Mini-batch training  
- Periodic evaluation on validation set  
- Best model checkpoint saving  
- Loss tracking  

### Text Generation

- Temperature sampling  
- Top-k sampling  
- Context-aware sequence generation  

### System

- GPU support (CUDA)  
- Efficient batching  

### Artifact Saving

- Model checkpoint  
- Training metrics (JSON)  
- Generated text output  

---

## Model Architecture

The model follows a standard GPT-style Transformer:

- Embedding Layer (Token + Position)  
- Stacked Transformer Blocks:
  - Multi-head self-attention  
  - Feed-forward network  
  - Residual connections + LayerNorm  
- Final Linear Head → Vocabulary logits  

---

## Training Configuration

```python
block_size = 128
batch_size = 32
n_embd = 96
n_head = 4
n_layer = 3
dropout = 0.2
learning_rate = 3e-4
max_iters = 6000
```

## Results

| Metric                | Value    |
|:---------------------|:--------:|
| Best Validation Loss | `1.8154` |
| Test Loss            | `1.8594` |
| Device               | `CUDA`   |

## Model Capabilities

The model successfully learns:

- Sentence structure  
- Character-level word formation  
- Dialogue-style formatting  
- Punctuation patterns  
- Context-aware generation  

## Generation Settings

```python
temperature = 0.8
top_k = 20
```

## Project Structure

- **MiniGPT/**
  - **data/**
    - train.csv  
    - validation.csv  
    - test.csv  
  - **src/**
    - model.py  
    - dataset.py  
    - train.py  
  - **artifacts/**
    - best_model.pt  
    - training_results.json  
    - generated_text.txt  
  - README.md  

## Saved Artifacts

After training, the following files are generated:

- **best_model.pt** → Model weights, configuration, and vocabulary  
- **training_results.json** → Loss curves and evaluation metrics  
- **generated_text.txt** → Sample generated output  

## Training Workflow

1. Load dataset from CSV files  
2. Build vocabulary (`stoi` / `itos`)  
3. Train model using mini-batch sampling  
4. Evaluate periodically on validation set  
5. Save best model checkpoint  
6. Reload best model after training  
7. Evaluate on test set  
8. Generate sample text  
9. Save results  

## Requirements

Make sure:

- CSV files exist inside `data/`  
- PyTorch is installed  
- CUDA is available (optional but recommended)  

## Output

```
Bust your by: your may feight cours?

FORAMEONIUS:
Be his in upon my lady loveds.

BRINA:
And sus wome did that as lo courgfrows to towed to heam a ding
Who have these I not your I must ofes deaves,
What I hown ell to peack have well with in truthose anseland to befand.

YORK:
What I come my arm trauch I say thee of their en you blood thim.

DUCKESS WARD:
That thou stran as sills that godly weep to mone are
the thou dough old say the agair morety, are wark,
Then streitiong that and the lacke
Be
```

> Note: The generated text reflects early-stage character-level learning. While it captures structure and formatting, it may still contain nonsensical words.

## Notes

This project is built entirely from scratch to understand:

- Transformer internals  
- Self-attention mechanism  
- Token generation process  
- Training dynamics of language models  

## Future Improvements

- Top-p (nucleus) sampling  
- KV cache optimization  
- Mixed precision training  
- Larger context window  
- Fine-tuning on custom datasets  