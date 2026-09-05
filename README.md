# Transformer From Scratch

A from-scratch PyTorch implementation of the original Transformer architecture ("Attention Is All You Need"), trained as a machine translation model. Built for learning purposes — every component (embeddings, positional encoding, multi-head attention, encoder/decoder stacks, projection layer) is implemented manually rather than using `nn.Transformer`.

## Project Structure

```
.
├── config.py     # Hyperparameters and path helpers
├── dataset.py    # Bilingual dataset class + causal mask
├── model.py      # Transformer architecture (encoder, decoder, attention, etc.)
├── train.py      # Training loop, tokenizer building, validation
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Uses the [`opus_books`](https://huggingface.co/datasets/opus_books) dataset via Hugging Face `datasets`, translating between the language pair configured in `config.py` (default: English → Italian). The dataset is downloaded automatically on first run.

## Configuration

All hyperparameters live in `config.py`:

| Key | Description | Default |
|---|---|---|
| `batch_size` | Training batch size | 8 |
| `num_epochs` | Number of training epochs | 20 |
| `lr` | Learning rate | 1e-4 |
| `seq_len` | Max sequence length | 350 |
| `d_model` | Embedding/model dimension | 512 |
| `lang_src` / `lang_tgt` | Source/target language codes | en / it |
| `model_folder` | Directory for saved checkpoints | weights |
| `preload` | Epoch checkpoint to resume from (or `None`) | None |
| `tokenizer_file` | Tokenizer save path template | tokenizer_{0}.json |
| `experiment_name` | TensorBoard log directory | runs/tmodel |

## Training

```bash
python train.py
```

This will:
1. Download the dataset and train (or load) WordLevel tokenizers for each language.
2. Build the Transformer model (`build_transformer` in `model.py`).
3. Train for the configured number of epochs, logging loss to TensorBoard.
4. Run greedy-decode validation after each epoch (prints example translations, logs CER/WER/BLEU).
5. Save a checkpoint (`model_state_dict`, `optimizer_state_dict`, `epoch`, `global_step`) after every epoch.

View training curves with:
```bash
tensorboard --logdir runs/tmodel
```

## Known Issues / Things to Fix

This is a learning project and currently has a few bugs worth fixing before serious training runs:

1. **Checkpoint path bug (`config.py`)** — `get_weigths_file_path()` references `config['model_basename']`, which doesn't exist in the config dict (`model_filename` is defined instead). This raises a `KeyError` on save/load. Fix by making the key names consistent.

2. **LayerNorm math bug (`model.py`)** — `LayerNormalization.forward` has an operator-precedence bug:
   ```python
   return self.alpha * x-mean/std + self.eps +self.bias
   ```
   This does **not** compute `alpha * (x - mean) / (std + eps) + bias`. It needs explicit parentheses:
   ```python
   return self.alpha * (x - mean) / (std + self.eps) + self.bias
   ```

3. **Decoder block reuses one residual connection (`model.py`)** — `DecoaderBlock.forward` applies `self.residual_connection[0]` to all three sublayers (self-attention, cross-attention, feed-forward) instead of `[0]`, `[1]`, `[2]` respectively. This means two of the three residual/dropout modules are never actually used.

4. **Missing `map_location` on checkpoint load (`train.py`)** — `torch.load(model_filename)` should include `map_location=device` to safely resume training across different devices (e.g., GPU-trained checkpoint loaded on a CPU-only machine).

5. **Loss `ignore_index` uses the wrong tokenizer (`train.py`)** — `nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'))` should use `tokenizer_tgt`, since the loss and labels are computed entirely in the target language. This currently works only by coincidence (both tokenizers share the same special-token ordering).

6. **Naming inconsistencies / typos** — `dataset.py` has `tokrnizer_tgt` and `lable`; `casual_mask` should be `causal_mask`; `model.py` has `DecoaderBlock`, `encoader_*` variables. These don't break functionality but are worth cleaning up for readability and to avoid confusing future edits.

## Acknowledgements

Architecture based on Vaswani et al., *"Attention Is All You Need"* (2017).
