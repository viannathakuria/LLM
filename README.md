# MyLLM

MyLLM is the foundation for an independently built GPT-style language model and
chat application. This first milestone contains a Next.js frontend and a FastAPI
backend. **The chat responses are placeholders; no language model is connected.**

There are no external AI APIs or pretrained models in this project. Model
training runs only when explicitly started with a prepared local dataset.

## Project structure

```text
frontend/   Next.js App Router application (TypeScript, Tailwind CSS)
backend/    FastAPI application and pytest tests
model/      Independent Python package for tokenizer and model development
```

The frontend checks the backend health endpoint. Chat submissions are handled
locally by the frontend and return a clearly labeled placeholder response.

## Requirements

- Node.js 20.9 or newer and npm
- Python 3.11 or newer

## Setup

### Backend

From the repository root:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

The API is available at <http://localhost:8000>; its health endpoint is
<http://localhost:8000/health>. Interactive API documentation is at
<http://localhost:8000/docs>.

Run backend tests from `backend/` with:

```bash
python -m pytest
```

### Model and tokenizer

The tokenizer remains usable without PyTorch or Hugging Face dependencies. Set
up the tokenizer and its tests from the repository root with:

```bash
cd model
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest tests/test_tokenizer.py
```

Its tests and package are independent of the FastAPI backend. The tokenizer
builds and saves vocabularies; the Transformer is provided separately by the
`myllm_model` package. The command above runs tokenizer-only tests without
installing PyTorch.

The vocabulary is built deterministically from the Unicode code points in the
provided text. Unknown characters encode as token ID `0` and decode as `�`.
Vocabulary files use a versioned JSON format.

### Dataset preparation

Install the optional Hugging Face loader and test dependencies from `model/`:

```bash
python -m pip install -e ".[data,dev]"
```

The default example uses the `Salesforce/wikitext` dataset,
`wikitext-2-raw-v1` configuration, and its `train` split. It processes at most
1,000 source rows unless `max_examples` is changed. The raw subset is about
7.7 MB to download; the row limit bounds preparation, not the initial dataset
download or cache. The dataset card's licensing section states CC BY-SA 4.0;
its metadata also lists CC BY-SA 3.0 and GFDL.
Review the current [dataset card](https://huggingface.co/datasets/Salesforce/wikitext)
and applicable terms before using the data downstream. Dataset downloads and
prepared output are not committed; the example output path is covered by the
repository's `datasets/` ignore rule.

From `model/`, prepare the default dataset with:

```bash
python -c 'from myllm_tokenizer.dataset import prepare_hf_dataset; prepare_hf_dataset("../datasets/wikitext-2", validation_fraction=0.1, seed=42, max_examples=1000)'
```

The output directory contains `train.jsonl` and `validation.jsonl`, one
`{"input_ids": [...]}` record per line, plus `vocabulary.json` and
`metadata.json`. Load and validate it with:

```python
from myllm_tokenizer.dataset import load_prepared_dataset

prepared = load_prepared_dataset("../datasets/wikitext-2")
```

The split is by text record, reproducible from its seed, and uses a vocabulary
built from training text only. Local/offline unit tests use sample rows and do
not download the Hugging Face dataset.

### Tiny Transformer

Install PyTorch and the development dependencies from `model/`:

```bash
cd model
python -m pip install -e ".[model,dev]"
```

On an Apple Silicon Mac (including M2), use Python 3.11 or newer in a native
arm64 virtual environment. The standard PyPI PyTorch package includes MPS
support; no separate MPS package is needed. Verify whether MPS is available:

```bash
python -c 'import torch; print(torch.__version__); print(torch.backends.mps.is_available())'
```

MyLLM detects MPS availability at runtime and uses CPU when MPS is unavailable.
See the [official PyTorch installation selector](https://pytorch.org/get-started/locally/)
if you need platform-specific installation guidance. The model dependency is an
optional extra, so tokenizer-only installations remain lightweight.

The model is a decoder-only Transformer with learned token and position
embeddings, causal multi-head self-attention, pre-layer normalization, residual
connections, and GELU feed-forward layers. Its defaults are 2 layers, 2 heads,
embedding dimension 64, context length 64, and recommended batch size 4. The
batch size is a starting point for future training code; the model forward pass
accepts any non-empty batch. The weights are randomly initialized and the
forward pass returns logits only.

From `model/`, for example:

```python
import torch

from myllm_model import TinyGPT, TransformerConfig, select_device
from myllm_tokenizer import CharacterTokenizer

tokenizer = CharacterTokenizer.from_text("hello world")
config = TransformerConfig(vocab_size=tokenizer.vocabulary_size)
device = select_device()
model = TinyGPT(config).to(device).eval()
batch = torch.tensor(
    [tokenizer.encode("hello"), tokenizer.encode("world")],
    dtype=torch.long,
).to(device)

with torch.no_grad():
    logits = model(batch)

print(logits.shape)  # [2, 5, tokenizer.vocabulary_size]
```

The model accepts sequences up to the configured context length. It starts with
random weights; generation, pretrained weights, and external AI APIs are not
used. Run the offline model and tokenizer tests from `model/` with:

```bash
python -m pytest
```

These tests use synthetic/local token IDs and do not download datasets or
weights.

### Training

Training requires an already prepared dataset directory; it never downloads a
dataset automatically. To prepare the default WikiText sample, follow the
dataset preparation instructions above. For datasets already prepared locally,
the Hugging Face `data` extra is not needed.

From `model/`, start a small training run with:

```bash
python -m myllm_model.train \
  --dataset-dir ../datasets/wikitext-2 \
  --output-dir ../checkpoints/wikitext-small \
  --learning-rate 0.0003 \
  --batch-size 4 \
  --steps 100 \
  --eval-frequency 10 \
  --seed 42 \
  --weight-decay 0.01 \
  --context-length 64
```

The CLI also accepts `--num-steps` as an alias for `--steps`. Training uses
AdamW, deterministically shuffles windows using the seed, advances through
each record with stride equal to the context length, and never joins separate
records. A window contains up to `context_length + 1` IDs; its input excludes
the final ID and its target excludes the first. Batches are right-padded with
valid token ID `0`; padded targets use ignore index `-100`, so padding does not
contribute to cross-entropy. Records shorter than two tokens are counted and
skipped; a split with no usable token pairs is an error.

At each evaluation interval and at the final step, the CLI reports training
and validation loss over their complete, separate splits. Both are evaluated
with dropout disabled and weighted by the number of unmasked target tokens.
Only individual batches are moved to the runtime-selected MPS or CPU device.
Training settings can be changed with `--learning-rate`, `--batch-size`,
`--steps`, `--eval-frequency`, `--seed`, and `--weight-decay`; the architecture
context can be changed with `--context-length`.

`--output-dir` must be a new or empty directory. A checkpoint contains:

```text
model.pt          PyTorch state_dict only, loaded with weights_only=True
config.json       Versioned Transformer and training configuration
vocabulary.json   Existing versioned CharacterTokenizer format
metadata.json     Format version, completed steps, loss history, skipped counts
```

Load and validate a checkpoint with:

```python
from myllm_model import load_checkpoint

checkpoint = load_checkpoint("../checkpoints/wikitext-small")
model = checkpoint.model
tokenizer = checkpoint.tokenizer
```

The loader validates checkpoint versions, configuration, vocabulary size,
training metadata, and the final evaluation step before loading model weights.
Training tests use tiny local datasets and perform only a few optimizer steps;
they do not download datasets or model weights.

### Text generation

Generation is a standalone model-package command. It loads the existing
validated checkpoint format, using the checkpoint's model configuration and
vocabulary. From `model/`, install the model extra if it is not already
installed:

```bash
python -m pip install -e ".[model]"
```

Generate text from a trained checkpoint with:

```bash
python -m myllm_model.generate \
  --checkpoint-dir ../checkpoints/wikitext-small \
  --prompt "Once upon a time" \
  --max-new-tokens 80 \
  --temperature 0.8 \
  --top-k 20 \
  --seed 42
```

`--checkpoint-dir` and a non-empty `--prompt` are required. Generation uses
the checkpoint's vocabulary to encode the prompt and decode the generated
suffix; prompt characters outside the vocabulary map to the tokenizer's
unknown token. `--max-new-tokens` sets the exact maximum number of new
characters (default `100`; `0` returns the prompt unchanged). A temperature of
`0` uses greedy selection; a positive temperature samples from the model's
probabilities. `--top-k` optionally limits sampling to the most likely
candidate characters and must not exceed the vocabulary size. `--seed` makes
sampling reproducible on the same device/runtime. Seeded sampling uses a local
CPU random generator for compatibility with both CPU and Apple MPS model
inference; cross-device output is not guaranteed to match.

The model runs in evaluation and inference modes during generation and restores
its original training/evaluation mode afterward. It keeps only the most recent
tokens up to the checkpoint's context length. The character-level model has no
end-of-sequence token, so generation stops at the requested token limit; the
small model may produce low-quality or repetitive text. This command does not
connect to FastAPI or the frontend, which still returns clearly labeled
placeholder responses. Generation tests use local synthetic checkpoints and
do not download datasets or model weights.

### Frontend

In another terminal, from the repository root:

```bash
cd frontend
cp .env.example .env.local
npm ci
npm run dev
```

Open <http://localhost:3000>. The frontend uses
`NEXT_PUBLIC_API_BASE_URL` to find the backend; the example value points to
`http://localhost:8000`.

Run frontend tests and checks from `frontend/` with:

```bash
npm test
npm run typecheck
npm run build
```

## Environment variables

- `frontend/.env.example` sets `NEXT_PUBLIC_API_BASE_URL`, the base URL used for
  the backend health check. `NEXT_PUBLIC_` values are exposed to the browser and
  must not contain secrets.
- `backend/.env.example` sets `MYLLM_CORS_ORIGINS`, a comma-separated list of
  browser origins allowed to call the API.

Keep local `.env` files out of version control. Neither service needs API keys.

## Current limitations

- The chat interface returns a fixed placeholder message, not generated text.
- The backend currently provides a health check only.
- Conversation history is held only in the current browser page and is not
  persisted.
- The tokenizer does not train or generate text; it supplies character
  vocabularies to the separate model package.
- The tiny Transformer starts with random weights and must be trained on
  prepared local data before its generated text is useful. Standalone
  generation is available, but inference API and chatbot integration remain
  out of scope.

## Next milestone

Consider integrating the standalone generator with a separate inference
service in a future milestone. Do not present an existing pretrained model as
a MyLLM-trained model; review dataset licenses before using any training data.
