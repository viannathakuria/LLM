# MyLLM

MyLLM is the foundation for an independently built GPT-style language model and
chat application. This first milestone contains a Next.js frontend and a FastAPI
backend. **The chat responses are placeholders; no language model is connected.**

There are no external AI APIs, pretrained models, or training code in this
project.

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

The tokenizer is a standalone Python package with no runtime dependencies. Set
it up from the repository root with:

```bash
cd model
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
```

Its tests and package are independent of the FastAPI backend. The tokenizer
builds and saves vocabularies but does not train or load a Transformer.

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
- The tokenizer does not train a model or generate text.
- Transformer training, datasets, model checkpoints, and inference remain out
  of scope.

## Next milestone

Build and test MyLLM's own small Transformer in the separate `model/` component.
Do not present an existing pretrained model as a MyLLM-trained model; review
dataset licenses before using any training data.
