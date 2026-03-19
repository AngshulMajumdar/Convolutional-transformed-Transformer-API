# Transformed Transformer API

A reusable PyTorch research-software package for comparing standard transformers with two proposed convolution-informed variants:

- a **decoder-only** family for miniSLM-style causal language modelling
- an **encoder-decoder** family for conditional generation and seq2seq-style tasks

The repository includes clean public APIs, runnable demos, smoke tests, and compiled benchmark results.

## Statement of need

Transformer research code is often difficult to reuse because architecture changes, benchmarking scripts, and demos are tightly coupled and poorly exposed. This repository addresses that gap by packaging a from-scratch transformer implementation into a small reusable API that makes it easy to:

- instantiate **standard baselines** and **proposed variants** through the same interface,
- compare them on fixed experiment suites,
- run smoke tests quickly before larger experiments,
- extend the package toward additional model families without rewriting the core.

The specific software contribution here is a unified package for comparing:

1. **Standard decoder-only transformers** vs **proposed convolutional-transform decoder-only transformers**
2. **Standard encoder-decoder transformers** vs **proposed analysis-synthesis convolutional encoder-decoder transformers**
3. **miniBERT** standard vs proposed convolutional-transform variants
4. **miniSLM (encoder-decoder conditional generation)** standard vs proposed analysis-synthesis variants

This makes the repository suitable as reusable research software rather than a one-off benchmark dump.

## Software summary

### Public APIs

The package exposes two top-level public APIs:

- `DecoderOnlyAPI`
- `EncoderDecoderAPI`

with corresponding config dataclasses:

- `DecoderOnlyAPIConfig`
- `EncoderDecoderAPIConfig`

These APIs provide methods for running:

- baseline comparisons
- low-data sweeps
- corruption sweeps

### Model builders

The lower-level package also exposes builders for direct use:

- `build_mini_slm(...)`
- `build_mini_seq2seq(...)`
- `build_analysis_synthesis_transformer(...)`
- `build_mini_bert_classifier(...)`
- `build_mini_bert_mlm(...)`

### Proposed variants included

- `transformed_conv` for decoder-only and miniBERT-style models
- `analysis_synthesis_conv` for encoder-decoder and miniSLM-style conditional generation

## Implementation notes

- All models are implemented in **PyTorch**.
- The proposed convolutional-transform modules are trained **end-to-end with Adam/backprop**.
- The encoder-decoder proposed model is **inspired by** convolutional transform learning on the encoder side and convolutional dictionary learning on the decoder side.
- The implementation is intended as reusable research software, not as a paper-faithful alternating CTL/CDL solver.

## Repository structure

```text
src/transformed_transformer/
  attention/
  models/
  public_api/
  utils/
scripts/
examples/
tests/
results/
docs/
```

## Installation

```bash
pip install -e .
pip install -e .[dev]
```

## Quickstart

### Decoder-only API

```python
from transformed_transformer.public_api import DecoderOnlyAPI, DecoderOnlyAPIConfig

api = DecoderOnlyAPI(DecoderOnlyAPIConfig())
summary = api.run_baseline()
print(summary)
```

### Encoder-decoder API

```python
from transformed_transformer.public_api import EncoderDecoderAPI, EncoderDecoderAPIConfig

api = EncoderDecoderAPI(EncoderDecoderAPIConfig())
summary = api.run_baseline()
print(summary)
```

## Included demos

```bash
PYTHONPATH=src python examples/minibert_api_demo.py
PYTHONPATH=src python examples/minislm_encdec_api_demo.py
PYTHONPATH=src python examples/seq2seq_smoke_demo.py
```

## Smoke tests

Only smoke-level verification is required for the packaged repo:

```bash
PYTHONPATH=src pytest -q tests/test_smoke.py tests/test_minibert.py tests/test_minislm.py tests/test_seq2seq.py tests/test_api.py tests/test_text_seq2seq_lm.py
```

## Reproducing the compiled benchmark suites

### First type: decoder-only full suite

```bash
PYTHONPATH=src python scripts/run_decoder_only_api_full_benchmark.py
```

Outputs:
- `results/api_decoder_only_full_suite/all_experiments_summary.csv`
- `results/api_decoder_only_full_suite/all_experiments_per_seed.csv`
- `results/api_decoder_only_full_suite/all_experiments_summary.json`

### Second type: encoder-decoder full suite

```bash
PYTHONPATH=src python scripts/run_encoder_decoder_api_full_benchmark.py
```

Outputs:
- `results/api_encoder_decoder_full_suite/all_experiments_summary.csv`
- `results/api_encoder_decoder_full_suite/all_experiments_per_seed.csv`
- `results/api_encoder_decoder_full_suite/all_experiments_summary.json`

### miniBERT full suite

```bash
PYTHONPATH=src python scripts/run_minibert_full_benchmark.py
```

Outputs:
- `results/minibert_full_suite/all_experiments_summary.csv`
- `results/minibert_full_suite/all_experiments_per_seed.csv`
- `results/minibert_full_suite/all_experiments_summary.json`

### miniSLM second-type full suite

```bash
PYTHONPATH=src python scripts/run_minislm_encdec_full_benchmark.py
```

Outputs:
- `results/minislm_encdec_full_suite/all_experiments_summary.csv`
- `results/minislm_encdec_full_suite/all_experiments_per_seed.csv`
- `results/minislm_encdec_full_suite/all_experiments_summary.json`

## Included compiled results

The repository already contains compiled CSV/JSON outputs for:

- decoder-only baseline, low-data, and corruption experiments
- encoder-decoder baseline, low-data, and corruption experiments
- miniBERT baseline, low-data, and corruption experiments
- miniSLM encoder-decoder baseline, low-data, and corruption experiments

These are meant to make the software immediately inspectable after cloning.

## Reuse potential

The package is designed to be extended in three directions:

1. new transformer backends under the shared core,
2. additional benchmark suites using the public APIs,
3. larger downstream demos without changing the model-construction interface.

That makes it suitable for reuse in small comparative transformer studies and software-focused publications.

## Important honesty notes

- The convolutional modules are the proposed reusable research-software variants used in this repository.
- The package is strongest as a **software artifact** and **comparative benchmarking framework**.
- Runtime is consistently higher for the proposed variants, so included results should be interpreted jointly with accuracy/loss improvements.
