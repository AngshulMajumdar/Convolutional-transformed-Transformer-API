# Repo positioning notes

This repository should be presented as a clean engineering artifact.

Best framing:

- from-scratch PyTorch implementation
- nonstandard attention backend abstraction
- reusable package, not a one-off notebook
- honest surrogate sparse path
- miniBERT and miniSLM extensions built on the same core API


## Corrected seq2seq benchmark scope

The valid encoder-decoder narrative in this repository is a fixed-architecture comparison between the standard seq2seq transformer and the proposed analysis-synthesis convolutional seq2seq transformer. Low-data and corruption sweeps are included because they keep the architecture fixed. Depth-variation is intentionally excluded from the seq2seq benchmark scope for this packaged version.
