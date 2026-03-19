# Architecture Notes

The package is organized around one core idea: attention backends are swappable, while the model shells stay stable.

`build_attention(config)` converts `attention_type` into one of three backend modules:

- `StandardAttention`
- `TransformDenseAttention`
- `TransformSparseAttention`

The higher-level builders reuse that attention constructor:

- `build_tiny_classifier`
- `build_mini_bert_classifier`
- `build_mini_bert_mlm`
- `build_mini_slm`

This means the same API surface can later support richer model families without rewriting backend-selection logic.
