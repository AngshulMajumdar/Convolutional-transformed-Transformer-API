from dataclasses import dataclass
from typing import Literal

AttentionType = Literal["standard", "transformed_dense", "transformed_sparse", "transformed_conv"]


@dataclass(slots=True)
class ModelConfig:
    vocab_size: int
    seq_len: int
    d_model: int = 32
    ff_dim: int = 64
    num_classes: int = 2
    attention_type: AttentionType = "standard"
    sparse_k: int = 4
    reg_weight: float = 1e-3
    conv_kernel_size: int = 3
    conv_groups: int = 1
    num_layers: int = 2
    type_vocab_size: int = 2
    mask_token_id: int = 1
