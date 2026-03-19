from __future__ import annotations

from transformed_transformer.attention import (
    BaseAttention,
    StandardAttention,
    TransformDenseAttention,
    TransformSparseAttention,
    TransformConvAttention,
)
from transformed_transformer.configs import ModelConfig
from transformed_transformer.models import (
    MiniBertForMaskedLM,
    MiniBertForSequenceClassification,
    MiniSLMForCausalLM,
    MiniSeq2SeqForConditionalGeneration,
    TinyEncoderClassifier,
)
from transformed_transformer.models.seq2seq import StandardCrossAttention, TransformConvCrossAttention


def build_attention(config: ModelConfig) -> BaseAttention:
    if config.attention_type == "standard":
        return StandardAttention(d_model=config.d_model)
    if config.attention_type == "transformed_dense":
        return TransformDenseAttention(d_model=config.d_model, reg_weight=config.reg_weight)
    if config.attention_type == "transformed_sparse":
        return TransformSparseAttention(
            d_model=config.d_model,
            sparsity=config.sparse_k,
            reg_weight=config.reg_weight,
        )
    if config.attention_type == "transformed_conv":
        return TransformConvAttention(
            d_model=config.d_model,
            kernel_size=config.conv_kernel_size,
            groups=config.conv_groups,
            reg_weight=config.reg_weight,
        )
    raise ValueError(f"Unknown attention_type: {config.attention_type}")


def build_tiny_classifier(config: ModelConfig) -> TinyEncoderClassifier:
    attention = build_attention(config)
    return TinyEncoderClassifier(
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
        d_model=config.d_model,
        ff_dim=config.ff_dim,
        num_classes=config.num_classes,
        attention=attention,
    )


def build_mini_bert_classifier(config: ModelConfig) -> MiniBertForSequenceClassification:
    return MiniBertForSequenceClassification(
        vocab_size=config.vocab_size,
        max_seq_len=config.seq_len,
        d_model=config.d_model,
        ff_dim=config.ff_dim,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        attention_factory=lambda: build_attention(config),
    )


def build_mini_bert_mlm(config: ModelConfig) -> MiniBertForMaskedLM:
    return MiniBertForMaskedLM(
        vocab_size=config.vocab_size,
        max_seq_len=config.seq_len,
        d_model=config.d_model,
        ff_dim=config.ff_dim,
        num_layers=config.num_layers,
        attention_factory=lambda: build_attention(config),
    )


def build_mini_slm(config: ModelConfig) -> MiniSLMForCausalLM:
    return MiniSLMForCausalLM(
        vocab_size=config.vocab_size,
        max_seq_len=config.seq_len,
        d_model=config.d_model,
        ff_dim=config.ff_dim,
        num_layers=config.num_layers,
        attention_factory=lambda: build_attention(config),
    )


def build_mini_seq2seq(config: ModelConfig) -> MiniSeq2SeqForConditionalGeneration:
    return MiniSeq2SeqForConditionalGeneration(
        vocab_size=config.vocab_size,
        max_seq_len=config.seq_len,
        d_model=config.d_model,
        ff_dim=config.ff_dim,
        num_layers=config.num_layers,
        encoder_attention_factory=lambda: build_attention(config),
        decoder_self_attention_factory=lambda: build_attention(config),
        cross_attention_factory=lambda: (
            TransformConvCrossAttention(
                d_model=config.d_model, kernel_size=config.conv_kernel_size, reg_weight=config.reg_weight
            )
            if config.attention_type == "transformed_conv"
            else StandardCrossAttention(d_model=config.d_model)
        ),
        use_dictionary_decoder=False,
        conv_kernel_size=config.conv_kernel_size,
        reg_weight=config.reg_weight,
    )


def build_analysis_synthesis_transformer(config: ModelConfig) -> MiniSeq2SeqForConditionalGeneration:
    return MiniSeq2SeqForConditionalGeneration(
        vocab_size=config.vocab_size,
        max_seq_len=config.seq_len,
        d_model=config.d_model,
        ff_dim=config.ff_dim,
        num_layers=config.num_layers,
        encoder_attention_factory=lambda: build_attention(config),
        decoder_self_attention_factory=lambda: build_attention(config),
        cross_attention_factory=lambda: (
            TransformConvCrossAttention(
                d_model=config.d_model, kernel_size=config.conv_kernel_size, reg_weight=config.reg_weight
            )
            if config.attention_type == "transformed_conv"
            else StandardCrossAttention(d_model=config.d_model)
        ),
        use_dictionary_decoder=True,
        conv_kernel_size=config.conv_kernel_size,
        reg_weight=config.reg_weight,
    )
