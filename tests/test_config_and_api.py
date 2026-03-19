import pytest
from transformed_transformer import ModelConfig, build_attention, build_analysis_synthesis_transformer, build_mini_bert_classifier, build_mini_seq2seq, build_mini_slm, build_tiny_classifier


def test_invalid_attention_type_raises() -> None:
    config = ModelConfig(vocab_size=16, seq_len=8, attention_type="standard")
    config.attention_type = "not_real"  # type: ignore[assignment]
    with pytest.raises(ValueError):
        build_attention(config)


def test_public_api_builds_multiple_models() -> None:
    assert build_tiny_classifier(ModelConfig(vocab_size=16, seq_len=8, attention_type="standard")) is not None
    assert build_mini_bert_classifier(ModelConfig(vocab_size=16, seq_len=8, attention_type="transformed_dense")) is not None
    assert build_mini_slm(ModelConfig(vocab_size=16, seq_len=8, attention_type="transformed_sparse")) is not None
    assert build_mini_seq2seq(ModelConfig(vocab_size=16, seq_len=8, attention_type="standard")) is not None
    assert build_analysis_synthesis_transformer(ModelConfig(vocab_size=16, seq_len=8, attention_type="transformed_conv")) is not None
