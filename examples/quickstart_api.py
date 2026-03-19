from transformed_transformer import ModelConfig, build_mini_bert_classifier, build_mini_bert_mlm, build_mini_slm, build_tiny_classifier


def main() -> None:
    tiny = build_tiny_classifier(ModelConfig(vocab_size=32, seq_len=12, attention_type="standard"))
    bert_cls = build_mini_bert_classifier(ModelConfig(vocab_size=64, seq_len=16, d_model=48, ff_dim=96, num_layers=2, attention_type="transformed_dense"))
    bert_mlm = build_mini_bert_mlm(ModelConfig(vocab_size=64, seq_len=16, d_model=48, ff_dim=96, num_layers=2, attention_type="standard"))
    slm = build_mini_slm(ModelConfig(vocab_size=64, seq_len=16, d_model=48, ff_dim=96, num_layers=2, attention_type="transformed_sparse"))
    print("tiny:", type(tiny).__name__)
    print("bert classifier:", type(bert_cls).__name__)
    print("bert mlm:", type(bert_mlm).__name__)
    print("slm:", type(slm).__name__)


if __name__ == "__main__":
    main()
