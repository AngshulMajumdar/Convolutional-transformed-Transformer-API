from transformed_transformer import (
    EncoderDecoderAPI,
    EncoderDecoderAPIConfig,
    build_proposed_encoder_decoder,
    build_standard_encoder_decoder,
)


def test_encoder_decoder_api_builders() -> None:
    config = EncoderDecoderAPIConfig(d_model=16, ff_dim=32, num_layers=2, n_samples=8, seeds=(0,), epochs=1)
    standard = build_standard_encoder_decoder(config)
    proposed = build_proposed_encoder_decoder(config)
    assert standard is not None
    assert proposed is not None


def test_encoder_decoder_api_baseline_runs() -> None:
    config = EncoderDecoderAPIConfig(
        d_model=16,
        ff_dim=32,
        num_layers=2,
        n_samples=8,
        seeds=(0,),
        epochs=1,
        output_dir='results/test_encoder_decoder_api',
    )
    api = EncoderDecoderAPI(config)
    summary = api.run_baseline()
    assert any('standard_seq2seq' in key for key in summary)
    assert any('analysis_synthesis_conv' in key for key in summary)
