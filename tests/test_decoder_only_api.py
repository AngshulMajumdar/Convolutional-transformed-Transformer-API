from transformed_transformer.public_api import DecoderOnlyAPI, DecoderOnlyAPIConfig


def test_decoder_only_api_builds_and_runs_baseline(tmp_path):
    config = DecoderOnlyAPIConfig(epochs=1, seeds=(0,), n_samples=16, output_dir=str(tmp_path / "decoder_only_api"))
    api = DecoderOnlyAPI(config)
    standard = api.build_standard()
    proposed = api.build_proposed()
    assert standard is not None
    assert proposed is not None
    summary = api.run_baseline()
    assert any("standard_decoder_only" in key for key in summary)
    assert any("transformed_conv_decoder_only" in key for key in summary)
