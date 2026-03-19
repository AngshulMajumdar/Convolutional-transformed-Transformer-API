from transformed_transformer.public_api import DecoderOnlyAPI, DecoderOnlyAPIConfig


if __name__ == "__main__":
    config = DecoderOnlyAPIConfig(epochs=2, seeds=(0,), n_samples=32, output_dir="results/demo_decoder_only")
    api = DecoderOnlyAPI(config)
    print(api.run_baseline())
