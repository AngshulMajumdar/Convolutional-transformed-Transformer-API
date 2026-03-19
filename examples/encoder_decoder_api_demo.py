from transformed_transformer.public_api import EncoderDecoderAPI, EncoderDecoderAPIConfig

config = EncoderDecoderAPIConfig(epochs=2, seeds=(0,), n_samples=16, output_dir='results/demo_encoder_decoder_api')
api = EncoderDecoderAPI(config)
summary = api.run_baseline()
print(summary)
