# Usage notes

Start with the tiny encoder classifier because it is the shortest path from config to model to forward pass.

Then show:

1. `build_attention(config)`
2. `build_tiny_classifier(config)`
3. `build_mini_bert_classifier(config)`
4. `build_mini_slm(config)`


## Current comparison focus

For the next iteration, the clean experiment is `standard` versus `transformed_conv`. The convolutional variant is the main proposed backend; the dense and sparse variants remain secondary engineering baselines.
