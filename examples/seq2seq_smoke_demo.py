from __future__ import annotations

import torch
import torch.nn.functional as F

from transformed_transformer import ModelConfig, build_analysis_synthesis_transformer, build_mini_seq2seq


def make_batch(vocab_size: int, seq_len: int, batch_size: int = 4):
    src = torch.randint(3, vocab_size, (batch_size, seq_len))
    tgt = torch.flip(src, dims=[1])
    decoder_input_ids = torch.cat([torch.full((batch_size, 1), 2, dtype=torch.long), tgt[:, :-1]], dim=1)
    return src, decoder_input_ids, tgt


def run_demo(attention_type: str, use_dictionary_decoder: bool) -> None:
    config = ModelConfig(vocab_size=33, seq_len=8, d_model=16, ff_dim=32, num_layers=2, attention_type=attention_type)
    model = build_analysis_synthesis_transformer(config) if use_dictionary_decoder else build_mini_seq2seq(config)
    src, decoder_input_ids, tgt = make_batch(config.vocab_size, config.seq_len)
    logits, stats = model(src, decoder_input_ids)
    loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), tgt.reshape(-1)) + model.extra_loss()
    print({
        "attention_type": attention_type,
        "dictionary_decoder": use_dictionary_decoder,
        "logits_shape": tuple(logits.shape),
        "loss": float(loss.detach()),
        "num_stats": len(stats),
    })


if __name__ == "__main__":
    torch.manual_seed(0)
    run_demo(attention_type="standard", use_dictionary_decoder=False)
    run_demo(attention_type="transformed_conv", use_dictionary_decoder=True)
