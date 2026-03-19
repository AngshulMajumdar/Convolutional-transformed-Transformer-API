from __future__ import annotations
import argparse
import torch
import torch.nn.functional as F
from transformed_transformer.api import build_mini_bert_classifier, build_mini_bert_mlm, build_mini_slm, build_tiny_classifier
from transformed_transformer.configs import ModelConfig
from transformed_transformer.utils import make_synthetic_classification_data, set_seed

MODEL_CHOICES = ["tiny", "bert_cls", "bert_mlm", "slm"]
ATTENTION_CHOICES = ["standard", "transformed_dense", "transformed_sparse", "transformed_conv"]


def _masked_batch(vocab_size: int, seq_len: int, batch_size: int, mask_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(2, vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    input_ids[:, 1] = mask_token_id
    return input_ids, labels


def run_smoke(model_name: str, attention_type: str) -> None:
    set_seed(0)
    config = ModelConfig(vocab_size=64, seq_len=12, d_model=32, ff_dim=64, num_layers=2, num_classes=3, attention_type=attention_type, sparse_k=4, conv_kernel_size=3, conv_groups=1)
    if model_name == "tiny":
        model = build_tiny_classifier(config)
        x, y = make_synthetic_classification_data(n_samples=8, seq_len=config.seq_len, vocab_size=config.vocab_size)
        logits, _ = model(x)
        loss = F.cross_entropy(logits, y) + model.extra_loss()
    elif model_name == "bert_cls":
        model = build_mini_bert_classifier(config)
        x, y = make_synthetic_classification_data(n_samples=8, seq_len=config.seq_len, vocab_size=config.vocab_size)
        logits, _ = model(x)
        loss = F.cross_entropy(logits, y) + model.extra_loss()
    elif model_name == "bert_mlm":
        model = build_mini_bert_mlm(config)
        x, labels = _masked_batch(config.vocab_size, config.seq_len, 8, config.mask_token_id)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), labels.reshape(-1)) + model.extra_loss()
    elif model_name == "slm":
        model = build_mini_slm(config)
        ids = torch.randint(0, config.vocab_size, (8, config.seq_len))
        inputs, targets = ids[:, :-1], ids[:, 1:]
        logits, _ = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), targets.reshape(-1)) + model.extra_loss()
    else:
        raise ValueError(model_name)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(f"OK model={model_name} attention={attention_type} loss={float(loss.detach()):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Transformed Transformer CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke", help="Run one smoke check")
    smoke.add_argument("--model", choices=MODEL_CHOICES, required=True)
    smoke.add_argument("--attention", choices=ATTENTION_CHOICES, required=True)
    args = parser.parse_args()
    if args.command == "smoke":
        run_smoke(args.model, args.attention)


if __name__ == "__main__":
    main()
