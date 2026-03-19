from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import csv
import json
import time

import torch
import torch.nn.functional as F

from transformed_transformer.api import build_analysis_synthesis_transformer, build_mini_seq2seq
from transformed_transformer.configs import ModelConfig
from transformed_transformer.utils.synthetic_data import set_seed


@dataclass(slots=True)
class EncoderDecoderAPIConfig:
    vocab_size: int = 48
    seq_len: int = 10
    d_model: int = 32
    ff_dim: int = 64
    num_layers: int = 2
    conv_kernel_size: int = 3
    conv_groups: int = 1
    reg_weight: float = 1e-3
    learning_rate: float = 1e-3
    epochs: int = 10
    batch_size: int = 16
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    n_samples: int = 64
    corruption: float = 0.0
    output_dir: str = 'results/api_encoder_decoder'
    device: str = 'cpu'

    def as_model_config(self, attention_type: str) -> ModelConfig:
        return ModelConfig(
            vocab_size=self.vocab_size,
            seq_len=self.seq_len,
            d_model=self.d_model,
            ff_dim=self.ff_dim,
            num_layers=self.num_layers,
            attention_type=attention_type,
            conv_kernel_size=self.conv_kernel_size,
            conv_groups=self.conv_groups,
            reg_weight=self.reg_weight,
        )



def _make_seq2seq_batch(vocab_size: int, seq_len: int, n_samples: int, noise_prob: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    src = torch.randint(3, vocab_size, (n_samples, seq_len))
    if noise_prob > 0:
        mask = torch.rand(n_samples, seq_len) < noise_prob
        noise = torch.randint(3, vocab_size, (n_samples, seq_len))
        src = torch.where(mask, noise, src)
    tgt = torch.flip(src, dims=[1])
    decoder_input_ids = torch.cat([torch.full((n_samples, 1), 2, dtype=torch.long), tgt[:, :-1]], dim=1)
    return src, decoder_input_ids, tgt


def build_standard_encoder_decoder(config: EncoderDecoderAPIConfig):
    return build_mini_seq2seq(config.as_model_config('standard'))


def build_proposed_encoder_decoder(config: EncoderDecoderAPIConfig):
    return build_analysis_synthesis_transformer(config.as_model_config('transformed_conv'))


class EncoderDecoderAPI:
    def __init__(self, config: EncoderDecoderAPIConfig | None = None) -> None:
        self.config = config or EncoderDecoderAPIConfig()

    def build_standard(self):
        return build_standard_encoder_decoder(self.config)

    def build_proposed(self):
        return build_proposed_encoder_decoder(self.config)

    def run_baseline(self) -> dict[str, dict[str, float]]:
        return self._run_variant_suite(sample_sizes=(self.config.n_samples,), corruptions=(self.config.corruption,), write_prefix='baseline')

    def run_low_data(self, sample_sizes: tuple[int, ...] = (16, 32, 64, 128)) -> dict[str, dict[str, float]]:
        return self._run_variant_suite(sample_sizes=sample_sizes, corruptions=(self.config.corruption,), write_prefix='low_data')

    def run_corruption(self, corruptions: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3)) -> dict[str, dict[str, float]]:
        return self._run_variant_suite(sample_sizes=(self.config.n_samples,), corruptions=corruptions, write_prefix='corruption')

    def _run_variant_suite(self, sample_sizes: tuple[int, ...], corruptions: tuple[float, ...], write_prefix: str) -> dict[str, dict[str, float]]:
        if self.config.device == 'cpu':
            try:
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
        outdir = Path(self.config.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        per_seed_path = outdir / f'{write_prefix}_per_seed.csv'
        summary_path = outdir / f'{write_prefix}_summary.json'
        rows: list[dict[str, float | int | str]] = []
        for n_samples in sample_sizes:
            for corruption in corruptions:
                for variant_name, builder in (
                    ('standard_seq2seq', self.build_standard),
                    ('analysis_synthesis_conv', self.build_proposed),
                ):
                    for seed in self.config.seeds:
                        rows.append(self._run_single(seed, n_samples, corruption, variant_name, builder))
        self._write_csv(per_seed_path, rows)
        summary = self._summarize(rows, sample_sizes=sample_sizes, corruptions=corruptions)
        summary_path.write_text(json.dumps(summary, indent=2))
        return summary

    def _run_single(self, seed: int, n_samples: int, corruption: float, variant_name: str, builder) -> dict[str, float | int | str]:
        set_seed(seed)
        device = torch.device(self.config.device)
        model = builder().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        src, decoder_input_ids, tgt = _make_seq2seq_batch(self.config.vocab_size, self.config.seq_len, n_samples, noise_prob=corruption)
        src = src.to(device)
        decoder_input_ids = decoder_input_ids.to(device)
        tgt = tgt.to(device)
        best_acc = 0.0
        start = time.perf_counter()
        for _ in range(self.config.epochs):
            model.train()
            optimizer.zero_grad()
            logits, _ = model(src, decoder_input_ids)
            loss = F.cross_entropy(logits.reshape(-1, self.config.vocab_size), tgt.reshape(-1))
            total = loss + model.extra_loss()
            total.backward()
            optimizer.step()
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                acc = (pred == tgt).float().mean().item()
                best_acc = max(best_acc, acc)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            'variant': variant_name,
            'seed': seed,
            'n_samples': n_samples,
            'corruption': corruption,
            'final_loss': float(loss.detach().cpu()),
            'best_token_accuracy': float(best_acc),
            'elapsed_ms': float(elapsed_ms),
        }

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
        if not rows:
            return
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _summarize(rows: list[dict[str, float | int | str]], sample_sizes: tuple[int, ...], corruptions: tuple[float, ...]) -> dict[str, dict[str, float]]:
        summary: dict[str, dict[str, float]] = {}
        for row in rows:
            key = f"{row['variant']}|n={row['n_samples']}|c={row['corruption']}"
            bucket = summary.setdefault(key, {'count': 0, 'final_loss': 0.0, 'best_token_accuracy': 0.0, 'elapsed_ms': 0.0})
            bucket['count'] += 1
            bucket['final_loss'] += float(row['final_loss'])
            bucket['best_token_accuracy'] += float(row['best_token_accuracy'])
            bucket['elapsed_ms'] += float(row['elapsed_ms'])
        for bucket in summary.values():
            count = bucket.pop('count')
            bucket['final_loss'] /= count
            bucket['best_token_accuracy'] /= count
            bucket['elapsed_ms'] /= count
        return summary
