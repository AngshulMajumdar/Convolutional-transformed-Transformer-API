from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random

import torch

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

SMALL_TEXT_CORPUS = [
    "the cat sat on the warm window sill and watched the street below",
    "a small transformer can still learn useful local patterns from text",
    "the model reads the first phrase and predicts the words that follow",
    "careful engineering matters when a research idea becomes software",
    "the quick brown fox jumps over the lazy dog near the river bank",
    "we build compact language models to test architectural ideas quickly",
    "the student wrote a clear report and the mentor suggested two revisions",
    "a good inductive bias can help when the model is shallow or data is scarce",
    "the meeting started late but the discussion remained focused and practical",
    "small benchmarks are useful when they expose a consistent engineering trend",
    "the decoder generates the continuation one token at a time from context",
    "an encoder decoder model can condition on the source before producing output",
    "the paper proposed a transform and the next draft added a synthesis stage",
    "a locality aware operator may help more than a generic dense mixing layer",
    "the new experiment compared a standard transformer against the proposed variant",
    "the researcher changed the setup after the first results looked inconclusive",
    "the package exposes a clean api and the tests verify the basic training path",
    "the notebook explains the architecture and the readme keeps the claims modest",
    "the city was quiet after the rain and the road reflected the evening lights",
    "a short example can be enough to show that the implementation is real",
    "the sequence begins with a prefix and the model predicts the remaining words",
    "the stronger model reduced loss but the runtime cost was still significant",
    "the benchmark used several seeds so the comparison would not depend on luck",
    "the final narrative should stay honest about gains speed and limitations",
]

@dataclass(slots=True)
class TextVocab:
    stoi: dict[str, int]
    itos: list[str]

    @property
    def size(self) -> int:
        return len(self.itos)


def build_vocab(min_freq: int = 1) -> TextVocab:
    counter: Counter[str] = Counter()
    for sent in SMALL_TEXT_CORPUS:
        counter.update(sent.lower().split())
    tokens = [tok for tok, freq in sorted(counter.items()) if freq >= min_freq]
    itos = ["<pad>", "<bos>", "<eos>", "<unk>"] + tokens
    stoi = {tok: i for i, tok in enumerate(itos)}
    return TextVocab(stoi=stoi, itos=itos)


def encode_sentence(sentence: str, vocab: TextVocab) -> list[int]:
    return [vocab.stoi.get(tok, UNK_ID) for tok in sentence.lower().split()]


def _pad(seq: list[int], length: int, pad_id: int = PAD_ID) -> list[int]:
    if len(seq) >= length:
        return seq[:length]
    return seq + [pad_id] * (length - len(seq))


def make_text_seq2seq_lm_data(
    n_samples: int,
    src_len: int = 6,
    tgt_len: int = 6,
    noise_prob: float = 0.0,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, TextVocab]:
    rng = random.Random(seed)
    vocab = build_vocab()
    src_rows: list[list[int]] = []
    dec_rows: list[list[int]] = []
    tgt_rows: list[list[int]] = []

    for _ in range(n_samples):
        sent = rng.choice(SMALL_TEXT_CORPUS)
        ids = encode_sentence(sent, vocab)
        if len(ids) < 4:
            ids = ids + [EOS_ID]
        split = max(2, min(len(ids) - 1, len(ids) // 2))
        src = ids[:split]
        tgt = ids[split:] + [EOS_ID]
        src = _pad(src, src_len)
        tgt = _pad(tgt, tgt_len)
        dec = _pad([BOS_ID] + tgt[:-1], tgt_len)
        if noise_prob > 0.0:
            for i, tok in enumerate(src):
                if tok != PAD_ID and rng.random() < noise_prob:
                    src[i] = rng.randint(4, vocab.size - 1)
        src_rows.append(src)
        dec_rows.append(dec)
        tgt_rows.append(tgt)

    return (
        torch.tensor(src_rows, dtype=torch.long),
        torch.tensor(dec_rows, dtype=torch.long),
        torch.tensor(tgt_rows, dtype=torch.long),
        vocab,
    )
