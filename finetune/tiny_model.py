"""A tiny download-free model plus a character-level tokenizer (smoke test only).

Why: downloading a multi-GB checkpoint every time just to check that the training
script runs is wasteful, yet leaving the code untested until a GPU is available
is not acceptable either.

So a randomly initialized two-layer Llama is built on the fly and paired with a
character-level tokenizer, which exercises the full path on CPU in seconds:

    JSONL loading -> prompt/answer tokenization -> prompt-span masking -> LoRA injection
    -> training loop (loss decreases) -> generation -> JSON parsing -> scoring

Not verified: the performance of a real model. That requires `--model <HF id>`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CharTokenizer:
    """Character-level tokenizer whose vocabulary is built on the fly from the data."""
    itos: list[str]
    stoi: dict[str, int]
    pad_id: int
    eos_id: int

    @classmethod
    def from_texts(cls, texts: list[str]) -> "CharTokenizer":
        chars = sorted({c for t in texts for c in t})
        itos = ["<pad>", "</s>"] + chars
        return cls(itos=itos, stoi={c: i for i, c in enumerate(itos)}, pad_id=0, eos_id=1)

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids
                       if 1 < int(i) < len(self.itos))


def build_tiny_causal_lm(vocab_size: int):
    """Randomly initialized Llama (2 layers, hidden size 96). No download."""
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=96,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        pad_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=False,   # untied so lm_head can be trained separately (smoke test)
    )
    return LlamaForCausalLM(cfg)


class HFTokenizerAdapter:
    """Wrap an AutoTokenizer in the same minimal interface as CharTokenizer.

    The `__call__` signature differs slightly across transformers versions, so the
    training loop relies only on encode/decode to stay version-independent.
    """

    def __init__(self, tok):
        self.tok = tok
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        self.pad_id = tok.pad_token_id
        self.eos_id = tok.eos_token_id

    def __len__(self) -> int:
        return len(self.tok)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)
