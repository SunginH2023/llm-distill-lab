"""다운로드 없이 도는 초소형 모델 + 문자 단위 토크나이저 (스모크 전용).

왜 필요한가: 학습 스크립트가 정말 도는지 확인하려고 매번 수 GB짜리 체크포인트를
받을 수는 없다. 그렇다고 코드를 짜 놓고 "GPU 생기면 돌려보죠" 하는 것도 곤란하다.

그래서 무작위 초기화된 2층짜리 Llama 를 그 자리에서 만들고, 문자 단위 토크나이저를
붙여 CPU에서 몇 초 만에 전 경로를 통과시킨다. 검증되는 것:

    JSONL 적재 → 프롬프트/정답 토큰화 → 프롬프트 구간 마스킹 → LoRA 주입
    → 학습 루프(손실 감소) → 생성 → JSON 파싱 → 채점

검증되지 않는 것: 실제 모델의 성능. 그건 `--model <HF id>` 로 돌려야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CharTokenizer:
    """문자 단위 토크나이저. 어휘를 데이터에서 그 자리에서 만든다."""
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
    """무작위 초기화 Llama (2층, hidden 96). 다운로드 없음."""
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
        tie_word_embeddings=False,   # lm_head 를 따로 학습할 수 있게 (스모크용)
    )
    return LlamaForCausalLM(cfg)


class HFTokenizerAdapter:
    """AutoTokenizer 를 CharTokenizer 와 같은 최소 인터페이스로 감싼다.

    transformers 버전마다 `__call__` 시그니처가 조금씩 달라서, 학습 루프가
    버전에 흔들리지 않도록 encode/decode 두 개만 쓰기로 고정한다.
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
