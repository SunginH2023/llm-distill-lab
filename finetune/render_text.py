"""Render the synthetic world as Korean text.

Experiments e1-e5 operate on vector representations. An LLM, however, reads
text, so the same world has to be rendered as text before it can feed a LoRA
fine-tuning run.

Key design choice: phrases are sampled in proportion to the posterior, not from
the true label. Ambiguous items therefore produce ambiguous text, mixing phrases
from two categories. If ambiguity lived only in the labels while the text stayed
clean, the model would see plain label noise rather than realistic boundary cases.

Language note: the dataset is intentionally Korean. Every string constant below
(CLASS_NAMES, PHRASES, OPENERS, CLOSERS, PROMPT) is training/evaluation data:
it is rendered into finetune/data/*.jsonl, stored in meta.json, and fed to the
model as the prompt. Translating any of them would change the dataset, so they
are kept verbatim; English glosses are given in the comments.

Disclaimer: all categories and phrases are fictional strings produced for this
synthetic generator. They are unrelated to any real service, organization, or
dataset.
"""

from __future__ import annotations

import numpy as np

# Twelve fictional categories, mapped 1:1 onto the structure of world.py.
#   index 0           = "other" magnet class (where ambiguous items drift)
#   (3,4) (5,6) (7,8) = confusable pairs with overlapping definitions
#   (2,9) (10,11)     = pairs that can genuinely hold at the same time
# The Korean names are data (they appear in the prompt and in meta.json).
CLASS_NAMES = [
    "기타/분류불가",      # 0  other / unclassifiable (magnet)
    "배송지연",           # 1  shipping delay
    "상품문의",           # 2  product question      ─┐ can co-occur with 9
    "결제오류",           # 3  payment error         ─┐ confusable pair
    "환불요청",           # 4  refund request        ─┘
    "계정로그인",         # 5  account login         ─┐ confusable pair
    "비밀번호재설정",     # 6  password reset        ─┘
    "앱오류",             # 7  app error             ─┐ confusable pair
    "알림설정",           # 8  notification settings ─┘
    "재고품절",           # 9  out of stock          ─┘ can co-occur with 2
    "쿠폰혜택",           # 10 coupon / promotion    ─┐ can co-occur
    "멤버십등급",         # 11 membership tier       ─┘
]

# Four Korean phrases per category (row c belongs to CLASS_NAMES[c]); data, kept verbatim.

PHRASES: list[list[str]] = [
    ["뭐라고 해야 할지 모르겠는데 좀 이상합니다", "그냥 전반적으로 불편해요",
     "설명하기 애매한 문제가 있습니다", "어디에 문의해야 할지 모르겠어요"],
    ["주문한 지 일주일이 넘었는데 아직입니다", "배송조회가 며칠째 그대로예요",
     "도착 예정일이 계속 밀립니다", "출고가 안 되고 있어요"],
    ["이 제품 사이즈가 어떻게 되나요", "색상 선택지가 더 있는지 궁금합니다",
     "상세 스펙을 알고 싶어요", "다른 모델과 뭐가 다른가요"],
    ["결제가 중간에 튕겼습니다", "카드가 승인되지 않습니다",
     "결제창에서 오류코드가 떴어요", "두 번 결제된 것 같습니다"],
    ["주문 취소하고 돈 돌려받고 싶어요", "환불 처리가 언제 되나요",
     "결제 취소 요청합니다", "입금이 아직 안 들어왔습니다"],
    ["로그인이 안 됩니다", "아이디를 입력해도 넘어가지 않아요",
     "계정이 잠긴 것 같습니다", "인증 단계에서 막힙니다"],
    ["비밀번호를 잊어버렸습니다", "재설정 메일이 안 옵니다",
     "새 비밀번호가 등록되지 않아요", "인증번호 입력이 계속 실패합니다"],
    ["앱이 실행하자마자 꺼집니다", "화면이 하얗게만 나옵니다",
     "특정 메뉴에서 멈춰버려요", "업데이트 후로 계속 튕깁니다"],
    ["푸시가 하나도 안 옵니다", "알림을 꺼도 계속 옵니다",
     "설정을 저장해도 원래대로 돌아가요", "메일 수신 설정이 반영되지 않습니다"],
    ["원하는 옵션이 계속 품절입니다", "재입고 알림을 신청했는데 소식이 없어요",
     "장바구니에 담으면 품절로 바뀝니다", "언제 다시 들어오나요"],
    ["쿠폰이 적용되지 않습니다", "받은 할인코드가 만료됐다고 나와요",
     "이벤트 혜택이 반영이 안 됐어요", "적립금이 안 들어왔습니다"],
    ["등급이 내려간 이유를 모르겠습니다", "우수회원 조건이 어떻게 되나요",
     "등급 혜택이 안 보입니다", "실적이 반영되지 않은 것 같아요"],
]

# Optional greeting / sign-off around the body ("" means none); data, kept verbatim.
OPENERS = ["안녕하세요.", "문의드립니다.", "확인 부탁드려요.", ""]
CLOSERS = ["확인 후 답변 부탁드립니다.", "빠른 처리 부탁드려요.", "감사합니다.", ""]


def render(post: np.ndarray, seed: int = 0, n_phrases: int = 3) -> list[str]:
    """Map an (n, C) posterior matrix to n Korean texts.

    Phrases are sampled from the posterior, so items near a decision boundary
    mix phrases from two categories.
    """
    rng = np.random.default_rng(seed)
    n, C = post.shape
    out = []
    for i in range(n):
        picks = rng.choice(C, size=n_phrases, p=post[i], replace=True)
        parts = [PHRASES[c][rng.integers(len(PHRASES[c]))] for c in picks]
        # Drop duplicate phrases while preserving order.
        seen, uniq = set(), []
        for p in parts:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        body = " ".join(uniq)
        opener = OPENERS[rng.integers(len(OPENERS))]
        closer = CLOSERS[rng.integers(len(CLOSERS))]
        out.append(" ".join(s for s in (opener, body, closer) if s))
    return out


def code(i: int) -> str:
    """Format a category code with a fixed width of two digits.

    Many LLM tokenizers split numbers digit by digit. Mixing widths ('3' vs
    '11') makes the generation length vary and increases parse failures.
    With a fixed width every code occupies the same number of tokens.
    """
    return f"{i:02d}"


def category_list() -> str:
    return ", ".join(f"{code(i)} {name}" for i, name in enumerate(CLASS_NAMES))


# Classification prompt (data, kept verbatim in Korean). Gloss: "Classify the
# following customer inquiry into one of the categories below. Rules: pick the
# single best category; pick up to two only when both genuinely apply; do not
# pick two merely because the case is ambiguous; answer with a single line of
# JSON. Inquiry: {text}  Answer:"
PROMPT = """다음 고객 문의를 아래 카테고리로 분류하세요.

카테고리: {categories}

규칙:
- 가장 잘 맞는 카테고리 1개를 고릅니다.
- 두 카테고리가 모두 실제로 성립할 때만 2개까지 고를 수 있습니다.
- 애매하다는 이유로 2개를 고르지 마세요.
- JSON 한 줄로만 답하세요.

문의: {text}

정답:"""


def build_prompt(text: str) -> str:
    return PROMPT.format(categories=category_list(), text=text)


def build_completion(label_indices: list[int]) -> str:
    return '{"labels": [' + ", ".join(f'"{code(i)}"' for i in label_indices) + "]}"
