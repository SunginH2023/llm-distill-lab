# finetune — pseudo-label 기반 LoRA / QLoRA 학습

e1~e5 는 벡터 표현 위에서 수행된다. 동일한 결론이 실제 LLM 학습 경로에서도
성립하는지는 별도 확인이 필요하다. 이 폴더가 그 연결부에 해당한다.

```
합성 세계 → 한국어 텍스트 → LLM 호출(투표 + 게이트) → pseudo-label
                                                          ↓
                       세 채점표로 채점 ← 생성·JSON 파싱 ← LoRA / QLoRA SFT
```

## 빠른 실행

```bash
pip install -r finetune/requirements.txt

python finetune/build_sft_data.py            # 데이터 생성 (~3초)
python finetune/train_qlora.py --smoke \     # 다운로드·GPU 없이 전 경로 검증 (CPU ~70초)
    --epochs 8 --bs 8 --grad-accum 1 --lr 8e-3 --limit 256
python finetune/evaluate.py --smoke --limit 64
```

실제 모델:

```bash
python finetune/train_qlora.py --model Qwen/Qwen2.5-0.5B-Instruct --load-4bit \
    --epochs 1 --bs 4 --grad-accum 4 --lr 2e-4 --lora-r 32 --grad-checkpoint
python finetune/evaluate.py --model Qwen/Qwen2.5-0.5B-Instruct --adapter finetune/runs/lora
```

## `--smoke` 의 검증 범위

학습 코드를 실행하지 않은 상태로 두면 실제 실행 시점에 결함이 드러난다.
반면 검증 때마다 수 GB 체크포인트를 내려받는 것도 비효율적이다.

`--smoke` 는 무작위 초기화된 2층 Llama(0.23M)와 문자 단위 토크나이저를 즉석에서 구성해
CPU에서 전 경로를 실행한다.

| 검증되는 것 | 검증되지 않는 것 |
|---|---|
| JSONL 적재 → 프롬프트/정답 토큰화 | 실제 모델의 분류 성능 |
| 프롬프트 구간 -100 마스킹 | 4bit 양자화 경로 (CUDA 필요) |
| LoRA 주입 / 학습 파라미터 비율 | 대규모 배치·분산 학습 |
| 학습 루프 · 손실 감소 · 스케줄러 | 긴 문맥에서의 형식 붕괴 |
| 어댑터 저장 → 재로딩 → 생성 | |
| JSON 파싱 · 세 채점표 채점 | |

실측 (CPU, 256건, 8 epoch, ~67초):

| 항목 | 값 | 비고 |
|---|---:|---|
| 학습 손실 | 5.65 → 0.15 | 8 epoch |
| 형식 준수율 (JSON 파싱) | 1.000 | |
| ① pseudo-label 평가셋 EM | 0.234 | |
| ② 외부 신호 일관성 (top1) | 0.266 | |
| ③ 정답 EM | 0.156 | 무작위 추측은 1/12 ≈ 0.083 |
| pseudo-label 자체의 정확도 | 0.672 | 학습 라벨의 상한 |

0.23M 무작위 초기화 모델의 정답 EM은 무작위 추측의 약 2배 수준이다.
분류 성능 지표가 아니라 파이프라인 동작 확인용 수치다.
`pseudo-label 자체의 정확도 0.672` 를 함께 기록하는 이유는 학습 라벨의 상한을 명시하기 위함이다.

## 구현 시 주의점

- 프롬프트 마스킹 누락 시 모델이 카테고리 목록과 규칙 암기에 용량을 소비한다.
  손실은 정상적으로 감소하지만 분류 성능은 오르지 않는다. 검출이 어려워 `SFTDataset` 에 주석으로 기록했다.
- 어댑터는 base 가중치와 쌍으로만 유효하다. 스모크 base 는 무작위 초기화이므로, 저장하지 않고
  평가 시점에 재생성하면 다른 난수의 base 위에 어댑터가 적용된다. 이 경우 학습 손실은 0.13까지
  하락하지만 생성 결과는 전부 무효가 된다(`train_qlora.py` 주석 참조).
  사전학습 모델 사용 시에는 발생하지 않으나, 학습 정상·생성 비정상 상황의 원인 후보에 해당한다.
- 카테고리 코드는 두 자리로 고정한다. 다수 LLM이 숫자를 자릿수 단위로 토큰화하므로
  `3` 과 `11` 이 혼재하면 생성 길이가 변동하고 파싱 실패율이 상승한다.
- 행별 가중 손실을 직접 구현한다. HF 기본 손실은 배치 단위 평균이라 행 가중치를 적용할 수 없다.
  e1 의 등급(agree/partial/review)을 학습 가중치로 사용하려면(`--grade-weight`) 별도 손실이 필요하다.
- gradient checkpointing 은 DDP 의 reduction 훅과 충돌한다. 단일 프로세스에서만 활성화한다.
  멀티모달 계열 LoRA 에서는 미사용 타워의 파라미터가 unused 로 판정되어
  `ddp_find_unused_parameters=True` 가 필요하다.

## 데이터 형식

`finetune/data/*.jsonl` 한 줄:

```json
{"id": 12, "text": "안녕하세요. 로그인이 안 됩니다 ...",
 "completion": "{\"labels\": [\"05\"]}",
 "pseudo_labels": ["05"], "grade": "partial", "n_calls": 7,
 "vote_dist": [0.0, ...],
 "truth": ["05"], "outside_signal": 1}
```

- `pseudo_labels` / `grade` / `n_calls` / `vote_dist` — 학습 사용. LLM 호출로만 생성된다.
- `truth` / `outside_signal` — 평가 전용. 학습 스크립트는 이 필드를 참조하지 않는다.
  합성 실험에서 정답이 학습 경로에 유입되면 결과 해석이 불가능해진다.
- `prompt` 는 저장하지 않고 `text` 에서 재생성한다
  (행마다 카테고리 목록을 중복 저장하면 파일 크기가 3배가 된다).

## 카테고리·문구에 대한 고지

`render_text.py` 의 카테고리 12개와 문구는 전부 합성 생성기가 만든 가상의 문자열이며,
특정 서비스·조직·실제 데이터와 무관하다.
`world.py` 의 구조(흡수 클래스 1개, 경계 중첩 쌍 3개, 공존 허용 쌍 2개)에 명칭만 부여한 것이다.
