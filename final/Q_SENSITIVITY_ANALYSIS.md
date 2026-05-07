# Winner Q 민감도 분석 — 정당화 및 해석 가이드

본 문서는 [99_analyze.ipynb](99_analyze.ipynb) **섹션 M (M1~M5)** 의 q 민감도 분석 파이프라인이 **왜 필요한가**, **각 단계가 무엇을 답하는가**, **어디까지 해야 충분한가**를 정리.

대상 슬롯: `mat_eq_eq_lam_pap` (winner) — 12개 q_value 변형으로 sweep.

---

## 1. 핵심 질문

```
"winner의 q=0.003 선택은 합리적인가? 다른 q로 바꾸면 결과가 크게 달라지나?"
```

이 한 문장에 답하기 위해 다섯 단계.

---

## 2. 분석 5단계 — 정당화 (필요성)

### M1. 메트릭 테이블

**무엇**: 12개 q 변형 × 메트릭 (Sharpe/Sortino/Sortino_IR/CAGR/MDD/Beta/Alpha/Turnover/레짐별 분해)

**왜 필요**:
- 모든 후속 분석의 **raw data**. 이게 없으면 시작 불가.
- 절대 필요 (★★★)

**대안 없음**.

---

### M2. 라인 플롯 (전체기간 + 레짐별 6 subplot)

**무엇**: q_value 변화에 따른 Sharpe/Sortino/MDD 곡선 시각화. winner 위치(빨간점) + optimum 위치(초록 화살표) annotation.

**왜 필요**:
- 표만 봐서는 **곡선 모양** 파악 어려움 (역U자? 단조? 평탄?)
- 레짐별 분해(R1/R2/R3)로 **레짐 의존성** 즉각 확인 가능
- 발표·논문 핵심 그림 1장으로 작동
- 절대 필요 (★★★)

**대안**: 표만으로 결론 가능하지만 청자/독자 설득력 크게 떨어짐.

---

### M3. 통계 검정 (Jobson-Korkie + Block Bootstrap)

**무엇**:
- Jobson-Korkie z-test (Memmel 보정): Sharpe 차이의 통계적 유의성
- Block Bootstrap (B=2000, block=3M): Sortino 95% 신뢰구간

**왜 필요**:
- 168개월 표본에서 Sortino 차이 0.011 (winner 1.978 vs optimum 1.989) 같은 **작은 차이가 통계적으로 의미 있는지** 판정
- 통계 검정 없으면 "0.6% 더 좋은 q로 바꿔야 하나?" 같은 잘못된 결론 가능
- 학술적·발표용 "robust 결론" 정당화 핵심
- **상황별**:
  - 학부 졸업 프로젝트: ★★ (있으면 좋고, 없어도 그래프로 설득 가능)
  - 학회 발표/논문: ★★★ (필수 — 통계적 유의성 없는 비교는 거부됨)
  - 운용 의사결정: ★★ (실무는 보수적이라 plateau 안에 있는지 확인이 더 중요)

**대안**: M2 곡선의 plateau 폭만으로도 robust 주장 가능. 하지만 정량 근거 약함.

---

### M4. 실현 Q 분포 분석

**무엇**: q_mode='lambda'의 동적 변환 (Q_t = q_base × clip(λ_t/2.5, 0.1, 3.0))이 q_base 변화를 어떻게 반영하는지. Box plot + 시계열.

**왜 필요**:
- "**메커니즘 검증**" 단계. q_base 차이가 실효 Q에 반영되는지 확인.
- 만약 clip(0.1, 3.0)에 자주 걸리면 q_base 차이가 흡수됨 → 곡선 평탄의 **원인** 설명 가능
- 본 데이터에선: winner의 mean/base = 1.96 → λ가 강세장 시기 dominant. 의미 있는 정보.
- **상황별**:
  - 발표 30분 이상: ★ (메커니즘 깊이 설명 시간 있을 때)
  - 발표 짧음(15분 이내): △ (생략 가능)
  - 논문: ★★ (학술 검증에 필요)

**대안**: 생략해도 결론 도출 가능. 하지만 "왜 평탄한가?" 같은 질문 받으면 답 못 함.

---

### M5. Drawdown event attribution

**무엇**: Worst 5월 union → 각 q의 그 월 수익률 비교. Heatmap + 막대그래프.

**왜 필요**:
- "**위기 시 q별 보호 효과**" 정량화 — 본 프로젝트 "성향별 ETF" 컨셉과 직결
- 단일 사건(2020-03 COVID, 2022-09 베어)이 R3 변동 sortino 큰 폭 차이를 만든 주범인지 확인
- "안정형 q=0.001은 winner보다 X%p 보호" 같은 **마케팅 가능한 수치** 산출
- **상황별**:
  - 본 프로젝트(성향별 ETF): ★★ (안정형 차별화 근거)
  - 일반 sensitivity 분석: △ (생략 가능)

**대안**: M2 레짐 곡선만으로도 R3 sensitivity 보임. M5는 그것의 **시점·크기 attribution**.

---

## 3. 우선순위 요약

| 단계 | 학부 프로젝트 | 학회/논문 | 운용 결정 | 발표 (15분) | 발표 (30분+) |
|---|---|---|---|---|---|
| M1 (표) | ★★★ | ★★★ | ★★★ | ★★★ | ★★★ |
| M2 (곡선) | ★★★ | ★★★ | ★★★ | ★★★ | ★★★ |
| M3 (통계) | ★★ | ★★★ | ★★ | ★ | ★★★ |
| M4 (메커니즘) | ★ | ★★ | △ | △ | ★★ |
| M5 (event) | ★★ | ★★ | ★★ | ★★ | ★★★ |

**최소 권장 (M1+M2)**: q 민감도 검증의 핵심. 곡선 모양 + 레짐별 패턴 보고 결론 도출.
**표준 (M1+M2+M3)**: 통계적 robustness 주장 가능. 학술 standard.
**확장 (M1~M5)**: 메커니즘 + event attribution까지. 학회 발표/논문 수준.

---

## 4. 본 프로젝트의 권고

### 이번 분석에 5단계 모두 필요한가?

**M1, M2, M3은 필수** — winner robustness 주장의 근간.

**M4는 선택** — 발표 시간이 충분(20분 이상)하고 "왜 lambda mode를 썼는가" 같은 질문 예상되면 포함. 그렇지 않으면 생략 가능.

**M5는 권장** — 본 프로젝트의 "성향별 ETF" 컨셉과 정확히 매칭. 안정형/공격형 q 차별화 정당화에 직접 활용 가능. 발표·심사에서 강점.

### 만약 시간 제약으로 줄여야 한다면

**최소 버전 (M1 + M2 + M3)**:
- 결론: "winner q=0.003은 plateau 영역의 robust optimum (Bonferroni 후 다른 q와 통계 차이 무의미)"
- 충분히 학부 졸업 프로젝트 수준 보장

**표준 버전 (M1 + M2 + M3 + M5)**:
- 위 + "위기 시 보수적 q (0.001~0.002)가 안정형 변형으로 활용 가능 (X%p 보호)"
- 발표 강력, 시간 ~20분

**완전 버전 (M1~M5)**:
- 위 + lambda mode 메커니즘 검증
- 학술적 정밀도 ★ + 시간 ~25분

---

## 5. 산출물 위치

```
final/outputs/99_analyze/
├─ M_winner_q_sensitivity.png       (M2 — 6 subplot 라인플롯)
├─ M3_winner_q_bootstrap_ci.png     (M3 — Sortino CI)
├─ M4_realized_q_distribution.png   (M4 — Box + 시계열)
└─ M5_drawdown_events.png           (M5 — Heatmap + 막대)
```

CSV 산출 없음 (모두 노트북 안 display로 확인). PNG만 발표·문서용 보관.

---

## 6. 결론 narrative 템플릿

본 sensitivity 분석에서 도출 가능한 5가지 핵심 메시지:

### A. Plateau optimum
> "winner q=0.003은 [0.001, 0.005] 영역에서 sortino 1.88~1.99 plateau에 위치한 robust optimum. 절대 optimum q=0.002 대비 차이 0.6%로 통계적 노이즈 수준."

### B. Regime dependence
> "정상기(R2 확장)는 큰 q(0.005~0.006), 위기기(R3 변동)는 작은 q(0.001~0.002)가 최적. winner q=0.003은 양 레짐 절반씩 양보한 합리적 타협점."

### C. Statistical robustness
> "11개 비교 중 Bonferroni 보정 후 winner와 통계적으로 유의한 차이 0개. Sortino 95% bootstrap CI도 모두 winner CI와 겹침 → q 변경의 실질 효과 입증 안 됨."

### D. Mechanism (M4 추가 시)
> "q_mode='lambda'의 동적 스케일링으로 winner 실현 Q는 q_base 0.003의 평균 1.96배(0.0059). λ_mkt가 평균 lam_mean(2.5)보다 큰 시기 dominant — 강세장 환경 반영."

### E. Risk profile differentiation (M5 추가 시)
> "위기 사건 [2020-03, 2022-09 등]에서 q=0.001은 winner보다 평균 X%p 작은 손실. 본 프로젝트의 안정형 ETF 변형은 q=0.001~0.002로 차별화 가능."

---

## 7. 비판적 자기 점검

### 본 분석의 한계

1. **표본 크기**: 168개월(K_CUT 적용)은 sortino 차이 ±0.05 노이즈 흔함. peak 차이가 그 이내면 우연 가능성.

2. **R2 dominance**: post-cut R2(~84M) > R3(48M) > R1(30M). 전체기간 곡선은 R2가 끌고감. 더 긴 위기 데이터 있으면 보수적 q의 우위가 더 커질 수도.

3. **q_mode='lambda' clip 효과**: 실효 Q ∈ [q_base × 0.1, q_base × 3.0]. 큰 q_base는 비례 확대가 아니라 조건부 확대 → "절대 q"가 아닌 "base scaling 효과"로 해석 필요.

4. **단일 검정 통과 ≠ 진실**: Jobson-Korkie 비유의 → "차이 없다"는 결론이 아니라 "유의성 못 입증"이 정확. 표본 더 크면 유의 차이 발견될 수도.

5. **MDD V-shape (q=0.004)**: 통계적 우연일 가능성. 다른 시드/기간으로 검증 시 사라질 수 있음.

### 이 한계를 발표에서 어떻게 다루나

- 보수적 표현 사용: "통계적 차이 입증 못 함" (not "차이 없음")
- 결론은 robust optimum 표현 (절대 우월 아님)
- M4/M5로 메커니즘과 attribution까지 보여 종합적 판단 가능하게

---

## 8. 후속 분석 (현재 안 함)

| 분석 | 의의 | 우선순위 |
|---|---|---|
| Walk-forward q* stability | rolling 36M에서 optimum q 안정성 | 후속 연구 |
| Pareto frontier (Sortino, MDD) | 다목적 최적 q 시각화 | 낮음 (현재 단조성으로 충분) |
| TC 분리 (gross vs net) | turnover 영향 격리 | 낮음 (이미 net 기준) |
| Information Ratio vs SPY 곡선 | 액티브 운용 품질 | 중 (Phase 2 holdout에서 유의미) |
| lam_mean 변화 sensitivity | q 외 다른 hyperparameter | 낮음 |

지금 M1~M5만으로 발표·논문에 필요한 q 민감도 분석은 완성. 나머지는 follow-up 영역.

---

## 9. 메타 — 왜 5단계 구조인가

각 단계는 다음 질문에 답:

```
M1: WHAT? — 어떤 값인가? (raw)
M2: HOW LOOK? — 어떤 모양인가? (visualization)
M3: SIGNIFICANT? — 우연인가 진짜인가? (inference)
M4: WHY? — 왜 그런가? (mechanism)
M5: WHERE? — 어디서 왔나? (attribution)
```

데이터 분석의 표준 흐름 (Tukey 1977, EDA → confirmatory): WHAT → HOW LOOK → SIGNIFICANT → WHY → WHERE.

본 문서의 5단계는 이 표준에 winner sensitivity의 도메인 특수성(레짐, lambda mode, 위기 사건)을 결합한 형태.

---

**문서 버전**: 2026-05-08 (initial)
**관련 노트북**: [99_analyze.ipynb](99_analyze.ipynb) §M
**관련 가이드**: [BL_EXPERIMENT_GUIDE.md](BL_EXPERIMENT_GUIDE.md)
