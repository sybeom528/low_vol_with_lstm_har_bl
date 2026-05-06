# 2026-05-05 작업 정리

> 오늘 추가/생성된 파일과 주요 대화 결론 모음.
> 충돌 해결 작업은 제외.

---

## 0. 새로 추가된 파일

### 코드 / 스크립트
| 경로 | 역할 |
|---|---|
| [김윤서/low_risk/p_weight_stability_eda.py](p_weight_stability_eda.py) | P 가중치 4종 안정성 EDA (HHI/turnover/섹터 노출) |
| [김윤서/low_risk/p_weight_mechanism_check.py](p_weight_mechanism_check.py) | mcap 강점 메커니즘 검증 (메가캡 익스포저, 위기 drawdown) |
| [김윤서/low_risk/all_slots_summary.py](all_slots_summary.py) | 4 슬롯 OFAT 종합 정리 |
| [김윤서/low_risk/regime_analysis.py](regime_analysis.py) | 레짐 기반 서브기간 분석 (Sortino + MDD + Sharpe) |

### 결과 문서 (RESULTS.md)
| 경로 | 내용 |
|---|---|
| [p_weight_stability_eda_RESULTS.md](p_weight_stability_eda_RESULTS.md) | P 가중치 4종 비교 결과 |
| [all_slots_summary_RESULTS.md](all_slots_summary_RESULTS.md) | prior/q/p/Ω 4 슬롯 종합 |
| [regime_analysis_RESULTS.md](regime_analysis_RESULTS.md) | 5개 레짐 분석 + SPY 필터 |

### 실험 추가 ([final/bl_config.py](../../final/bl_config.py))
- **omega 14개**: scaled_half/double/rmse 6개 (BASELINE 레벨) + q_lambda/q_raw_lam × Ω 4종 8개
- **inv_lambda × omega 4개**: 매트릭스 완성용
- **prior_rp 16개**: prior=rp + LSTM × q × p × Ω 18-cell 매트릭스 (실행 진행 중)
- (외부 추가) HRP 2개: hrp_trailing, hrp_lstm

총 실험 수: 67 → 85개

---

## 1. q_lambda / q_raw_lam / inv_lambda 방향성 정리

### 공식
| q_mode | 공식 | 약세장 동작 |
|---|---|---|
| `lambda` | Q = q_base × clip(λ/λ̄, 0.1, 3.0) | clip 0.1로 **최소 view 유지** |
| `raw_lam` | Q = max(0, q_base × λ_raw/λ̄) | **Q = 0 자연 OFF** |
| `inv_lambda` | Q = q_base × clip(λ̄/λ, 0.1, 3.0) | Q **강화** (위기에 view↑) |

### 학계 가설(저위험 anomaly가 위기에 강화)과의 정합성

| q_mode | 강세장 | 약세장 | 학계 일치? |
|---|---|---|---|
| lambda | view↑ | view 최소 유지 | **반대** |
| raw_lam | view↑ | view OFF | **반대** (lambda와 동일 방향) |
| **inv_lambda** | view↓ | **view↑** | **일치 ★** |

→ **raw_lam은 lambda와 같은 방향**이지 inv_lambda 방향이 아님. raw_lam의 약세장 OFF는 손실 보호 메커니즘이지 "위기에 anomaly 강화" 가설을 구현한 게 아님.

### 실증 (prior_eq + LSTM + p=mcap + Ω=he)

| 기간 | lambda | raw_lam | inv_lambda |
|---|---:|---:|---:|
| 2010-14 회복기 | 1.71 | 1.72 | **1.97 ★** |
| 2015-19 정상 | 0.93 | 0.97 | **1.05 ★** |
| 2020-24 강세 | 0.90 | 0.90 | 0.71 |
| **full** | 1.14 | **1.15** | 1.08 |

inv_lambda는 회복기·위기에 학계 가설대로 강력하지만, 표본이 강세 편향이라 풀 Sharpe는 raw_lam에 밀림.

---

## 2. P 가중치 4종 — mcap이 단연 우세

### 4 스킴 공식 (변동성 정렬 후 30% 컷)

| p_weight | long | short | 유니버스 |
|---|---|---|---|
| **mcap** | +m_i / Σm_low | −m_i / Σm_high | 30% 컷 |
| eq | +1/n_g | −1/n_g | 30% 컷 |
| rp | +(1/σ_i) / Σ(1/σ) | −σ_i / Σσ | 30% 컷 |
| vol_mcap | +(1−r)·m / Σ((1−r)·m) | −r·m / Σ(r·m) | **컷 없음** |

### 2020-2024 무너짐 — 진짜 원인은 메가캡이 아니라 **섹터 분산**

초기 가설("mcap이 메가캡 많이 보유해서 강함")은 데이터로 부정됨:
- 2020-24에 mcap도 메가캡 net **−2.6% short** (eq/rp는 +2~3% net long)
- 즉 메가캡 long 노출은 mcap이 더 적은데도 단독 1위

진짜 메커니즘 — **방어주 over-tilt 차이**:

| p_weight | 방어주(Util+Staples) | 경기민감(Fin+Ind) | 차이 | 2020-24 Sharpe |
|---|---:|---:|---:|---:|
| **mcap** | **38.7%** | **27.0%** | +11.7%p | **0.90** |
| eq | 49.8% | 22.5% | +27.3%p | 0.71 |
| rp | 45.4% | 25.1% | +20.3%p | 0.70 |
| **vol_mcap** | **63.7%** | **17.4%** | **+46.3%p** | **0.53** |

→ **방어주 초과 노출이 클수록 강세장에서 무너짐.** 거의 단조 관계.

### vol_mcap의 동작 (정정)

수식상으로는 모든 종목이 long_w와 short_w **두 basket 모두에 양수 비중**으로 들어감:
- AAPL 2020-12: long_w=3.7%, short_w=14.4%
- MSFT 2020-12: long_w=9.2%, short_w=2.4%

하지만 BL에 들어가는 P는 **net = long_w − short_w**라 종목당 한 부호:
- AAPL net P = **−10.7%** (실질 short)
- MSFT net P = **+6.7%** (실질 long)

### 섹터 분산 메커니즘

- **mcap**: 30% 컷 안 시총 가중 → 큰 종목이 섹터별 골고루 → 자연 분산
- **eq**: 30% 컷 안 동일가중 → 작은 utility 종목까지 동일 비중 → utilities 26.7%
- **rp**: 1/σ 가중 → σ 가장 작은 utilities/staples 집중
- **vol_mcap**: 컷 없음 → 거대 utility(ED, WEC) 시총 가중 → defensives 64%

---

## 3. Omega 5종 비교

### 옵션 정리 ([bl_functions.py:336-408](../../final/bl_functions.py#L336-L408))

| Ω | 공식 | 신뢰도 조절 신호 | 단위 |
|---|---|---|---|
| `he_litterman` (default) | τ·P·Σ·Pᵀ | 표준 He-Litterman | 수익률² |
| `scaled` (×0.5/2.0) | he × scale | 사용자 지정 신뢰도 | 수익률² |
| **`rmse`** | he × (LSTM RMSE/base)² | LSTM **변동성 예측 정확도** | log-vol error |
| **`paper`** (ff3_paper) | (전월 Q − 실제 P@r)² | view **수익률 예측 오차** | 수익률² |

**rmse는 변동성 측면, paper는 수익률 측면** — 학문적으로 paper가 BL 정합. rmse는 LSTM의 가용 정보(vol 예측 정확도)를 직접 활용하는 간접 방식.

### 결과 (prior_eq + LSTM + p=mcap + q=raw_lam)

| Ω | full | 2010-14 | 2015-19 | 2020-24 |
|---|---:|---:|---:|---:|
| he | 1.148 | 1.715 | 0.972 | 0.903 |
| **rmse** | **1.162 ★** | 1.713 | 0.978 | 0.929 |
| paper | 1.125 | 1.466 | 0.932 | **1.008 ★** |
| scaled_half | 1.142 | **1.832 ★** | 0.939 | 0.823 |
| scaled_double | 1.105 | 1.643 | 0.974 | 0.862 |

### Q × Ω 12-cell 매트릭스

| Q \ Ω | he | paper | rmse | scaled_half | scaled_double |
|---|---:|---:|---:|---:|---:|
| q_lambda | 1.140 | 1.065 | 1.134 | 1.143 | 1.082 |
| **q_raw_lam** | 1.148 | 1.125 | **1.162 ★** | 1.142 | 1.105 |
| q_inv_lambda | 1.079 | 1.042 | 1.085 | 1.051 | 1.045 |

omega_paper의 2020-24 구원 효과:
- q_lambda: 0.902 → 0.997 (+0.10)
- q_raw_lam: 0.903 → 1.008 (+0.10)
- **q_inv_lambda: 0.706 → 0.870 (+0.16) ★ 가장 큰 boost**

---

## 4. 4 슬롯 종합 정리

| 슬롯 | 효과 크기 | "기본값" 추천 | 빼도 되는 옵션 |
|---|---|---|---|
| **prior** | 작음 (±0.04) | `capm_eq` | rp (eq와 거의 동일) |
| **q** | 보통 (±0.07) | `raw_lam` | vol_spread, ff3 류 |
| **p_weight** | **큼 (±0.30)** | **`mcap` (단연)** | eq/rp/vol_mcap 모두 |
| **omega** | 보통 (±0.10) | **`rmse`** (안정), `paper` (위기 보호) | scaled_double |

### 단일 default 조합
```
prior=capm_eq + LSTM + p=mcap + q=raw_lam + Ω=rmse
→ Sharpe 1.162 (모든 서브기간 0.93+)
```

**※ inv_lambda는 빼지 말 것.** 풀 Sharpe로는 밀리지만 회복기·위기 시뮬레이션 시 단독 최강 + 학계 가설 일치 → 보험 가치 있음.

---

## 5. 레짐 기반 분석 (5개 레짐 + Sortino > MDD > Sharpe)

### 5개 레짐 정의 (데이터 기반 — SPY 누적·12M·DD 검증)

| 레짐 | 기간 | 개월 | 성격 | SPY Sortino |
|---|---|---:|---|---:|
| R1 회복 | 2010-01~2014-12 | 60 | Post-GFC | 1.90 |
| R2 확장 | 2015-01~2018-12 | 48 | 저변동 + 18Q4 | 1.15 |
| R3 COVID | 2019-01~2020-12 | 24 | Pre-COVID + 크래시 | 1.35 |
| R4 베어 | 2021-01~2022-12 | 24 | 22 베어 | 0.82 |
| R5 AI 랠리 | 2023-01~2024-12 | 23 | Mag 7 강세 | **4.63** |

기존 5년 균등분할(2010-14/2015-19/2020-24)은 시장 레짐 무시. 새 분할은 SPY drawdown 기반.

### 평가 우선순위 = **Sortino > MDD > Sharpe**

우리 포트폴리오 Sortino/Sharpe ≈ 1.8 → 변동성이 주로 상승 방향이라 Sortino가 더 정확한 평가.

### SPY 필터 결과

| 기준 | 5/5 | 4+/5 |
|---|---:|---:|
| Sortino+MDD+Sharpe (최엄격) | 0 | 2 |
| Sortino+MDD | 0 | 6 |
| Sortino만 | 0 | **9 ✓** |

**구조적 한계**: R5 AI 랠리에서 SPY Sortino 4.63은 BL 포트폴리오로 못 따라감. 저위험 anomaly(utilities/staples 비중)의 트레이드오프.

### Tier 4 우승 후보 (Sortino 4/5 SPY 초과, 모두 R5에서만 미달)

| 후보 | prior | full Sortino | full MDD |
|---|---|---:|---:|
| **q_raw_lam** | mcap | **2.17 ★** | -12.7% |
| q_lambda | mcap | 2.11 | -12.7% |
| prior_eq_q_raw_lam | eq | 2.06 | -14.3% |
| prior_eq_q_lambda | eq | 2.04 | -14.3% |
| p_vol_mcap | mcap | 2.02 | -13.9% |
| omega_rmse | mcap | 1.95 | -13.0% |
| mat_eq_eq_raw_he | eq | 1.93 | -15.2% |
| mat_rp_rp_lam_he | rp | 1.81 | -17.8% |
| mat_rp_rp_fix_he | rp | 1.55 | -17.5% |

---

## 6. 99_run.ipynb 보완 계획 (백그라운드 종료 후)

현재 H 섹션은 **Sharpe만 + 5년 균등분할 3 서브기간**. 추가할 사항:

1. **Sortino 서브기간 히트맵** — Sortino 수치 표시 (cell H에서 sortino도 이미 계산 중인데 sharpe만 사용)
2. **5개 레짐 분할로 변경** — R1 회복 / R2 확장 / R3 COVID / R4 베어 / R5 AI 랠리
3. **prior별 분리 히트맵 3장** — mcap / eq / rp 그룹
4. **SPY 4+/5 초과 후보 강조** — 우승자 표시

기존 .pkl 파일에서 `ret` 시리즈로 모두 재계산 가능 → 재실행 불필요.

---

## 7. 진행 중 / 다음 단계

- ⏳ **백그라운드 `bign5hu0h`** 실행 중 (prior_rp 7 + HRP 2 = 9개, 약 60분 남음)
- 📋 완료 후 **99_run.ipynb H 섹션에 Sortino 매트릭스 추가**
- 🎯 **prior_rp 18-cell 매트릭스 완성 후 재분석** — regime_analysis.py 재실행
- 📚 **HRP 결과 비교** — 기존 BL 조합 대비 robust 여부 확인
