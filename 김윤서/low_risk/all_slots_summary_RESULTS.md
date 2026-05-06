# 4 슬롯 옵션별 효과 종합 — Results

> 데이터: 67개 실험 (`final/results/*.pkl`), 2010-2024 walk-forward
> 분석 스크립트: [all_slots_summary.py](all_slots_summary.py)
> CSV/그림: [outputs/slots_summary/](outputs/slots_summary/)
> 평가 지표: 연환산 Sharpe (full + 3 서브기간 2010-14 / 2015-19 / 2020-24)

---

## 0. 한눈에 보는 결론

| 슬롯 | 옵션 수 | 효과 크기 | "기본값" 추천 | 빼도 되는 옵션 | 반드시 봐야 하는 옵션 |
|---|---:|---|---|---|---|
| **prior** | 3 (mcap, eq, rp) | 작음 (±0.04) | `capm_eq` | **rp 빼도 됨** (eq와 거의 동일) | mcap (baseline 비교용) |
| **q** | 5+ (fixed, lambda, raw_lam, inv_lambda, vol_spread, …) | 보통 (±0.07) | `raw_lam` | vol_spread, ff3 류 | **inv_lambda는 빼지 말 것** (회복기 1.97 최강, 보험 가치) |
| **p_weight** | 4 (mcap, eq, rp, vol_mcap) | **큼 (±0.30 in 2020-24)** | **`mcap` (단연 최적)** | eq/rp/vol_mcap 모두 약함 | mcap 외 다 EDA 비교용 |
| **omega** | 5 (he, paper, rmse, scaled_half, scaled_double) | 보통 (±0.10) | **`rmse`** (안정성), `paper` (위기 보호) | scaled_double (view↓로 손해) | rmse, paper |

**최강 단일 조합** = `prior_eq + LSTM + p=mcap + q=raw_lam + Ω=rmse`
→ full Sharpe **1.162**, 모든 서브기간 0.93+

---

## 1. Prior (시장 사전분포 가중치)

### 옵션별 계산
| prior | 공식 | 의미 |
|---|---|---|
| `capm_mcap` | w_i = m_i / Σm | 시총 비례 — 표준 CAPM 균형수익률 π = λΣw |
| `capm_eq` | w_i = 1/n | 모든 종목 동일 비중 — "정보 없음" 가정 |
| `capm_rp` | w_i = (1/σ_i) / Σ(1/σ) | 역변동성 (Risk Parity) — 안정 종목에 더 비중 |

### 효과 (LSTM + p=mcap + q=fixed + Ω=he 고정 OFAT)

| prior | full | 2010-14 | 2015-19 | 2020-24 |
|---|---:|---:|---:|---:|
| capm_mcap | **1.097** | 1.608 | 0.929 | **0.869** |
| capm_eq | 1.090 | 1.813 | 0.993 | 0.761 |
| capm_rp | 1.077 | **1.874** | 0.987 | 0.748 |

### 효과 (trailing + p=mcap + q=fixed + Ω=he 고정 OFAT)

| prior | full | 2010-14 | 2015-19 | 2020-24 |
|---|---:|---:|---:|---:|
| baseline (mcap) | **1.218** | 1.645 | 1.246 | **0.915** |
| prior_eq | 1.211 | 1.885 | 1.204 | 0.864 |
| prior_rp | 1.179 | **1.936** | 1.194 | 0.856 |

### 해석
- **mcap**: 강세장(2020-24) 보호. 시총 비례라 메가캡 prior π가 자연스럽게 잡혀, 이후 BL이 view를 더 안정적으로 풀어냄.
- **eq**: 회복기(2010-14)에 강함. 1/N은 모든 종목에 동일 베팅 → 회복기에 광범위한 종목 반등을 흡수.
- **rp**: 회복기 가장 강함(1.94 vs 1.65). 1/σ로 안정 종목에 prior 가중 → 회복기 변동성 축소 시기에 알파 증폭. 강세장엔 mcap에 밀림.

### 옵션별 추천 사용 시나리오
| 환경 가정 | 추천 prior |
|---|---|
| 표본 평균 (2010-2024 같은 강세 편향) | mcap 또는 eq |
| 회복기 위주 운용 | rp |
| **고객별 risk profile (보수적)** | rp (1/σ 자체가 보수적) |
| **고객별 risk profile (공격적/중립)** | eq |

### "rp 안해도 되나?" 답

**거의 그래도 됩니다.** OFAT 풀 Sharpe 차이는 0.02~0.04로 **노이즈 수준**. 다만:
- 회복기(2010-14)에 rp가 단독 1.94로 가장 강한 패턴이 일관됨 → **위기 회복기 시뮬레이션이나 보수적 포트폴리오 운용 시 쓸만함**
- 04 EDA에서 "1/N과 RP의 w* 상관 0.98"이 portfolio level에서도 그대로 확인됨

→ **eq를 기본으로, rp는 보수형 상품용으로 보관** 정도가 합리적.

---

## 2. Q (뷰 기대수익률)

### 옵션별 공식

| q_mode | 공식 | 의미 |
|---|---|---|
| `fixed` | Q = q_base (=0.003) | 정적 view (월 0.3%) |
| `lambda` | Q = q_base × clip(λ/λ̄, 0.1, 3.0) | λ 큼(강세) → view↑ |
| `raw_lam` | Q = max(0, q_base × λ_raw/λ̄) | λ_raw 음수(약세) → view OFF (자연 게이팅) |
| `inv_lambda` | Q = q_base × clip(λ̄/λ, 0.1, 3.0) | λ 작음(위기) → view↑ |
| `vol_spread` | Q = q_base × LSTM 예측 vol 스프레드/기준 | 예측 vol 격차 비례 |

여기서 λ = SPY 초과수익 / σ²_market.

### 효과 (prior_eq + LSTM + p=mcap + Ω=he 고정 OFAT)

| q_mode | full | 2010-14 | 2015-19 | 2020-24 |
|---|---:|---:|---:|---:|
| fixed | 1.090 | 1.813 | 0.993 | 0.761 |
| lambda | 1.140 | 1.707 | 0.927 | 0.902 |
| **raw_lam** | **1.148 ★** | 1.715 | 0.972 | **0.903** |
| inv_lambda | 1.079 | **1.974 ★** | **1.046 ★** | 0.706 |
| vol_spread | 1.076 | 1.833 | 1.042 | 0.713 |

### lambda vs raw_lam 차이 — 상세

두 방식 모두 "λ 큼 → view 강화"의 같은 방향이지만:

| 측면 | `lambda` | `raw_lam` |
|---|---|---|
| 입력 λ | walk_forward에서 clip(0.5, 10.0) 적용된 정규화 λ | clip 없는 raw λ (음수 가능) |
| 강세장 동작 | λ 커서 Q 강화 (clip 3.0 상한) | λ_raw 양수 → Q 양수 |
| 약세장 동작 | clip 0.1로 **최소 Q 유지** (절대 0 아님) | λ_raw 음수 → **Q = 0 자연 OFF** |
| 손실 보호 | 약함 (약세에 view 끌고 감) | **강함** (약세에 view 비활성, prior로 회귀) |
| 표본 결과 | full 1.140 | full 1.148 (+0.01) |

→ **raw_lam이 약세장 손실 보호 메커니즘 덕에 약간 더 안전**. 큰 차이는 아니지만 일관되게 우월.

### inv_lambda — 학계 가설과의 일치

저위험 anomaly 문헌(Frazzini-Pedersen 2014 등)은 **위기/고변동성 시기에 anomaly 강화**를 시사. inv_lambda가 정확히 그 방향:
- λ 작음(위기) → Q↑ → view 강화
- λ 큼(강세) → Q↓ → view 약화

데이터로 확인:
- 2010-14 회복기: **1.974 ★** (모든 옵션 중 최고)
- 2015-19 (2018Q4 약세 포함): **1.046 ★**
- 2020-24 강세장: 0.706 (학계 가설대로 view 약화 효과)
- 풀: 1.079 (강세장 비중 큰 표본이라 raw_lam에 밀림)

### 옵션별 추천 사용 시나리오

| 시장 환경 가정 | 추천 q_mode |
|---|---|
| 표본대로 강세 지속 | `raw_lam` 또는 `lambda` |
| 위기/조정 다수 (정상 사이클 가정) | `inv_lambda` |
| 시장 신호 무시, 단순 view | `fixed` |
| LSTM vol 신호 직접 활용 | `vol_spread` |

### "inv_lambda 안해도 되나?" 답

**빼지 마세요.** 이유:
1. **풀 Sharpe 차이는 작음** (1.079 vs raw_lam 1.148, 0.07 차이)
2. **2010-14, 2015-19에 raw_lam보다 0.27, 0.07 우세** — 표본의 절반에서 단독 최강
3. **"표본은 강세 편향"** — 향후 약세/조정 사이클이 평균만큼 돌아오면 inv_lambda가 raw_lam보다 robust
4. **학계 가설과 일치하는 유일한 옵션** — 이론적 정당성

→ raw_lam을 default로 운용하되 inv_lambda는 **위기 시뮬레이션·시나리오 분석용**으로 유지 권장.

---

## 3. P 가중치 (뷰 행렬)

### 옵션별 공식 (변동성 정렬 후 30% 컷)

| p_weight | long | short | 유니버스 |
|---|---|---|---|
| `mcap` | +m_i / Σm_low | −m_i / Σm_high | 30% 컷 |
| `eq` | +1/n_g | −1/n_g | 30% 컷 |
| `rp` | +(1/σ_i) / Σ(1/σ_low) | −σ_i / Σσ_high | 30% 컷 |
| `vol_mcap` | +(1−r)·m / Σ((1−r)·m) | −r·m / Σ(r·m) | **컷 없음, 전체** |

### 효과 (prior_eq + LSTM + q=raw_lam + Ω=he 고정 OFAT)

| p_weight | full | 2010-14 | 2015-19 | 2020-24 |
|---|---:|---:|---:|---:|
| **mcap** | **1.148 ★** | 1.715 | 0.972 | **0.903 ★** |
| eq | 1.069 | 1.793 | 0.932 | 0.716 |
| rp | 1.052 | 1.226 | **1.258 ★** | 0.720 |
| vol_mcap | 1.039 | **1.804 ★** | 0.943 | **0.526 ⚠** |

### 해석 — 메커니즘 핵심

**mcap이 단독 우세인 진짜 이유는 섹터 분산** (참고: [p_weight_stability_eda_RESULTS.md](p_weight_stability_eda_RESULTS.md)):

| p_weight | 2020-24 방어주 노출 | 경기민감 노출 | 차이 |
|---|---:|---:|---:|
| **mcap** | **38.7%** | **27.0%** | **+11.7%p** |
| eq | 49.8% | 22.5% | +27.3%p |
| rp | 45.4% | 25.1% | +20.3%p |
| vol_mcap | 63.7% | 17.4% | +46.3%p |

- **mcap**: 30% 컷 안에서 시총 가중 → low-vol 그룹의 큰 종목들이 섹터별로 골고루 → utilities/staples 비중 38.7%로 제한
- **eq**: 30% 컷 + 1/n → low-vol 그룹의 작은 utility들도 동일 비중 → utilities 26.7%로 폭증
- **rp**: 30% 컷 + 1/σ → σ 가장 작은 utilities/staples 집중 → defensives 45.4%
- **vol_mcap**: 컷 없음 + (1−vol_rank)·m → 거대 utility(ED, WEC) 시총 가중 → defensives 64% (top10 39%)

### 옵션별 효과 한 줄

| p_weight | 한 줄 |
|---|---|
| **mcap** | 30% 컷 + 시총 가중이 만드는 자연 섹터 분산. 모든 기간 0.90+, **유일한 robust 선택** |
| eq | 30% 컷 안 1/n. utilities 비중 폭증. 회복기엔 강하지만 강세장 약함 |
| rp | 1/σ 집중. utility/staples로 편중. 정상장(2015-19)엔 강함 (1.26) |
| vol_mcap | 컷 없는 연속함수. 거대 utility 집중. **2020-24에 0.53으로 무너짐** |

### "p=mcap이 제일 적합?" 답

**YES, 단연.** OFAT 풀 Sharpe 차이가 0.08~0.11로 4 슬롯 중 가장 큼. 2020-24 차이는 **0.18~0.38**까지 벌어짐. 다른 P 가중치는 모두 방어주 over-tilt를 만들어 강세장 알파 부족.

→ **p_weight=mcap을 고정 default**. 다른 옵션은 EDA/벤치마크용으로만.

---

## 4. Omega (뷰 불확실성)

### 옵션별 공식

| omega_label | 공식 | 의미 |
|---|---|---|
| `he_litterman` (default) | ω = τ·P·Σ·Pᵀ | 표준 He-Litterman 1999 |
| `scaled_half` | ω = he × 0.5 | view 더 신뢰 (2배 자신감) |
| `scaled_double` | ω = he × 2.0 | view 덜 신뢰 (절반 자신감) |
| `rmse` | scale = (LSTM RMSE / base)², ω = he × scale | LSTM 예측 정확도로 동적 신뢰도 |
| `paper` (ff3_paper) | ω = (전월 Q − 실제 P@r)² | 자기-적응: 직전월 view 오차² |

### 효과 (prior_eq + LSTM + p=mcap + q=raw_lam 고정 OFAT)

| omega | full | 2010-14 | 2015-19 | 2020-24 |
|---|---:|---:|---:|---:|
| he | 1.148 | 1.715 | 0.972 | 0.903 |
| **rmse** | **1.162 ★** | 1.713 | 0.978 | 0.929 |
| paper | 1.125 | 1.466 | 0.932 | **1.008 ★** |
| scaled_half | 1.142 | **1.832 ★** | 0.939 | 0.823 |
| scaled_double | 1.105 | 1.643 | 0.974 | 0.862 |

### 해석

- **he_litterman**: 균형. 모든 기간 적당히 잘 나옴. baseline.
- **rmse**: LSTM 예측 RMSE의 변동을 ω 신뢰도에 반영. 2020-24에 he 대비 +0.03 boost. **풀 Sharpe 최강 1.162, 모든 서브기간 0.93+** — 가장 robust.
- **paper**: 자기-적응이 강세장에 view 신뢰도 자동 차단. **2020-24 1.008 ★ (모든 옵션 중 최고)**. 단 회복기(2010-14)는 1.47로 양보 (자기-적응 학습 시간 필요).
- **scaled_half** (×0.5, view↑): 회복기 boost (1.83 ★). 강세장엔 0.82로 손해 — view 더 신뢰가 잘못된 방향이면 손실 키움.
- **scaled_double** (×2.0, view↓): 거의 모든 곳에서 약간 손해. view 약화로 prior에 가까워져 알파 손실. **추천 안 함**.

### Q × Ω 매트릭스 (prior_eq + LSTM + p=mcap)

| Q \ Ω | he | paper | rmse | scaled_half | scaled_double |
|---|---:|---:|---:|---:|---:|
| q_lambda | 1.140 | 1.065 | 1.134 | 1.143 | 1.082 |
| **q_raw_lam** | 1.148 | 1.125 | **1.162 ★** | 1.142 | 1.105 |
| q_inv_lambda | 1.079 | 1.042 | 1.085 | 1.051 | 1.045 |

**서브기간별 winner**:
- 2010-14: q_inv_lambda × he (1.974)
- 2015-19: q_inv_lambda × rmse (1.064)
- 2020-24: q_raw_lam × paper (1.008)
- Full: q_raw_lam × rmse (1.162)

### 옵션별 추천 사용 시나리오

| 운용 목표 | 추천 omega |
|---|---|
| **풀 기간 안정 Sharpe** | `rmse` |
| **강세장 보호 (위기 후반)** | `paper` |
| **회복기 알파 추구** | `scaled_half` |
| **표준 baseline** | `he_litterman` |
| 빼도 됨 | `scaled_double` |

---

## 5. 종합 — 운용 결정 트리

### 단일 default 조합 (가장 robust)
```
prior   = capm_eq
p_mode  = lstm_predicted
p_weight= mcap
q_mode  = raw_lam
omega   = rmse
```
**→ Sharpe 1.162, 모든 서브기간 0.93+** (`mat_eq_mcap_raw_rms`)

### 보수형 상품 (대안)
```
prior=capm_rp, p_weight=mcap, q=inv_lambda, omega=rmse
```
회복기·위기에 강함, 강세장 약함을 감수.

### 위기 보호 강화
```
omega=paper로 변경
```
2020-24 1.008로 강세장 단독 최강.

---

## 6. 빼도 되는 옵션 정리

| 슬롯 | 빼도 되는 옵션 | 이유 |
|---|---|---|
| **prior** | `capm_rp` | eq와 0.02 차이, 거의 동일 (보수형 상품 시만 유지) |
| **q** | `vol_spread`, `inv_lambda`, `ff3_*` | 풀 Sharpe 0.04~0.5 손해. 단 inv_lambda는 학계 가설 일치, 위기 시뮬레이션용 보관 |
| **p_weight** | `eq`, `rp`, `vol_mcap` | 모두 mcap에 0.08+ 패배. EDA 비교용으로만 |
| **omega** | `scaled_double` | 거의 모든 곳에서 약간 손해 |

→ 본 운용 코드는 **prior_eq, p=mcap, q=raw_lam(or lambda), Ω=rmse** 4개 핵심 + Ω=paper(위기 보호) + Ω=scaled_half(회복기 알파) 정도면 충분.

---

## 부록: CSV/그림

- [outputs/slots_summary/all_results.csv](outputs/slots_summary/all_results.csv) — 67개 실험 전체 결과
- [outputs/slots_summary/q_at_ref.csv](outputs/slots_summary/q_at_ref.csv) — Q 슬롯 OFAT
- [outputs/slots_summary/omega_at_ref.csv](outputs/slots_summary/omega_at_ref.csv) — Ω 슬롯 OFAT
- [outputs/slots_summary/pweight_at_ref.csv](outputs/slots_summary/pweight_at_ref.csv) — P_weight 슬롯 OFAT
- [outputs/slots_summary/heatmap_top30.png](outputs/slots_summary/heatmap_top30.png) — Top-30 실험 서브기간 히트맵
- [outputs/slots_summary/heatmap_top2024.png](outputs/slots_summary/heatmap_top2024.png) — 2020-24 강세장 robust 조합 Top-20
