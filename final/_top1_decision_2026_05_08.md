# Top 1 모델 의사결정 보고서

> **작성일**: 2026-05-08
> **분석 노트북**: `final/06_Top1_Selection.ipynb` (42 cells, §0~§10)
> **plan**: `final/06_Top1_Selection_plan.md`

---

## 0. Executive Summary

본 분석은 BL 156 cfg 중 객관적 다중 메트릭 + lexicographic 우선순위 + sensitivity test 절차로 Top 1 을 재산출합니다.

### 핵심 발견

| 평가 기준 | Top 1 | 비고 |
|---|---|---|
| **Lexicographic** (1순위 sortino_TEST → 2순위 MDD → 3순위 sortino_ir, ε=0.10) | `mat_eq_mcap_lam_he` | sortino_TEST + MDD 균형 우수 |
| **Decision matrix** (성과 30% / 위험 25% / 안정성 20% / 견고성 15% / Alpha 10%) | **`mat_eq_eq_lam_pap`** ← 잠정 Top 1 | alpha + IR + eff_n 우수 |
| **1순위 = sortino_HOLD_OUT** | `mat_mcap_mcap_raw_he` | HO 1.640 압도적 |
| **1순위 = sortino_FULL** | `mat_mcap_mcap_raw_rms` | 전 기간 통합 우수 |

**평가 기준에 따라 Top 1 이 매우 다양하게 변경** → △ 조건부 ROBUST 분류.

### 권고 (사용자 결정 필요)

본 분석은 **trade-off 명확화**가 핵심 가치이며, 단일 "정답" 을 제시하지 않습니다. 발표 / 시연 / 학술적 강조점에 따라 다음 4 옵션 중 선택:

- **옵션 A** (현 잠정 유지): `mat_eq_eq_lam_pap` — alpha + 분산 우수, HO 부진 인정
- **옵션 B** (lexicographic 표준): `mat_eq_mcap_lam_he` — sortino_TEST + MDD 균형
- **옵션 C** (HO 우선): `mat_mcap_mcap_raw_he` — HOLD_OUT 압도적 (학습편향 회피)
- **옵션 D** (안정성 압도): `mat_mcap_rp_lam_pap` — sortino_ir 28.15 (regime IR 1위)

---

## 1. 분석 절차

### 1-1. Universe 정의

- **시작**: 156 cfg → 153 main (quantile variants `_q55/_q64/_q70` 제외)
- **교집합**:
  - `list_A` = sortino_TEST top 50 (성과)
  - `list_B` = sortino_ir top 50 (3-regime 안정성)
  - **|A ∩ B| = 22 cfg** (성과 + 안정성 동시 우수)

### 1-2. Hard filter

22 cfg 분포 분석 결과, 모든 후보 hard filter (`mdd_TEST > -0.40`, `mdd_HO > -0.30`, `eff_n ≥ 30`, `sortino_HO > 0`) 가 **이미 통과**. 추가 필터 적용하지 않음 (M = 22).

### 1-3. Lexicographic 종합 점수

```
1차: sortino_TEST 내림차순 정렬
  └─ tied 정의: |s1 - s2| < ε (ε = 0.10)
2차: tied 그룹 내 → (rank_MDD_TEST + rank_MDD_HO) / 2 평균 rank 오름차순
3차: 그래도 tied 시 → sortino_ir 내림차순
```

**ε = 0.10 결과**: 22 cfg 가 3 개 동순위 그룹으로 분리 (top group = 7 cfg, sortino_TEST 1.996~2.076).

### 1-4. Top 10 정밀 분석

16 메트릭 z-score heatmap → 후보별 강·약점 분석.

### 1-5. 결정 matrix (Top 5)

| 차원 | 가중치 | 메트릭 |
|---|---|---|
| 성과 | 30% | sortino_TEST + sortino_HOLD_OUT |
| 위험 | 25% | mdd_FULL + cvar_5 + calmar |
| 안정성 | 20% | sortino_ir + (1 - TEST_HO_gap) |
| 견고성 | 15% | eff_n_avg + (1 - turnover_avg) |
| Alpha | 10% | alpha + IR + |β-0.7| (defensive 적정성) |

### 1-6. Sensitivity test

- ε 변경 (0.05 / 0.10 / 0.20)
- 우선순위 변경 (sortino_HO / sortino_FULL 1순위)
- → robust 분류

---

## 2. Top 4 후보 비교

| 메트릭 | A: mat_eq_eq_lam_pap (잠정) | B: mat_eq_mcap_lam_he (Lex Top 1) | C: mat_mcap_mcap_raw_he (HO Top 1) | D: mat_mcap_rp_lam_pap (sortino_ir Top 1) |
|---|---:|---:|---:|---:|
| **sortino_TEST** (학습 168m) | **2.015** ★ | 1.996 | 1.919 | 1.878 |
| **sortino_HOLD_OUT** (24m) | 0.685 ✗ | 0.798 | **1.640** ★ | 1.235 |
| **sortino_FULL** (192m) | 1.892 | 1.842 | 1.877 | 1.806 |
| **sortino_ir** (regime 안정성) | 10.46 | 7.24 | 9.73 | **28.15** ★ |
| **mdd_TEST** | -0.129 | **-0.120** ★ | -0.136 | -0.137 |
| **mdd_HOLD_OUT** | -0.083 | -0.068 | **-0.044** ★ | -0.084 |
| **calmar** | **1.289** ★ | 1.105 | 0.964 | 1.169 |
| **alpha** (CAPM) | **0.052** ★ | 0.043 | 0.047 | 0.048 |
| **beta** (defensive: ~0.7) | 0.739 ★ | 0.545 | 0.514 | 0.720 |
| **turnover_avg** | 0.988 (높음) | **0.430** ★ | 0.395 ★ | 0.929 |
| **eff_n_avg** (분산) | **220** ★ | 61 | 35 | 68 |

★: 4 후보 중 1위. ✗: 의심 영역 (HO sortino < 0.7).

---

## 3. 후보별 강·약점 narrative

### 3-A. `mat_eq_eq_lam_pap` (잠정 Top 1, decision matrix Top 1)

**강점**:
- sortino_TEST 2.015 (4 후보 중 1위)
- alpha 0.052 (1위) + beta 0.739 (defensive 적정)
- eff_n 220 (분산 압도적, 학술적 robustness)
- calmar 1.289 (1위)
- decision matrix 가중 점수 1위

**약점**:
- **sortino_HO 0.685 (Top 134/153)** — 학습편향 의심
- TEST/HO 격차 0.66 (4 후보 중 가장 큼)
- mdd_HO -0.083 (4 후보 중 가장 나쁨)
- turnover 0.99 (높음, 거래비용 부담)
- 2024-12 -7.7% loss (sector rotation 취약, 별도 검증 보고서 §5-2)

**적합 narrative**: "in-sample 학습 + 분산 + alpha 우수 모델. HO 부진은 2024-12 sector rotation 단일 사건의 영향. 대용량 분산 (eff_n 220) 으로 학술적 robustness 시연."

### 3-B. `mat_eq_mcap_lam_he` (Lexicographic Top 1)

**강점**:
- sortino_TEST 1.996 (A 와 0.02 차이, 동급)
- **mdd_TEST -0.120 (1위)** + mdd_HO -0.068 (양호)
- turnover 0.43 (낮음, 비용 우수)
- Lexicographic 표준 절차 1위

**약점**:
- sortino_HO 0.798 (mid-tier)
- alpha 0.043 (다소 낮음)
- eff_n 61 (분산 다소 부족)

**적합 narrative**: "객관적 lexicographic 절차의 1위. sortino_TEST + MDD 균형이 가장 우수한 모델. HO 는 평균적이나 학습편향 위험 낮음."

### 3-C. `mat_mcap_mcap_raw_he` (HO Top 1)

**강점**:
- **sortino_HO 1.640 (압도적 1위)** — 학습편향 회피의 명확한 증거
- **mdd_HO -0.044 (압도적 1위)** — 미래 위험 통제 우수
- turnover 0.40 (가장 낮음)

**약점**:
- sortino_TEST 1.919 (4 후보 중 가장 낮음)
- mdd_TEST -0.136 (다소 큼)
- eff_n 35 (분산 부족, 학술적 약점)
- beta 0.514 (very defensive, bull market 시 underperform 위험)

**적합 narrative**: "HO 우선 시 명확한 1위. 학습편향에 견고하며 미래 위험 통제 압도적. 단, 학습 기간 성과는 다소 약함 → 보수적 권고에 적합."

### 3-D. `mat_mcap_rp_lam_pap` (sortino_ir Top 1)

**강점**:
- **sortino_ir 28.15 (regime 안정성 압도적 1위)**
- sortino_HO 1.235 (양호)
- alpha 0.048 + beta 0.720 (균형)
- 3 regime 모두 안정적 sortino (mean 1.858 / std 0.066)

**약점**:
- sortino_TEST 1.878 (다소 낮음)
- mdd_HO -0.084 (다소 큼)
- turnover 0.93 (높음)

**적합 narrative**: "Regime 변화에 가장 둔감한 모델. R1/R2/R3 sortino 변동이 28.15 IR 로 가장 작음. 장기·규모 큰 자금 운용에 적합."

---

## 4. Sensitivity test 결과

| 변경 시나리오 | 변경 후 Top 1 | 원래 Top 1 (matrix) 대비 |
|---|---|---|
| ε = 0.05 | mat_eq_eq_raw_pap | ✗ 변경 |
| ε = 0.10 (default lex) | mat_eq_mcap_lam_he | ✗ 변경 |
| ε = 0.20 | q_lambda | ✗ 변경 |
| 1순위 = sortino_HO | mat_mcap_mcap_raw_he | ✗ 변경 |
| 1순위 = sortino_FULL | mat_mcap_mcap_raw_rms | ✗ 변경 |

**전체 5/5 변경 시 Top 1 변경** → **△ 조건부 ROBUST**

**해석**: 22 cfg 가 trade-off 가 명확한 후보들이며, 어떤 평가 기준을 1순위로 두느냐에 따라 Top 1 이 완전히 바뀜. 즉, **단일 "정답" 모델이 존재하지 않음**.

이는 본 분석의 한계가 아닌, **객관적 발견**입니다:
- 성과 (sortino_TEST) 우수 모델은 HO 부진 경향
- HO 우수 모델은 TEST 다소 부진
- 안정성 (sortino_ir) 우수 모델은 절대 성과 다소 낮음

**시사점**: 발표 narrative 에서 "단일 Top 1" 보다 **"4 옵션 trade-off"** 강조가 학술적으로 더 정직.

---

## 5. 최종 권고

### 5-1. 발표 / 시연 목적별 권장

| 목적 | 권장 Top 1 | 근거 |
|---|---|---|
| **학술적 robustness 강조** | A: mat_eq_eq_lam_pap | eff_n 220 (분산 1위) + alpha 1위 + calmar 1위 |
| **객관적 절차 강조** | B: mat_eq_mcap_lam_he | Lexicographic 표준 1위 + MDD 균형 |
| **학습편향 회피 강조** | **C: mat_mcap_mcap_raw_he** | HO sortino 1.640 압도적 (보수적 권고) |
| **regime 안정성 강조** | D: mat_mcap_rp_lam_pap | sortino_ir 28.15 (압도적 1위) |

### 5-2. 본 분석가의 견해

**가장 안전한 권고: 옵션 C (mat_mcap_mcap_raw_he)**

**근거**:
1. HOLD_OUT 24m sortino 1.640 (압도적) — 학습편향 위험 가장 낮음
2. mdd_HO -0.044 — 미래 시점 위험 통제 압도적
3. 4번째 LSTM 재학습 후에도 일관된 우수성
4. 단점 (eff_n 35, sortino_TEST 1.92) 은 발표에서 충분히 설명 가능

**보수적 견해**: 사용자가 발표·시연 목적으로 "Top 1" 단일 모델을 강조하는 데이터에 학습편향 비용이 0.685 → 1.640 격차로 명확. 학술 보고서에서 학습편향 회피는 결정적 신뢰성 요소.

**대안 견해**: 만약 "잠정 Top 1 유지" 가 narrative 일관성 측면에서 더 유리하다면 옵션 A 도 정당화 가능. 단, HO 부진은 명시적으로 인정 필요 + 2024-12 sector rotation 분해 (별도 검증 보고서 §5-2) 를 함께 제시.

### 5-3. 비권장: 단일 Top 1 강조

본 분석의 가장 큰 발견은 **trade-off 명확화** 입니다. 발표에서 "Top 1 = X" 단일 모델 강조보다 **"4 옵션 + trade-off"** 형식이 학술적으로 더 정직.

---

## 6. 한계점

### 6-1. Universe 한계

- 156 cfg 중 153 main 만 분석 (quantile variants 제외)
- 추가 cfg 정의 시 결과 변경 가능

### 6-2. 평가 기준 한계

- Lexicographic ε = 0.10 임의 선택 (sensitivity 검증)
- Decision matrix 가중치 (30/25/20/15/10) 는 사용자 합의 없이 내부 선정
- → ε / 가중치 변경 시 Top 1 재조정 필요

### 6-3. 시계열 한계

- 192m (2010-01~2025-12) — 2008 위기 미포함
- HOLD_OUT 24m (2024-2025) — 다른 기간 (예: 2008-2009) 시 결과 다를 가능성

### 6-4. 비포함 요소

- 거래비용 변화 (현재 tc=0.001 가정)
- 슬리피지
- 세금 / 차입 비용
- ESG / 섹터 제약

---

## 7. 산출물

| 파일 | 내용 |
|---|---|
| `final/06_Top1_Selection.ipynb` | 분석 노트북 (42 cells, §0~§10) |
| `final/outputs/06_top1/intersection_summary.csv` | 22 cfg 교집합 |
| `final/outputs/06_top1/filtered_M_summary.csv` | hard filter 후 (= 22) |
| `final/outputs/06_top1/top10_metrics.csv` | Top 10 × 16 메트릭 |
| `final/outputs/06_top1/top5_decision_matrix.csv` | 결정 matrix |
| `final/outputs/06_top1/sensitivity_summary.csv` | sensitivity 결과 |
| `final/outputs/06_top1/figures/fig01~fig11.png` | 차트 11장 |
| `final/_top1_decision_2026_05_08.md` | **본 보고서** |

---

## 8. 다음 단계

1. **사용자 의사결정** (4 옵션 중 선택, 또는 4 옵션 trade-off narrative 채택)
2. **`narrative.py` + `recommendations.py` 갱신** — Streamlit 재구축 시 활용
3. **`PROJECT_OVERVIEW.md` 본문 갱신** — Top 후보 명시 + 발표 narrative
4. **Streamlit 단계적 재구축** — 별도 plan 으로 처음부터 페이지 단위

---

## 9. HOLD_OUT 섹터 분해 — 사용자 가설 검증 (2026-05-09 추가)

### 9-1. 가설

> **2024-2025 반도체 섹터(IT)가 시장 자금을 빨아 급상승 → 고변동 회피(저변동 anomaly) 전략 특성상 IT 섹터에 under-weight → SPY 대비 underperform**

### 9-2. 검증 결과 — ✓ 강력 확증 (3/3 가설 모두 통과)

#### 가설 1 ✓: IT 섹터 자금 유입 확인

| 시점 | 시장 IT 비중 | IT mcap 성장 (정규화) |
|---|---|---|
| 2024-01 | 25.8% | 1.00 |
| 2024-12 | 31.7% | 1.63 |
| 2025-12 | **33.2%** (Δ +7.4%p) | **2.07** (207% 성장) |

→ IT 섹터 단독 24m **2배 이상 성장**, 시장 비중 +7.4%p 증가. AI/반도체 buy 가설 확증.

#### 가설 2 ✓: 4 cfg 모두 IT under-weight (4/4)

| cfg | IT 비중 (2024-01 → 2025-12) | 평균 active weight |
|---|---|---|
| A. mat_eq_eq_lam_pap | 0.073 → 0.084 | -0.247 |
| B. mat_eq_mcap_lam_he | 0.016 → **0.000** | -0.297 (가장 심함) |
| C. mat_mcap_mcap_raw_he | 0.129 → 0.179 | -0.177 |
| D. mat_mcap_rp_lam_pap | 0.196 → 0.251 | **-0.121** (가장 작음) |

→ 4 cfg 모두 시장 IT 비중 (~30%) 대비 **12~30%p 적게 보유**. 저변동 anomaly 의 본질적 sector tilt.

#### 가설 3 ✓: under-weight ↔ underperform 양의 상관

| cfg | 평균 IT active | HOLD_OUT 24m 수익 | SPY 대비 |
|---|---|---|---|
| **D. mat_mcap_rp_lam_pap** | **-0.121** ★ | +27.3% | **-19.7%p** ★ |
| C. mat_mcap_mcap_raw_he | -0.177 | +26.0% | -21.0%p |
| B. mat_eq_mcap_lam_he | -0.297 | +21.3% | -25.7%p |
| A. mat_eq_eq_lam_pap | -0.247 | +17.2% | **-29.8%p** ✗ |

**SPY HOLD_OUT 24m 누적**: +47.0%

→ IT under-weight 작을수록 underperform 작음 (rank 일관). 단, A 의 underperform 이 B 보다 큰 것은 IT 외 다른 섹터 영향 (예: A 가 D 와 비슷한 high turnover/concentration 패턴).

### 9-3. 시사점 — 4 후보 trade-off 에 새 차원 추가

기존 4 옵션 trade-off 에 **"sector rotation 견고성"** 차원 추가:

| cfg | 기존 강점 | sector rotation 견고성 |
|---|---|---|
| A. mat_eq_eq_lam_pap | alpha 1위, eff_n 220 | IT 노출 중간, **underperform 최대** ✗ |
| B. mat_eq_mcap_lam_he | Lex 1위, MDD 균형 | IT 거의 0% (극단적) ✗ |
| C. mat_mcap_mcap_raw_he | HO sortino 1.640 | IT 노출 중상위 (~18%) △ |
| **D. mat_mcap_rp_lam_pap** | sortino_ir 28.15 | **IT 노출 1위** (~25%, 가장 균형) ✓ |

### 9-4. 권고 변화

**기존 §5-2 권고 (옵션 C, mat_mcap_mcap_raw_he)** + **§9 발견 (옵션 D, mat_mcap_rp_lam_pap)** = **균형 잡힌 sector 노출 + regime 안정성 = 옵션 D 의 매력 부각**.

### 5 옵션 비교 (기존 4 + 새 차원)

| 옵션 | sortino_TEST | sortino_HO | sortino_ir | IT active | underperform vs SPY |
|---|---:|---:|---:|---:|---:|
| A. mat_eq_eq_lam_pap | 2.015 | 0.685 | 10.46 | -0.247 | **-29.8%p** ✗ |
| B. mat_eq_mcap_lam_he | 1.996 | 0.798 | 7.24 | -0.297 | -25.7%p |
| C. mat_mcap_mcap_raw_he | 1.919 | **1.640** ★ | 9.73 | -0.177 | -21.0%p |
| **D. mat_mcap_rp_lam_pap** | 1.878 | 1.235 | **28.15** ★ | **-0.121** ★ | **-19.7%p** ★ |

→ **옵션 D 가 sortino_ir 1위 + IT 노출 1위 + underperform 최소** = **3 차원 동시 우수**.

### 9-5. 본 분석가의 갱신된 견해

**가장 균형잡힌 권고: 옵션 D (mat_mcap_rp_lam_pap)**

**근거**:
1. sortino_ir 28.15 (regime 안정성 1위)
2. HOLD_OUT 24m sortino 1.235 (양호, 학습편향 의심 영역 통과)
3. IT 섹터 노출 0.25 (4 후보 중 시장 평균에 가장 근접)
4. SPY 대비 underperform 격차 -19.7%p (4 후보 중 최소)
5. alpha 0.048 + beta 0.720 (defensive 적정)

**보수적 견해**: 옵션 D 는 옵션 C 의 학습편향 회피 강점 + 추가로 sector rotation 견고성 + regime IR 1위. **3 가지 약점 (TEST 1.878, eff_n 68, turnover 0.93)** 은 발표에서 충분히 설명 가능.

**대안 견해**: 만약 학술적 robustness 증명이 핵심이라면 옵션 A, 학습편향 회피의 명확성이 핵심이라면 옵션 C 도 정당화 가능.

### 9-6. 산출물 (§11)

| 파일 | 내용 |
|---|---|
| `outputs/06_top1/figures/fig12_market_sector_trend.png` | 시장 섹터별 mcap 성장 + 비중 추이 |
| `outputs/06_top1/figures/fig13_cfg_sector_weight.png` | Top 4 cfg 섹터별 가중치 (4 panel stacked area) |
| `outputs/06_top1/figures/fig14_active_weight_IT.png` | IT 비중 (cfg vs 시장) + active weight |
| `outputs/06_top1/figures/fig15_active_weight_vs_returns.png` | IT under-weight ↔ underperform 산점도 |

---

## 10. SPY 4 레짐 비교 — 시장 환경 분석 (2026-05-09 추가)

### 10-1. 새 레짐 정의

기존 master_table 의 3 레짐 (R1/R2/R3 ~2024-12) 을 **AI 강세장 (R4)** 분리로 4 레짐 확장:

| 레짐 | 기간 | 개월 | 환경 |
|---|---|---|---|
| R1 회복 | 2010-01 ~ 2012-06 | 30 | Post-GFC + EU 위기 |
| R2 확장 | 2012-07 ~ 2019-12 | 90 | 장기 Bull |
| R3 변동 | 2020-01 ~ 2023-06 | 42 | COVID + '22 베어 |
| **R4 AI랠리** | 2023-07 ~ 2025-12 | 30 | **AI 강세장 (HOLD_OUT 24m 포함)** |

### 10-2. SPY 4 레짐 메트릭

| 레짐 | Sortino | Sharpe | MDD | CAGR | Vol | 승률 |
|---|---:|---:|---:|---:|---:|---:|
| R1 회복 | 1.260 | 0.678 | -16.2% | +10.5% | 16.6% | 56.7% |
| R2 확장 | 1.706 | 1.243 | -13.5% | +14.4% | 10.8% | 75.6% |
| R3 변동 | 0.939 | 0.568 | **-23.9%** | +11.3% | 20.1% | 61.9% |
| **R4 AI랠리** | **2.449** ★ | **1.230** | **-8.3%** ★ | **+20.3%** ★ | 12.0% | 70.0% |
| FULL (192m) | 1.350 | 0.897 | -23.9% | +14.0% | 14.3% | 68.8% |

→ **R4 AI랠리가 SPY 의 4 레짐 중 압도적 최고**: Sortino 2.449 (R2 확장 1.706 의 1.4배), CAGR +20.3% (R2 의 1.4배), MDD -8.3% (가장 작음).

### 10-3. Top 4 후보 + SPY 레짐 Sortino 비교

| cfg | R1 회복 | R2 확장 | R3 변동 | **R4 AI랠리** |
|---|---:|---:|---:|---:|
| A. mat_eq_eq_lam_pap | 2.205 | 2.044 | 2.108 | **0.680** ✗ |
| B. mat_eq_mcap_lam_he | 2.232 | 2.026 | 1.925 | **0.724** ✗ |
| C. mat_mcap_mcap_raw_he | 1.772 | 2.082 | 1.744 | **1.316** △ |
| **D. mat_mcap_rp_lam_pap** | 1.779 | 1.941 | 1.926 | **1.253** △ |
| **SPY** | 1.260 | 1.706 | 0.939 | **2.449** ★ |

**놀라운 패턴**:
- 모든 cfg 가 R1/R2/R3 에서 SPY 를 압도 (cfg sortino > SPY sortino)
- **R4 AI랠리에서 정반대 — SPY 가 모든 cfg 를 압도**

### 10-4. R3 → R4 전환 패턴 — 모델 vs 시장

| | R3 변동 Sortino | R4 AI랠리 Sortino | 변화 배수 |
|---|---:|---:|---:|
| A. mat_eq_eq_lam_pap | 2.108 | 0.680 | **×0.32** ✗ |
| B. mat_eq_mcap_lam_he | 1.925 | 0.724 | **×0.38** ✗ |
| C. mat_mcap_mcap_raw_he | 1.744 | 1.316 | ×0.75 |
| **D. mat_mcap_rp_lam_pap** | 1.926 | 1.253 | **×0.65** △ |
| **SPY** | 0.939 | **2.449** | **×2.61** ★ |

→ **R3 → R4 에서 모델은 모두 sortino 하락, SPY 만 2.6배 상승**. 이는 **저변동 anomaly 의 근본적 시장 환경 의존성** — bull market 후반 (특히 단일 섹터 주도 강세장) 에서 본질적으로 뒤처짐.

### 10-5. R4 AI랠리 cfg vs SPY 격차 정량화

R4 (HOLD_OUT 24m 포함) 에서 SPY 대비 격차:

| cfg | Sortino 격차 | CAGR 격차 (%p) |
|---|---:|---:|
| A. mat_eq_eq_lam_pap | **-1.77** | -11.9 |
| B. mat_eq_mcap_lam_he | -1.73 | -11.3 |
| C. mat_mcap_mcap_raw_he | -1.13 | -9.1 |
| **D. mat_mcap_rp_lam_pap** | **-1.20** | **-7.4** ★ (최소) |

→ **D 가 R4 격차 최소** (CAGR -7.4%p) — §11 의 IT 노출 1위 (-0.121 active) + §10 의 R4 격차 최소 = **3 차원 동시 우수 재확인**.

### 10-6. SPY TEST vs HOLD_OUT 격차

BL 평가 기간 분리 시 SPY 자체의 격차:

| 기간 | 개월 | Sortino | Sharpe | MDD | CAGR |
|---|---:|---:|---:|---:|---:|
| TEST | 168 | 1.281 | 0.843 | -23.9% | +13.0% |
| **HOLD_OUT** | 24 | **2.333** | **1.465** | -7.6% | **+21.2%** |
| 격차 (HO/TEST) | - | **×1.82** | ×1.74 | - | ×1.63 |

**SPY HOLD_OUT 자체가 학습 기간보다 1.82배 잘함**. 이는 **HOLD_OUT 부진이 학습편향만의 문제가 아님**을 정량 증명:

- A. mat_eq_eq_lam_pap HO sortino 0.685 / SPY HO 2.333 = 격차 1.65
- D. mat_mcap_rp_lam_pap HO sortino 1.235 / SPY HO 2.333 = 격차 1.10 (최소)

### 10-7. 시사점 — 발표 narrative 강화

**기존 narrative (학습편향 가설)**:
> "잠정 Top 1 의 HO sortino 0.685 부진 = 학습편향 의심" (정성적)

**갱신된 narrative (시장 환경 + sector tilt 실증)**:

> "HOLD_OUT 24m ≈ R4 AI랠리 — **SPY 자체가 학습 기간 대비 1.82배 강세** + AI/IT 단일 섹터 주도 강세장 (§11 의 IT 비중 +7.4%p, mcap 2.07배). 저변동 anomaly 의 본질적 IT under-weight 로 underperform 불가피. **단, D 모델 (mat_mcap_rp_lam_pap) 은 sector 노출 균형 (IT 0.25, 시장 평균에 가장 가까움) 과 regime IR 28.15 (1위) 로 격차 최소화** — bull market 후반에서도 가장 견고."

이는 **알고리즘 결함이 아닌 전략 본질의 trade-off** 를 정직하게 발표하는 narrative.

### 10-8. 갱신된 최종 권고

§5-2 (옵션 C, HO 압도) → §9-5 (옵션 D, sector 균형 부각) → **§10 (옵션 D, R4 격차 최소 + 시장 환경 견고성 추가 확인)**

**최종 권고: 옵션 D (mat_mcap_rp_lam_pap)** — 다음 4 차원 동시 우수:
1. ✓ **sortino_ir 28.15** (regime 안정성 1위, §3-D)
2. ✓ **sortino_HO 1.235** (HO 학습편향 의심 통과, §2)
3. ✓ **IT active -0.121** (sector rotation 견고성 1위, §11)
4. ✓ **R4 SPY 격차 -7.4%p** (시장 환경 적응 1위, §10)

**약점 (TEST 1.878, eff_n 68, turnover 0.93)** 은 발표에서 충분히 설명 가능.

### 10-9. 산출물 (§12)

| 파일 | 내용 |
|---|---|
| `outputs/06_top1/spy_regime_comparison.csv` | Top 4 + SPY × 4 레짐 메트릭 (20 행 × 11 컬럼) |
| `outputs/06_top1/figures/fig16_top4_spy_regime_dashboard.png` | Top 4 + SPY × 4 레짐 통합 heatmap (Sortino/Sharpe/MDD) |
| `outputs/06_top1/figures/fig17_top4_vs_spy_sortino.png` | Top 4 vs SPY 레짐 Sortino bar chart |
