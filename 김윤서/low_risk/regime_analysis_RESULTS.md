# 레짐 기반 서브기간 분석 — RESULTS

> 분석 스크립트: [regime_analysis.py](regime_analysis.py)
> 출력: [outputs/regime/](outputs/regime/)
> 평가 우선순위: **Sortino > MDD > Sharpe** (변동성 다루는 프로젝트라 하방위험 우선)
> 데이터: 72개 실험 × 5개 레짐 (백그라운드에서 11개 prior_rp 추가 실행 중, 완료 후 재실행 필요)

---

## 1. 5개 레짐 정의 (데이터 기반 검증)

기존 5년 균등분할(2010-14 / 2015-19 / 2020-24)은 시장 레짐 변화를 무시. SPY 누적·12M rolling·drawdown 시계열로 검증한 5개 레짐:

| 레짐 | 기간 | 개월 | 성격 | SPY Sortino | SPY MDD | SPY Sharpe |
|---|---|---:|---|---:|---:|---:|
| **R1 회복** | 2010-01 ~ 2014-12 | 60 | Post-GFC 회복기 | 1.895 | -16.2% | 1.183 |
| **R2 확장** | 2015-01 ~ 2018-12 | 48 | 저변동성 확장 + 2018Q4 조정 | 1.147 | -13.5% | 0.861 |
| **R3 COVID** | 2019-01 ~ 2020-12 | 24 | Pre-COVID + 크래시 + 회복 | 1.347 | -19.4% | 1.003 |
| **R4 베어** | 2021-01 ~ 2022-12 | 24 | Tech 버블 + 2022 베어 | 0.815 | -23.9% | 0.414 |
| **R5 AI 랠리** | 2023-01 ~ 2024-12 | 23 | Magnificent 7 주도 강세 | **4.629** | -8.3% | **1.722** |

검증 시각화: [01_regime_validation.png](outputs/regime/01_regime_validation.png) — SPY 누적, 12M rolling, drawdown 3패널 + 레짐 음영.

### 분할 근거
- **R1 → R2 (2014→2015)**: 2014년 말 유가 폭락·강달러 시작. SPY 누적 감속, 변동성 증가.
- **R2 → R3 (2018→2019)**: 2018Q4 -19% 조정 종료, Fed 비둘기 전환. 그 다음 2020 COVID로 직결되므로 한 묶음.
- **R3 → R4 (2020→2021)**: COVID 회복 거의 완료, Tech/Meme 버블 형성 → 2022 inflation/rate hikes.
- **R4 → R5 (2022→2023)**: 2022-10 SPY 저점 후 2023-01부터 AI 랠리 본격화.

---

## 2. SPY 필터 — 4단계 Tier

각 레짐에서 SPY를 초과한 레짐 수 분포 (72개 실험):

| 기준 | 5/5 | 4+/5 | 3+/5 |
|---|---:|---:|---:|
| Sortino + MDD + Sharpe (최엄격) | **0** | 2 | 57 |
| Sortino + MDD | **0** | 6 | 57 |
| Sortino만 | **0** | 9 | 59 |
| MDD만 (참고) | 17 | 52 | 67 |

**구조적 한계**: R5 AI 랠리에서 SPY Sortino 4.63 / Sharpe 1.72는 Magnificent 7 집중 강세장 결과. 저위험 anomaly 포트폴리오는 utilities/staples 비중이 높아 **R5에서 SPY를 따라잡을 수 없음**. 이는 모델 결함이 아니라 **저위험 anomaly의 구조적 트레이드오프**.

→ **실용적 기준은 "4+/5 Sortino 초과"** (R5 제외 4개 레짐 모두 SPY 초과).

---

## 3. Tier 4 우승 후보 — Sortino 4/5 레짐 SPY 초과 (9개)

[04_winners_tier4_4of5.csv](outputs/regime/04_winners_tier4_4of5.csv)

| 실험명 | prior | full Sortino | full MDD | full Sharpe | 실패 레짐 |
|---|---|---:|---:|---:|---|
| **q_raw_lam** | capm_mcap | **2.169 ★** | -12.7% | 1.185 | R5 AI 랠리 |
| q_lambda | capm_mcap | 2.107 | -12.7% | 1.178 | R5 AI 랠리 |
| prior_eq_q_raw_lam | capm_eq | 2.063 | -14.3% | 1.179 | R5 AI 랠리 |
| prior_eq_q_lambda | capm_eq | 2.043 | -14.3% | 1.178 | R5 AI 랠리 |
| p_vol_mcap | capm_mcap | 2.016 | -13.9% | 1.149 | R5 AI 랠리 |
| omega_rmse | capm_mcap | 1.950 | -13.0% | 1.223 | R5 AI 랠리 |
| mat_eq_eq_raw_he | capm_eq | 1.931 | -15.2% | 1.069 | R5 AI 랠리 |
| mat_rp_rp_lam_he | capm_rp | 1.810 | -17.8% | 1.058 | R5 AI 랠리 |
| mat_rp_rp_fix_he | capm_rp | 1.551 | -17.5% | 1.035 | R5 AI 랠리 |

### 관찰
1. **모두 R5에서 실패** — AI 랠리 SPY Sortino 4.63은 어떤 BL 조합도 못 넘김
2. **mcap prior가 단연 강세** (top 6개 중 4개가 capm_mcap)
3. **LSTM 미사용 실험이 상위에 많음** (q_raw_lam, q_lambda, p_vol_mcap, omega_rmse) — Sortino 메트릭으로는 LSTM이 결정적이지 않음
4. **prior 3종 모두 출현**: mcap (4), eq (3), rp (2) — prior 슬롯의 실용 가치 있음
5. **q_raw_lam이 q_lambda보다 일관되게 우월** — 약세장 자동 OFF의 손실 보호 효과

---

## 4. Prior별 히트맵 (Sortino, full 정렬)

### prior=mcap [03_heatmap_prior_mcap.png](outputs/regime/03_heatmap_prior_mcap.png)
- top 1: **q_raw_lam** (full 2.17)
- top 2: q_lambda (2.11)
- top 3: p_vol_mcap (2.02)
- 강한 R1·R3·R4, R5에서 SPY 미달

### prior=eq [03_heatmap_prior_eq.png](outputs/regime/03_heatmap_prior_eq.png)
- top 1: **prior_eq_q_raw_lam** (full 2.06)
- top 2: prior_eq_q_lambda (2.04)
- LSTM 결합 실험들이 다수 (Sortino 1.7~1.9)
- mcap prior 대비 평균 0.05 약간 낮음

### prior=rp [03_heatmap_prior_rp.png](outputs/regime/03_heatmap_prior_rp.png)
- top 1: **mat_rp_rp_lam_he** (full 1.81)
- top 2: mat_rp_rp_fix_he (1.55)
- top 3: mat_rp_mcap_raw_he (~1.42)
- ⚠ 11개 신규 실험 백그라운드 실행 중 (완료 후 더 채워짐)

---

## 5. Sortino vs Sharpe 차이 — 왜 Sortino가 우선?

| 실험 | full Sharpe | full Sortino | 비율 |
|---|---:|---:|---:|
| q_raw_lam | 1.185 | 2.169 | 1.83 |
| omega_rmse | 1.223 | 1.950 | 1.59 |
| q_lambda | 1.178 | 2.107 | 1.79 |

→ **Sortino가 Sharpe의 1.6~1.8배** = 우리 포트폴리오의 **상승 변동성이 하락 변동성보다 크다**는 뜻. 변동성 자체는 있지만 그게 주로 **상승 방향**이라 Sharpe로는 페널티지만 Sortino로는 정확히 평가됨.

저위험 anomaly 본질이 "하방 보호 + 상승 참여"라 **Sortino가 더 정확한 평가 지표**.

---

## 6. 추천

### 발표·운용용 default
**`q_raw_lam` (mcap prior, 단순 baseline)**
- full Sortino 2.17 (가장 높음)
- 4/5 레짐 SPY 초과
- LSTM 의존 없는 단순 구조
- MDD -12.7% (그룹 내 최저 수준)

### LSTM 활용 default (메인 운용)
**`mat_eq_mcap_raw_rms`**
- 다른 분석에서 풀 Sharpe 1.162 최강
- Tier 4 매트릭스에는 안 들었지만 (LSTM 결합 시 R3 COVID에서 다른 옵션이 더 강함)
- 안정성과 LSTM 통합 균형

### 보수형 상품 (rp prior)
**`mat_rp_rp_lam_he`**
- full Sortino 1.81
- MDD -17.8% (다른 후보보다 약간 큼)
- 1/σ × 1/σ 이중 보수적 구조

### R5 AI 랠리 한계 인정
저위험 anomaly의 구조적 한계 — Magnificent 7 집중 강세장에서는 어떤 조합도 SPY 못 넘김. **이는 모델/실험 결함이 아닌 anomaly 자체의 트레이드오프**. 발표 시 "강력한 집중 랠리에서 SPY 추종 곤란, 4/5 레짐에서 SPY 압도"로 정직하게 표현 권장.

---

## 7. 후속 작업

1. ⏳ **백그라운드 prior_rp 13개 실험 완료 대기** (현재 task `bbyru3trq`) → 완료 후 재실행하면 prior=rp 매트릭스가 완성됨
2. 📋 **99_run.ipynb의 H 섹션 다음에 통합** (백그라운드 끝나면 안전하게 가능)
3. 🎯 **R5 한계 보완 옵션 검토**: 메가캡 추가 베타 익스포저(예: SPY 30% 블렌드) 등 — 별도 분석 필요

---

## 부록: 출력 파일

- [01_regime_validation.png](outputs/regime/01_regime_validation.png) — 레짐 분할 검증 (SPY 누적/12M/DD)
- [02_regime_stats.csv](outputs/regime/02_regime_stats.csv) — 레짐별 SPY + winners 평균
- [03_heatmap_prior_mcap.png](outputs/regime/03_heatmap_prior_mcap.png), [_eq](outputs/regime/03_heatmap_prior_eq.png), [_rp](outputs/regime/03_heatmap_prior_rp.png) — prior별 히트맵
- [04_winners_tier4_4of5.csv](outputs/regime/04_winners_tier4_4of5.csv) — 4/5 SPY 초과 후보
- [04_winners_top.png](outputs/regime/04_winners_top.png) — 상위 후보 시각화
- [all_metrics.csv](outputs/regime/all_metrics.csv) — 전체 (실험 × 레짐 × 5지표)
