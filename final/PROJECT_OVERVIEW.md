# 성향별 액티브 ETF 펀드 — Low-Risk BL 프로젝트 개요

> **최종 갱신: 2026-05-06** (LSTM 변동성 피봇 + Q 민감도 실험 반영)

---

## 1. 핵심 목표

> **저변동성 anomaly 가설을 LSTM σ 예측 + Black-Litterman 단일 view로 정형화하고,**
> **위험성향(공격/균형/안정)별 ETF 펀드 후보를 175+ 슬롯 조합으로 탐색.**

---

## 2. 단일 View Black-Litterman 구조 (K=1)

```
π = λΣw_mkt          (Prior: CAPM 균형 — mcap/eq/rp 가중)
+
P (1×N)              (저변동 30% long, 고변동 30% short — vol_pred 기준)
Q (스칼라)            (그룹 spread view 강도)
Ω (스칼라)            (view 분산)
─────────────────────────────────────────────────────────────
사후 (Sherman-Woodbury):
μ_BL = π + (τΣPᵀ)·(Q − Pπ) / (PτΣPᵀ + Ω)
↓
MVO 최적화 (max_weight 0.10, 자기자금 sum=1)
```

**왜 단일 spread view?** "고변동 종목은 risk-adjusted return으로 저변동을 못 따라간다"는 가설을 자산별 view로 표현하기 부자연스럽고 noisy(자산별 Q 예측 R² −0.95 실패 사례). 그룹 간 spread 1개 view가 직관·강건·해석 모두 우수.

---

## 3. 7-Step 파이프라인 (현 진행도)

```
Step 1: Data Collection         ✅ portfolio_prices, FRED, yfinance
Step 2: Preprocessing & EDA     ✅ monthly_panel, daily_returns
Step 3: LSTM σ 예측              ✅ ensemble_predictions_stockwise.csv
                                 (피봇: Q 직접 예측 → 변동성 예측, 2026-05-06)
Step 4: Black-Litterman (1-view) ✅ μ_BL, Σ_BL — 175 슬롯 조합
Step 5: MVO by Risk Profile      🟡 99_analyze에서 후보 선정 진행
Step 6: Backtesting (5-레짐)     ✅ 175 슬롯 + 39 Q 민감도 = 214 실험
Step 7: Streamlit Dashboard      ❌ 미착수
```

---

## 4. 파일 구조

```
final/
├── bl_config.py              ← EXPERIMENTS 정의 (현 214개)
├── bl_functions.py           ← BL 핵심 함수 (Σ/π/P/Q/Ω/BL/TC/Metrics)
├── master_table.py           ← results/*.pkl → mt/rt DataFrame 빌더
├── analyze_plots.py          ← 시각화 (대시보드, 매트릭스, 비교 차트)
│
├── 99_run.ipynb              ← walk_forward 실행 → results/*.pkl
├── 99_analyze.ipynb          ← 24-cell 분석 (J1~J6, K1~K6, K2-H)
│
├── results/                  ← 214 pkl (config + ret + comp + meta)
├── data/                     ← monthly_panel.csv, daily_returns.pkl, ff3_monthly.csv
├── phase3(data_outputs)/     ← LSTM 예측 산출물
│   └── data/ensemble_predictions_stockwise.csv
├── outputs/99_analyze/       ← 차트 PNG
│
├── PROJECT_OVERVIEW.md       ← 본 문서
├── BL_EXPERIMENT_GUIDE.md    ← 실험 정의/추가 가이드
├── 99_ANALYZE_GUIDE.md       ← 분석 노트북 셀별 해설
├── DATA_COLLECTION.md        ← 데이터 수집 파이프라인
├── ANOMALY_ANALYSIS.md       ← 저변동 anomaly EDA (02_LowRiskAnomaly)
└── Exploiting_LowRisk_Anomaly_BL_Summary.md   ← Pyo & Lee 2018 논문 정리
```

---

## 5. 슬롯 정의 (BL 변형 선택지)

| 슬롯 | 선택지 | 설명 |
|---|---|---|
| `prior` | `capm_mcap` / `capm_eq` / `capm_rp` | Prior π 가중 방식 |
| `p_mode` | `trailing_vol21` / `trailing_vol252` / `lstm_predicted` | P 분류 기준 변동성 |
| `p_weight` | `mcap` / `eq` / `rp` / `vol_mcap` | P 행렬 가중 |
| `q_mode` | `fixed` / `vol_spread` / `lambda` / `raw_lam` / `inv_lambda` / `none` / `capm` | Q 결정 방식 |
| `q_value` | float (기본 0.003) | fixed/vsp의 Q 값 또는 base |
| `omega_mode` | `he_litterman` / `scaled` / `rmse` / `ff3_paper` | Ω 계산 방식 |
| `omega_scale` | float (기본 1.0) | scaled에서 배수 |
| `tc` | float (기본 0.001 = 10bp) | 편도 거래비용 |
| `max_weight` | float (기본 0.10) | 단일 종목 상한 |

상세 슬롯 설명: [BL_EXPERIMENT_GUIDE.md](BL_EXPERIMENT_GUIDE.md)

---

## 6. 명명 체계 (5-token canonical)

| 컬럼 | 형식 | 예시 |
|---|---|---|
| `name` | 디스크 파일명 (mat_안: 4-token / 밖: semantic) | `mat_mcap_eq_raw_pap` / `baseline` |
| `canonical` | 분석용 5-token: `{prior}_{p_mode}_{p_weight}_{q}_{omega}` | `mcap_ls_eq_raw_pap` |

| 토큰 | 가능 값 |
|---|---|
| prior | `mcap` / `eq` / `rp` |
| p_mode | `tr` (trailing) / `ls` (LSTM) |
| p_weight | `mcap` / `eq` / `rp` / `volm` |
| q | `fix` / `lam` / `raw` / `inv` / `vsp` / `none` / `capm` |
| omega | `he` / `pap` / `rms` / `sh` (scaled_half) / `sd` (scaled_double) |

---

## 7. 5-레짐 안정성 평가

```
R1_회복   2010-01 ~ 2014-12   Post-GFC + 유럽위기
R2_확장   2015-01 ~ 2018-12   미국 회복 + 다중충격
R3_COVID  2019-01 ~ 2020-12   코로나 크래시 + 재개
R4_베어   2021-01 ~ 2022-12   인플레 + 22년 베어
R5_AI랠리 2023-01 ~ 2024-12   Mag7 강세
```

각 후보의 레짐별 Sortino/Sharpe/MDD → 안정성 지표(`sortino_ir`, `sharpe_ir`, `sortino_min`, `mdd_worst`)로 종합. 자세한 해석: [99_ANALYZE_GUIDE.md](99_ANALYZE_GUIDE.md).

---

## 8. 175+ 실험 매트릭스 + Q 민감도

**핵심 매트릭스 (LSTM 고정, 108 cells)**:
- prior(3) × p_weight(3) × q_mode(4: fix·lam·raw·inv) × omega(3: he·pap·rms) = 108
- + vol_spread q_mode 매트릭스 27 cells
- + trailing/scaled/he_double 등 비교군 약 40 cells
- = 175 base experiments

**Q 민감도 추가 (2026-05-06, 39 experiments)**:
- 5-레짐 sortino_ir Top 20 중 q_mode ∈ {fixed, vol_spread} 후보 13개
- × q_value ∈ {0.0055, 0.0064, 0.0070} (0.003 기존 baseline 재사용)
- 학술 근거: BAB(Frazzini & Pedersen 2014) US 0.0070 / global 0.0064 / 보수치 0.0055

총 EXPERIMENTS = 214

---

## 9. Look-ahead Bias 체크리스트

| 변수 | 사용 시점 | 상태 |
|---|---|---|
| `vol_21d`, `vol_252d` | pred_date 이전 N일 | ✅ 안전 |
| `log_mcap` | pred_date 시점 | ✅ 안전 |
| `fwd_ret_1m` | 평가(net_ret 계산)에만 사용 | ✅ 안전 |
| LSTM 예측 (`y_pred_ensemble`) | walk-forward 사전 생성 | ⚠️ 학습 분할 검증 필요 |
| Σ (LW 공분산) | 직전 60개월 일별 1260일 | ✅ 안전 |
| Q (`vol_spread`/`lambda`/`raw_lam`/`inv_lambda`) | 직전 월 vol_pred / λ | ✅ 안전 |
| Ω (`ff3_paper`) | 직전 월 예측오차² | ✅ 안전 (Bayesian rolling) |

> ⚠️ **omega_mode='ff3_paper' 명명 주의**: `compute_omega_paper`(FF3 회귀)는 dispatcher에서 호출되지 않음. 실제 로직은 walk_forward 안의 `Ω_t = (Q_{t-1} − actual_P_return_{t-1})²` (직전월 예측오차² 기반 적응형). FF3 논문과 무관하므로 발표 시 "직전월 예측오차² 적응형"으로 정확히 표현 권장.

---

## 10. 다음 단계

1. **Q 민감도 결과 분석** (39 백테스트 완료 후)
   - q_value별 IR 단조성 검증
   - BAB 학술 평균(0.0064/0.0070)이 0.003보다 robust한가?

2. **위험성향별 최종 후보 매핑** (K6)
   - 공격형: sortino_mean 최고
   - 균형형: stability_score 최고
   - 안정형: sortino_min > 0 + mdd_worst 얕음

3. **Step 7: Streamlit 대시보드** — 리밸런싱 추천 UI

---

## 11. 참고 문서

- [BL_EXPERIMENT_GUIDE.md](BL_EXPERIMENT_GUIDE.md) — 슬롯 정의 + 실험 추가 방법
- [99_ANALYZE_GUIDE.md](99_ANALYZE_GUIDE.md) — 분석 노트북 셀별 해설
- [DATA_COLLECTION.md](DATA_COLLECTION.md) — 01_DataCollection 파이프라인
- [ANOMALY_ANALYSIS.md](ANOMALY_ANALYSIS.md) — 저변동 anomaly EDA
- [Exploiting_LowRisk_Anomaly_BL_Summary.md](Exploiting_LowRisk_Anomaly_BL_Summary.md) — Pyo & Lee 2018 정리
- [../서윤범/project_design_v3.md](../서윤범/project_design_v3.md) — 전체 아키텍처 (피봇 전 기준이라 일부 outdated)
- [../김윤서/전체_프로젝트_프로세스.md](../김윤서/전체_프로젝트_프로세스.md) — End-to-end 프로세스
