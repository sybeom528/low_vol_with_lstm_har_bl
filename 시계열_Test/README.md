# 시계열_Test — COL-BL 프로젝트 통합 명세서

> **상위 프로젝트명**: COL-BL (Su et al. 2026 ESWA 295 + Pyo & Lee 2018 PBFJ 51 결합)
> 핵심 가설: **"딥러닝 기반 변동성 예측을 Black-Litterman 포트폴리오에 통합하면 위험조정 수익이 향상되는가?"**

---

## 0. 폴더 전체 구조

```
시계열_Test/
├── README.md                                        # 본 문서 (통합 명세서)
├── Objective Black-Litterman views through deep learning.pdf   # 원조 논문 (Su et al. 2026)
├── 학습자료_주의사항.md                             # 협업 시 학습자료 활용 시 유의 사항
│
├── GARCH/                          # GARCH(1,1) 변동성 baseline (논문 비교용 외부 트랙)
├── Phase1_LSTM/                    # Phase 1 — LSTM 수익률 예측 (보존)
├── Phase1_GRU/                     # Phase 1 변형 — GRU 로 LSTM 결과 개선 시도
├── Phase1_5_Volatility/            # Phase 1.5 — 타깃 변동성 예측으로 분기 (LSTM v8 ensemble)
├── Phase2_BL_Integration/          # Phase 2 — Top 50 종목으로 BL 통합 (12.7년)
└── Phase3_Robust_Extensions/       # Phase 3 — 전체 S&P 500 + 17년 확장 (최종 단계)
```

> **단계 진화**: Phase 1 (수익률 LSTM) → Phase 1.5 (변동성 LSTM v8) → Phase 2 (BL Top 50) → Phase 3 (BL 전체 universe + Robust)
> **별도 트랙**: GARCH (논문 비교용 baseline)

---

## 1. 폴더 간 상호 의존성

### 1-1. 시간 순 진화 흐름

```
Phase1_LSTM (수익률 예측, FAIL: R²_OOS<0)
    ├──→ Phase1_GRU (LSTM → GRU 교체 시도)
    └──→ Phase1_5_Volatility (타깃을 변동성으로 변경 → PASS)
              │
              ↓
         Phase2_BL_Integration (Top 50, 12.7년)
              │
              ↓
         Phase3_Robust_Extensions (전체 S&P 500, 17년) ← 최종 확정 단계
```

### 1-2. 데이터·코드 의존 그래프

| 종속 폴더 | 의존 대상 | 의존 형태 |
|---|---|---|
| **Phase1_GRU** | Phase1_LSTM/results/raw_data/ | raw 가격 데이터 공유 (재다운로드 회피) |
| **Phase1_5_Volatility** | Phase1_LSTM/results/raw_data/ | raw 가격 데이터 공유 |
| **Phase2_BL_Integration** | Phase1_5_Volatility | `scripts/volatility_ensemble.py` 가 import 시 Phase 1.5 의 4개 모듈 (`dataset.py`, `models.py`, `train.py`, `baselines_volatility.py`) 동적 로드 |
| **Phase3_Robust_Extensions** | Phase1_5_Volatility | (재학습 시) Phase 2 와 동일한 동적 import |
| **Phase3_Robust_Extensions** | Phase2_BL_Integration | (선택) 02a §6 BL sanity check 에서 `ensemble_predictions_top50.csv` 비교 검증 read |
| **Phase3_Robust_Extensions** | (외부) `../서윤범/low_risk/` | 학술 baseline 비교 대상 (재현 검증, 결과 비교만) |

### 1-3. Phase 3 단독 동작 가능성 요약

| 시나리오 | 필요 폴더 |
|---|---|
| **분석·시각화·통계만** (현재 학습 결과 보유) | Phase 3 만 |
| `02a/02b` 재학습 | Phase 3 + Phase 1.5 (필수) |
| 02a §6 Phase 2 비교 sanity check | Phase 3 + Phase 1.5 + Phase 2 |

자세한 내용은 [`Phase3_Robust_Extensions/README.md`](Phase3_Robust_Extensions/README.md) 의 "외부 Phase 폴더 의존성" 섹션 참조.

---

## 2. 폴더별 명세

### 2-1. 📁 `GARCH/` — GARCH(1,1) 변동성 baseline (별도 트랙)

| 항목 | 내용 |
|---|---|
| **목적** | 전통적 시계열 변동성 모델 (GARCH(1,1)) 의 변동성 예측 + BL 통합 결과를 LSTM 기반 (Phase 1.5+) 과 비교 |
| **위치** | Phase 1~3 와 **독립적** (별도 baseline 트랙) |
| **주요 파일** | `garch_11_volatility.ipynb`, `garch_11_walkforward.ipynb`, `garch_m_bl_pipeline.ipynb` |
| **결과 문서** | `garch_11_해석.md`, `garch_11_walkforward_해석.md`, `garch_m_bl_결과.md`, `결과_요약.md` |
| **의존성** | 외부 `yfinance` 만 (다른 Phase 폴더와 무관) |

> 본 트랙은 학술 보고서 작성 시 "전통적 GARCH 대비 딥러닝 우위 검증" 용도로 활용됩니다.

---

### 2-2. 📁 `Phase1_LSTM/` — LSTM 수익률 예측 베이스라인 (보존)

| 항목 | 내용 |
|---|---|
| **목적** | LSTM 으로 SPY/QQQ 의 21일 누적 수익률 예측 — BL Q 입력의 정량 평가 |
| **결론** | **FAIL** — 일별 수익률의 자기상관이 약하여 R²_OOS < 0, Hit Rate < 0.55 (이론적 한계 확인) |
| **상태** | **보존, 변경 금지** — Phase 1.5/2/3 의 raw 데이터 공급원 |

#### 하위 구조

```
Phase1_LSTM/
├── README.md                       # 협업 진입점 문서
├── PLAN.md                         # 전체 구현 계획서
├── scripts_정의서.md               # scripts/*.py 공개 API 정의
├── 재천_WORKLOG.md, 윤서_WORKLOG.md, 하연_WORKLOG.md   # 팀원별 작업 일지
├── 00_setup_and_utils.ipynb        # 환경 부트스트랩
├── 01_data_download_and_eda.ipynb  # yfinance 다운로드 + EDA
├── 01_eda_statistics.ipynb         # 추가 통계 분석
├── 02_setting_A_daily14.ipynb      # 14일 후 수익률 예측
├── 02_setting_A_daily21.ipynb      # 21일 후 수익률 예측 (메인)
├── 02_setting_A_daily{1,5,21}_ceemdan.ipynb   # CEEMDAN 분해 + LSTM
├── results/                        # 학습 결과 (raw_data, predictions 등)
├── scripts/                        # 재사용 모듈 (dataset, models, train, metrics 등)
└── 논의사항/                       # 회의·결정 사항 메모
```

#### 핵심 의사결정

- 자산: SPY, QQQ (각각 독립 학습)
- 기간: 2016-01 ~ 2025-12 (10년)
- Walk-Forward: IS 231 / Purge 21 / Embargo 21 / OOS 21 / step 21 (López de Prado 2018)
- 손실: Huber(δ=0.01)
- 평가: Hit Rate, R²_OOS (Campbell & Thompson 2008)

---

### 2-3. 📁 `Phase1_GRU/` — GRU 변형 실험 (Phase 1 의 결과 개선 시도)

| 항목 | 내용 |
|---|---|
| **목적** | Phase1_LSTM 의 과적합 (R²_OOS<0) 개선을 위해 GRU (파라미터 27% 감소) 로 교체 후 동일 조건 재실험 |
| **결론** | LSTM 과 유사한 한계 — 변동성 예측으로 분기 결정 (Phase 1.5) |
| **상태** | 비교용 보존 |

#### 하위 구조

```
Phase1_GRU/
├── README.md, PLAN.md, scripts_정의서.md
├── 00_setup_and_utils.ipynb
├── 02_setting_A_daily21.ipynb      # GRU 일별 21일 후 예측
├── results/, scripts/              # LSTM 과 동일 구조
└── 논의사항/
```

#### Phase1_LSTM 와 비교

| 항목 | LSTM | GRU |
|---|---|---|
| 게이트 수 | 4 | 2 |
| 파라미터 (h=32) | ~4,480 | ~3,264 (-27%) |
| 셀 상태 | 있음 | 없음 |
| 소규모 데이터 과적합 | 높음 | 낮음 |

---

### 2-4. 📁 `Phase1_5_Volatility/` — 변동성 예측 분기 (핵심 입력 생성기)

| 항목 | 내용 |
|---|---|
| **목적** | 타깃을 "21일 누적 수익률" → **"21일 forward 실현변동성 (log-RV)"** 로 변경하여 PASS 도달 |
| **결론** | **PASS** — LSTM v8 Performance-Weighted Ensemble (LSTM v4 + HAR-RV) 이 baseline 대비 우위 확인 |
| **역할** | **Phase 2/3 의 ML 변동성 예측 모듈 공급원** (Phase 3 의 `volatility_ensemble.py` 가 4개 모듈 동적 import) |

#### 하위 구조

```
Phase1_5_Volatility/
├── README.md, PLAN.md, REPORT.md, 재천_WORKLOG.md
├── 00_setup_and_utils.ipynb
├── 01_volatility_eda.ipynb
├── 02_volatility_lstm.ipynb        # LSTM 단독 (1ch)
├── 02_v2_volatility_lstm_har3ch.ipynb  # 3ch (rv_d, rv_w, rv_m)
├── 02_v3_lstm_optuna.ipynb         # Optuna 하이퍼파라미터 탐색
├── 02_v4_lstm_optuna.ipynb         # 4ch (3ch + vix_log) Optuna
├── 02_v4_final_evaluation.ipynb    # v4 최종 평가
├── 03_baselines_and_compare.ipynb  # HAR-RV/EWMA/Naive/Train-Mean 비교
├── 04_har_rv_evaluation.ipynb      # HAR-RV 단독 평가
├── 05_multi_asset_evaluation.ipynb # SPY+QQQ+DIA+EEM+XLF+GOOGL+WMT 7종
├── 06_lstm_external_indicators.ipynb # 외부 지표 (VIX 등) 추가
├── 07_ablation_study.ipynb         # 각 구성요소 기여도 분리
├── 08_ensemble_evaluation.ipynb    # v8 Performance-Weighted Ensemble (최종)
├── results/
│   ├── lstm_v4_final/              # v4 최종 결과
│   ├── lstm_ensemble/              # v8 ensemble (Phase 2/3 의 핵심 입력)
│   ├── lstm_optuna_v4/             # Optuna 결과
│   ├── lstm_v6_9ch/, lstm_v7_ablation/  # 추가 실험
│   ├── comparison_report.md, ensemble_report.md, har_rv_diagnostics.md
│   └── ...
└── scripts/                         # ⭐ Phase 3 가 동적 import 하는 모듈
    ├── dataset.py                  # build_fold_datasets, walk_forward_folds
    ├── models.py                   # LSTMRegressor
    ├── train.py                    # train_one_fold
    ├── baselines_volatility.py     # fit_har_rv (HAR-RV)
    ├── targets.py                  # 변동성 타깃 생성
    ├── metrics.py                  # RMSE, QLIKE, R²_train_mean
    └── setup.py                    # 환경 부트스트랩
```

#### Phase 1 vs Phase 1.5 비교

| 항목 | Phase 1 | Phase 1.5 |
|---|---|---|
| 입력 | `log_ret` univariate | `log_ret²` univariate (1ch) → +`vix_log` (3ch_vix) |
| 타깃 | 21일 누적 log-return | 21일 forward log-RV |
| 손실 | Huber(δ=0.01) | MSE |
| 1차 지표 | Hit Rate, R²_OOS | RMSE on Log-RV, QLIKE, R²_train_mean |
| Baseline | zero/previous/mean | HAR-RV/EWMA/Naive/Train-Mean |
| 결과 | FAIL | PASS (v8 ensemble) |

---

### 2-5. 📁 `Phase2_BL_Integration/` — Black-Litterman 통합 (Top 50 → 12.7년)

| 항목 | 내용 |
|---|---|
| **목적** | Phase 1.5 ensemble 변동성 예측을 BL 의 P 행렬에 통합하여 Top 50 종목 12.7년 백테스트 |
| **핵심 질문** | "Phase 1.5 의 변동성 예측 향상이 BL 포트폴리오 위험조정 수익으로 이전되는가?" |
| **결론** | 부분적 PASS — BL_ml 이 BL_trailing 대비 일부 우위, 단 universe 가 Top 50 으로 제한적 |
| **다음 단계** | Phase 3 에서 전체 universe + 17년으로 확장 |

#### 하위 구조

```
Phase2_BL_Integration/
├── README.md, PLAN.md, REPORT.md, 재천_WORKLOG.md
├── 01_universe_construction.ipynb  # S&P 500 Top 50 (매년 갱신) universe 빌드
├── 02_data_collection.ipynb        # 일별 panel + 시장 데이터
├── 03_phase15_ensemble_top50.ipynb # Phase 1.5 ensemble 50 종목 학습
├── 04_BL_yearly_rebalance.ipynb    # BL 매년 리밸런싱 백테스트
├── 05_sensitivity_and_report.ipynb # τ/Q/거래비용 민감도 + 학술 보고
├── _build_*.py                     # 노트북 빌드 스크립트 (5개)
├── data/
│   ├── ensemble_predictions_top50.csv  # Phase 3 의 02a §6 비교 대상
│   ├── daily_panel.csv, market_data.csv, ff3_monthly.csv
│   ├── bl_weights_*.csv (5개 시나리오)
│   ├── metrics_7scenarios.csv
│   ├── jobson_korkie_test.csv, dm_test_results.csv, bootstrap_sharpe_diff.csv
│   └── ...
├── outputs/                        # 시각화, 학술 보고용 figure
└── scripts/                        # ⭐ Phase 3 의 scripts/ 와 동일 구조 (재사용)
    ├── setup.py                    # 환경 부트스트랩
    ├── universe.py                 # S&P 500 멤버십 추적
    ├── data_collection.py          # yfinance OHLCV 수집
    ├── covariance.py               # LedoitWolf 공분산
    ├── volatility_ensemble.py      # ⚠️ Phase 1.5 의 4개 모듈 동적 import
    ├── black_litterman.py          # BL 5함수 (서윤범 99 추출)
    ├── benchmarks.py               # Equal/Mcap Weight + SPY
    └── backtest.py                 # 백테스트 엔진
```

#### Phase 1.5 vs Phase 2 비교

| 항목 | Phase 1.5 | Phase 2 |
|---|---|---|
| 자산군 | 7 종목 | S&P 500 Top 50 (매년 갱신) |
| 데이터 빈도 | 일별 → 21일 forward | 일별 ret → ×21 월별 환산 |
| 평가 지표 | RMSE/QLIKE/R²_train_mean | Sharpe/Alpha/MDD/CumRet |
| Baseline | HAR-RV/EWMA/Naive | SPY/1/N/Mcap/BL_trailing |
| 분석 기간 | 2016~2025 (10년) | 2013-04~2025-12 (12.7년) |

---

### 2-6. 📁 `Phase3_Robust_Extensions/` — 전체 S&P 500 + 17년 확장 (최종 단계)

| 항목 | 내용 |
|---|---|
| **목적** | Phase 2 결과를 더 robust 한 환경에서 검증: (1) universe 확장 (Top 50 → 전체 624 종목), (2) 기간 확장 (12.7년 → 17년), (3) Cross-Sectional LSTM 추가 (02b), (4) 서윤범 99 baseline 직접 비교 |
| **결론** | (1) 서윤범 99 재현 PASS (Sharpe 1.203 vs 1.157, +3.96%), (2) ML 통합 효과 미확인 (BL_ml 가 BL_trailing 대비 Sharpe -0.11), (3) pct=0.30 최적값 아님 |
| **상태** | **현재 진행 중인 최종 분석 단계** |

#### 하위 구조 (요약)

```
Phase3_Robust_Extensions/
├── README.md                       # 종합 명세서 (분석 vs 재학습 의존성 명시)
├── PLAN.md, NOTEBOOK_TODO.md
├── 01_universe_extended.ipynb      # 624 종목 + 17년 panel
├── 02a_phase15_stockwise_extended.ipynb, 02a_v2.ipynb  # 종목별 LSTM 624개
├── 02b_phase15_cross_sectional.ipynb                   # Cross-Sectional LSTM (Ticker Embedding)
├── 03_BL_backtest_extended.ipynb, 03_v2.ipynb, 03_BL_..._legacy_static.ipynb
├── 04_compare_stockwise_vs_cross.ipynb, 04_v2.ipynb
├── 05a_eval_stockwise.ipynb, 05a_v2.ipynb              # 02a 단독 평가 (4-Layer)
├── 05b_eval_crosssec.ipynb, 05b_v2.ipynb               # 02b 단독 평가 (+Embedding)
├── 05c_eval_compare.ipynb, 05c_v2.ipynb                # 시나리오 통계 검정
├── 06_pct_sensitivity.ipynb        # pct 0.10~0.40 민감도
├── 03_v2.py, 06_pct_sensitivity.py # 백그라운드 실행용 스크립트 변환본
├── _build_*.py                     # 노트북 빌드 스크립트 (10개)
├── data/                           # 624 종목 panel + ML 학습 결과 + BL 캐시
└── scripts/                        # Phase 2 와 동일 구조 + diagnostics.py + models_cs.py
```

상세 내용은 [`Phase3_Robust_Extensions/README.md`](Phase3_Robust_Extensions/README.md) 참조.

#### Phase 2 vs Phase 3 비교

| 항목 | Phase 2 | Phase 3 |
|---|---|---|
| Universe | Top 50 yearly (74~95 종목) | **전체 S&P 500 (~624 종목)** |
| 분석 기간 | 2013-04~2025-12 (12.7년) | **2009-01~2025-12 (17년)** |
| ML 모델 | Stockwise LSTM 만 | Stockwise + **Cross-Sectional (Ticker Embedding)** |
| BL TAU | 0.05 | **0.1** (서윤범 99 일관) |
| BL LAM | dynamic clip | **2.5 fixed** (서윤범 99 일관) |
| Universe 모드 | Static | **Dynamic-Membership + Stale 필터** (look-ahead bias 차단) |
| 시나리오 | 5 | **9** (mcap/eq/rp 가중치 변형 포함) |
| 학술 비교 | 자체 baseline | **서윤범 99_baseline 직접 재현 검증** |

---

## 3. 외부 의존성 (시계열_Test 외부 폴더·자원)

| 자원 | 용도 | 위치 |
|---|---|---|
| **서윤범 baseline** | Phase 3 의 BL_trailing 비교 대상 (Sharpe 1.157) | `../서윤범/low_risk/99_baseline.ipynb` |
| **yfinance API** | 모든 Phase 의 raw 가격·시장 데이터 (SPY, VIX, ^TNX 등) | 인터넷 |
| **Wikipedia S&P 500** | Phase 2/3 universe 멤버십 추적 (역사적 편입·제외) | 인터넷 |
| **FRED (DGS3MO)** | 무위험 수익률 fallback (yfinance ^IRX 실패 시) | 인터넷 |
| **Ken French Library** | Fama-French 3팩터 (Phase 2 의 alpha 검정) | 인터넷 |
| **원조 논문** | Su et al. 2026 ESWA 295 — CGL-BL 재현 대상 | `Objective Black-Litterman views through deep learning.pdf` |

---

## 4. 권장 학습 순서 (신규 합류자)

```
[1] 시계열_Test/README.md (본 문서) — 전체 구조 파악
        ↓
[2] Phase3_Robust_Extensions/README.md — 최종 단계 명세 + 의존성
        ↓
[3] Phase1_5_Volatility/README.md — ML 모델 구조 (Phase 3 의 입력 생성기)
        ↓
[4] Phase2_BL_Integration/README.md — BL 통합 방법론 + 8 결정사항
        ↓
[5] Phase1_LSTM/README.md — 초기 시도와 한계 (배경 이해)
        ↓
[6] 각 폴더의 PLAN.md, REPORT.md — 상세 구현·결과 문서
        ↓
[7] 각 폴더의 노트북 — 실제 분석·학습·백테스트 흐름
```

### 단계별 보조 문서

| Phase | 주요 보조 문서 |
|---|---|
| 모든 Phase | `README.md`, `PLAN.md` |
| Phase 1, 1_GRU, 1_5 | `scripts_정의서.md`, `재천_WORKLOG.md` |
| Phase 1.5, 2, 3 | `REPORT.md` (학술 보고서 형식 결과 문서) |

---

## 5. 단계별 PASS/FAIL 게이트 요약

| Phase | 1차 지표 | 게이트 조건 | 결과 |
|---|---|---|---|
| **Phase 1** (수익률) | Hit Rate, R²_OOS | Hit Rate > 0.55 AND R²_OOS > 0 | **FAIL** (수익률 자기상관 한계) |
| **Phase 1_GRU** | 동일 | 동일 | FAIL (LSTM 과 유사) |
| **Phase 1.5** (변동성) | RMSE on Log-RV, QLIKE | HAR-RV baseline 대비 우위 | **PASS** (v8 ensemble) |
| **Phase 2** (BL Top50) | Sharpe, Alpha, MDD | BL_trailing baseline 대비 Sharpe ↑ | 부분 PASS |
| **Phase 3** (BL 전체) | Sharpe, ML 통합 효과 | 서윤범 99 재현 + ML 우위 | **재현 PASS, ML 우위 미확인** |

---

## 6. 핵심 학술 결과 (Phase 3 기준)

1. **서윤범 99_baseline 재현 PASS**: BL_trailing_mcap Sharpe = 1.203 (목표 1.157, +3.96%, ±5% 이내)
2. **ML 통합 효과 미확인**: BL_ml_sw_mcap (1.082), BL_ml_cs (1.094) 모두 BL_trailing_mcap (1.203) 대비 Sharpe 하락
3. **Weighting 효과**: ML 전략에서 eq/rp 가중치 (1.140, 1.135) 가 mcap (1.082) 보다 우수 — ML 의 변동성 ranking 신호가 시총 가중에 의해 오염
4. **pct 민감도**: 현행 pct=0.30 최적값 아님 — BL_ml_sw 는 0.15, BL_trailing 은 0.20 이 Sharpe 최적
5. **시기별**: 정상 강세장 (2012~2019) BL_trailing 압도, COVID+AI (2020~2025) 시총 가중 (McapWeight, SPY) 우위

---

## 7. 참고 문헌

- **Su, Z., et al. (2026)** — *Objective Black-Litterman views through deep learning*, Expert Systems with Applications 295 — ⭐ 본 프로젝트의 직접 재현 대상
- **Pyo, U. & Lee, J. (2018)** — Pacific-Basin Finance Journal 51 (BAB factor + 30% 양극단 분류)
- **Black, F. & Litterman, R. (1992)** — 원조 BL 모델
- **He, G. & Litterman, R. (1999)** — Goldman Sachs 백서 (TAU 표준)
- **Markowitz, H. (1952)** — 평균-분산 최적화
- **Ledoit, O. & Wolf, M. (2004)** — Shrinkage 공분산 추정
- **Campbell, J. & Thompson, S. (2008)** — R²_OOS (Phase 1 지표)
- **DeMiguel, V. et al. (2009)** — 1/N 의 강력한 baseline
- **López de Prado, M. (2018)** — Walk-Forward + Purge + Embargo (모든 Phase 표준)
- **Maillard, S. et al. (2010)** — Risk Parity (inverse-vol)
- **Diebold, F. & Mariano, R. (1995)** — 예측 정확도 검정
- **Patton, A. (2011)** — QLIKE 손실 함수
- **Jobson, J. & Korkie, B. (1981) / Memmel, C. (2003)** — Sharpe 차이 검정
- **Hansen, P. et al. (2011)** — Model Confidence Set
- **Gu, S., Kelly, B. & Xiu, D. (2020)** — Empirical Asset Pricing via ML (Ticker Embedding)
- **서윤범 99_baseline** — Phase 3 의 직접 비교 대상 (외부, `../서윤범/low_risk/`)

---

## 8. 변경 이력 (시계열_Test 통합 README)

| 날짜 | 변경 |
|---|---|
| 2026-05-01 | 시계열_Test/README.md 최초 작성 (Phase 1~3 + GARCH + 외부 의존성 통합 명세) |

---

## 9. 작성자 / 라이선스

- 작성: 본 프로젝트의 모든 Phase 노트북·코드·문서 분석을 기반으로 자동 생성
- 외부 데이터 출처: yfinance (Yahoo Finance), Wikipedia, FRED, Ken French Library
- 학술 비교 대상: 서윤범 `99_baseline.ipynb` (외부 폴더)
- 원조 논문: Su et al. 2026 ESWA 295 (`Objective Black-Litterman views through deep learning.pdf`)
