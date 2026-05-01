# low_risk 파이프라인 전체 구조

> 저위험 이상현상(Low-Risk Anomaly) 기반 액티브 ETF — GARCH + Black-Litterman 포트폴리오 구축

---

## 전체 흐름 요약

```
01_DataCollection
      ↓  monthly_panel.csv
02_LowRiskAnomaly        03_VolatilityEDA
(투자 테제 검증)           (ML 적용 타당성 검증)
      ↓                         ↓
04_Prior_Universe_Analysis (유니버스·Prior 설계 결정)
      ↓
05_VolatilityPrediction   (GARCH walk-forward 예측 + 평가)
      ↓  vol_predicted.csv
┌─────────────────────────────────────────────────────────┐
│  실험 단계 (파라미터 탐색)                                  │
│  06_Q_Sensitivity     최적 Q 값 민감도 분석 (GARCH+Baseline)│
│  08_BL_Q_Methods      GARCH × Q 추정 방식 5종 비교          │
└─────────────────────────────────────────────────────────┘
      ↓  Q_OPTIMAL (06에서 자동 로드)
09_Regime_Q_Portfolio    ← 레짐 적응형 BL (3단계 이산 Q)
10_Q_Adaptive_Comparison ← Q 적응 전략 비교 (6종 전략 실험)

── 참고용 ──────────────────────────────────────────────────
98_2006_baseline         금융위기 포함 20년 BL 베이스라인
99_baseline              GARCH 없이 현재 vol 직접 사용 베이스라인
```

---

## 공통 유틸리티: `bl_utils.py`

코어 BL 파이프라인(노트북 04, 06~10, 98, 99)에서 공유하는 함수 모음. 각 노트북은 imports 셀에서 `from bl_utils import *` 한 줄로 모든 공통 함수를 사용한다.

| 카테고리 | 함수 |
|---------|------|
| 핵심 BL | `compute_sigma`, `compute_pi`, `build_P`, `compute_omega`, `black_litterman`, `optimize_portfolio` |
| 성과 평가 | `performance(ret, rf, label, verbose=False)` — Sharpe / Calmar / MDD / 누적수익 dict 반환 |
| Q 추정 | `compute_Q_hist`, `compute_Q_momentum`, `compute_Q_lambda`, `compute_Q_pi_ratio`, `compute_Q_ff3` |
| FF3 | `download_ff3()` — Tuck 서버에서 월별 FF3 다운로드 (소수 단위) |
| 시각화 | `drawdown(ret)`, `rolling_sharpe(ret, rf, window=12)`, `shade_high_vol(ax, dates, ...)` |

이 모듈 하나로 이전에 7~8개 노트북에 중복 정의되어 있던 함수들을 단일 진실 원천(single source of truth)으로 통합했다. 함수 시그니처/이름 통일:
- `q_lambda` → `compute_Q_lambda`, `q_pi_ratio` → `compute_Q_pi_ratio`, `_download_ff3` → `download_ff3`
- `performance(ret, label)` → `performance(ret, rf, label)` (rf를 명시적 인자로)

---

## 파일별 상세 설명

### 01_DataCollection.ipynb

**역할**: 전체 파이프라인의 원시 데이터 생성

**입력**
- Wikipedia S&P500 역사적 편출입 내역 (scraping)
- yfinance: S&P500 전종목 + SPY + ^IRX 일별 OHLCV (2004~2025)

**처리 단계**
| 단계 | 내용 |
|------|------|
| 1 | Wikipedia 편출입 히스토리 역방향 재구성 → 월별 멤버십 구축 (생존편향 완화) |
| 2 | GOOGL/GOOG 중복 제거 (`DUPLICATE_DROP`) |
| 3 | 일별 가격 이상값 탐지 (하루 10배↑ 변화 = yfinance 오류) |
| 4 | 일별 피처 계산: 로그수익률, vol_252d, beta_252d, log_mcap |
| 5 | 월말 리샘플링 + S&P500 편입 종목만 필터링 → 패널 구성 |

**출력**: `data/monthly_panel.csv` — `(date, ticker)` 멀티인덱스, 컬럼: `ret_1m, vol_21d, vol_252d, beta_252d, log_mcap, spy_ret, rf_1m`

**파라미터**
- `PANEL_START = '2004-01-01'` (수집 시작)
- 이상값 기준: 하루 10배 이상 가격 변화

---

### 02_LowRiskAnomaly.ipynb

**역할**: 투자 테제 검증 — "저위험 주식이 고위험보다 위험 대비 수익이 높은가?"

**입력**: `data/monthly_panel.csv`

**방법**: 포트폴리오 정렬(Portfolio Sort)
- `beta_252d` / `vol_252d` 각각 기준으로 5분위 분류 (Q1=저위험, Q5=고위험)
- 동일가중 포트폴리오, 다음 달 수익률 측정 (look-ahead bias 방지)
- 전기간(2004~2025) + 최근(2016~2025) 구간 분석

**확인 지표**
- 분위별 연환산수익률, 변동성, Sharpe Ratio
- 누적 수익률 시계열 (vs SPY)
- Q별 평균 시가총액 순위 (소형주 편향 여부 확인)

**결론**: 저변동성 Q1이 고변동성 Q5 대비 높은 누적수익률 + 높은 Sharpe → 이상현상 확인

---

### 03_VolatilityEDA.ipynb

**역할**: ML 변동성 예측 적용의 통계적 타당성 검증

**입력**: `data/monthly_panel.csv` (vol_21d 컬럼 사용)

**검정 항목**

| 검정 | 목적 | 기준 |
|------|------|------|
| ADF (Augmented Dickey-Fuller) | 정상성 확인 — ML 회귀 적용 전제조건 | p < 0.05 |
| Ljung-Box (lag-1) | 자기상관 유의성 — 예측 가능성 존재 여부 | p < 0.05 |
| ARCH LM | 변동성 군집 확인 — 단순 평균보다 ARCH/GARCH가 유리한가 | p < 0.05 |

**출력**: 전종목 검정 결과 테이블 + 비율 요약  
(정상성 ~90%↑ / Ljung-Box ~85%↑ / ARCH ~85%↑ 유의 → GARCH 적용 근거 확보)

---

### 04_Prior_Universe_Analysis.ipynb

**역할**: BL 설계 파라미터 확정 — 유니버스 크기, Prior 타입, PCT_GROUP 결정

**입력**: `data/monthly_panel.csv`

**분석 항목**

| Part | 내용 |
|------|------|
| 1 | 커버리지 기준별 유니버스 크기 (50%~100%) |
| 1-1 | p/T 비율 진단 (Ledoit-Wolf 공분산 추정 품질) |
| 2 | Prior 비교 — 시가총액 vs 1/N |
| 3 | 유니버스 규모별 저위험 이상현상 신호 강도 |
| 4 | Mini Backtest — Prior 선택에 따른 BL 성과 비교 |
| 5 | P 행렬 가중치 비교 |
| 6 | 종합 판단 |
| 7 | 동적 유니버스 실험 (N=50/100/150/200/full) |
| 8 | PCT_GROUP IC 단조성 분석 (|t|≥2.0 기준 최소 pct 선택) |

**출력**: `outputs/04_*.png` 시각화 파일들, `outputs/04_dynamic_universe_stats.csv`

---

### 05_VolatilityPrediction.ipynb

**역할**: GARCH(1,1) walk-forward 변동성 예측 + Baseline(`vol_21d`) 대비 성능 평가

**입력**: `data/monthly_panel.csv`

**모델**: GARCH(1,1) — 03 EDA에서 ARCH 효과 85%↑ 유의 확인

**Walk-forward 설계**
- 학습 윈도우(`TRAIN_WINDOW=60`): 직전 60개월 월별 수익률 → GARCH 파라미터 피팅에 사용
- 최소 관측치(`MIN_OBS=36`): 36개월 미만이면 GARCH 미실행, 역사적 변동성(fallback)으로 대체
- 타겟: 다음 달 `vol_21d` (연환산, 1-step ahead)
- 스케일: 수익률 × 100 (percent, 수치 안정성)
- 예측 기간: **2011-01 ~ 2025-12** (`START_PRED = '2011-01-01'`)

**GARCH vs Baseline 평가** (구 04_5_GARCH_Evaluation 통합)

| 분석 | 핵심 지표 |
|------|----------|
| Rank IC 비교 | GARCH IC vs Baseline IC — 절대 예측 정확도 |
| P 행렬 일치율 | 저위험/고위험 분류가 두 방법 간 겹치는 비율 |
| P 분류 정확도 | Precision — 실제 저위험 종목을 맞히는 비율 |
| 변동성 분리도 | 예측 저위험 그룹의 실제 vol이 고위험보다 낮은가 |
| P 팩터 수익률 | long 저위험 − short 고위험 실현 스프레드 |
| 레짐별 분석 | SPY 12M 롤링 변동성 기준 고/중/저변동성 구간별 IC |

**핵심 발견**
- 전기간 IC: GARCH ≈ Baseline (근소 우위)
- **저변동성 레짐**: GARCH IC +0.051 우위 (73.3% 승률)
- **고변동성 레짐**: GARCH IC -0.006 역전 (45.0% 승률) → 고변동 구간에서 GARCH 신뢰도 저하 → 09 레짐 적응형 Q의 동기

**출력**
- `data/vol_predicted.csv` — `(date, ticker, vol_pred)` long format
- `outputs/05_garch/` — IC 비교, factor returns, p_overlap, p_precision 등 평가 산출물

---

### 06_Q_Sensitivity.ipynb

**역할**: Q_FIXED 민감도 분석 — **GARCH** + **Baseline(vol_21d)** 두 가지 vol 소스에서 최적 Q 탐색

> **TRAIN_WINDOW의 역할 (BL 노트북 공통)**: 05~08의 `TRAIN_WINDOW=60`은 GARCH 피팅이 아닌, **BL 입력값 추정에 사용되는 lookback 기간**이다. 구체적으로 직전 60개월 수익률로 Ledoit-Wolf 공분산 Σ, CAPM 균형 수익률 π = λΣw_mkt 등을 계산한다.

**입력**: `data/monthly_panel.csv`, `data/vol_predicted.csv`  
**분석 기간**: 2011-01 ~ 2025-12

**설계**
- Vol 소스: GARCH(04번 출력) + Baseline(vol_21d) 각각 독립 실행
- Q 후보: `[0.001, 0.002, 0.003, 0.005, 0.007, 0.010, 0.015, 0.020]`

**파라미터**
- `TRAIN_WINDOW = 60` / `TAU = 0.1` / `PCT_GROUP = 0.30`

**출력**
- `Q_CANDIDATES`별 성과 비교표 + 시각화 (GARCH·Baseline 겹쳐서 Sharpe vs Q 곡선)
- `outputs/06_Q_Sensitivity/q_sensitivity_stats.csv` — GARCH 성과표 (09_Regime_Q_Portfolio에서 자동 로드)
- `outputs/06_Q_Sensitivity/q_sensitivity_baseline_stats.csv` — Baseline 성과표 (06에서 자동 로드)

**관련 문서**: `06_Q_Sensitivity_RESULTS.md`

---

### 08_BL_Q_Methods.ipynb

**역할**: 예측 vol(GARCH 또는 LSTM) × Q 추정 방식 5종 성과 비교

> **이전 구조 정리(2026-05-01)**: 옛 `07_BL_Q_Comparison`(Baseline×5Q)과 옛 `08_BL_VolQ_Grid`(2×5 격자)를 합쳐, 예측 모델 vol 1개 × 5 Q 방식만 비교하는 단일 노트북으로 단순화. Baseline(`vol_21d`) 직접 비교는 06번 민감도 분석으로 충분하므로 제거.

**입력**: `data/monthly_panel.csv`, `data/vol_predicted.csv`, `outputs/06_Q_Sensitivity/q_sensitivity_stats.csv` (Q_FIXED 자동 로드)  
**분석 기간**: 2011-01 ~ 2025-12

**비교 대상 Q 방식**

| 방식 | 공식 | 특징 |
|------|------|------|
| `Q_FIXED` | 06번 최적값 자동 로드 | Sharpe 최대 기준 |
| `Q_hist` | mean(저위험 − 고위험) 60M window | 학습 기간 전체 평균 실현수익 |
| `Q_momentum` | mean(저위험 − 고위험) 12M window | 단기 추세 |
| `Q_lambda` | Q_FIXED × clip(λ_t/λ_mean, 0.1, 3.0) | 위험회피계수 스케일 |
| `Q_ff3` | FF3 회귀 → P·r̂ | 팩터모델 기반 |

**벤치마크**: CAPM (BL 없음), SPY

**파라미터**
- `TRAIN_WINDOW = 60` / `TAU = 0.1` / `PCT_GROUP = 0.30`
- `Q_FIXED` (06 자동 로드) / `MOMENTUM_WINDOW = 12` / `LAM_MEAN = 2.5`

**출력**: 7개 전략(Q 5종 + CAPM + SPY) 성과 비교표 + 3-panel 시각화 (누적수익/Sharpe/MDD), `data/q_methods_returns.csv`, `outputs/08_BL_Q_Methods/q_methods_stats.csv`, `q_methods_log.csv`(시점별 Q 값)  
**관련 문서**: `08_BL_Q_Methods_RESULTS.md`

> **vol 소스 교체**: 노트북은 `vol_predicted.csv`만 읽으므로, GARCH가 아닌 LSTM/Transformer 등 다른 예측 모델로도 동일 파일 스키마(date, ticker, vol_pred)만 맞추면 코드 수정 없이 그대로 적용된다. 자세한 가이드는 본 문서 하단 "## 예측 모델 교체 가이드" 참조.

---

### 09_Regime_Q_Portfolio.ipynb

**역할**: 최종 포트폴리오 — 레짐 적응형 BL. 시장 변동성 레짐에 따라 Q를 동적으로 조정하여 고변동성 구간의 GARCH 예측력 저하를 방어.

**동기**: 05 GARCH 평가 결과에서 고변동성 레짐의 GARCH 예측력 저하 확인 (IC -0.006 역전). Q를 레짐별로 다르게 설정해 잘못된 뷰 신호를 억제.

**입력**: `data/monthly_panel.csv`, `data/vol_predicted.csv`, `outputs/06_Q_Sensitivity/q_sensitivity_stats.csv` (Q_OPTIMAL 자동 로드; 없으면 기본값 0.003 사용)  
**분석 기간**: 2011-01 ~ 2025-12

**레짐 정의** (expanding quantile — look-ahead bias 없음)
```python
spy_roll_vol = spy_series.rolling(12).std() * sqrt(12)
q33 = spy_roll_vol.expanding(min_periods=24).quantile(0.33)  # 각 시점 기준
q67 = spy_roll_vol.expanding(min_periods=24).quantile(0.67)
```

| 레짐 | 조건 | Q 설정 |
|------|------|--------|
| 저변동성 | vol ≤ q33 | `Q_OPTIMAL` (GARCH 우위 최대) |
| 중간 | q33 < vol < q67 | `Q_OPTIMAL × 0.5` |
| 고변동성 | vol ≥ q67 | `0.0` (CAPM으로 수렴, BL 뷰 억제) |

**파라미터**
- `TRAIN_WINDOW = 60` / `TAU = 0.1` / `PCT_GROUP = 0.30`
- `Q_MID_SCALE = 0.5` / `Q_HIGH_VOL = 0.0`

**Walk-forward**: Regime-Q + Fixed-Q + CAPM + SPY 동시 실행 (공정 비교)

**출력**
- 4-panel 시각화 (누적수익률, 롤링 Sharpe, 레짐 타임라인, Q 추이)
- 레짐별 성과 breakdown 표
- Q_HIGH_VOL 민감도 분석 (0.0 / 0.001 / 0.0015 / 0.002 / Q_OPTIMAL)
- `outputs/09_Regime_Q_Portfolio/` 저장

---

### 10_Q_Adaptive_Comparison.ipynb

**역할**: Q 적응 전략 6종 성능 비교 실험. 08의 이산 레짐 방식 외에 연속 스케일링·신뢰도 기반 등 다양한 Q 결정 방법을 동일 walk-forward 루프에서 공정하게 비교.

**입력**: `data/monthly_panel.csv`, `data/vol_predicted.csv`, `outputs/06_Q_Sensitivity/q_sensitivity_stats.csv`  
**분석 기간**: 2011-01 ~ 2025-12

**비교 전략**

| 전략 | Q 산출 방식 | 핵심 아이디어 |
|------|-----------|--------------|
| Fixed-Q | Q = Q_OPTIMAL (고정) | 기준선 |
| Regime3 | 저→Q_OPTIMAL / 중→×0.5 / 고→0 | 08과 동일 방식 |
| Q_lambda | Q_OPTIMAL × clip(λ/λ_mean, 0.1, 3.0) | 위험회피계수 연속 스케일링 |
| Regime+λ | 고변동성→0, 나머지→Q_lambda | 하드스탑 + λ 미세조정 |
| GARCH_conf | Q_OPTIMAL × GARCH예측신뢰도 | GARCH 정확도(MAPE 기반)에 비례 |
| π_ratio | Q_OPTIMAL × clip(\|P·π_t\| / 기준스프레드, 0.1, 3.0) | CAPM 스프레드 비율 고정 |

**파라미터**
- `TRAIN_WINDOW = 60` / `TAU = 0.1` / `PCT_GROUP = 0.30`
- `LAM_MEAN = 2.5` / `Q_MID_SCALE = 0.5` / `CONF_WINDOW = 12`

**출력**
- 4-panel 시각화 (누적수익률, 낙폭, Sharpe 막대, Q 시계열)
- 레짐별 성과 분해 (전략별 최강 레짐 확인)
- `outputs/10_Q_Adaptive_Comparison/` 저장

---

### 98_2006_baseline.ipynb (참고용)

**역할**: BL 베이스라인의 장기 검증 — 금융위기(2008~2009) 포함

99_baseline과 동일 구조, 학습 윈도우만 24개월로 단축해 예측 기간을 2006년까지 확장.

| 항목 | 98_baseline | 99_baseline |
|------|------------|-------------|
| `TRAIN_WINDOW` | 24개월 | 60개월 |
| 예측 시작 | 2006-01 | 2009-01 |
| 금융위기 포함 | O | O (일부) |

---

### 99_baseline.ipynb (참고용)

**역할**: GARCH 없이 현재 vol_21d로 직접 분류하는 BL 베이스라인

- 논문 구현 참고: Pyo & Lee (2018)
- Baseline + GARCH 두 전략을 동일 노트북에서 비교
- Q: FF3 팩터 회귀 추정 기대수익률 (`q = P·r̂`)
- 08_BL_VolQ_Grid의 원형(prototype) 역할

---

## 데이터 파일 의존 관계

```
monthly_panel.csv              ← 01_DataCollection 출력
      ├── 02_LowRiskAnomaly
      ├── 03_VolatilityEDA
      ├── 04_Prior_Universe_Analysis
      ├── 05_VolatilityPrediction
      │         ↓
      │   vol_predicted.csv          ← GARCH(현재) 또는 LSTM(교체 가능)
      │         ├── 06_Q_Sensitivity   (예측 vol + Baseline 두 소스 민감도)
      │         │         ↓
      │         │   outputs/06_Q_Sensitivity/q_sensitivity_stats.csv          (예측 vol)
      │         │   outputs/06_Q_Sensitivity/q_sensitivity_baseline_stats.csv (Baseline)
      │         │         ├── 08_BL_Q_Methods         (예측 vol × 5Q 방식)
      │         │         │         ↓  data/q_methods_returns.csv
      │         │         ├── 09_Regime_Q_Portfolio   (예측 vol stats → Q_OPTIMAL 자동)
      │         │         └── 10_Q_Adaptive_Comparison(예측 vol stats → Q_OPTIMAL 자동)
      │         └── 99_baseline                       (Baseline vol 직접 사용)
      └── 98_2006_baseline
```

---

## 핵심 파라미터 일람 (코어 파이프라인 공통)

| 파라미터 | 값 | 의미 |
|---------|-----|------|
| `TRAIN_WINDOW` | 60 | BL 학습 윈도우 (월) |
| `TAU` | 0.1 | BL 불확실성 스케일 |
| `PCT_GROUP` | 0.30 | 저위험/고위험 분류 비율 (상하위 30%) |
| `MAX_WEIGHT` | 0.10 | 종목 최대 비중 (MVO 제약) |
| `START_PRED` | 2011-01 | 코어 파이프라인 예측 시작 (04~08 통일) |
| λ 범위 | [0.5, 10.0] | CAPM 위험회피계수 클리핑 |

---

## 파일 번호 체계

| 번호 | 파일 | 역할 | 성격 |
|------|------|------|------|
| 01~03 | DataCollection / LowRiskAnomaly / VolatilityEDA | 데이터 수집 / 검증 | 전처리·탐색 |
| **04** | **Prior_Universe_Analysis** | **유니버스·Prior·PCT_GROUP 설계 결정** | **설계** |
| 05 | VolatilityPrediction | 변동성 예측 (GARCH; LSTM 등으로 교체 가능) | 핵심 모델 |
| **06** | **Q_Sensitivity** | **최적 Q 탐색 (예측 vol + Baseline)** | **실험** |
| **08** | **BL_Q_Methods** | **예측 vol × Q 방식 5종 비교** | **실험** |
| **09** | **Regime_Q_Portfolio** | **최종 포트폴리오 (레짐 적응형 BL)** | **결과** |
| 10 | Q_Adaptive_Comparison | Q 적응 전략 비교 (6종) | 실험 |
| 98~99 | 베이스라인 | 참고·비교 | 참고 |

> 옛 `07_BL_Q_Comparison`(Baseline×5Q)과 옛 `08_BL_VolQ_Grid`(2×5 격자)는 2026-05-01 정리에서 제거·통합되었다. 새 `08_BL_Q_Methods`가 두 노트북의 핵심 분석을 단일 파일로 흡수한다.

---

## 예측 모델 교체 가이드 (GARCH ↔ LSTM ↔ ...)

### 핵심 인터페이스: `data/vol_predicted.csv`

다운스트림 노트북(06, 08, 09, 10, 99)은 **모두 `data/vol_predicted.csv` 한 파일**만 읽는다. 이 파일은 model-agnostic이며 활성 예측 모델의 출력으로 덮어써진다.

**스키마 (long format)**

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `date` | datetime (월말) | 예측 대상 월 |
| `ticker` | str | 종목 코드 |
| `vol_pred` | float | 다음 달 연환산 변동성 예측값 |

LSTM(또는 다른 모델)이 동일 스키마로 `vol_predicted.csv`를 생성하기만 하면 06~10, 99의 코드 수정 없이 그대로 작동한다.

### GARCH/LSTM 동시 보존 (발표 시나리오)

05_VolatilityPrediction 실행 시 두 곳에 저장된다:
- `data/vol_predicted.csv` — **활성 예측 파일** (다운스트림이 읽음)
- `data/vol_predicted_garch.csv` — GARCH 영구 백업

LSTM 노트북(추후 추가 예정)도 동일 패턴을 따른다:
- `data/vol_predicted.csv` ← LSTM 결과로 덮어씀
- `data/vol_predicted_lstm.csv` ← LSTM 영구 백업

### 모델 전환 절차

```bash
# GARCH 결과로 다운스트림 분석
cp data/vol_predicted_garch.csv data/vol_predicted.csv
# → 06, 08, 09, 10, 99를 차례로 재실행

# LSTM 결과로 다운스트림 분석
cp data/vol_predicted_lstm.csv data/vol_predicted.csv
# → 06, 08, 09, 10, 99를 차례로 재실행
```

또는 LSTM 노트북 마지막 셀에서 `vol_predicted.csv`로 저장하면 자동으로 활성 모델이 LSTM이 된다.

### 변수명 규약

다운스트림 노트북에서 모델별 라벨이 박힌 변수명은 사용하지 않는다:

| 의미 | 사용할 변수명 |
|------|--------------|
| 예측 vol에 대한 최적 Q | `Q_PRED` (옛 `Q_GARCH` 폐기) |
| Baseline vol에 대한 최적 Q | `Q_BASELINE` (06에서만 사용) |
| 예측 모델 비의존 통합 Q | `Q_OPTIMAL` (09, 10) |

이 규약 덕분에 LSTM/Transformer 등 다른 모델로 교체해도 변수명 의미가 흐려지지 않는다.

### 99_baseline의 Q 방식 적용

99_baseline은 `vol_21d`(현재 실현 vol)을 P 행렬 분류에 직접 사용하는 baseline 노트북으로, FF3·Hist·Momentum·Lambda 등 다양한 Q 추정 방식을 통합 비교한다. 이 Q 방식들은 모두 [bl_utils.py](bl_utils.py)에 함수 형태로 추출되어 있어, **LSTM vol을 사용하는 새 노트북에서도 한 줄 import로 즉시 재사용 가능**하다:

```python
from bl_utils import (compute_Q_hist, compute_Q_momentum,
                     compute_Q_lambda, compute_Q_pi_ratio, compute_Q_ff3)
```
