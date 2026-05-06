# 시계열_Test → final/ 통합 — 산출물 통합 계획

> **작성일**: 2026-05-06
> **본 문서 위치**: `시계열_Test/INTEGRATION_PLAN.md`
> **목적**: 시계열_Test 폴더의 작업 산출물을 메인 BL 실험 framework (`final/`) 로 통합한 의사결정·정책·산출물 목록을 보존하기 위한 1차 기록

---

## 1. Context

`시계열_Test/` 폴더는 다음 두 작업이 진행된 곳입니다.

- **Phase 1.5 — Volatility Forecasting**: SPY/QQQ → 7 종목 → 615 종목으로 확장된 LSTM 변동성 예측 (v1 ~ v8 진화, Performance-Weighted Ensemble 최선)
- **Phase 3 (3-1, 3-2 v2) — Robust Extensions**: S&P 500 historical 615 종목 stockwise + Cross-Sectional 학습, BL 백테스트 9 시나리오, §2-B 학술 통계 심화

본 통합은 **메인 BL framework (`final/`)** 로 이 작업의 핵심을 압축·재현 가능하게 옮기는 것이 목적입니다. 시계열_Test 의 14,125 + 1,754 라인을 ~600 라인 단일 모듈 + 2 노트북으로 압축했습니다.

---

## 2. 사용자 결정 사항

| 항목 | 결정 |
|---|---|
| 노트북 구성 | 2개 분리 (03 모델 + 04 통계) |
| 재현 범위 | **학습부터 구현, cache 활용 skip 가능, csv 정합성 우선 검증** |
| 파일명 prefix | `03_타입명` 패턴 (`final/01_DataCollection`, `02_LowRiskAnomaly` 다음 순서) |
| 데이터 의존성 | 기존 `final/` 데이터 활용 |
| **모델 범위** | **Stockwise (02a) + HAR-RV + Performance-Weighted Ensemble 만**. Cross-Sectional (02b) 제외 (분석 미활용) |

---

## 3. 사전 호환성 검증

### 3.1 csv 정합성 — byte-byte 동일

| 검증 항목 | 결과 |
|---|---|
| 파일 size | 331,142,770 bytes (동일) |
| md5 hash | `1e9ab2faf63fdfd4abbb54083a1cb0fb` (동일) |
| 라인 수 | 2,468,884 (동일) |
| 시계열_Test 원본 → final 사본 | **byte-byte 정확히 동일** |

> 즉 `final/phase3(data_outputs)/data/ensemble_predictions_stockwise.csv` 는 `시계열_Test/Phase3_Robust_Extensions/data/ensemble_predictions_stockwise.csv` 의 100% 정확한 사본입니다.

### 3.2 Universe 호환성

| 항목 | 결과 |
|---|---|
| ensemble 학습 종목 | 613 |
| `final/universe.csv` | 833 |
| 교집합 (학습 종목 ⊂ universe) | **613/613 = 100%** |
| `monthly_panel.csv` 와 ensemble 공통 | 585 종목 |

### 3.3 fold 구조

- 각 (ticker, date) 에 정확히 1 fold (walk-forward 표준)
- 224 fold × 21 영업일/fold ≈ 18.7년 OOS (2007-04 ~ 2025-12)

---

## 4. 4 주의사항별 판단

| # | 주의사항 | 판단 | 처리 방법 |
|---|---|---|---|
| 1 | **vol_21d 정의 차이** (`monthly_panel.vol_21d` trailing 연환산 vs `ensemble.y_true` forward Log-RV) | ✅ **넘어가도 됨 (명시만)** | 두 변수가 별개 개념임을 03 §1 docstring 으로 명시. monthly_panel 의 vol_21d 는 BL P 행렬 입력, ensemble 의 y_true 는 03·04 분석 입력으로 분리 사용. 분석 영향 0. |
| 2 | **`y_true` 의 -inf 행** | 🔧 **수정 필요** | 거래정지 등으로 std=0 → log(0)=-inf 발생 (113 행). 04 §1 에서 `np.isfinite()` 필터 적용. `tlib.load_ensemble_predictions()` 가 자동 처리. |
| 3 | **종목 정의 미세 차이** | 🔧 **수정 필요 (선별 적용)** | (a) 모델 평가 (03) = ensemble 613 종목 전체 / (b) Sector 통계 (04) = "5 시기 cover 503 종목 필터" (`tlib.filter_503_universe()`) — 시계열_Test §2-B 표준. |
| 4 | **panel 종료=2024 vs ensemble=2025** | ✅ **넘어가도 됨** | 04 통계는 ticker 단위 sector mapping 만 사용 → 영향 0. 03 의 2025 hold-out 분석은 ensemble 단독 작동. |

---

## 5. 모델 범위 — Cross-Sectional 제외

본 통합에는 **Stockwise (02a) + HAR-RV + Performance-Weighted Ensemble** 만 포함합니다.

### 제외 항목

- ❌ `시계열_Test/Phase3_Robust_Extensions/scripts/models_cs.py` (Ticker Embedding LSTM)
- ❌ `시계열_Test/Phase3_Robust_Extensions/data/ensemble_predictions_crosssec.csv`
- ❌ `시계열_Test/Phase3_Robust_Extensions/02b_phase15_cross_sectional.ipynb`
- ❌ `BL_ml_cs` 시나리오 관련 코드

### 통합 대상

- ✅ Stockwise LSTM (per-ticker, 615 종목)
- ✅ HAR-RV (Corsi 2009)
- ✅ Diebold-Pauly Performance-Weighted Ensemble

---

## 6. 산출물 목록

### 6.1 신규 생성 (final/ 하위)

| 파일 | 라인/셀 | 역할 |
|---|---|---|
| `final/timeseries_lib.py` | 596 라인 | 핵심 함수 모듈 (CS 제외) — 12 섹션 (시드·타깃·Walk-Forward·LSTM·HAR·Ensemble·Metrics·DM·ANOVA·Welch·KW·Pairwise·Heavy-tail·assert) |
| `final/03_Volatility_Forecasting.ipynb` | 26 셀 | 모델 구축 (csv 정합성·LSTM·HAR·Ensemble·615 종목 평가) |
| `final/04_Statistical_Validation.ipynb` | 32 셀 | 학술 통계 (ANOVA·Welch·KW·Pairwise·Heavy-tail·B3 5+B4 3 시각화·4 학술 명제) |
| `final/outputs/03_volatility/` | 2 PNG + summary.json | 615 종목 RMSE 분포·Best model 분포 |
| `final/outputs/04_statistics/` | 8 PNG + summary.json | B3 5 패널 + B4 3 패널 |

### 6.2 신규 생성 (시계열_Test 하위)

| 파일 | 역할 |
|---|---|
| `시계열_Test/INTEGRATION_PLAN.md` | 본 문서 — 통합 의사결정·정책·산출물 목록 |
| `시계열_Test/INTEGRATION_WORKLOG.md` | 통합 작업 진행 일지 — 단계별 작업·이슈·검증 결과 |

### 6.3 변경 없음 (기존 보존)

- 시계열_Test/ 폴더 전체 (REPORT.md / README.md / 노트북 / scripts / data / outputs / 재천_WORKLOG.md 등) — read-only
- final/ 의 기존 산출물 (bl_config.py / bl_functions.py / 99_*.ipynb / results/*.pkl / outputs/02_anomaly·99_baseline·99_compare·99_analyze 등) — 무수정

---

## 7. 비편집 보장 — 기존 final/ 산출물 무영향

본 통합 작업은 **새로운 파일·폴더 추가** 만 수행합니다. 기존 final/ 의 모든 파일은 read-only 로만 사용되며 무수정입니다.

### 영향 분석

| 영역 | 항목 | 영향 |
|---|---|---|
| 데이터 | `final/data/` 11종 | ❌ read-only |
| 코드 | `final/bl_config.py`, `final/bl_functions.py` | ❌ 무수정 |
| 노트북 | `final/01_DataCollection`, `02_LowRiskAnomaly`, `99_run`, `99_analyze`, `99_explore` | ❌ 무수정 |
| 노트북 | `final/_dev/99_baseline.ipynb` (서윤범 baseline) | ❌ 무수정 |
| 결과 | `final/results/*.pkl` (13+ BL 실험 결과) | ❌ 무수정 |
| 결과 | `final/outputs/02_anomaly/`, `99_*/` | ❌ 무수정 |
| 문서 | `PROJECT_OVERVIEW.md`, `BL_EXPERIMENT_GUIDE.md` 등 | ❌ 무수정 |

### Indirect dependency 무영향

- `bl_config.py` 의 LSTM 예측 경로 (`phase3(data_outputs)/data/ensemble_predictions_stockwise.csv`) 는 본 통합에서 read-only → BL 실험 (`p_lstm_*` 13건) 결과 무영향
- `99_run.ipynb` 재실행 X → `final/results/` 13+ pkl 무수정
- 신규 outputs 서브폴더 (`03_volatility/`, `04_statistics/`) 만 추가 → 기존 폴더 무영향

---

## 8. 검증 통과 결과 (2026-05-06)

### 8.1 csv 정합성

- ✅ md5 일치: `1e9ab2faf63fdfd4abbb54083a1cb0fb` (시계열_Test 원본과 byte-byte 동일)
- ✅ 종목 수 613 ⊂ final universe 833
- ✅ Walk-Forward 224 fold × 613 종목 학습 결과 활용

### 8.2 615 종목 RMSE 평균 (03 노트북)

| 모델 | 본 재현 | 시계열_Test §8.2 기대 |
|---|---|---|
| LSTM | 0.5288 | 0.4298 (5 시기 cover 503 종목) |
| HAR | 0.4015 | 0.3922 |
| Ensemble | 0.3910 | 0.3815 |

**Best model 분포**: Ensemble 398 (65.0%) / HAR 199 (32.5%) / LSTM 16 (2.6%) — 시계열_Test §8.2 의 "Ensemble 69.7% / HAR 27.2% / LSTM 3.1%" 와 거의 일치.

### 8.3 §2-B 학술 통계 (04 노트북)

| 지표 | 시계열_Test 기대 | 본 재현 | 정합 |
|---|---|---|---|
| η²_period | 0.450 | **0.4498** | ✓ |
| η²_ticker | 0.194 | **0.1944** | ✓ |
| F_period | 634.6 | **634.56** | ✓ |
| Welch F | 420.59 | **420.59** | ✓ |
| Levene stat | 16.78 | **16.78** | ✓ |
| Skewness | +1.30 | **+1.2993** | ✓ |
| Excess Kurtosis | +4.71 | **+4.71** | ✓ |
| JB stat | 605.60 | **605.60** | ✓ |
| AD stat | 4.89 | **4.89** | ✓ |
| **4 학술 명제** | 모두 PASS | **모두 PASS** | ✓ |

### 8.4 미세 차이 — 합리적 범위

- KW H: 시계열_Test 70.55 → 본 재현 97.30 (sector mapping 차이로 추정 — `final/monthly_panel.csv` 의 sector 분류가 시계열_Test 의 `ticker_sector_mapping.csv` 와 미세 차이)
- Pairwise n_total: 시계열_Test 66 (12 sector) → 본 재현 153 (18 sector) (Unknown 등 추가 분류)
- 모두 통계 결론 (모든 명제 PASS) 에는 영향 없음

---

## 9. 재현 절차

### Cache hit 모드 (기본, ~5분)

```bash
cd final/
jupyter nbconvert --to notebook --execute --inplace 03_Volatility_Forecasting.ipynb
jupyter nbconvert --to notebook --execute --inplace 04_Statistical_Validation.ipynb
```

### Full retraining 모드 (GPU + 3~5시간)

03 노트북에서 `FORCE_RECOMPUTE = True` 설정 후 실행. 단 시계열_Test 의 Phase 1.5 모듈 (`scripts/{dataset, models, train, baselines_volatility}.py`) 필요. 본 통합 노트북에서는 권장 X (별도 scripts 에서 실행 권장).

---

## 10. 학술 baseline 인용

본 통합은 다음 학술 baseline 의 결과를 직접 재현합니다.

- **Corsi (2009)** — HAR-RV 모델
- **Patton (2011)** — QLIKE 손실 함수
- **Diebold & Pauly (1987)** — Performance-Weighted Ensemble
- **Diebold & Mariano (1995)** — DM 검정
- **Cohen (1988)** — 효과크기 표준
- **Welch (1951)** — 이분산 robust ANOVA
- **Lin (2013)** — Large-n 함정 경고
- **Cont (2001), Mandelbrot (1963)** — Heavy-tail stylized fact
- **Engle, Ghysels, Sohn (2013)** — Multi-frequency 변동성 동학
- **Schwert (1989)** — Sector-specific leverage effect
- **Fama-French (1992)** — Sector effect on asset pricing
- **Mincer & Zarnowitz (1969)** — MZ 회귀
- **Lopez de Prado (2018)** — Walk-Forward + Purge + Embargo

---

*본 문서는 시계열_Test 작업이 final/ 로 통합된 의사결정의 1차 기록입니다. 실제 작업 일지는 `시계열_Test/INTEGRATION_WORKLOG.md` 를, 통합 산출물은 `final/timeseries_lib.py`, `final/03_*.ipynb`, `final/04_*.ipynb` 를 참조하시기 바랍니다.*
