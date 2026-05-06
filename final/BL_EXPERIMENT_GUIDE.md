# Black-Litterman 실험 프레임워크 — 상세 가이드

> **최종 갱신: 2026-05-06** — LSTM σ 피봇, vol_spread/lambda/raw_lam/inv_lambda q_mode, ff3_paper omega, Q 민감도 39 추가 반영. 현재 EXPERIMENTS 총 **214개**.

---

## 1. 전체 구조

```
final/
├── bl_config.py              ← EXPERIMENTS 정의 (214개)
├── bl_functions.py           ← BL 수식 핵심 함수 (Σ/π/P/Q/Ω/BL/TC/Metrics)
├── master_table.py           ← results/*.pkl → mt/rt 빌더
├── analyze_plots.py          ← 시각화 모듈
│
├── 99_run.ipynb              ← walk_forward 실행 → results/*.pkl
├── 99_analyze.ipynb          ← 24-cell 분석 (J1~J6, K1~K6, K2-H)
│
├── results/                  ← 214 pkl
├── data/                     ← monthly_panel.csv, daily_returns.pkl, ff3_monthly.csv
└── phase3(data_outputs)/     ← LSTM 예측 결과
    └── data/ensemble_predictions_stockwise.csv
```

---

## 2. 실험 실행 순서

```
① bl_config.py에 실험 dict 추가 (필요 시)
② 99_run.ipynb 셀 순서대로 실행
   cell-00 : 패키지 임포트 + 경로 설정
   cell-01 : 데이터 로드 (monthly_panel, daily_returns, FF3)
   cell-02 : LSTM 예측 로드 + monthly_cache 빌드 (Σ 등 사전계산)
   cell-03 : Dispatcher 함수 정의 (get_vol_series, get_Q, get_omega 등)
   cell-04 : walk_forward(cfg) 함수 정의
   cell-05 : run_list = 미생성 실험만 → 자동 스킵
   cell-06 : 빠른 성과 확인
③ 99_analyze.ipynb 실행 (Cell 1 → J1 → K1 순)
```

> **자동 스킵**: `(RESULTS_DIR / f'{cfg["name"]}.pkl').exists()` → 이미 있으면 스킵.
> 재실행하려면 해당 pkl 삭제.

---

## 3. 실험 추가 방법

### Case 1 — 기존 슬롯 파라미터만 변경 (가장 흔함)

`bl_config.py`의 `EXPERIMENTS`에 dict 한 줄 추가. **다른 파일 수정 불필요**.

```python
{**BASELINE, 'name': 'q_0007', 'q_value': 0.007},                            # Q값 변경
{**BASELINE, 'name': 'no_tc',  'tc': 0.0},                                   # 거래비용 제거
{**BASELINE, 'name': 'mat_eq_rp_vsp_pap', 'prior': 'capm_eq', 'p_mode': 'lstm_predicted',
 'p_weight': 'rp', 'q_mode': 'vol_spread', 'q_value': 0.003, 'omega_mode': 'ff3_paper'},
```

`{**BASELINE, ...}` 패턴: BASELINE 모든 값 상속 + 변경 슬롯만 덮어씀.

### Case 2 — 새 계산 방식 추가

3개 파일을 **이 순서대로** 수정:
1. `bl_functions.py` — 새 함수 추가 (예: `compute_Q_my_method`)
2. `bl_config.py` — 슬롯 주석 + 실험 dict 추가
3. `99_run.ipynb` cell-03 — dispatcher (`get_Q`, `get_omega` 등)에 elif 추가

---

## 4. 슬롯 키 레퍼런스

| 슬롯 | 선택지 | 기본값 | 설명 |
|---|---|---|---|
| `prior` | `capm_mcap` / `capm_eq` / `capm_rp` | `capm_mcap` | Prior π 가중 |
| `p_mode` | `trailing_vol21` / `trailing_vol252` / `lstm_predicted` | `trailing_vol21` | P 분류 변동성 |
| `p_weight` | `mcap` / `eq` / `rp` / `vol_mcap` | `mcap` | P 행렬 가중 |
| `q_mode` | `fixed` / `vol_spread` / `lambda` / `raw_lam` / `inv_lambda` / `none` / `capm` | `fixed` | Q 결정 방식 |
| `q_value` | float | `0.003` | fixed/vsp의 Q 값(또는 base) |
| `omega_mode` | `he_litterman` / `scaled` / `rmse` / `ff3_paper` | `he_litterman` | Ω 계산 방식 |
| `omega_scale` | float | `1.0` | scaled에서 배수 |
| `tc` | float | `0.001` | 편도 거래비용 (10bp) |
| `max_weight` | float | `0.10` | 단일 종목 상한 |
| `lstm_pred_path` | str | 자동 탐색 | LSTM 예측 파일 경로 |

---

### `prior` 상세

| 값 | π 가중 |
|---|---|
| `capm_mcap` | 시총 비례 — 시장 균형 표준 (He-Litterman 1999) |
| `capm_eq` | 균등 가중 — naïve 비교군 |
| `capm_rp` | 1/σ 역변동성 — Risk Parity 사상 |

### `p_mode` 상세

| 값 | 사용 변동성 | 설명 |
|---|---|---|
| `trailing_vol21` | `vol_21d` (과거 21일 실현) | look-ahead 없음, 단순 |
| `trailing_vol252` | `vol_252d` (과거 252일) | 장기 안정 |
| `lstm_predicted` | LSTM+HAR 앙상블 예측 | Phase3 산출 (`y_pred_ensemble` → exp() → σ) |

### `p_weight` 상세 — Long(저변동) / Short(고변동)

| 값 | Long (하위 30%) | Short (상위 30%) | 비고 |
|---|---|---|---|
| `mcap` | 시총 비례 | 시총 비례 | 표준 |
| `eq` | 동일 | 동일 | 단순·견고 |
| `rp` | 1/σ | 1/σ | Pyo & Lee 2018 |
| `vol_mcap` | (1/σ)×mcap | σ×mcap | 전체 유니버스 (30% 컷 없음) |

### `q_mode` 상세 — Q (단일 view 강도) 결정

| 값 | Q 계산 | 비고 |
|---|---|---|
| `fixed` | `Q = q_value` | 가장 단순 |
| `vol_spread` (vsp) | `Q = q_value × clip(vol_spread_t / spread_ref, 0.1, 3.0)` | 위기 시 확대(저변동 anomaly 강화) |
| `lambda` | `Q = q_value × λ_t / λ_ref` | 시장 위험회피도에 비례 |
| `raw_lam` | `Q = q_value × λ_t` | λ 직접 곱 |
| `inv_lambda` | `Q = q_value × λ_ref / λ_t` | λ 역수(반대 방향) |
| `none` | Q 없음 → BL 스킵, 직접 보유 | naive 비교군 |
| `capm` | BL 없음 → CAPM π 직접 최적화 | capm_no_bl 비교군 |

### `omega_mode` 상세 — Ω (view 분산) 결정

| 값 | Ω 계산 | 비고 |
|---|---|---|
| `he_litterman` (he) | `Ω = τ·P·Σ·Pᵀ` (스칼라) | He-Litterman 1999 표준 |
| `scaled` | `he × omega_scale` | scale<1 view 신뢰↑ / >1 신뢰↓ |
| `rmse` (rms) | LSTM 예측 RMSE 기반 스케일링 | RMSE↑ → Ω↑ |
| `ff3_paper` (pap) | `Ω_t = (Q_{t-1} − actual_P_return_{t-1})²` | 직전월 예측오차² Bayesian rolling |

> ⚠️ `ff3_paper` 명명 주의: `compute_omega_paper`(FF3 회귀 잔차분산)는 dispatcher에서 호출 안 됨. 실제 로직은 walk_forward 안의 적응형 갱신. FF3 논문 무관, 발표 시 "직전월 예측오차² 적응형"으로 정확히 표현.

---

## 5. 실험 카테고리 (총 214)

### 5-1. 매트릭스 (LSTM 고정, 4-token mat_*)

```
mat_{prior}_{p_weight}_{q}_{omega}
```

| 차원 | 값 |
|---|---|
| prior | mcap / eq / rp |
| p_weight | mcap / eq / rp |
| q_mode | fix / lam / raw / inv / vsp |
| omega | he / pap / rms |

3 × 3 × 5 × 3 = **135 cells** (현재 일부만 실현, 약 108)

### 5-2. 비교군 / 변형군 (semantic naming)

| 카테고리 | 예시 |
|---|---|
| BL 없음 | `capm_no_bl`, `naive_lowvol`, `naive_lowvol_rp` |
| Trailing 21일 | `baseline`, `omega_rmse`, `omega_paper`, `q_lambda`, `q_raw_lam` |
| Scaled Ω | `omega_scaled_double`, `omega_scaled_half_p_lstm`, `omega_scaled_double_p_lstm` |
| vol_mcap | `p_vol_mcap`, `p_lstm_vol_mcap`, `prior_eq_p_vol_mcap` |

### 5-3. Q 민감도 (2026-05-06 추가, 39개)

5-레짐 sortino_ir Top 20 중 q_mode ∈ {fixed, vol_spread} 후보 13개 × q_value ∈ {0.0055, 0.0064, 0.0070}.

이름 규칙: `{원본_이름}_q55` / `_q64` / `_q70`.

학술 근거 (Frazzini-Pedersen 2014, BAB):
- 0.0055: 4팩터 알파 보수치
- 0.0064: 글로벌 19개국 평균
- 0.0070: 미국 평균

목적: q_value 단조 증가 시 IR 일관 패턴 검증. 일관되면 Q 상향이 robust, 그렇지 않으면 0.003 유지 + 학술 mention만.

---

## 6. LSTM 실험 전제조건

`p_mode='lstm_predicted'` 실험은 `phase3(data_outputs)/data/ensemble_predictions_stockwise.csv` 필요. 없으면 자동 스킵 (경고 메시지).

파일 구조:
```
date        ticker  y_pred_lstm  y_pred_har  y_pred_ensemble  y_true  ...
2007-04-23  AAPL    -4.12        -4.36       -4.24            -3.99
```

`y_pred_ensemble`(log-RV) → `np.exp()` → σ_pred → 월말 기준 종목 랭킹.

---

## 7. 결과 파일 구조 (`results/{name}.pkl`)

| 키 | 타입 | 내용 |
|---|---|---|
| `config` | dict | 실험 cfg 전체 |
| `ret` | pd.Series | 월별 **순수익률** (`net_ret = gross - turnover×tc`) |
| `gross_ret` | pd.Series | 월별 총수익률 (TC 차감 전) |
| `spy_ret` | pd.Series | 월별 SPY 수익률 |
| `weights` | pd.DataFrame | 월별 × 종목 가중치 |
| `comp` | pd.DataFrame | 월별 구성지표 (eff_n, top10_share, avg_vol, **turnover**, tc_cost 등) |
| `meta` | pd.DataFrame | 월별 Q값, λ |
| `errors` | list | 에러 발생 월 |

> **TC 반영**: `ret` Series는 이미 TC 차감 후. mt/rt의 모든 sharpe/sortino/cagr는 net-of-TC.

---

## 8. 재실행 방법

```bash
# 특정 실험만 재실행
rm final/results/{name}.pkl
# 99_run.ipynb 실행 (해당 cfg만 walk_forward)
```

전체 재실행: cell-05의 `SKIP_IF_EXISTS = True`를 `False`로 변경.

---

## 9. 주의사항

| 항목 | 내용 |
|---|---|
| **Look-ahead bias** | `fwd_ret_1m`은 평가 전용. BL 입력에 절대 사용 금지 |
| **LSTM 학습** | Phase3에서 walk-forward 사전 생성. 재학습 시 GPU + 수 시간 |
| **rp 가중방식** | Pyo & Lee 2018 — 30% 선별 후 그룹 내 1/σ 가중 |
| **vol_mcap** | 전체 유니버스, 30% 컷 없음 |
| **거래비용 단위** | `tc=0.001` = 편도 10bp, 월 TC = `turnover × tc` |
| **데이터 선행** | `01_DataCollection.ipynb` 실행 후 `data/` 채워진 상태에서 실행 |
| **monthly_cache** | `99_run` cell-02에서 빌드 후 캐시 — 재시작 시 자동 재사용 |

---

## 10. 분석 흐름 (다음 단계)

`99_run` 완료 → `99_analyze` 실행 → 슬롯 효과 + 매트릭스 히트맵 + 5-레짐 안정성 → 위험성향별 최종 후보.

상세: [99_ANALYZE_GUIDE.md](99_ANALYZE_GUIDE.md).
