# Black-Litterman 실험 프레임워크 사용 가이드

## 파일 구조

```
low_risk/
├── bl_functions.py        ← BL 수식 함수 (수정 거의 불필요)
├── bl_config.py           ← 실험 정의 (자주 수정)
├── 99_run.ipynb           ← 실험 실행 + 결과 저장
├── 99_analyze.ipynb       ← 결과 분석 + 차트
└── results/               ← 실험 결과 pkl 저장 위치 (자동 생성)
```

---

## 실행 순서

```
1. bl_config.py에 실험 추가 (필요 시)
2. 99_run.ipynb 전체 실행 → results/*.pkl 생성
3. 99_analyze.ipynb 전체 실행 → 비교 차트 + 성과 테이블
```

---

## 실험 추가 방법

### Case 1. 기존 파라미터 조합만 바꿀 때

`bl_config.py`의 `EXPERIMENTS` 리스트에 dict 한 줄 추가.

```python
# 예: Q를 0.005로 바꾼 실험
{**BASELINE, 'name': 'q_0005', 'q_value': 0.005},

# 예: 거래비용 없는 실험
{**BASELINE, 'name': 'no_tc', 'tc': 0.0},

# 예: vol252 + rp 조합
{**BASELINE, 'name': 'p_vol252_rp',
 'p_mode': 'trailing_vol252', 'p_weight': 'rp'},
```

→ 다른 파일은 건드리지 않아도 됨.

### Case 2. 새 계산 방식을 추가할 때

3개 파일을 순서대로 수정:

```
① bl_functions.py  → 새 함수 추가
② bl_config.py     → 실험 dict 추가
③ 99_run.ipynb     → cell-03 dispatcher에 elif 한 줄 추가
```

---

## 슬롯 키 레퍼런스

| 키 | 선택지 | 기본값 | 설명 |
|---|---|---|---|
| `p_mode` | `trailing_vol21` / `trailing_vol252` / `lstm_predicted` | `trailing_vol21` | P 행렬 분류 기준 변동성 |
| `p_weight` | `mcap` / `eq` / `rp` / `asymmetric` / `vol_mcap` | `mcap` | P 행렬 내 가중 방식 |
| `q_mode` | `fixed` / `ff3_regression` / `realized_spread` / `regime` / `none` | `fixed` | Q 추정 방식 |
| `q_value` | float | `0.003` | `q_mode='fixed'`일 때 Q값 (월 기준) |
| `q_regime_table` | dict | — | `q_mode='regime'`일 때 레짐별 Q값 |
| `omega_mode` | `he_litterman` / `scaled` / `rmse` | `he_litterman` | Omega 계산 방식 |
| `omega_scale` | float | `1.0` | `omega_mode='scaled'`일 때 배수 |
| `prior` | `capm_mcap` / `capm_eq` | `capm_mcap` | 시장 Prior 가중 방식 |
| `tc` | float | `0.001` | 편도 거래비용 (10bp = 0.001) |
| `max_weight` | float | `0.10` | 단일 종목 최대 비중 |
| `lstm_pred_path` | str | Phase3 경로 | LSTM 예측 파일 경로 |

### p_weight 상세

| 값 | 롱 가중 | 숏 가중 |
|---|---|---|
| `mcap` | 시총 비례 | 시총 비례 |
| `eq` | 동일가중 | 동일가중 |
| `rp` | 1/σ | 1/σ |
| `asymmetric` | 1/σ | σ×mcap |
| `vol_mcap` | (1/σ)×mcap | σ×mcap |

---

## 현재 실험 목록 (16개)

| 실험명 | 변경 슬롯 | 설명 |
|---|---|---|
| `baseline` | — | 기준선 |
| `p_vol252` | p_mode | 장기(252일) 변동성으로 분류 |
| `p_rp` | p_weight | 1/σ 역변동성 가중 P |
| `p_eq` | p_weight | 동일가중 P |
| `p_asymmetric` | p_weight | 롱 1/σ, 숏 σ×mcap |
| `p_vol_mcap` | p_weight | 롱 (1/σ)×mcap, 숏 σ×mcap |
| `q_ff3` | q_mode | FF3 회귀 기반 Q |
| `q_spread` | q_mode | 훈련구간 실현 스프레드 Q |
| `q_regime` | q_mode | 시장변동성 레짐별 Q |
| `omega_tight` | omega_mode/scale | Omega × 0.5 (뷰 더 신뢰) |
| `omega_loose` | omega_mode/scale | Omega × 2.0 (뷰 덜 신뢰) |
| `prior_eq` | prior | 1/N 균등가중 Prior |
| `p_lstm` | p_mode | LSTM 예측 변동성 P (파일 필요) |
| `p_lstm_omega_rmse` | p_mode + omega_mode | LSTM P + RMSE 기반 Omega |
| `naive_lowvol` | q_mode | BL 없음, 저변동 시총가중 보유 |
| `naive_lowvol_rp` | q_mode + p_weight | BL 없음, 저변동 1/σ 가중 보유 |

---

## 결과 파일 구조

`results/{실험명}.pkl`에 저장되는 dict:

| 키 | 타입 | 내용 |
|---|---|---|
| `config` | dict | 이 실험의 bl_config 설정 |
| `ret` | pd.Series | 월별 순수익률 (거래비용 차감 후) |
| `gross_ret` | pd.Series | 월별 총수익률 (거래비용 전) |
| `spy_ret` | pd.Series | 월별 SPY 수익률 |
| `weights` | pd.DataFrame | 월별 × 종목 가중치 |
| `comp` | pd.DataFrame | 월별 구성 지표 (eff_n, top10, avg_vol, turnover 등) |
| `meta` | pd.DataFrame | 월별 Q값, lambda 등 실험별 메타데이터 |
| `errors` | list | 에러 발생 월 목록 |

---

## 성과 지표 목록 (compute_metrics 반환값)

| 지표 | 설명 |
|---|---|
| `sharpe` | 연환산 Sharpe Ratio |
| `sortino` | 하방 변동성 기준 Sharpe |
| `cagr` | 연환산 수익률 |
| `vol` | 연환산 변동성 |
| `mdd` | 최대 낙폭 (MDD) |
| `calmar` | CAGR / MDD |
| `cum_ret` | 전체 누적 수익률 |
| `skewness` | 수익률 분포 왜도 |
| `win_rate` | 양수 수익 월 비율 |
| `avg_win` | 수익 월 평균 수익률 |
| `avg_loss` | 손실 월 평균 손실률 |
| `cvar_5` | 최악 5% 구간 평균 손실 |
| `mdd_duration` | MDD 지속 기간 (월) |
| `beta` | SPY 대비 베타 (mkt_ret 필요) |
| `alpha` | CAPM 알파, 연환산 (mkt_ret 필요) |
| `treynor` | 초과수익 / 베타 (mkt_ret 필요) |

---

## 주의사항

- **데이터 선행 필요**: `01_DataCollection.ipynb` 실행 후 `data/` 폴더가 채워진 상태에서 실행
- **LSTM 실험**: `bl_config.py`의 `_LSTM_PRED_DEFAULT` 경로에 Phase3 결과 파일이 있어야 함. 없으면 자동 스킵
- **재실행 방지**: `SKIP_IF_EXISTS = True`가 기본 → 이미 저장된 실험은 스킵. 재실행하려면 `results/{실험명}.pkl` 삭제 후 실행
- **거래비용 단위**: `tc=0.001`은 편도 10bp. turnover × tc = 월 TC 비용
- **Look-ahead 금지**: `fwd_ret_1m`은 타겟 변수 전용. BL 입력으로 절대 사용 금지
