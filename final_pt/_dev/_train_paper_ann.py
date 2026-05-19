"""논문 방식 ANN per-ticker 학습 (walk-forward) — final_pt 환경.

Pyo & Lee (2018) Eq. σ̂_i,t = f(σ_i,t-1, σ_i,t-2, ..., σ_i,t-10)

Spec (논문):
    Input  : 10개월 변동성 (벡터 길이 10)
    Output : 다음 1개월 변동성
    Activation : ReLU
    Optimizer  : Backpropagation (Adam 사용)
    Training data : 60개월 윈도우 (rolling)

Spec (본 구현 — 논문 미공개 부분):
    Hidden : 10 → 4 → 1 (49 params, sample/param 비율 ≈ 1.0)
    L2 reg : alpha=0.01
    Max iter : 50 (early termination)
    Scaling : X, y 모두 StandardScaler (per-ticker, per-window)
    LR     : 0.001 (small data 안정성)

데이터 소스 (LSTM 과 정합)
-------------------------
입력: data/daily_returns.pkl (univ 824 종목, 2004 ~ 2026, daily log return).
종목별 vol_21d (annualized) 를 직접 계산: rolling(21).std(ddof=1) × √252.
(monthly_panel 의 vol_21d 와 수치 일치 검증 완료, 2026-05-16.)

universe 선택
~~~~~~~~~~~~~
LSTM 과 동일한 학습 대상 종목 (= monthly_panel 의 617 종목) 사용.
단, 학습 데이터는 종목별 **전체 daily history** (시점별 S&P500 멤버십 무관).
이유: monthly_panel 의 strict 멤버십 필터를 학습에 적용하면 신규 상장주 (예 CVNA,
DDOG, ABNB) 의 옛 daily 데이터가 무시되어 LSTM 과의 학습 데이터 비대칭 발생.
BL 단계의 universe 필터링은 04_BL_Walkforward 의 valid_tix (panel 기반) 에서 별도 수행.

Look-ahead bias 감사
--------------------
- pred_date M 에서 train_window = all_dates[M-59 : M+1]  → 월 M 까지 포함 (slice end exclusive)
- 학습 페어: X = months [M-k-9..M-k], y = month M-k  for k in 49..0 → 가장 마지막 y 는 month M
- 예측: X_pred = months [M-9..M] → ŷ = month M+1
- 학습/예측 모두 t ≤ M 데이터만 사용. y_true (month M+1) 은 evaluation 전용, 모델 학습 미사용.
- vol_21d[M] 은 월말 기준 trailing 21d std → end-of-M 에서 완전 관측 가능. 미래 leakage 없음.

출력: data/paper_ann_predictions.csv
    LSTM CSV (ensemble_predictions_stockwise.csv) 와 동일 컬럼 스키마
    y_pred_ensemble 은 log(daily_std) 공간 (LSTM 호환)

실행: python _dev/_train_paper_ann.py  (final_pt/ 경로에서)
"""
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed


# ── 경로 ────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / 'data'
OUT_PATH = DATA_DIR / 'paper_ann_predictions.csv'

# ── 파라미터 ────────────────────────────────────────────────
LOOKBACK = 10               # 입력 윈도우 (논문)
TRAIN_WINDOW = 60           # 학습 윈도우 (논문)
HIDDEN = (4,)               # 은닉층 (1 layer × 4 neurons, 49 params)
N_JOBS = 4                  # 종목별 병렬 worker
START_PRED = '2010-01-01'

# 학습 hyperparameter
ALPHA = 0.01                # L2 regularization
MAX_ITER = 50               # 빠른 학습 (small data)
SEED = 42


def train_predict_ticker(ticker, vol_history_log, pred_date_idx, all_dates):
    """한 종목의 한 pred_date 시점 ANN 학습 + 다음달 예측.

    Returns dict {'date', 'ticker', 'y_pred_ensemble', 'y_true'} 또는 None.
    """
    pred_date = all_dates[pred_date_idx]

    # 학습 윈도우: 현재 시점까지 최근 60개월 (t ≤ pred_date)
    train_start_idx = max(0, pred_date_idx - TRAIN_WINDOW + 1)
    train_window = all_dates[train_start_idx:pred_date_idx + 1]
    train_vols = vol_history_log.reindex(train_window).dropna()

    if len(train_vols) < TRAIN_WINDOW:
        return None

    # 학습 페어 생성: (X = 10개월 시퀀스, y = 다음달)
    arr = train_vols.values
    n_pairs = len(arr) - LOOKBACK
    if n_pairs < 5:    # 최소 5 페어
        return None

    X_train = np.array([arr[i:i+LOOKBACK] for i in range(n_pairs)])
    y_train = np.array([arr[i+LOOKBACK] for i in range(n_pairs)])

    # 표준화 (각 종목·윈도우 단위, look-ahead 없음 — train 데이터로만 fit)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

    # ANN 학습 (Pyo & Lee 2018 spec)
    model = MLPRegressor(
        hidden_layer_sizes=HIDDEN,
        activation='relu',
        solver='adam',
        alpha=ALPHA,
        max_iter=MAX_ITER,
        learning_rate_init=0.001,
        random_state=SEED,
    )
    try:
        model.fit(X_train_scaled, y_train_scaled)
    except Exception:
        return None

    # 다음달 예측: 가장 최근 10개월 (months M-9..M) 사용
    X_pred = arr[-LOOKBACK:].reshape(1, -1)
    X_pred_scaled = scaler_X.transform(X_pred)
    try:
        y_pred_scaled = float(model.predict(X_pred_scaled)[0])
        y_pred = float(scaler_y.inverse_transform([[y_pred_scaled]])[0, 0])
    except Exception:
        return None

    # 실제값 (다음달 데이터 있으면) — evaluation 전용, 학습 미사용
    next_idx = pred_date_idx + 1
    if next_idx < len(all_dates):
        next_date = all_dates[next_idx]
        y_true = vol_history_log.get(next_date, np.nan)
    else:
        y_true = np.nan

    return {
        'date': pred_date,
        'ticker': ticker,
        'y_pred_ensemble': y_pred,
        'y_true': float(y_true) if pd.notna(y_true) else np.nan,
    }


def main():
    # ── 데이터 로드 — daily_returns.pkl 에서 직접 vol 계산 (LSTM 정합) ─────────
    print(f'데이터 로드: {DATA_DIR}/daily_returns.pkl')
    daily = pd.read_pickle(DATA_DIR / 'daily_returns.pkl')
    print(f'  daily shape: {daily.shape} ({daily.index.min().date()} ~ {daily.index.max().date()})')

    # 학습 universe = monthly_panel 의 ticker (LSTM 과 동일 학습 대상)
    panel_meta = pd.read_csv(DATA_DIR / 'monthly_panel.csv',
                              usecols=['date', 'ticker'], parse_dates=['date'])
    target_tix = sorted(set(panel_meta['ticker'].unique()) & set(daily.columns))
    print(f'  학습 universe: panel ∩ daily = {len(target_tix)} ticker')

    # vol_21d (annualized) 동등 계산: rolling(21).std(ddof=1) × √252
    # monthly_panel.vol_21d 와 수치 일치 (검증 완료 — _train_paper_ann 도큐스트링 참조).
    # 단, 종목별 전체 daily history 사용 (S&P500 멤버십 무관).
    daily_sub = daily[target_tix]
    rolling_daily_std = daily_sub.rolling(21, min_periods=21).std(ddof=1)

    # 월말 매핑 — panel 의 calendar 월말 index 와 정합
    # ('ME': month-end frequency; 각 월의 마지막 거래일 값을 calendar 월말로 매핑)
    month_end_daily_std = rolling_daily_std.resample('ME').last()

    # log(daily_std) — LSTM target 과 동일 공간
    log_vol_panel = np.log(month_end_daily_std.clip(lower=1e-10))

    # panel 의 월말 date 와 정렬 (calendar 월말로 통일)
    panel_dates = sorted(panel_meta['date'].unique())
    log_vol_panel = log_vol_panel.reindex(panel_dates)

    print(f'log_vol 패널: {log_vol_panel.shape}')
    print(f'  종목 수: {log_vol_panel.shape[1]}')
    print(f'  월 수: {log_vol_panel.shape[0]}')
    print(f'  종목별 non-NaN 월 수 분포: '
          f'min={log_vol_panel.notna().sum(axis=0).min()}, '
          f'median={int(log_vol_panel.notna().sum(axis=0).median())}, '
          f'max={log_vol_panel.notna().sum(axis=0).max()}')

    all_dates = log_vol_panel.index.sort_values()
    pred_dates = all_dates[all_dates >= START_PRED]
    print(f'예측 기간: {pred_dates[0].date()} ~ {pred_dates[-1].date()} ({len(pred_dates)}개월)')

    # ── walk-forward 메인 루프 ───────────────────────────────
    print(f'\nWalk-Forward ANN 학습 시작 (n_jobs={N_JOBS})')
    print('-' * 60)
    records = []
    _t0 = time.time()

    for i, pred_date in enumerate(pred_dates):
        pred_date_idx = all_dates.get_loc(pred_date)

        # 학습 가능한 종목 추출 (충분한 과거 데이터 보유)
        train_start_idx = max(0, pred_date_idx - TRAIN_WINDOW + 1)
        train_window = all_dates[train_start_idx:pred_date_idx + 1]
        n_valid_per_ticker = log_vol_panel.reindex(train_window).notna().sum(axis=0)
        tickers_with_data = n_valid_per_ticker[
            n_valid_per_ticker >= LOOKBACK + 5
        ].index.tolist()

        if not tickers_with_data:
            continue

        # 종목별 병렬 학습
        results_per_date = Parallel(n_jobs=N_JOBS, backend='threading')(
            delayed(train_predict_ticker)(
                t, log_vol_panel[t], pred_date_idx, all_dates
            )
            for t in tickers_with_data
        )

        records.extend([r for r in results_per_date if r is not None])

        # 진행 상황
        if i % 12 == 0 or i == len(pred_dates) - 1:
            elapsed = time.time() - _t0
            pct = (i + 1) / len(pred_dates)
            eta = elapsed / pct * (1 - pct) if pct > 0.01 else 0
            print(
                f'  {pred_date.strftime("%Y-%m")} ({i+1}/{len(pred_dates)}, {pct:.0%}) | '
                f'records {len(records):,} | '
                f'경과 {elapsed/60:.1f}분 | ETA {eta/60:.1f}분',
                flush=True,
            )

    elapsed = (time.time() - _t0) / 60
    print(f'\n학습 완료: {len(records):,}개 예측 / {elapsed:.1f}분')

    # ── CSV 저장 ──────────────────────────────────────────
    df = pd.DataFrame(records)
    print(f'\n저장 전: {df.shape}')
    print(f'  종목 수: {df["ticker"].nunique()}')
    print(f'  날짜 수: {df["date"].nunique()}')

    cols = ['date', 'ticker', 'y_true', 'y_pred_ensemble']
    df = df[cols].sort_values(['date', 'ticker']).reset_index(drop=True)

    df.to_csv(OUT_PATH, index=False)
    print(f'\n저장: {OUT_PATH}')
    print(f'총 {len(df):,}행 × {len(df.columns)}컬럼')

    # 간단 통계
    print('\n=== 예측 통계 ===')
    print(f'  y_pred_ensemble  mean={df["y_pred_ensemble"].mean():.3f}, '
          f'std={df["y_pred_ensemble"].std():.3f}')
    print(f'  y_true (NaN 제외)  mean={df["y_true"].dropna().mean():.3f}, '
          f'std={df["y_true"].dropna().std():.3f}')

    # RMSE 계산
    valid = df.dropna(subset=['y_true'])
    rmse = float(np.sqrt(((valid['y_pred_ensemble'] - valid['y_true']) ** 2).mean()))
    print(f'\n  전체 RMSE (log(daily_std) 공간): {rmse:.4f}')
    print(f'  (참고: LSTM v4 RMSE ≈ 0.43, HAR ≈ 0.39, Ensemble ≈ 0.38)')


if __name__ == '__main__':
    main()
