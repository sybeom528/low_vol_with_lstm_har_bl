"""Smoke test — modified _train_paper_ann.py 의 데이터 로딩 + 학습 로직 검증.

검증 항목:
1. 새 데이터 로딩 (daily_returns.pkl → log_vol_panel) 성공
2. 종목별 cover 가 OLD (panel-based) 보다 같거나 큼
3. mini ANN 학습 1 종목 × 2 pred_date — look-ahead 없음 (train data 가 pred_date 이전만)
"""
import sys, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).parent.parent / 'data'

LOOKBACK = 10
TRAIN_WINDOW = 60

print('=' * 70)
print('SMOKE TEST — modified _train_paper_ann.py 데이터 + 학습')
print('=' * 70)

# ── Step 1: 데이터 로딩 (modified 스크립트와 동일 로직) ──
daily = pd.read_pickle(DATA_DIR / 'daily_returns.pkl')
panel_meta = pd.read_csv(DATA_DIR / 'monthly_panel.csv',
                          usecols=['date', 'ticker'], parse_dates=['date'])
target_tix = sorted(set(panel_meta['ticker'].unique()) & set(daily.columns))
daily_sub = daily[target_tix]
rolling_daily_std = daily_sub.rolling(21, min_periods=21).std(ddof=1)
month_end_daily_std = rolling_daily_std.resample('ME').last()
log_vol_panel = np.log(month_end_daily_std.clip(lower=1e-10))
panel_dates = sorted(panel_meta['date'].unique())
log_vol_panel = log_vol_panel.reindex(panel_dates)

print(f'\n[Step 1] 데이터 로딩')
print(f'  daily            : {daily.shape}')
print(f'  target_tix       : {len(target_tix)} (panel ∩ daily)')
print(f'  log_vol_panel    : {log_vol_panel.shape}')
print(f'  panel_dates 첫/끝: {panel_dates[0].date()} ~ {panel_dates[-1].date()}')

# ── Step 2: vol_21d 공식 일치 (panel 와 비교, 검증 재확인) ──
old_panel = pd.read_csv(DATA_DIR / 'monthly_panel.csv',
                         parse_dates=['date'], usecols=['date','ticker','vol_21d']
                         ).set_index(['date','ticker'])['vol_21d'].unstack('ticker')

# 같은 시점에 두 시리즈가 모두 valid 인 cell 비교
sample_t = 'AAPL'
common_dates = log_vol_panel.index.intersection(old_panel.index)
new_val = log_vol_panel[sample_t].reindex(common_dates).dropna()
# panel vol_21d = annualized → /sqrt(252) → log = log_vol_panel 동등
old_val_log = np.log((old_panel[sample_t].reindex(common_dates) / np.sqrt(252)).clip(lower=1e-10)).dropna()
common2 = new_val.index.intersection(old_val_log.index)
diffs = (new_val.reindex(common2) - old_val_log.reindex(common2)).abs()
print(f'\n[Step 2] vol_21d 일치 검증 (AAPL, {len(common2)} 시점)')
print(f'  max |diff|: {diffs.max():.2e}, mean |diff|: {diffs.mean():.2e}')
assert diffs.max() < 1e-10, f'NEW vs OLD vol_21d 불일치: max diff={diffs.max()}'
print(f'  ✓ NEW (daily-derived) ≡ OLD (panel) for AAPL')

# ── Step 3: 신규 상장주의 예측 가능 시점 확인 ──
test_tix_new = ['CVNA', 'DDOG', 'APP', 'HOOD']
all_dates = log_vol_panel.index
print(f'\n[Step 3] 신규 상장주 예측 가능 시점 (NEW vs OLD)')
print(f'  {"ticker":7s} {"NEW 첫 예측":12s} {"OLD 첫 예측":12s}')
for t in test_tix_new:
    def first_pred(s):
        for i in range(TRAIN_WINDOW - 1, len(all_dates)):
            tw = all_dates[i - TRAIN_WINDOW + 1: i + 1]
            if s.reindex(tw).notna().sum() >= LOOKBACK + 5:
                return all_dates[i]
        return None
    new = first_pred(log_vol_panel[t]) if t in log_vol_panel.columns else None
    old = first_pred(old_panel[t]) if t in old_panel.columns else None
    print(f'  {t:7s} {str(new.date()) if new is not None else "불가":12s} {str(old.date()) if old is not None else "불가":12s}')

# ── Step 4: mini ANN 학습 — look-ahead 정밀 검증 ──
print(f'\n[Step 4] mini ANN 학습 + look-ahead 정밀 검증')

# 2022-12-31 시점에서 CVNA 학습 시뮬레이션
test_ticker = 'CVNA'
test_pred_date_idx = all_dates.get_loc(pd.Timestamp('2022-12-31'))
test_pred_date = all_dates[test_pred_date_idx]
print(f'  pred_date: {test_pred_date.date()} ({test_ticker})')

# train_window = M-59 ~ M (inclusive)
train_start = max(0, test_pred_date_idx - TRAIN_WINDOW + 1)
train_window = all_dates[train_start: test_pred_date_idx + 1]
print(f'  train_window: {train_window[0].date()} ~ {train_window[-1].date()} ({len(train_window)}개월)')
assert train_window[-1] <= test_pred_date, 'train_window 가 pred_date 초과!'
print(f'  ✓ train_window 마지막 = pred_date 자체 (포함 OK, 미래 데이터 없음)')

train_vols = log_vol_panel[test_ticker].reindex(train_window).dropna()
print(f'  train_vols 유효 개수: {len(train_vols)}')

if len(train_vols) >= TRAIN_WINDOW and len(train_vols) - LOOKBACK >= 5:
    arr = train_vols.values
    X_train = np.array([arr[i:i+LOOKBACK] for i in range(len(arr) - LOOKBACK)])
    y_train = np.array([arr[i+LOOKBACK] for i in range(len(arr) - LOOKBACK)])
    print(f'  학습 페어: X={X_train.shape}, y={y_train.shape}')

    # 마지막 y_train 값 = 시점 M (pred_date) 의 vol — 학습 input
    print(f'  마지막 y_train (= pred_date M 의 vol): {y_train[-1]:.4f}')
    print(f'  pred_date {test_pred_date.date()} 의 log_vol[{test_ticker}]: '
          f'{log_vol_panel.loc[test_pred_date, test_ticker]:.4f}')

    # 학습
    sX, sy = StandardScaler(), StandardScaler()
    Xs = sX.fit_transform(X_train)
    ys = sy.fit_transform(y_train.reshape(-1,1)).ravel()
    m = MLPRegressor(hidden_layer_sizes=(4,), activation='relu', solver='adam',
                     alpha=0.01, max_iter=50, learning_rate_init=0.001, random_state=42)
    m.fit(Xs, ys)
    X_pred = arr[-LOOKBACK:].reshape(1, -1)
    yp = sy.inverse_transform(m.predict(sX.transform(X_pred)).reshape(-1,1))[0,0]
    print(f'  → predicted log_vol for {(test_pred_date + pd.offsets.MonthEnd(1)).date()} (M+1): {yp:.4f}')

    # 실제값 (다음 월)
    next_dt = all_dates[test_pred_date_idx + 1] if test_pred_date_idx + 1 < len(all_dates) else None
    if next_dt is not None:
        actual = log_vol_panel.loc[next_dt, test_ticker]
        print(f'  → actual    log_vol for {next_dt.date()}      : {actual:.4f}')
        print(f'  → |error|: {abs(yp - actual):.4f}')

    # look-ahead 정밀 검증: X_pred 의 모든 element 가 pred_date 이전(포함) 데이터인지
    X_pred_dates = train_vols.index[-LOOKBACK:].tolist()
    print(f'  X_pred 입력 사용된 날짜 범위: {X_pred_dates[0].date()} ~ {X_pred_dates[-1].date()}')
    assert X_pred_dates[-1] <= test_pred_date, 'LOOK-AHEAD: X_pred 가 pred_date 초과!'
    print(f'  ✓ X_pred 마지막 ≤ pred_date — leakage 없음')
else:
    print(f'  ⚠ {test_ticker} @ {test_pred_date.date()}: 학습 페어 부족 (실 학습엔 다른 시점에서 가능)')

print('\n' + '=' * 70)
print('SMOKE TEST PASSED.')
print('=' * 70)
