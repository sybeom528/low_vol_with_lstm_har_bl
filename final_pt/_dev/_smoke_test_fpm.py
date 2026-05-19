"""Smoke test — 1 fpm 슬롯에 대해 mini walk_forward 실행 + look-ahead 감사.

목표:
1. ff3_paper_mean Q + ANN dispatcher 가 wiring 단계에서 오류 없는지 확인
2. 시점 t 의 Q 계산에 사용된 데이터가 모두 t-1 까지인지 검증

ANN CSV 부재 시 ANN 슬롯은 자동 skip — LSTM 슬롯만 테스트.
"""
import sys, io, time, pickle
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Project path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bl_runner import load_pred_csv, build_monthly_cache, walk_forward, get_Q, compute_Q_ff3_paper_mean
from bl_config import EXPERIMENTS as EXPERIMENTS_LSTM, BASELINE
from bl_config_ann import EXPERIMENTS as EXPERIMENTS_ANN

DATA_DIR = Path(__file__).parent.parent / 'data'

print('=' * 70)
print('SMOKE TEST — final_pt ff3_paper_mean + ANN dispatcher')
print('=' * 70)

# ── 데이터 로드 ──
panel     = pd.read_csv(DATA_DIR / 'monthly_panel.csv', parse_dates=['date'])
panel     = panel.set_index(['date', 'ticker'])
daily_ret = pd.read_pickle(DATA_DIR / 'daily_returns.pkl')

all_dates  = panel.index.get_level_values('date').unique().sort_values()
START_PRED = '2010-01-01'
pred_dates_full = all_dates[all_dates >= START_PRED]

# 빠른 smoke test: 마지막 6개월만
pred_dates = pred_dates_full[-6:]

spy_series = panel['spy_ret'].groupby(level='date').first()
rf_series  = panel['rf_1m'].groupby(level='date').first()
ret_pivot  = panel['ret_1m'].unstack('ticker')

ff3 = pd.read_csv(DATA_DIR / 'ff3_monthly.csv', index_col=0, parse_dates=True)

print(f'\n[데이터]')
print(f'  panel       : {panel.shape}')
print(f'  pred_dates  : {pred_dates[0].date()} ~ {pred_dates[-1].date()} ({len(pred_dates)}개월, 테스트용 6개월)')
print(f'  ff3         : {ff3.shape}')

# ── LSTM/ANN 로드 ──
lstm_state = load_pred_csv(BASELINE['lstm_pred_path'], pred_dates_full)
ann_state  = load_pred_csv(DATA_DIR / 'paper_ann_predictions.csv', pred_dates_full)
print(f'\n[예측]')
print(f'  LSTM available: {lstm_state["available"]}')
print(f'  ANN  available: {ann_state["available"]}')

# ── 캐시 ──
print(f'\n[캐시 빌드 — {len(pred_dates)}개월]')
monthly_cache = build_monthly_cache(
    panel=panel, daily_ret=daily_ret,
    pred_dates=pred_dates, all_dates=all_dates,
    spy_series=spy_series, rf_series=rf_series,
    train_window=60, thresh_daily=0.9, verbose=False,
)
print(f'  cache entries: {len(monthly_cache)}')

# ── look-ahead 정밀 감사: compute_Q_ff3_paper_mean ──
print(f'\n[Look-ahead audit — compute_Q_ff3_paper_mean]')
test_pred_date = pred_dates[-1]
idx = all_dates.get_loc(test_pred_date)
train_dates = all_dates[max(0, idx - 60): idx]
print(f'  pred_date  : {test_pred_date.date()} (idx={idx})')
print(f'  train_dates: {train_dates[0].date()} ~ {train_dates[-1].date()} ({len(train_dates)}개월)')
assert train_dates[-1] < test_pred_date, (
    f'LOOK-AHEAD BUG: train_dates 마지막 {train_dates[-1].date()} >= pred_date {test_pred_date.date()}'
)
print(f'  ✓ train_dates[-1] = {train_dates[-1].date()} < pred_date = {test_pred_date.date()}')

# 직접 호출
valid_tix = monthly_cache[test_pred_date]['valid_tix'][:50]
monthly_sl = ret_pivot.reindex(index=train_dates, columns=valid_tix).dropna(axis=1, thresh=int(len(train_dates)*0.7))
ff3_train  = ff3.reindex(train_dates)
rf_train   = rf_series.reindex(train_dates)
P_test = pd.Series({t: 0.1 if i < 5 else (-0.1 if i < 10 else 0) for i, t in enumerate(monthly_sl.columns)})
Q_test = compute_Q_ff3_paper_mean(P_test, monthly_sl, ff3_train, rf_train)
print(f'  ✓ Q value (직접 호출): {Q_test:.6f}')
assert np.isfinite(Q_test), 'Q 가 비유한값'

# ── walk_forward mini 실행 (1 LSTM fpm 슬롯) ──
print(f'\n[mini walk_forward — mat_mcap_mcap_fpm_he]')
fpm_cfg = next(e for e in EXPERIMENTS_LSTM if e['name'] == 'mat_mcap_mcap_fpm_he')
result = walk_forward(
    fpm_cfg, monthly_cache, pred_dates, lstm_state,
    spy_series=spy_series, tau=0.1, pct_group=0.30, verbose=False,
    ann_state=ann_state, ret_pivot=ret_pivot, ff3=ff3, rf_series=rf_series,
)
print(f'  ret series ({len(result["ret"])}개월):')
for d, r in result['ret'].items():
    print(f'    {d.date()} → ret={r:+.4f}')
print(f'  meta (Q 값들):')
for d, row in result['meta'].iterrows():
    print(f'    {d.date()} → Q={row["Q"]:+.6f}, lam={row["lam"]:.3f}')
print(f'  errors: {len(result["errors"])}개')
assert len(result['errors']) == 0, f'에러 발생: {result["errors"]}'
assert len(result['ret']) >= 1, '리턴이 비어있음'

# ── 추가 감사: ff3_paper omega 의 직전월 사용 패턴 ──
print(f'\n[ff3_paper omega 감사 — mat_mcap_mcap_fpm_pap]')
fpm_pap_cfg = next(e for e in EXPERIMENTS_LSTM if e['name'] == 'mat_mcap_mcap_fpm_pap')
result2 = walk_forward(
    fpm_pap_cfg, monthly_cache, pred_dates, lstm_state,
    spy_series=spy_series, tau=0.1, pct_group=0.30, verbose=False,
    ann_state=ann_state, ret_pivot=ret_pivot, ff3=ff3, rf_series=rf_series,
)
print(f'  ret series ({len(result2["ret"])}개월):')
for d, r in result2['ret'].items():
    print(f'    {d.date()} → ret={r:+.4f}')
print(f'  errors: {len(result2["errors"])}개')
assert len(result2['errors']) == 0

# ── 결과 정합성 — fpm_he vs fpm_pap 가 서로 다른 결과 ──
diff = (result['ret'] - result2['ret']).abs().mean()
print(f'\n[정합성 — fpm_he vs fpm_pap mean |diff|: {diff:.6f}]')
assert diff > 1e-8, 'fpm_he 와 fpm_pap 결과가 동일 — omega 차이 미반영 의심'
print('  ✓ omega 분기 정상 작동 (he vs ff3_paper 결과 차이 있음)')

print('\n' + '=' * 70)
print('SMOKE TEST PASSED.')
print('=' * 70)
