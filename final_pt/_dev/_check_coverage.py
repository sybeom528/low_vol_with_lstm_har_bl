"""LSTM vs ANN prediction coverage 차이 source 직접 확인."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).parent.parent
DATA_DIR = PROJECT / 'data'

# LSTM daily → month-end
lstm = pd.read_csv(DATA_DIR / '03b_lstm' / 'data' / 'ensemble_predictions_stockwise.csv',
                   parse_dates=['date'])
lstm_clean = lstm.replace([np.inf,-np.inf], np.nan).dropna(subset=['y_pred_ensemble','y_true'])
lstm_clean['ym'] = lstm_clean['date'].dt.to_period('M')
lstm_m = (lstm_clean.sort_values('date')
          .groupby(['ym','ticker'], as_index=False).last())

# ANN monthly
ann = pd.read_csv(DATA_DIR / 'paper_ann_predictions.csv', parse_dates=['date'])
ann_clean = ann.replace([np.inf,-np.inf], np.nan).dropna(subset=['y_pred_ensemble','y_true'])
ann_clean['ym'] = ann_clean['date'].dt.to_period('M')

print('=' * 72)
print('(1) 전체 row 수')
print('=' * 72)
print(f'  LSTM raw daily rows         : {len(lstm):>10,}')
print(f'  LSTM cleaned (no inf/nan)   : {len(lstm_clean):>10,}')
print(f'  LSTM monthly (last per ym×t): {len(lstm_m):>10,}')
print(f'  ANN  raw monthly rows       : {len(ann):>10,}')
print(f'  ANN  cleaned                : {len(ann_clean):>10,}')
print(f'  → gap (cleaned)             : {len(lstm_m)-len(ann_clean):>10,}')

print('\n' + '=' * 72)
print('(2) 고유 ticker / 고유 month 수')
print('=' * 72)
l_tix = set(lstm_m['ticker'].unique())
a_tix = set(ann_clean['ticker'].unique())
l_ym  = set(lstm_m['ym'].unique())
a_ym  = set(ann_clean['ym'].unique())
print(f'  LSTM tickers          : {len(l_tix):>5}')
print(f'  ANN  tickers          : {len(a_tix):>5}')
print(f'  ∩ (공통 ticker)        : {len(l_tix & a_tix):>5}')
print(f'  LSTM only             : {len(l_tix - a_tix):>5}')
print(f'  ANN  only             : {len(a_tix - l_tix):>5}')
print(f'  LSTM months           : {len(l_ym):>5}')
print(f'  ANN  months           : {len(a_ym):>5}')
print(f'  공통 months           : {len(l_ym & a_ym):>5}')
print(f'  LSTM only months      : {sorted(l_ym - a_ym)[:5]}{"..." if len(l_ym-a_ym)>5 else ""}')
print(f'  ANN  only months      : {sorted(a_ym - l_ym)[:5]}{"..." if len(a_ym-l_ym)>5 else ""}')

print('\n' + '=' * 72)
print('(3) 공통 (month, ticker) vs 비대칭')
print('=' * 72)
lset = set(zip(lstm_m['ym'].astype(str), lstm_m['ticker']))
aset = set(zip(ann_clean['ym'].astype(str), ann_clean['ticker']))
print(f'  LSTM (ym,ticker) pairs        : {len(lset):>8,}')
print(f'  ANN  (ym,ticker) pairs        : {len(aset):>8,}')
print(f'  공통                          : {len(lset & aset):>8,}')
print(f'  LSTM only (ANN 못한 ym,ticker): {len(lset - aset):>8,}')
print(f'  ANN  only (LSTM 못한 ym,ticker): {len(aset - lset):>8,}')

print('\n' + '=' * 72)
print('(4) 비대칭 source - LSTM only 의 ticker 분포')
print('=' * 72)
lstm_only = lset - aset
if lstm_only:
    # 어떤 ticker 가 LSTM 만 있냐
    lstm_only_df = pd.DataFrame(list(lstm_only), columns=['ym','ticker'])
    cnt = lstm_only_df['ticker'].value_counts()
    print(f'  LSTM-only (ym,ticker) entries 가 가장 많은 ticker top 15:')
    for t, c in cnt.head(15).items():
        in_ann = '(ANN universe 에 아예 없음)' if t not in a_tix else f'(ANN 도 가짐, {sum(1 for ym,tt in aset if tt==t)} 월)'
        print(f'    {t:>6}: {c:>4} 월  {in_ann}')

    # 그 중 ANN universe 에 아예 없는 ticker 수
    tix_lstm_only = set(lstm_only_df['ticker']) - a_tix
    print(f'\n  LSTM-only ticker 중 ANN universe 에 아예 없는 ticker: {len(tix_lstm_only)}')
    print(f'    samples: {sorted(tix_lstm_only)[:20]}')

    # 그 중 ANN 에도 ticker 는 있지만 특정 월만 누락된 경우
    tix_partial = set(lstm_only_df['ticker']) & a_tix
    print(f'  LSTM-only ticker 중 ANN 도 ticker 보유 but 일부 월만 누락: {len(tix_partial)}')

print('\n' + '=' * 72)
print('(5) 시간대별 ANN 누락 - ANN 의 ticker count by year')
print('=' * 72)
ann_clean['year'] = ann_clean['date'].dt.year
lstm_m['year'] = lstm_m['ym'].dt.year
for yr in sorted(set(ann_clean['year']) | set(lstm_m['year'])):
    a_count = ann_clean[ann_clean['year']==yr]['ticker'].nunique()
    l_count = lstm_m[lstm_m['year']==yr]['ticker'].nunique()
    diff = l_count - a_count
    bar = '█' * (diff // 3) if diff > 0 else ''
    print(f'  {yr}: LSTM {l_count:>3} tix | ANN {a_count:>3} tix | gap {diff:>+4}  {bar}')
