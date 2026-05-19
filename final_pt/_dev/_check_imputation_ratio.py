"""각 BL pred_date 별 median imputation 비중 측정 (LSTM vs ANN)."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).parent.parent
DATA_DIR = PROJECT / 'data'

# 1) monthly_panel 의 valid_tix (each month 의 BL universe)
panel = pd.read_csv(DATA_DIR / 'monthly_panel.csv', parse_dates=['date'])
panel = panel.dropna(subset=['vol_21d'])   # BL 이 valid 로 보는 ticker
valid_by_month = panel.groupby(panel['date'].dt.to_period('M'))['ticker'].apply(set)

# 2) LSTM 예측 universe (month-end last)
lstm = pd.read_csv(DATA_DIR / '03b_lstm' / 'data' / 'ensemble_predictions_stockwise.csv',
                   parse_dates=['date'])
lstm = lstm.replace([np.inf,-np.inf], np.nan).dropna(subset=['y_pred_ensemble'])
lstm['ym'] = lstm['date'].dt.to_period('M')
lstm_by_month = lstm.groupby('ym')['ticker'].apply(set)

# 3) ANN 예측 universe
ann = pd.read_csv(DATA_DIR / 'paper_ann_predictions.csv', parse_dates=['date'])
ann = ann.replace([np.inf,-np.inf], np.nan).dropna(subset=['y_pred_ensemble'])
ann['ym'] = ann['date'].dt.to_period('M')
ann_by_month = ann.groupby('ym')['ticker'].apply(set)

# BL pred 기간 (2010-01 ~ 2023-12)
START = pd.Period('2010-01', 'M')
END   = pd.Period('2023-12', 'M')

REGIMES = [('R1_회복', pd.Period('2010-01','M'), pd.Period('2012-06','M')),
           ('R2_확장', pd.Period('2012-07','M'), pd.Period('2019-12','M')),
           ('R3_변동', pd.Period('2020-01','M'), pd.Period('2023-12','M')),
           ('전체',   START, END)]

rows = []
for ym in sorted(valid_by_month.index):
    if ym < START or ym > END: continue
    v = valid_by_month[ym]
    l = lstm_by_month.get(ym, set())
    a = ann_by_month.get(ym, set())
    rows.append({
        'ym': ym,
        'valid_tix': len(v),
        'lstm_have': len(v & l),
        'ann_have' : len(v & a),
        'lstm_imp' : len(v - l),
        'ann_imp'  : len(v - a),
    })

df = pd.DataFrame(rows)
df['lstm_imp_pct'] = df['lstm_imp'] / df['valid_tix'] * 100
df['ann_imp_pct']  = df['ann_imp']  / df['valid_tix'] * 100

print('=' * 80)
print('각 월 BL valid_tix 대비 median imputation 비중 (LSTM vs ANN)')
print('=' * 80)
print(f'{"regime":<12}{"n_mo":>6}{"valid_tix(μ)":>14}'
      f'{"LSTM imp #":>13}{"LSTM imp %":>12}'
      f'{"ANN imp #":>12}{"ANN imp %":>11}')
print('-' * 80)
for lbl, s, e in REGIMES:
    sub = df[(df['ym'] >= s) & (df['ym'] <= e)]
    if len(sub) == 0: continue
    print(f'{lbl:<12}{len(sub):>6}{sub["valid_tix"].mean():>14.1f}'
          f'{sub["lstm_imp"].mean():>13.1f}{sub["lstm_imp_pct"].mean():>11.1f}%'
          f'{sub["ann_imp"].mean():>12.1f}{sub["ann_imp_pct"].mean():>10.1f}%')

# 가장 ANN imputation 비중 큰 월 top 10
print('\n' + '=' * 80)
print('ANN imputation 비중 top 10 월')
print('=' * 80)
print(f'{"ym":<10}{"valid_tix":>11}{"ann_have":>10}{"ann_imp":>9}{"imp%":>7}')
top = df.nlargest(10, 'ann_imp_pct')
for _, r in top.iterrows():
    print(f'{str(r["ym"]):<10}{r["valid_tix"]:>11}{r["ann_have"]:>10}{r["ann_imp"]:>9}{r["ann_imp_pct"]:>6.1f}%')

# 시계열 분포
print('\n' + '=' * 80)
print('연도별 ANN imputation 추이')
print('=' * 80)
df['year'] = df['ym'].astype(str).str[:4].astype(int)
year_avg = df.groupby('year')[['valid_tix','lstm_imp_pct','ann_imp_pct']].mean()
for yr, r in year_avg.iterrows():
    bar = '█' * int(r['ann_imp_pct'])
    print(f'  {yr}: valid {r["valid_tix"]:.0f} | LSTM imp {r["lstm_imp_pct"]:>4.1f}% | '
          f'ANN imp {r["ann_imp_pct"]:>4.1f}%  {bar}')
