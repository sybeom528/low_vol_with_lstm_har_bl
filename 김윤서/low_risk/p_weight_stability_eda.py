"""
P 가중치(mcap / eq / rp / vol_mcap) 안정성 EDA

목적
  mat_eq_mcap_lam_he / mat_eq_mcap_raw_he 처럼
  p_weight='mcap' 조합이 서브기간별 Sharpe 편차가 가장 작고 평탄한 이유를
  P-벡터 단계 진단(집중도/회전율/극단노출) + 포트폴리오 단계 결과(서브 Sharpe)로 설명.

산출물
  김윤서/low_risk/outputs/pweight_eda/
    - 01_pweight_concentration.png       (eff_n, max|w|, top10 share, 시계열)
    - 02_pweight_turnover.png            (월간 P 회전율)
    - 03_pweight_distribution.png        (월말 P 분포 violin)
    - 04_pweight_signal_corr.png         (4개 스킴 P 시그널 상관)
    - 05_subperiod_sharpe.png            (3 서브기간 × 4 P가중치, q_lambda/q_raw_lam)
    - 06_cumulative_returns.png          (8 실험 누적수익률)
    - metrics_pvec.csv                   (월별 P 진단치 long format)
    - metrics_subperiod.csv              (서브기간 Sharpe table)
"""

from __future__ import annotations

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.set_loglevel('error')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / 'final'
OUTDIR = Path(__file__).resolve().parent / 'outputs' / 'pweight_eda'
OUTDIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(FINAL))
from bl_functions import build_P  # noqa: E402

PCT = 0.30
SCHEMES = ['mcap', 'eq', 'rp', 'vol_mcap']

# ── 데이터 로드 ────────────────────────────────────────────────────────────
panel = pd.read_csv(FINAL / 'data' / 'monthly_panel.csv', parse_dates=['date'])
panel = panel.set_index(['date', 'ticker']).sort_index()

lstm_path = FINAL / 'phase3(data_outputs)' / 'data' / 'ensemble_predictions_stockwise.csv'
lstm = pd.read_csv(lstm_path, parse_dates=['date'])
# 일별 → 월말 종가일 LSTM 예측 (가까운 일자 ffill 후 월말 추출)
lstm = lstm.sort_values(['ticker', 'date'])
lstm['month_end'] = lstm['date'] + pd.offsets.MonthEnd(0)
lstm_me = (lstm.groupby(['month_end', 'ticker'])['y_pred_ensemble']
                .mean()
                .reset_index()
                .rename(columns={'month_end': 'date', 'y_pred_ensemble': 'vol_pred_log'}))
# y_pred_ensemble은 log(vol)에 가까운 스케일이므로 양수로 변환
lstm_me['vol_pred'] = np.exp(lstm_me['vol_pred_log'])
lstm_me = lstm_me.set_index(['date', 'ticker'])

START, END = '2010-01-31', '2024-12-31'
months = pd.date_range(START, END, freq='ME')

# ── 1. 월별 P 진단 ─────────────────────────────────────────────────────────
def hhi(w: pd.Series) -> float:
    s = w.abs().sum()
    if s <= 0:
        return np.nan
    return float(((w.abs() / s) ** 2).sum())

def eff_n(w: pd.Series) -> float:
    h = hhi(w)
    return np.nan if not np.isfinite(h) or h <= 0 else 1.0 / h

records = []
P_history = {s: {} for s in SCHEMES}

for d in months:
    if d not in panel.index.get_level_values(0):
        continue
    snap = panel.loc[d]
    snap = snap.dropna(subset=['vol_21d', 'log_mcap'])
    if len(snap) < 50:
        continue

    # LSTM vol 예측 (log_mcap·vol_21d 정렬과 동일 ticker 기준)
    if d in lstm_me.index.get_level_values(0):
        v_pred = lstm_me.loc[d]['vol_pred']
        v_pred = v_pred.reindex(snap.index)
    else:
        v_pred = pd.Series(index=snap.index, dtype=float)
    # LSTM 미공급 종목은 trailing vol_21d로 대체 (final/walk_forward와 동일 정책)
    vol = v_pred.where(v_pred.notna(), snap['vol_21d'])
    mcap = np.exp(snap['log_mcap'])

    valid = vol.notna() & mcap.notna() & (mcap > 0)
    vol = vol[valid]
    mcap = mcap[valid]
    n = len(vol)
    if n < 50:
        continue

    for s in SCHEMES:
        P = build_P(vol, mcap, pct=PCT, weighting=s)
        P_history[s][d] = P
        long_w = P[P > 0]
        short_w = P[P < 0]
        records.append({
            'date': d, 'scheme': s, 'n_universe': n,
            'eff_n': eff_n(P),
            'max_abs': float(P.abs().max()),
            'top10_share': float(P.abs().nlargest(10).sum() / P.abs().sum()) if P.abs().sum() > 0 else np.nan,
            'long_sum': float(long_w.sum()),
            'short_sum': float(short_w.sum()),
            'n_nonzero': int((P != 0).sum()),
            'long_n': int((P > 0).sum()),
            'short_n': int((P < 0).sum()),
        })

df_p = pd.DataFrame(records)
df_p.to_csv(OUTDIR / 'metrics_pvec.csv', index=False)
print(f'[1] P 진단 기록: {len(df_p):,} rows ({df_p["date"].nunique()} months × {len(SCHEMES)} schemes)')

# ── P 회전율: ||P_t - P_{t-1}||_1 ─────────────────────────────────────────
turnover_records = []
for s in SCHEMES:
    hist = P_history[s]
    dates_sorted = sorted(hist.keys())
    prev = None
    for d in dates_sorted:
        cur = hist[d]
        if prev is not None:
            common = cur.index.union(prev.index)
            diff = cur.reindex(common, fill_value=0) - prev.reindex(common, fill_value=0)
            turnover_records.append({'date': d, 'scheme': s, 'turnover': float(diff.abs().sum())})
        prev = cur

df_turn = pd.DataFrame(turnover_records)
df_p = df_p.merge(df_turn, on=['date', 'scheme'], how='left')

# ── 시각화 1: 집중도 시계열 ─────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
for s in SCHEMES:
    sub = df_p[df_p['scheme'] == s].sort_values('date')
    axes[0].plot(sub['date'], sub['eff_n'], label=s, alpha=0.85)
    axes[1].plot(sub['date'], sub['max_abs'], label=s, alpha=0.85)
    axes[2].plot(sub['date'], sub['top10_share'], label=s, alpha=0.85)
axes[0].set_ylabel('Effective N (1/HHI)')
axes[0].set_title('P-vector 집중도 시계열')
axes[0].legend(ncol=4); axes[0].grid(alpha=0.3)
axes[1].set_ylabel('max |w|'); axes[1].grid(alpha=0.3)
axes[2].set_ylabel('top-10 share'); axes[2].set_xlabel('date'); axes[2].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUTDIR / '01_pweight_concentration.png', dpi=130); plt.close()

# ── 시각화 2: 회전율 ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4))
for s in SCHEMES:
    sub = df_p[df_p['scheme'] == s].sort_values('date')
    ax.plot(sub['date'], sub['turnover'], label=s, alpha=0.8)
ax.set_title('P-vector 회전율  ||P_t − P_{t−1}||_1')
ax.set_ylabel('L1 turnover'); ax.set_xlabel('date'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUTDIR / '02_pweight_turnover.png', dpi=130); plt.close()

# ── 시각화 3: 분포 (sample 12 month-ends) ────────────────────────────────
sample_months = pd.date_range('2010-12-31', '2024-12-31', freq='YE')
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
for ax, s in zip(axes, SCHEMES):
    data = []
    for d in sample_months:
        if d in P_history[s]:
            P = P_history[s][d]
            data.append(P[P != 0].values)
    ax.boxplot(data, showfliers=True, widths=0.7)
    ax.axhline(0, color='red', lw=0.5)
    ax.set_title(f'P 분포 (year-end, {s})')
    ax.set_xticklabels([d.year for d in sample_months], rotation=45, fontsize=8)
    ax.grid(alpha=0.3)
axes[0].set_ylabel('weight (long +, short −)')
plt.tight_layout(); plt.savefig(OUTDIR / '03_pweight_distribution.png', dpi=130); plt.close()

# ── 시각화 4: 시그널 상관 ──────────────────────────────────────────────
def flatten_pseries(hist: dict) -> pd.Series:
    parts = []
    for d, P in hist.items():
        s = P.copy(); s.index = pd.MultiIndex.from_product([[d], P.index], names=['date', 'ticker'])
        parts.append(s)
    return pd.concat(parts)

big = pd.DataFrame({s: flatten_pseries(P_history[s]) for s in SCHEMES}).fillna(0)
corr = big.corr()
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(SCHEMES))); ax.set_xticklabels(SCHEMES)
ax.set_yticks(range(len(SCHEMES))); ax.set_yticklabels(SCHEMES)
for i in range(len(SCHEMES)):
    for j in range(len(SCHEMES)):
        ax.text(j, i, f'{corr.values[i,j]:.2f}', ha='center', va='center', fontsize=10)
ax.set_title('P 시그널 상관 (전체 패널 flatten)')
plt.colorbar(im); plt.tight_layout(); plt.savefig(OUTDIR / '04_pweight_signal_corr.png', dpi=130); plt.close()

print('[1] P-vector EDA 시각화 4종 저장')

# ── 2. 포트폴리오 서브기간 Sharpe ─────────────────────────────────────────
EXP_BY_QP = {
    'q_lambda': {
        'mcap':     'mat_eq_mcap_lam_he',
        'eq':       'mat_eq_eq_lam_he',
        'rp':       'mat_eq_rp_lam_he',
        'vol_mcap': 'prior_eq_p_vol_mcap_mat_mcap_mcap_lam_he',
    },
    'q_raw_lam': {
        'mcap':     'mat_eq_mcap_raw_he',
        'eq':       'mat_eq_eq_raw_he',
        'rp':       'mat_eq_rp_raw_he',
        'vol_mcap': 'prior_eq_p_vol_mcap_q_raw_lam_p_lstm',
    },
}

SUBPERIODS = [('2010-2014', '2010-01-01', '2014-12-31'),
              ('2015-2019', '2015-01-01', '2019-12-31'),
              ('2020-2024', '2020-01-01', '2024-12-31')]

def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 12 or r.std() == 0:
        return np.nan
    return float(r.mean() / r.std() * np.sqrt(12))

rows = []
returns_all = {}
for q_mode, mapping in EXP_BY_QP.items():
    for pw, name in mapping.items():
        path = FINAL / 'results' / f'{name}.pkl'
        if not path.exists():
            print(f'  [WARN] missing {name}'); continue
        with open(path, 'rb') as f:
            d = pickle.load(f)
        ret = d['ret']
        returns_all[(q_mode, pw)] = ret
        for label, s, e in SUBPERIODS:
            sl = ret.loc[s:e]
            rows.append({'q_mode': q_mode, 'p_weight': pw, 'subperiod': label,
                         'sharpe': sharpe(sl), 'mean_m': float(sl.mean()), 'std_m': float(sl.std())})
        rows.append({'q_mode': q_mode, 'p_weight': pw, 'subperiod': 'full',
                     'sharpe': sharpe(ret), 'mean_m': float(ret.mean()), 'std_m': float(ret.std())})

df_sub = pd.DataFrame(rows)
df_sub.to_csv(OUTDIR / 'metrics_subperiod.csv', index=False)

# ── 시각화 5: 서브기간 Sharpe 히트맵 ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, q_mode in zip(axes, ['q_lambda', 'q_raw_lam']):
    sub = df_sub[(df_sub['q_mode'] == q_mode) & (df_sub['subperiod'] != 'full')]
    pivot = sub.pivot(index='subperiod', columns='p_weight', values='sharpe').reindex(
        index=[s[0] for s in SUBPERIODS], columns=SCHEMES)
    im = ax.imshow(pivot.values, cmap='RdYlGn', vmin=-0.5, vmax=2.5, aspect='auto')
    ax.set_xticks(range(len(SCHEMES))); ax.set_xticklabels(SCHEMES)
    ax.set_yticks(range(len(SUBPERIODS))); ax.set_yticklabels([s[0] for s in SUBPERIODS])
    ax.set_title(f'{q_mode} | sub-period Sharpe')
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f'{v:.2f}' if pd.notna(v) else '–', ha='center', va='center', fontsize=10)
    plt.colorbar(im, ax=ax)
plt.tight_layout(); plt.savefig(OUTDIR / '05_subperiod_sharpe.png', dpi=130); plt.close()

# ── 시각화 6: 누적수익률 ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for ax, q_mode in zip(axes, ['q_lambda', 'q_raw_lam']):
    for pw in SCHEMES:
        if (q_mode, pw) not in returns_all: continue
        r = returns_all[(q_mode, pw)]
        cum = (1 + r).cumprod()
        ax.plot(cum.index, cum.values, label=f'p={pw}', alpha=0.85)
    ax.set_title(f'{q_mode} | 누적수익 (1+r 곱)')
    ax.set_yscale('log'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUTDIR / '06_cumulative_returns.png', dpi=130); plt.close()

# ── 진단 요약 ────────────────────────────────────────────────────────────
print('\n=== P-벡터 진단 요약 (월별 평균) ===')
agg = df_p.groupby('scheme').agg(
    eff_n=('eff_n', 'mean'),
    max_abs=('max_abs', 'mean'),
    top10_share=('top10_share', 'mean'),
    turnover=('turnover', 'mean'),
    n_nonzero=('n_nonzero', 'mean'),
).round(3)
print(agg)

print('\n=== 서브기간 Sharpe table ===')
piv_l = df_sub[df_sub['q_mode'] == 'q_lambda'].pivot(
    index='subperiod', columns='p_weight', values='sharpe').reindex(
    index=[s[0] for s in SUBPERIODS] + ['full'], columns=SCHEMES)
piv_r = df_sub[df_sub['q_mode'] == 'q_raw_lam'].pivot(
    index='subperiod', columns='p_weight', values='sharpe').reindex(
    index=[s[0] for s in SUBPERIODS] + ['full'], columns=SCHEMES)
print('\n[q_lambda]'); print(piv_l.round(3))
print('\n[q_raw_lam]'); print(piv_r.round(3))

# 서브기간 std (낮을수록 평탄)
flat = df_sub[df_sub['subperiod'] != 'full'].groupby(['q_mode', 'p_weight'])['sharpe'].std().unstack()
print('\n=== 서브기간 Sharpe 표준편차 (낮을수록 평탄) ==='); print(flat.round(3))

# 결과 저장
agg.to_csv(OUTDIR / 'pvec_summary.csv')
piv_l.to_csv(OUTDIR / 'sharpe_q_lambda.csv')
piv_r.to_csv(OUTDIR / 'sharpe_q_raw_lam.csv')
flat.to_csv(OUTDIR / 'sharpe_subperiod_std.csv')

print(f'\n✓ 모든 출력 → {OUTDIR}')
