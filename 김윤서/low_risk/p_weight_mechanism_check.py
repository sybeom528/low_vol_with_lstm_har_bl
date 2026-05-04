"""
mcap 강점 메커니즘 검증
  가설1: "위기 시 메가캡 보유" → drawdown 시 mcap이 가장 덜 빠지는가?
  가설2: "메가캡 장세 정렬" → low30 안의 메가캡 비중이 시간에 따라 늘어나며 mcap의 long을 그쪽으로 옮기는가?
  가설3: "short 분산" → vol_mcap은 메가캡 net short, mcap은 메가캡 net long인 비율이 시기별로 어떻게 다른가?
"""
from __future__ import annotations
import sys, pickle
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); matplotlib.set_loglevel('error')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / 'final'
OUTDIR = Path(__file__).resolve().parent / 'outputs' / 'pweight_eda'
OUTDIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(FINAL))
from bl_functions import build_P  # noqa

panel = pd.read_csv(FINAL/'data'/'monthly_panel.csv', parse_dates=['date']).set_index(['date','ticker']).sort_index()

# 메가캡 정의: 그 시점 mcap 상위 20개
def top_mcap_set(snap: pd.DataFrame, k: int = 20) -> set:
    s = snap.dropna(subset=['log_mcap'])
    return set(s.nlargest(k, 'log_mcap').index)

months = pd.date_range('2010-01-31','2024-12-31', freq='ME')
SCHEMES = ['mcap','eq','rp','vol_mcap']
rows = []

for d in months:
    if d not in panel.index.get_level_values(0): continue
    snap = panel.loc[d].dropna(subset=['vol_21d','log_mcap'])
    if len(snap) < 50: continue
    mcap = np.exp(snap['log_mcap'])
    vol = snap['vol_21d']  # trailing — 비교 단순화
    mega = top_mcap_set(snap, 20)
    n_g = max(1, int(len(snap)*0.30))
    sorted_idx = vol.sort_values().index
    low30 = set(sorted_idx[:n_g])
    high30 = set(sorted_idx[-n_g:])

    rec = {'date': d, 'mega_in_low30': len(mega & low30), 'mega_in_high30': len(mega & high30)}

    for s in SCHEMES:
        P = build_P(vol, mcap, pct=0.30, weighting=s)
        # 메가캡 net P 합 (양수=long 익스포저, 음수=short 익스포저)
        P_mega = P.reindex(list(mega), fill_value=0)
        rec[f'{s}_mega_netP'] = float(P_mega.sum())
        rec[f'{s}_mega_long']  = float(P_mega[P_mega>0].sum())
        rec[f'{s}_mega_short'] = float(P_mega[P_mega<0].sum())
    rows.append(rec)

df = pd.DataFrame(rows).set_index('date')
df.to_csv(OUTDIR/'mega_exposure.csv')

# ── A. 메가캡 분포: low30 vs high30 ────────────────────────────────────
fig, ax = plt.subplots(figsize=(11,3.5))
ax.fill_between(df.index, 0, df['mega_in_low30'], alpha=0.6, label='mega in low30 (long candidate)')
ax.fill_between(df.index, 0, -df['mega_in_high30'], alpha=0.6, color='red', label='mega in high30 (short candidate)')
ax.axhline(0, color='k', lw=0.5); ax.legend(); ax.grid(alpha=0.3)
ax.set_title('top-20 메가캡 중 low-vol 30% / high-vol 30% 진입 갯수 시계열')
ax.set_ylabel('count'); plt.tight_layout()
plt.savefig(OUTDIR/'07_mega_lowhigh_count.png', dpi=130); plt.close()

# ── B. 스킴별 메가캡 net P 익스포저 ────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
for ax, s in zip(axes.flat, SCHEMES):
    ax.plot(df.index, df[f'{s}_mega_netP'], label='net', color='black')
    ax.fill_between(df.index, 0, df[f'{s}_mega_long'], alpha=0.3, color='green', label='long')
    ax.fill_between(df.index, 0, df[f'{s}_mega_short'], alpha=0.3, color='red', label='short')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title(f'{s}: top-20 메가캡 P 익스포저'); ax.legend(loc='upper left'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUTDIR/'08_mega_netP_by_scheme.png', dpi=130); plt.close()

# ── C. 서브기간 평균 메가캡 익스포저 ──────────────────────────────────
SUB = [('2010-2014','2010','2014'),('2015-2019','2015','2019'),('2020-2024','2020','2024')]
agg_rows = []
for label, s_, e_ in SUB:
    sub = df.loc[s_:e_]
    for sc in SCHEMES:
        agg_rows.append({'subperiod': label, 'scheme': sc,
                         'mega_netP_avg': sub[f'{sc}_mega_netP'].mean(),
                         'mega_long_avg': sub[f'{sc}_mega_long'].mean(),
                         'mega_short_avg': sub[f'{sc}_mega_short'].mean()})
agg = pd.DataFrame(agg_rows)
piv_net = agg.pivot(index='subperiod', columns='scheme', values='mega_netP_avg')[SCHEMES]
piv_short = agg.pivot(index='subperiod', columns='scheme', values='mega_short_avg')[SCHEMES]
print('\n=== 서브기간별 평균 top-20 메가캡 net P 익스포저 ==='); print(piv_net.round(4))
print('\n=== 서브기간별 평균 top-20 메가캡 short 노출 (음수, 작을수록 큰 short) ==='); print(piv_short.round(4))
piv_net.to_csv(OUTDIR/'mega_netP_subperiod.csv')
piv_short.to_csv(OUTDIR/'mega_short_subperiod.csv')

# ── D. 위기 drawdown 분석 ──────────────────────────────────────────────
EXP = {
    'q_lambda': {'mcap':'prior_eq_q_lambda_p_lstm','eq':'prior_eq_p_lstm_eq_q_lambda',
                 'rp':'prior_eq_p_lstm_rp_q_lambda','vol_mcap':'prior_eq_p_vol_mcap_q_lambda_p_lstm'},
}
CRISIS = [('COVID', '2020-02', '2020-04'),
          ('2022 약세장', '2022-01', '2022-10'),
          ('2018 4Q', '2018-10', '2018-12')]

returns = {}
for pw, name in EXP['q_lambda'].items():
    p = FINAL/'results'/f'{name}.pkl'
    if not p.exists(): continue
    with open(p,'rb') as f: returns[pw] = pickle.load(f)['ret']

print('\n=== 위기 구간 누적수익률 ===')
crisis_rows = []
for cname, s_, e_ in CRISIS:
    print(f'\n[{cname}] {s_} ~ {e_}')
    for pw in SCHEMES:
        if pw not in returns: continue
        r = returns[pw].loc[s_:e_]
        cum = (1+r).prod() - 1
        max_dd = ((1+r).cumprod() / (1+r).cumprod().cummax() - 1).min()
        print(f'  {pw:10s}  cum {cum:+.2%}  max DD {max_dd:+.2%}')
        crisis_rows.append({'crisis': cname, 'p_weight': pw, 'cum_ret': cum, 'max_dd': max_dd})

pd.DataFrame(crisis_rows).to_csv(OUTDIR/'crisis_drawdown.csv', index=False)

print(f'\n✓ outputs → {OUTDIR}')
