"""Clean rewrite of [9] — minimal, focused output."""
import json, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

nb_path = '99_main_analysis.ipynb'
nb = json.load(open(nb_path, encoding='utf-8'))

new_code = r'''# ── [9] R4 mechanism — fpm q sign + AI 섹터 노출 ──────────────
R4_S = pd.Timestamp('2023-07-31')
R4_E = pd.Timestamp('2025-12-31')

# 데이터 준비
_lstm_raw = pd.read_csv(DATA_DIR/'03b_lstm/data/ensemble_predictions_stockwise.csv',
                          parse_dates=['date'])[['date','ticker','y_pred_ensemble']]
_lstm_raw['ym'] = _lstm_raw['date'].dt.to_period('M')
_lstm_m = _lstm_raw.sort_values('date').groupby(['ym','ticker'], as_index=False).last()
_lstm_m['date'] = _lstm_m['ym'].dt.to_timestamp(how='end').apply(
    lambda d: pd.Timestamp(d.year, d.month, 1) + pd.offsets.MonthEnd(0))
_lstm_pivot = _lstm_m.pivot(index='date', columns='ticker', values='y_pred_ensemble')

_panel_full = pd.read_csv(DATA_DIR/'monthly_panel.csv', parse_dates=['date'])
_panel_full['ym'] = _panel_full['date'].dt.to_period('M')
AI_SECTORS = {'Information Technology', 'Technology', 'Communication Services'}
_sector_map = (_panel_full.dropna(subset=['gics_sector'])
                 .groupby('ticker')['gics_sector']
                 .agg(lambda s: s.mode().iloc[0] if len(s.mode())>0 else 'Unknown')
                 .to_dict())

def bucket_summary(slot_name, lstm_pivot, R4_S, R4_E):
    """ slot 의 R4 월별 q + low30%/hi30% bucket weight 계산. """
    w_all = loaded[slot_name]['weights']
    meta = loaded[slot_name]['meta']
    q_series = meta['Q']
    w_r4 = w_all[(w_all.index>=R4_S)&(w_all.index<=R4_E)]
    rows = []
    for dt in w_r4.index:
        wt = w_r4.loc[dt].dropna()
        sig = lstm_pivot.loc[dt] if dt in lstm_pivot.index else None
        q_val = q_series.get(dt, np.nan)
        if sig is None or pd.isna(q_val): continue
        common = wt.index.intersection(sig.dropna().index)
        if len(common) < 10: continue
        sig_c = sig.loc[common]
        n_b = max(1, int(len(sig_c)*0.30))
        rows.append(dict(
            date=dt, q=q_val,
            w_lo=wt.loc[sig_c.nsmallest(n_b).index].sum(),
            w_hi=wt.loc[sig_c.nlargest(n_b).index].sum(),
        ))
    return pd.DataFrame(rows)

# ── (1) fpm vs lam bucket weight 비교 — slot family (prior=mcap, p_w=eq) ──
fpm_slot = 'mat_mcap_eq_fpm_pap'   # LSTM-best (Q=fpm, sign-flexible)
lam_slot = 'mat_mcap_eq_lam_pap'   # 같은 family Q-swap (Q=lam, σ-direct, q>0 항상)

fpm = bucket_summary(fpm_slot, _lstm_pivot, R4_S, R4_E)
lam = bucket_summary(lam_slot, _lstm_pivot, R4_S, R4_E)

print('='*72)
print(f'R4 portfolio bucket weight — Q 비교 (prior=mcap, p_w=eq family)')
print(f'  fpm slot ({fpm_slot}): Q sign-flexible')
print(f'  lam slot ({lam_slot}): Q always positive (σ-direct)')
print('='*72)
print(f'{"":<15s}{"q":>15s}{"low30% w":>12s}{"hi30% w":>12s}{"hi−low":>10s}')
print('-'*64)
# fpm: q<0 / q>0 분리
neg = fpm[fpm['q']<0]; pos = fpm[fpm['q']>=0]
print(f'{"fpm q<0":<15s}{"mean " + f"{neg["q"].mean():+.4f}":>15s}'
      f'{neg["w_lo"].mean()*100:>11.2f}%{neg["w_hi"].mean()*100:>11.2f}%'
      f'{(neg["w_hi"]-neg["w_lo"]).mean()*100:>+9.2f}%   ← 음수 q → high-vol over')
print(f'{"fpm q>=0":<15s}{"mean " + f"{pos["q"].mean():+.4f}":>15s}'
      f'{pos["w_lo"].mean()*100:>11.2f}%{pos["w_hi"].mean()*100:>11.2f}%'
      f'{(pos["w_hi"]-pos["w_lo"]).mean()*100:>+9.2f}%')
print(f'{"lam (q always+)":<15s}{"mean " + f"{lam["q"].mean():+.4f}":>15s}'
      f'{lam["w_lo"].mean()*100:>11.2f}%{lam["w_hi"].mean()*100:>11.2f}%'
      f'{(lam["w_hi"]-lam["w_lo"]).mean()*100:>+9.2f}%   ← 양수 q → low-vol over 유지')
print(f'\nfpm q<0 months: {len(neg)}/{len(fpm)} ({len(neg)/len(fpm)*100:.0f}%)')

# ── (2) 대표 month (fpm q 최저) — top 10 weighted 종목 비교 ──
target_dt = fpm.loc[fpm['q'].idxmin(), 'date']
target_q = fpm['q'].min()
w_fpm_t = loaded[fpm_slot]['weights'].loc[target_dt].dropna().sort_values(ascending=False)
w_lam_t = loaded[lam_slot]['weights'].loc[target_dt].dropna().sort_values(ascending=False)
sig_t = _lstm_pivot.loc[target_dt] if target_dt in _lstm_pivot.index else pd.Series()
next_ym = (target_dt + pd.offsets.MonthEnd(1)).to_period('M')
real_ret = _panel_full[_panel_full['ym']==next_ym].set_index('ticker')['fwd_ret_1m']

print('\n' + '='*72)
print(f'대표 month {target_dt.strftime("%Y-%m")} (fpm q={target_q:+.5f}, R4 최저) — top 10 long weight')
print('='*72)
def _print_top(label, w_series):
    print(f'\n[{label}]')
    print(f'  {"ticker":<8s}{"weight":>9s}{"σ_pred":>10s}{"sector":>26s}{"next_ret":>11s}')
    for tk, w in w_series.head(10).items():
        sig_v = sig_t.get(tk, np.nan)
        sec = _sector_map.get(tk, 'Unknown')[:26]
        r = real_ret.get(tk, np.nan)
        r_str = f'{r*100:+.2f}%' if not pd.isna(r) else '  —  '
        print(f'  {tk:<8s}{w*100:>8.2f}%{sig_v:>10.4f}{sec:>26s}{r_str:>11s}')

_print_top(f'fpm slot ({fpm_slot})', w_fpm_t)
_print_top(f'lam slot ({lam_slot})', w_lam_t)
ovl = set(w_fpm_t.head(10).index) & set(w_lam_t.head(10).index)
print(f'\n  top10 overlap: {len(ovl)} 종목 — {sorted(ovl)}')
print(f'  fpm only: {sorted(set(w_fpm_t.head(10).index) - set(w_lam_t.head(10).index))}')
print(f'  lam only: {sorted(set(w_lam_t.head(10).index) - set(w_fpm_t.head(10).index))}')

# ── (3) prior 별 AI 섹터 노출 — fpm 4 슬롯 ─────────────────
fpm_slots = {
    'mat_mcap_mcap_fpm_pap': 'mcap/mcap',
    'mat_mcap_eq_fpm_pap':   'mcap/eq (best)',
    'mat_eq_eq_fpm_pap':     'eq/eq',
    'mat_rp_rp_fpm_pap':     'rp/rp',
}
print('\n' + '='*72)
print('R4 AI 섹터 (IT+Tech+CommSvc) weight 비중 — fpm 슬롯 비교')
print('='*72)
ai_by_slot = {}
for sn, lbl in fpm_slots.items():
    if sn not in loaded: continue
    ws = loaded[sn]['weights']
    wr = ws[(ws.index>=R4_S)&(ws.index<=R4_E)]
    shares = []
    for dt in wr.index:
        wt = wr.loc[dt].dropna()
        pos = wt[wt>0]
        ai = pos[[t for t in pos.index if _sector_map.get(t,'') in AI_SECTORS]]
        shares.append((dt, ai.sum()/pos.sum() if pos.sum()>0 else 0))
    ai_by_slot[sn] = shares
    print(f'  {lbl:<18s}: R4 평균 AI 섹터 비중 = {np.mean([s for _,s in shares])*100:.1f}%')

# 시각화: 2 panel
fig, axes = plt.subplots(2, 1, figsize=(14, 7.5))

ax = axes[0]
w_days = 12
ax.bar(fpm['date'] - pd.Timedelta(days=w_days/2),
       (fpm['w_hi']-fpm['w_lo'])*100, width=w_days,
       color=['#d62728' if q<0 else '#fdae61' for q in fpm['q']],
       alpha=0.85, label=f'fpm ({fpm_slot})')
ax.bar(lam['date'] + pd.Timedelta(days=w_days/2),
       (lam['w_hi']-lam['w_lo'])*100, width=w_days,
       color='#2ca02c', alpha=0.7, label=f'lam ({lam_slot})')
ax.axhline(0, color='black', lw=0.5)
ax.set_title('R4 bucket asymmetry  (>0: high-vol over, <0: low-vol over)\n'
             'fpm 빨강 = q<0 month, 주황 = q≥0 month;  lam 항상 q>0')
ax.set_ylabel('hi30% − low30% weight (%)')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=15)

ax = axes[1]
colors_p = {'mat_mcap_mcap_fpm_pap':'#d62728', 'mat_mcap_eq_fpm_pap':'#ff7f0e',
            'mat_eq_eq_fpm_pap':'#2ca02c', 'mat_rp_rp_fpm_pap':'#1f77b4'}
for sn, lbl in fpm_slots.items():
    if sn not in ai_by_slot: continue
    dts = [d for d,_ in ai_by_slot[sn]]
    vals = [v*100 for _,v in ai_by_slot[sn]]
    ax.plot(dts, vals, lw=1.6, marker='o', markersize=4,
             color=colors_p.get(sn,'gray'), label=lbl)
ax.set_title('R4 AI 섹터 (IT+Tech+CommSvc) weight 비중 — long 포지션 중')
ax.set_ylabel('AI sector weight (%)')
ax.legend(fontsize=9, loc='best')
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.show()
'''

for c in nb['cells']:
    src = ''.join(c['source'])
    if c['cell_type']=='code' and '── [9]' in src:
        c['source'] = new_code.splitlines(keepends=True)
        c['outputs'] = []; c['execution_count']=None
        print('Cleaned [9]')
        break
json.dump(nb, open(nb_path,'w',encoding='utf-8'), indent=1, ensure_ascii=False)
