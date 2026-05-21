"""Update [9] to use mat_mcap_eq_{fpm,lam}_pap pair + top-weighted stocks."""
import json, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

nb_path = '99_main_analysis.ipynb'
nb = json.load(open(nb_path, encoding='utf-8'))

new_code = r'''# ── [9] R4 mechanism — fpm q sign + AI 섹터 노출 ──────────────
R4_S = pd.Timestamp('2023-07-31')
R4_E = pd.Timestamp('2025-12-31')

# LSTM σ_pred 월말화
_lstm_raw = pd.read_csv(DATA_DIR/'03b_lstm/data/ensemble_predictions_stockwise.csv',
                          parse_dates=['date'])[['date','ticker','y_pred_ensemble']]
_lstm_raw['ym'] = _lstm_raw['date'].dt.to_period('M')
_lstm_m = _lstm_raw.sort_values('date').groupby(['ym','ticker'], as_index=False).last()
_lstm_m['date'] = _lstm_m['ym'].dt.to_timestamp(how='end').apply(
    lambda d: pd.Timestamp(d.year, d.month, 1) + pd.offsets.MonthEnd(0))
_lstm_pivot = _lstm_m.pivot(index='date', columns='ticker', values='y_pred_ensemble')

_panel_full = pd.read_csv(DATA_DIR/'monthly_panel.csv', parse_dates=['date'])
AI_SECTORS = {'Information Technology', 'Technology', 'Communication Services'}
_sector_map = (_panel_full.dropna(subset=['gics_sector'])
                 .groupby('ticker')['gics_sector']
                 .agg(lambda s: s.mode().iloc[0] if len(s.mode())>0 else 'Unknown')
                 .to_dict())
# realized return per ticker per month (for case study month 검증)
_panel_full['ym'] = _panel_full['date'].dt.to_period('M')

# ── (1) fpm Q 부호별 bucket weight — slot = mat_mcap_eq_fpm_pap (LSTM-best) ──
slot = 'mat_mcap_eq_fpm_pap'
w_all = loaded[slot]['weights']
meta = loaded[slot]['meta']
q_series = meta['Q']
w_r4 = w_all[(w_all.index>=R4_S)&(w_all.index<=R4_E)]

print('='*82)
print(f'(1) fpm q 부호별 bucket weight — slot = {slot}  (LSTM-best)')
print(f'    R4 (2023-07~2025-12) 의 월별 q 값과 portfolio σ bucket weight')
print('='*82)
print(f'{"month":<9s}{"q_fpm":>10s}{"q sign":>8s}{"low30%w":>10s}{"hi30%w":>10s}{"hi−low":>10s}')
print('-'*60)

q_sign_rows = []
for dt in w_r4.index:
    wt = w_r4.loc[dt].dropna()
    sig = _lstm_pivot.loc[dt] if dt in _lstm_pivot.index else None
    q_val = q_series.get(dt, np.nan)
    if sig is None or pd.isna(q_val): continue
    common = wt.index.intersection(sig.dropna().index)
    if len(common) < 10: continue
    sig_c = sig.loc[common]
    n_b = max(1, int(len(sig_c)*0.30))
    lo_set = sig_c.nsmallest(n_b).index
    hi_set = sig_c.nlargest(n_b).index
    w_lo = wt.loc[lo_set].sum()
    w_hi = wt.loc[hi_set].sum()
    qsg = '음 (−)' if q_val < 0 else '양 (+)'
    print(f'{dt.strftime("%Y-%m"):<9s}{q_val:>+10.5f}{qsg:>8s}{w_lo*100:>9.2f}%{w_hi*100:>9.2f}%{(w_hi-w_lo)*100:>+9.2f}%')
    q_sign_rows.append(dict(date=dt, q=q_val, w_lo=w_lo, w_hi=w_hi))

qdf = pd.DataFrame(q_sign_rows)
neg = qdf[qdf['q']<0]; pos = qdf[qdf['q']>=0]
print(f'\n  q<0 month (n={len(neg)}): avg low30% w = {neg["w_lo"].mean()*100:.2f}%, hi30% w = {neg["w_hi"].mean()*100:.2f}%, hi−low = {(neg["w_hi"]-neg["w_lo"]).mean()*100:+.2f}%')
print(f'  q≥0 month (n={len(pos)}): avg low30% w = {pos["w_lo"].mean()*100:.2f}%, hi30% w = {pos["w_hi"].mean()*100:.2f}%, hi−low = {(pos["w_hi"]-pos["w_lo"]).mean()*100:+.2f}%')
print(f'  → q<0 month 에서 hi−low > 0 이면: 음수 q 일 때 high-vol 종목 over-weight (가설 1 입증)')

# ── (1b) 대조: σ-direct Q (lam) — slot = mat_mcap_eq_lam_pap (same family) ──
slot_lam = 'mat_mcap_eq_lam_pap'
w_lam_all = loaded[slot_lam]['weights']
meta_lam = loaded[slot_lam]['meta']
q_lam_series = meta_lam['Q']
w_lam_r4 = w_lam_all[(w_lam_all.index>=R4_S)&(w_lam_all.index<=R4_E)]

print('\n' + '='*82)
print(f'(1b) 대조 — σ-direct Q (lam) slot = {slot_lam}  (same family, Q only swap)')
print(f'      lam Q 는 λ scaling 으로 항상 q>0 (== long low-vol view 유지)')
print('='*82)
print(f'{"month":<9s}{"q_lam":>10s}{"q sign":>8s}{"low30%w":>10s}{"hi30%w":>10s}{"hi−low":>10s}')
print('-'*60)

lam_rows = []
for dt in w_lam_r4.index:
    wt = w_lam_r4.loc[dt].dropna()
    sig = _lstm_pivot.loc[dt] if dt in _lstm_pivot.index else None
    q_val = q_lam_series.get(dt, np.nan)
    if sig is None or pd.isna(q_val): continue
    common = wt.index.intersection(sig.dropna().index)
    if len(common) < 10: continue
    sig_c = sig.loc[common]
    n_b = max(1, int(len(sig_c)*0.30))
    lo_set = sig_c.nsmallest(n_b).index
    hi_set = sig_c.nlargest(n_b).index
    w_lo = wt.loc[lo_set].sum()
    w_hi = wt.loc[hi_set].sum()
    qsg = '양 (+)' if q_val >= 0 else '음 (−)'
    print(f'{dt.strftime("%Y-%m"):<9s}{q_val:>+10.5f}{qsg:>8s}{w_lo*100:>9.2f}%{w_hi*100:>9.2f}%{(w_hi-w_lo)*100:>+9.2f}%')
    lam_rows.append(dict(date=dt, q=q_val, w_lo=w_lo, w_hi=w_hi))

lamdf = pd.DataFrame(lam_rows)
print(f'\n  lam q 항상 양수 확인 (n_pos={(lamdf["q"]>=0).sum()}, n_neg={(lamdf["q"]<0).sum()})')
print(f'  R4 평균 low30% weight = {lamdf["w_lo"].mean()*100:.2f}%, hi30% weight = {lamdf["w_hi"].mean()*100:.2f}%')
print(f'  R4 평균 hi−low = {(lamdf["w_hi"]-lamdf["w_lo"]).mean()*100:+.2f}%   (음수면: low-vol over 유지)')
print('\n  대조 요약:')
print(f'    fpm slot: q<0 month 에서 hi−low = {(neg["w_hi"]-neg["w_lo"]).mean()*100:+.2f}%  (음수 q → high-vol over)')
print(f'    lam slot: q>0 항상,        hi−low = {(lamdf["w_hi"]-lamdf["w_lo"]).mean()*100:+.2f}%  (양수 q → low-vol over 유지)')
print('  → Q sign behavior 가 portfolio view 방향을 결정한다는 메커니즘 데이터로 입증')

# ── (1c) 대표 R4 month 의 top 10 weighted 종목 — fpm vs lam 비교 ──
# fpm q 가 가장 음수인 month 선택
neg_sorted = qdf[qdf['q']<0].sort_values('q')
if len(neg_sorted) > 0:
    target_dt = neg_sorted.iloc[0]['date']
    print('\n' + '='*92)
    print(f'(1c) 대표 month = {target_dt.strftime("%Y-%m")}  (fpm q 가장 음수, q={neg_sorted.iloc[0]["q"]:+.5f})')
    print('     top 10 weighted 종목 비교 (fpm slot vs lam slot)')
    print('='*92)
    w_fpm_t = w_r4.loc[target_dt].dropna().sort_values(ascending=False)
    w_lam_t = w_lam_r4.loc[target_dt].dropna().sort_values(ascending=False)
    sig_t = _lstm_pivot.loc[target_dt] if target_dt in _lstm_pivot.index else pd.Series()
    # realized ret 그 다음달
    next_ym = (target_dt + pd.offsets.MonthEnd(1)).to_period('M')
    real_ret = _panel_full[_panel_full['ym']==next_ym].set_index('ticker')['fwd_ret_1m']

    print(f'\n  ── fpm slot ({slot}) top 10 long weights ──')
    print(f'  {"ticker":<8s}{"weight":>9s}{"σ_pred":>10s}{"sector":>22s}{"next_ret":>11s}')
    for tk, w in w_fpm_t.head(10).items():
        sig_v = sig_t.get(tk, np.nan)
        sec = _sector_map.get(tk, 'Unknown')
        r = real_ret.get(tk, np.nan)
        print(f'  {tk:<8s}{w*100:>8.2f}%{sig_v:>10.4f}{sec[:22]:>22s}{r*100 if not pd.isna(r) else 0:>+10.2f}%')

    print(f'\n  ── lam slot ({slot_lam}) top 10 long weights ──')
    print(f'  {"ticker":<8s}{"weight":>9s}{"σ_pred":>10s}{"sector":>22s}{"next_ret":>11s}')
    for tk, w in w_lam_t.head(10).items():
        sig_v = sig_t.get(tk, np.nan)
        sec = _sector_map.get(tk, 'Unknown')
        r = real_ret.get(tk, np.nan)
        print(f'  {tk:<8s}{w*100:>8.2f}%{sig_v:>10.4f}{sec[:22]:>22s}{r*100 if not pd.isna(r) else 0:>+10.2f}%')

    # overlap
    fpm_top10 = set(w_fpm_t.head(10).index)
    lam_top10 = set(w_lam_t.head(10).index)
    overlap = fpm_top10 & lam_top10
    print(f'\n  top10 overlap: {len(overlap)} 종목 ({sorted(overlap)[:5]}...)')
    print(f'  fpm only: {sorted(fpm_top10 - lam_top10)}')
    print(f'  lam only: {sorted(lam_top10 - fpm_top10)}')

# ── (2) prior 별 AI 섹터 노출 — fpm 슬롯 4개 ────────────────
print('\n' + '='*82)
print('(2) prior 별 AI 섹터 (IT+Tech+CommSvc) weight 합 — R4 fpm 슬롯들')
print('='*82)
fpm_slots = {
    'mat_mcap_mcap_fpm_pap': 'prior=mcap, p_w=mcap',
    'mat_mcap_eq_fpm_pap':   'prior=mcap, p_w=eq  (best All)',
    'mat_eq_eq_fpm_pap':     'prior=eq,   p_w=eq',
    'mat_rp_rp_fpm_pap':     'prior=rp,   p_w=rp',
}
ai_by_slot = {}
for slot_name, label in fpm_slots.items():
    if slot_name not in loaded: continue
    w_all_s = loaded[slot_name]['weights']
    w_r4_s = w_all_s[(w_all_s.index>=R4_S)&(w_all_s.index<=R4_E)]
    ai_shares = []
    for dt in w_r4_s.index:
        wt = w_r4_s.loc[dt].dropna()
        wt_pos = wt[wt>0]
        ai_tix = [t for t in wt_pos.index if _sector_map.get(t,'') in AI_SECTORS]
        ai_share = wt_pos.loc[ai_tix].sum() / wt_pos.sum() if wt_pos.sum()>0 else 0
        ai_shares.append((dt, ai_share))
    ai_by_slot[slot_name] = ai_shares
    avg = np.mean([s for _,s in ai_shares])
    print(f'{label:<32s}({slot_name:<25s}): R4 평균 AI 섹터 비중 = {avg*100:.1f}%')

# 시각화: 두 panel
fig, axes = plt.subplots(2, 1, figsize=(14, 7.5))

ax = axes[0]
qdf_sorted = qdf.sort_values('date')
lamdf_sorted = lamdf.sort_values('date')
width_days = 12
ax.bar(qdf_sorted['date'] - pd.Timedelta(days=width_days/2),
       (qdf_sorted['w_hi']-qdf_sorted['w_lo'])*100, width=width_days,
       color=['#d62728' if q<0 else '#fdae61' for q in qdf_sorted['q']],
       alpha=0.85, label=f'fpm ({slot}, q sign flexible)')
ax.bar(lamdf_sorted['date'] + pd.Timedelta(days=width_days/2),
       (lamdf_sorted['w_hi']-lamdf_sorted['w_lo'])*100, width=width_days,
       color='#2ca02c', alpha=0.7, label=f'lam ({slot_lam}, q always +)')
ax.axhline(0, color='black', lw=0.5)
ax.set_title('R4 portfolio bucket asymmetry — fpm vs lam (mcap-eq family)\n'
             '(>0: high-vol over-weight, <0: low-vol over-weight; fpm 빨강=q<0 month)')
ax.set_ylabel('hi30% − low30% weight (%)')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=15)

ax = axes[1]
colors_p = {'mat_mcap_mcap_fpm_pap':'#d62728', 'mat_mcap_eq_fpm_pap':'#ff7f0e',
            'mat_eq_eq_fpm_pap':'#2ca02c', 'mat_rp_rp_fpm_pap':'#1f77b4'}
for slot_name, label in fpm_slots.items():
    if slot_name not in ai_by_slot: continue
    dts = [d for d,_ in ai_by_slot[slot_name]]
    vals = [v*100 for _,v in ai_by_slot[slot_name]]
    ax.plot(dts, vals, lw=1.6, marker='o', markersize=4, color=colors_p.get(slot_name,'gray'), label=label)
ax.set_title('R4 portfolio AI 섹터 (IT+Tech+CommSvc) weight 비중 — long 포지션 중')
ax.set_ylabel('AI sector weight (%)')
ax.legend(fontsize=9, loc='best')
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.show()
'''

# Replace [9] code
for c in nb['cells']:
    src = ''.join(c['source'])
    if c['cell_type']=='code' and '── [9]' in src:
        c['source'] = new_code.splitlines(keepends=True)
        c['outputs'] = []; c['execution_count'] = None
        print('Updated [9] code')
        break

json.dump(nb, open(nb_path,'w',encoding='utf-8'), indent=1, ensure_ascii=False)
