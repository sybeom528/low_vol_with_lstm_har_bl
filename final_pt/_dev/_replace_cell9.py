"""Replace [9] with direct verification: fpm q sign vs bucket weight + AI sector exposure."""
import json, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

nb_path = '99_main_analysis.ipynb'
nb = json.load(open(nb_path, encoding='utf-8'))

new_md = """## [9] R4 mechanism — fpm q 부호별 bucket weight + AI 섹터 노출

**가설**:
1. **fpm q 음수 ↔ portfolio 가 high-vol 종목 over-weight** (BL view flip → long high-vol short low-vol)
2. **mcap prior → AI/Tech 섹터 over-weight** (대형주가 IT/Communication 섹터에 집중)

**확인 방법**:
1. fpm 슬롯 (`mat_mcap_mcap_fpm_pap`) 의 월별 q 값 (meta) + 그달 portfolio 의 low-vol bucket weight 합, high-vol bucket weight 합 → q<0 month 에서 high > low 인지 직접 확인
2. R4 의 각 prior (mcap/eq/rp) 별 fpm 슬롯 portfolio 의 **IT + Tech + Communication Services 섹터 weight 합** → prior=mcap 가 가장 높은지 확인
"""

new_code = r'''# ── [9] R4 mechanism — fpm q sign + AI 섹터 노출 ──────────────
R4_S = pd.Timestamp('2023-07-31')
R4_E = pd.Timestamp('2025-12-31')

# LSTM σ_pred 월말화 — bucket 판정용
_lstm_raw = pd.read_csv(DATA_DIR/'03b_lstm/data/ensemble_predictions_stockwise.csv',
                          parse_dates=['date'])[['date','ticker','y_pred_ensemble']]
_lstm_raw['ym'] = _lstm_raw['date'].dt.to_period('M')
_lstm_m = _lstm_raw.sort_values('date').groupby(['ym','ticker'], as_index=False).last()
_lstm_m['date'] = _lstm_m['ym'].dt.to_timestamp(how='end').apply(
    lambda d: pd.Timestamp(d.year, d.month, 1) + pd.offsets.MonthEnd(0))
_lstm_pivot = _lstm_m.pivot(index='date', columns='ticker', values='y_pred_ensemble')

# panel 의 sector 정보
_panel_full = pd.read_csv(DATA_DIR/'monthly_panel.csv', parse_dates=['date'])
AI_SECTORS = {'Information Technology', 'Technology', 'Communication Services'}
# 종목별 most-common sector (시간 무관 단일 매핑)
_sector_map = (_panel_full.dropna(subset=['gics_sector'])
                 .groupby('ticker')['gics_sector']
                 .agg(lambda s: s.mode().iloc[0] if len(s.mode())>0 else 'Unknown')
                 .to_dict())

# ── (1) fpm q 부호별 bucket weight (mcap-mcap-fpm-pap 슬롯) ───
slot = 'mat_mcap_mcap_fpm_pap'
w_all = loaded[slot]['weights']
meta = loaded[slot]['meta']
q_series = meta['Q']

w_r4 = w_all[(w_all.index>=R4_S)&(w_all.index<=R4_E)]
print('='*82)
print(f'(1) fpm q 부호별 bucket weight — slot = {slot}')
print(f'    R4 (2023-07~2025-12) 의 월별 q 값과 portfolio 의 σ bucket weight')
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

# ── (2) prior 별 AI 섹터 노출 (fpm 슬롯들) ─────────────────────
print('\n' + '='*82)
print('(2) prior 별 AI 섹터 (IT+Tech+CommSvc) weight 합 — R4 fpm 슬롯들')
print('='*82)
fpm_slots = {
    'mat_mcap_mcap_fpm_pap': 'prior=mcap',
    'mat_eq_eq_fpm_pap':     'prior=eq',
    'mat_rp_rp_fpm_pap':     'prior=rp',
}
ai_by_slot = {}
for slot_name, label in fpm_slots.items():
    if slot_name not in loaded: continue
    w_all = loaded[slot_name]['weights']
    w_r4 = w_all[(w_all.index>=R4_S)&(w_all.index<=R4_E)]
    ai_shares = []
    for dt in w_r4.index:
        wt = w_r4.loc[dt].dropna()
        wt_pos = wt[wt>0]   # long 포지션만
        ai_tix = [t for t in wt_pos.index if _sector_map.get(t,'') in AI_SECTORS]
        ai_share = wt_pos.loc[ai_tix].sum() / wt_pos.sum() if wt_pos.sum()>0 else 0
        ai_shares.append((dt, ai_share))
    ai_by_slot[slot_name] = ai_shares
    avg = np.mean([s for _,s in ai_shares])
    print(f'{label:<12s}({slot_name:<25s}): R4 평균 AI 섹터 비중 (long 포지션 중) = {avg*100:.1f}%')

# ── 시각화: 두 panel ─────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 7.5))

# panel 1: fpm q sign vs hi−low bucket weight
ax = axes[0]
qdf_sorted = qdf.sort_values('date')
ax.bar(qdf_sorted['date'], (qdf_sorted['w_hi']-qdf_sorted['w_lo'])*100, width=20,
       color=['#d62728' if q<0 else '#2ca02c' for q in qdf_sorted['q']], alpha=0.7)
ax.axhline(0, color='black', lw=0.5)
ax.set_title(f'fpm q sign vs portfolio bucket weight asymmetry  ({slot})\n'
             f'red bar = q<0 month, green bar = q≥0 month, height = hi30% − low30% weight')
ax.set_ylabel('hi30% − low30% weight (%)')
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=15)

# panel 2: AI 섹터 비중 시계열 by prior
ax = axes[1]
colors = {'mat_mcap_mcap_fpm_pap':'#d62728', 'mat_eq_eq_fpm_pap':'#2ca02c', 'mat_rp_rp_fpm_pap':'#1f77b4'}
for slot_name, label in fpm_slots.items():
    if slot_name not in ai_by_slot: continue
    dts = [d for d,_ in ai_by_slot[slot_name]]
    vals = [v*100 for _,v in ai_by_slot[slot_name]]
    ax.plot(dts, vals, lw=1.6, marker='o', markersize=4, color=colors.get(slot_name,'gray'), label=label)
ax.set_title('R4 portfolio AI 섹터 (IT+Tech+CommSvc) weight 비중 — long 포지션 중')
ax.set_ylabel('AI sector weight (%)')
ax.legend(fontsize=9, loc='best')
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.show()
'''

# find and replace [9]
md_idx = None; code_idx = None
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    if c['cell_type']=='markdown' and '## [9]' in src:
        md_idx = i
    elif c['cell_type']=='code' and '── [9]' in src:
        code_idx = i

if md_idx is None or code_idx is None:
    print(f'ERROR: cannot find [9] cells (md={md_idx}, code={code_idx})')
else:
    nb['cells'][md_idx]['source'] = new_md.splitlines(keepends=True)
    nb['cells'][code_idx]['source'] = new_code.splitlines(keepends=True)
    nb['cells'][code_idx]['outputs'] = []
    nb['cells'][code_idx]['execution_count'] = None
    json.dump(nb, open(nb_path,'w',encoding='utf-8'), indent=1, ensure_ascii=False)
    print(f'Replaced [9] at md={md_idx}, code={code_idx}')
