"""Add [9] R4 mechanism verification — fpm q sign + mcap prior Mag7 exposure."""
import json, uuid, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

nb_path = '99_main_analysis.ipynb'
nb = json.load(open(nb_path, encoding='utf-8'))

md = """## [9] R4 mechanism — fpm q sign + mcap prior 대형주 노출

**가설**:
1. **fpm q 음수 → portfolio 가 high-vol 종목 over-weight** (BL view 가 long high-vol 로 flip)
2. **mcap prior → top 시총 종목 (Mag7 등) over-weight** → R4 AI 랠리에서 이 종목들 상승 → portfolio 우위

**확인 방법**:
- R4 (2023-07~2025-12) 의 actual portfolio weights 에서 추출
- 1번: weighted σ_pred 평균 (portfolio 의 vol exposure) by Q
- 2번: top 20 mcap 종목 weight 합산 by prior
"""

code = r'''# ── [9] R4 mechanism verification ──────────────────────────────
from pathlib import Path

R4_S = pd.Timestamp('2023-07-31')
R4_E = pd.Timestamp('2025-12-31')

# LSTM σ_pred 월말화
_lstm_raw = pd.read_csv(DATA_DIR/'03b_lstm/data/ensemble_predictions_stockwise.csv',
                          parse_dates=['date'])[['date','ticker','y_pred_ensemble']]
_lstm_raw['ym'] = _lstm_raw['date'].dt.to_period('M')
_lstm_m = _lstm_raw.sort_values('date').groupby(['ym','ticker'], as_index=False).last()
_lstm_m['date'] = _lstm_m['ym'].dt.to_timestamp(how='end').dt.normalize() + pd.Timedelta(hours=0)
# month-end 정규화 (pkl weights index 와 매칭하기 위해)
_lstm_m['date'] = _lstm_m['date'].apply(lambda d: pd.Timestamp(d.year, d.month, 1) + pd.offsets.MonthEnd(0))
_lstm_pivot = _lstm_m.pivot(index='date', columns='ticker', values='y_pred_ensemble')

# mcap pivot — monthly_panel 에서
_panel_full = pd.read_csv(DATA_DIR/'monthly_panel.csv', parse_dates=['date'])
_mcap_pivot = _panel_full.pivot_table(index='date', columns='ticker', values='log_mcap', aggfunc='first')

# 분석 대상 슬롯 (LSTM)
SLOTS_R4 = [
    ('mat_mcap_mcap_fpm_pap', 'mcap prior, mcap p_w, fpm Q', 'red'),   # R4 best Sharpe
    ('mat_mcap_mcap_lam_pap', 'mcap prior, mcap p_w, lam Q', 'orange'),  # σ-direct, R4 weaker
    ('mat_eq_eq_fpm_pap',     'eq prior, eq p_w, fpm Q',     'blue'),    # different prior
    ('mat_rp_rp_fpm_pap',     'rp prior, rp p_w, fpm Q',     'purple'),  # different prior
]

# 각 슬롯의 R4 portfolio 특성 추출
print('='*92)
print('R4 portfolio exposure 분석 — 각 슬롯의 weighted σ_pred & top 20 mcap 비중')
print('='*92)
print(f'{"slot":<30s}{"avg L_Sh":>10s}{"avg σ exp":>12s}{"avg σ low":>11s}{"top20 mcap %":>14s}')
print('-'*78)

results = {}
for slot_name, slot_label, color in SLOTS_R4:
    if slot_name not in loaded: continue
    w = loaded[slot_name]['weights']  # rows=dates (construction), cols=tickers
    w_r4 = w[(w.index>=R4_S)&(w.index<=R4_E)].copy()
    if len(w_r4) == 0: continue
    ret = loaded[slot_name]['ret'].reindex(w_r4.index).dropna()

    sigma_exp_list = []
    top20_share_list = []
    sigma_low_share_list = []
    for dt in w_r4.index:
        wt = w_r4.loc[dt].dropna()
        if len(wt) == 0: continue
        # σ_pred for this date (use LSTM since slot is LSTM)
        # match construction date
        sig = _lstm_pivot.loc[dt] if dt in _lstm_pivot.index else None
        if sig is not None:
            common = wt.index.intersection(sig.dropna().index)
            if len(common) > 5:
                w_n = wt.loc[common] / wt.loc[common].sum()
                sig_n = sig.loc[common]
                sigma_exp = (w_n * sig_n).sum()
                sigma_exp_list.append(sigma_exp)
                # bottom 30% σ 종목 비중
                n_lo = max(1, int(len(sig_n)*0.30))
                lo_set = sig_n.nsmallest(n_lo).index
                sigma_low_share_list.append(wt.loc[lo_set].sum())
        # top 20 mcap 종목 비중
        mc = _mcap_pivot.loc[dt] if dt in _mcap_pivot.index else None
        if mc is not None:
            common_m = wt.index.intersection(mc.dropna().index)
            if len(common_m) > 5:
                top20 = mc.loc[common_m].nlargest(20).index
                top20_share_list.append(wt.loc[top20].sum())

    avg_sigma_exp = np.mean(sigma_exp_list) if sigma_exp_list else np.nan
    avg_low_share = np.mean(sigma_low_share_list) if sigma_low_share_list else np.nan
    avg_top20 = np.mean(top20_share_list) if top20_share_list else np.nan
    # Sharpe in R4
    if len(ret)>=6:
        sh = ret.mean()*12/(ret.std()*np.sqrt(12))
    else:
        sh = np.nan
    results[slot_name] = dict(sigma_exp=avg_sigma_exp, low_share=avg_low_share,
                                top20=avg_top20, sh=sh, label=slot_label, color=color)
    print(f'{slot_label:<30s}{sh:>+10.3f}{avg_sigma_exp:>12.4f}{avg_low_share*100:>10.1f}%{avg_top20*100:>13.1f}%')

# 해석 표
print('\n해석 요약:')
print('-'*78)
print('1. σ exposure (weighted σ_pred 평균):')
print('   → 높으면 portfolio 가 high-vol 종목 over-weight (= fpm q 음수 효과)')
print('2. σ low share (bottom 30% σ_pred 종목 weight 합):')
print('   → 낮으면 portfolio 가 low-vol 종목 underweight (= long-high-vol view)')
print('3. top20 mcap share:')
print('   → 높으면 portfolio 가 시총 대형주 over-weight (= mcap prior 효과, Mag7 노출)')

# 시각화: 4 슬롯의 R4 timeseries of σ exposure + top20 mcap share
fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True)

# σ exposure 시계열
ax = axes[0]
for slot_name, _, color in SLOTS_R4:
    if slot_name not in loaded: continue
    w = loaded[slot_name]['weights']
    w_r4 = w[(w.index>=R4_S)&(w.index<=R4_E)]
    sigma_ts = []
    for dt in w_r4.index:
        wt = w_r4.loc[dt].dropna()
        if dt in _lstm_pivot.index:
            sig = _lstm_pivot.loc[dt].dropna()
            common = wt.index.intersection(sig.index)
            if len(common) > 5:
                w_n = wt.loc[common] / wt.loc[common].sum()
                sigma_ts.append((dt, (w_n*sig.loc[common]).sum()))
    if sigma_ts:
        dts, vals = zip(*sigma_ts)
        ax.plot(dts, vals, lw=1.5, marker='o', markersize=4, color=color, label=results[slot_name]['label'])
ax.set_title('R4 portfolio σ exposure (weighted σ_pred 평균) — 높을수록 high-vol 종목 over-weight')
ax.set_ylabel('weighted σ_pred')
ax.legend(fontsize=9, loc='best')
ax.grid(True, alpha=0.3)

# top20 mcap share 시계열
ax = axes[1]
for slot_name, _, color in SLOTS_R4:
    if slot_name not in loaded: continue
    w = loaded[slot_name]['weights']
    w_r4 = w[(w.index>=R4_S)&(w.index<=R4_E)]
    top20_ts = []
    for dt in w_r4.index:
        wt = w_r4.loc[dt].dropna()
        if dt in _mcap_pivot.index:
            mc = _mcap_pivot.loc[dt].dropna()
            common = wt.index.intersection(mc.index)
            if len(common) > 5:
                top20 = mc.loc[common].nlargest(20).index
                top20_ts.append((dt, wt.loc[top20].sum()))
    if top20_ts:
        dts, vals = zip(*top20_ts)
        ax.plot(dts, vals, lw=1.5, marker='o', markersize=4, color=color, label=results[slot_name]['label'])
ax.set_title('R4 top 20 mcap 종목 weight 합 — 높을수록 Mag7/대형주 노출 큼')
ax.set_ylabel('weight in top 20 mcap')
ax.legend(fontsize=9, loc='best')
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.show()
'''

new_cells = [
    {'cell_type':'markdown','id':uuid.uuid4().hex[:8],'metadata':{},
     'source': md.splitlines(keepends=True)},
    {'cell_type':'code','id':uuid.uuid4().hex[:8],'metadata':{},
     'execution_count':None,'outputs':[],
     'source': code.splitlines(keepends=True)},
]
nb['cells'].extend(new_cells)
json.dump(nb, open(nb_path,'w',encoding='utf-8'), indent=1, ensure_ascii=False)
print(f'Appended [9]. Total cells: {len(nb["cells"])}')
