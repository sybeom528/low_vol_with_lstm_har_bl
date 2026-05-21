"""Insert mcap_eq_fpm_pap table between (1) and (1b), add q values to chart."""
import json, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

nb_path = '99_main_analysis.ipynb'
nb = json.load(open(nb_path, encoding='utf-8'))

# locate [9] code cell
for c in nb['cells']:
    src = ''.join(c['source'])
    if c['cell_type']=='code' and '── [9]' in src:
        break

# Insert (1.5) mcap_eq_fpm_pap section right before (1b) lam section
insert_after = """qdf = pd.DataFrame(q_sign_rows)
neg = qdf[qdf['q']<0]; pos = qdf[qdf['q']>=0]
print(f'\\n  q<0 month (n={len(neg)}): avg hi−low = {(neg["w_hi"]-neg["w_lo"]).mean()*100:+.2f}%  ← 음수 q → high-vol over')
print(f'  q≥0 month (n={len(pos)}): avg hi−low = {(pos["w_hi"]-pos["w_lo"]).mean()*100:+.2f}%  ← 양수 q → low-vol over')

# ── (1b) 대조: σ-direct Q (lam) — same family ───────────────"""

inserted_section = """qdf = pd.DataFrame(q_sign_rows)
neg = qdf[qdf['q']<0]; pos = qdf[qdf['q']>=0]
print(f'\\n  q<0 month (n={len(neg)}): avg hi−low = {(neg["w_hi"]-neg["w_lo"]).mean()*100:+.2f}%  ← 음수 q → high-vol over')
print(f'  q≥0 month (n={len(pos)}): avg hi−low = {(pos["w_hi"]-pos["w_lo"]).mean()*100:+.2f}%  ← 양수 q → low-vol over')

# ── (1.5) LSTM-best slot — mat_mcap_eq_fpm_pap ─────────────
slot_me = 'mat_mcap_eq_fpm_pap'
w_me_all = loaded[slot_me]['weights']
meta_me = loaded[slot_me]['meta']
q_me_series = meta_me['Q']
w_me_r4 = w_me_all[(w_me_all.index>=R4_S)&(w_me_all.index<=R4_E)]

print('\\n' + '='*72)
print(f'(1.5) LSTM-best slot — slot = {slot_me}  (prior=mcap, p_w=eq)')
print(f'      anchor 와 같은 Q (fpm) 이지만 p_w 가 eq 라 q magnitude 작음')
print('='*72)
print(f'{"month":<9s}{"q_fpm":>10s}{"q sign":>8s}{"low30%w":>10s}{"hi30%w":>10s}{"hi−low":>10s}')
print('-'*60)

me_rows = []
for dt in w_me_r4.index:
    wt = w_me_r4.loc[dt].dropna()
    sig = _lstm_pivot.loc[dt] if dt in _lstm_pivot.index else None
    q_val = q_me_series.get(dt, np.nan)
    if sig is None or pd.isna(q_val): continue
    common = wt.index.intersection(sig.dropna().index)
    if len(common) < 10: continue
    sig_c = sig.loc[common]
    n_b = max(1, int(len(sig_c)*0.30))
    w_lo = wt.loc[sig_c.nsmallest(n_b).index].sum()
    w_hi = wt.loc[sig_c.nlargest(n_b).index].sum()
    qsg = '음 (−)' if q_val < 0 else '양 (+)'
    print(f'{dt.strftime("%Y-%m"):<9s}{q_val:>+10.5f}{qsg:>8s}{w_lo*100:>9.2f}%{w_hi*100:>9.2f}%{(w_hi-w_lo)*100:>+9.2f}%')
    me_rows.append(dict(date=dt, q=q_val, w_lo=w_lo, w_hi=w_hi))

medf = pd.DataFrame(me_rows)
me_neg = medf[medf['q']<0]; me_pos = medf[medf['q']>=0]
print(f'\\n  q<0 month (n={len(me_neg)}): avg hi−low = {(me_neg["w_hi"]-me_neg["w_lo"]).mean()*100:+.2f}%')
print(f'  q≥0 month (n={len(me_pos)}): avg hi−low = {(me_pos["w_hi"]-me_pos["w_lo"]).mean()*100:+.2f}%')
print(f'  비교 (anchor mcap_mcap vs best mcap_eq):')
print(f'    q magnitude: anchor mean {qdf["q"].mean():+.5f},  best mean {medf["q"].mean():+.5f}  ({abs(qdf["q"].mean()/medf["q"].mean()):.1f}x 차이)')
print(f'    → best slot 의 view 가 약해서 prior 효과 흡수, q sign 이 bucket 방향 못 뒤집음')

# ── (1b) 대조: σ-direct Q (lam) — same family ───────────────"""

new_src = src.replace(insert_after, inserted_section)

# 시각화 panel 1 에 q 값 overlay 추가
old_panel1 = """ax = axes[0]
w_days = 12
qdf_s = qdf.sort_values('date'); lamdf_s = lamdf.sort_values('date')
ax.bar(qdf_s['date'] - pd.Timedelta(days=w_days/2),
       (qdf_s['w_hi']-qdf_s['w_lo'])*100, width=w_days,
       color=['#d62728' if q<0 else '#fdae61' for q in qdf_s['q']],
       alpha=0.85, label=f'fpm ({slot})')
ax.bar(lamdf_s['date'] + pd.Timedelta(days=w_days/2),
       (lamdf_s['w_hi']-lamdf_s['w_lo'])*100, width=w_days,
       color='#2ca02c', alpha=0.7, label=f'lam ({slot_lam})')
ax.axhline(0, color='black', lw=0.5)
ax.set_title('R4 bucket asymmetry — fpm (sign-flexible) vs lam (σ-direct, q+)\\n'
             '(>0: high-vol over, <0: low-vol over; fpm 빨강 = q<0 month)')
ax.set_ylabel('hi30% − low30% weight (%)')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=15)"""

new_panel1 = """ax = axes[0]
w_days = 9
qdf_s = qdf.sort_values('date'); medf_s = medf.sort_values('date'); lamdf_s = lamdf.sort_values('date')
ax.bar(qdf_s['date'] - pd.Timedelta(days=w_days),
       (qdf_s['w_hi']-qdf_s['w_lo'])*100, width=w_days,
       color=['#d62728' if q<0 else '#fdae61' for q in qdf_s['q']],
       alpha=0.85, label=f'fpm anchor ({slot})')
ax.bar(medf_s['date'],
       (medf_s['w_hi']-medf_s['w_lo'])*100, width=w_days,
       color=['#8c2d04' if q<0 else '#fed976' for q in medf_s['q']],
       alpha=0.85, label=f'fpm best ({slot_me})')
ax.bar(lamdf_s['date'] + pd.Timedelta(days=w_days),
       (lamdf_s['w_hi']-lamdf_s['w_lo'])*100, width=w_days,
       color='#2ca02c', alpha=0.7, label=f'lam ({slot_lam})')
ax.axhline(0, color='black', lw=0.5)
ax.set_ylabel('hi30% − low30% weight (%)')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=15)
ax.set_title('R4 bucket asymmetry — fpm (anchor, best) vs lam\\n'
             '(>0: high-vol over, <0: low-vol over; bar 빨강 톤 = q<0 month)')

# q 값 secondary axis overlay
ax2 = ax.twinx()
ax2.plot(qdf_s['date'], qdf_s['q'], color='#d62728', lw=1.2, marker='o', markersize=3,
          alpha=0.6, label='q (fpm anchor)')
ax2.plot(medf_s['date'], medf_s['q'], color='#8c2d04', lw=1.2, marker='s', markersize=3,
          alpha=0.6, linestyle='--', label='q (fpm best)')
ax2.plot(lamdf_s['date'], lamdf_s['q'], color='#2ca02c', lw=1.2, marker='^', markersize=3,
          alpha=0.6, linestyle=':', label='q (lam)')
ax2.axhline(0, color='gray', lw=0.5, ls='--')
ax2.set_ylabel('q value (line)', color='dimgray')
ax2.tick_params(axis='y', labelcolor='dimgray')
ax2.legend(loc='upper right', fontsize=8)"""

new_src = new_src.replace(old_panel1, new_panel1)
c['source'] = new_src.splitlines(keepends=True)
c['outputs'] = []; c['execution_count']=None
json.dump(nb, open(nb_path,'w',encoding='utf-8'), indent=1, ensure_ascii=False)
print('Updated [9] — added mcap_eq_fpm section + q overlay on chart')
