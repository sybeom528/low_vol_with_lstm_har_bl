"""Update [6] to show all 9 combinations (3 prior x 3 p_w) per subplot."""
import json, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

nb_path = '99_main_analysis.ipynb'
nb = json.load(open(nb_path, encoding='utf-8'))

new_code = r'''# ── [6] Q × regime × (prior × p_w) timeline — 9 라인 per subplot ──
REGIME_PERIODS = {
    'R1 회복 (2010-01~2012-06)':    (pd.Timestamp('2010-01-31'), pd.Timestamp('2012-06-30')),
    'R2 확장 (2012-07~2019-12)':    (pd.Timestamp('2012-07-31'), pd.Timestamp('2019-12-31')),
    'R3 위기 (2020-01~2023-06)':    (pd.Timestamp('2020-01-31'), pd.Timestamp('2023-06-30')),
    'R4 정상화 (2023-07~2025-12)':  (pd.Timestamp('2023-07-31'), pd.Timestamp('2025-12-31')),
}
REGIME_EVENTS = {
    'R1 회복 (2010-01~2012-06)': [('2010-05','Flash Crash'),('2011-03','Tohoku eq'),('2011-08','S&P downgrade')],
    'R2 확장 (2012-07~2019-12)': [('2015-08','CNY devalue'),('2018-02','Volmageddon'),('2018-12','Q4 selloff')],
    'R3 위기 (2020-01~2023-06)': [('2020-03','COVID crash'),('2022-06','Fed hike'),('2022-09','BoE mini-budget')],
    'R4 정상화 (2023-07~2025-12)': [('2023-10','Yields spike'),('2024-08','JPY carry')],
}
PRIORS=['mcap','eq','rp']; PW=['mcap','eq','rp']; Q=['lam','raw','inv','vsp','fpm']
Q_LABEL = {'lam':'lam  (P + λ scale)','raw':'raw  (P + λ raw)',
           'inv':'inv  (P + 1/λ)','vsp':'vsp  (P + σ spread, Q 진입)',
           'fpm':'fpm  (P only, q = FF3)'}
pw_colors = {'mcap':'#d62728', 'eq':'#2ca02c', 'rp':'#1f77b4'}
prior_styles = {'mcap':'-', 'eq':'--', 'rp':':'}   # solid / dashed / dotted

def build_one(start, end, q, pw, pr):
    L = f'mat_{pr}_{pw}_{q}_pap'; A = L + '_ann'
    if L not in loaded or A not in loaded: return pd.Series(dtype=float)
    rL = loaded[L]['ret'].dropna(); rA = loaded[A]['ret'].dropna()
    common = rL.index.intersection(rA.index)
    d = rL.loc[common] - rA.loc[common]
    d.index = d.index + pd.offsets.MonthEnd(1)
    return d[(d.index>=start)&(d.index<=end)]

# 평균표 (전체 9 슬롯 평균; reference)
print('='*86)
print('Q × regime 월평균 LSTM−ANN return  (각 셀: 3 prior × 3 p_w = 9 슬롯 평균)')
print('='*86)
print(f'{"Q":<8s}' + ''.join(f'{r.split()[0]:>16s}' for r in REGIME_PERIODS) + f'{"all (10y)":>14s}')
print('-'*86)
for q in Q:
    cells = []
    for label, (s, e) in REGIME_PERIODS.items():
        vals = []
        for pw in PW:
            for pr in PRIORS:
                vals.append(build_one(s,e,q,pw,pr).mean())
        cells.append(np.nanmean(vals))
    all_vals = []
    for pw in PW:
        for pr in PRIORS:
            L = f'mat_{pr}_{pw}_{q}_pap'; A = L+'_ann'
            if L not in loaded or A not in loaded: continue
            rL = loaded[L]['ret'].dropna(); rA = loaded[A]['ret'].dropna()
            common = rL.index.intersection(rA.index)
            all_vals.append((rL.loc[common]-rA.loc[common]).mean())
    print(f'{q:<8s}' + ''.join(f'{v:>+16.4f}' for v in cells) + f'{np.mean(all_vals):>+14.4f}')

# regime 별 figure — 5 Q rows × 1 col, 각 subplot 에 9 라인 (3 prior × 3 p_w)
for rlabel, (s, e) in REGIME_PERIODS.items():
    series = {q: {pw: {pr: build_one(s,e,q,pw,pr) for pr in PRIORS} for pw in PW} for q in Q}
    all_vals = [v for q in Q for pw in PW for pr in PRIORS for v in series[q][pw][pr].values]
    ymin, ymax = min(all_vals)*1.1, max(all_vals)*1.1

    fig, axes = plt.subplots(len(Q), 1, figsize=(14, 13), sharex=True)
    for i, q in enumerate(Q):
        ax = axes[i]
        for pw in PW:
            for pr in PRIORS:
                ser = series[q][pw][pr]
                if len(ser)==0: continue
                label = (f'p_w={pw}, prior={pr}' if i==0 else None)
                ax.plot(ser.index, ser.values,
                        color=pw_colors[pw], linestyle=prior_styles[pr],
                        lw=1.1, marker='o', markersize=2.5, alpha=0.85,
                        label=label)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel(Q_LABEL[q], fontsize=10)
        ax.grid(True, alpha=0.3)
        for ym, ev in REGIME_EVENTS.get(rlabel, []):
            matches = [dt for dt in series[q]['mcap']['mcap'].index if dt.strftime('%Y-%m') == ym]
            if matches:
                dt = matches[0]
                ax.axvline(dt, color='gray', ls='-.', lw=0.6, alpha=0.5)
                if i == 0:
                    ax.annotate(f'{ym}\n{ev}', xy=(dt, ymax*0.85),
                                fontsize=8, ha='center', color='dimgray')

    axes[0].legend(loc='upper right', fontsize=7, ncol=3,
                    bbox_to_anchor=(1.0, 1.55))
    fig.suptitle(f'{rlabel}  —  Monthly LSTM − ANN return\n'
                  f'color = p_w (red=mcap, green=eq, blue=rp),  '
                  f'linestyle = prior (solid=mcap, dashed=eq, dotted=rp)',
                  fontsize=12, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()
'''

# Replace [6] code cell content
target_id = '6452175a'
updated = False
for c in nb['cells']:
    if c.get('id') == target_id:
        c['source'] = new_code.splitlines(keepends=True)
        c['outputs'] = []
        c['execution_count'] = None
        updated = True
        break

if updated:
    json.dump(nb, open(nb_path,'w',encoding='utf-8'), indent=1, ensure_ascii=False)
    print('Updated [6] to show all 9 lines per subplot.')
else:
    print('ERROR: [6] code cell not found')
