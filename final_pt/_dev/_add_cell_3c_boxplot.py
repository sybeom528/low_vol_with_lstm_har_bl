"""Insert [3c] — 45-slot Δ Sharpe 분포 박스플롯 (regime marginal + dimension marginal)."""
import json, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

nb_path = '99_main_analysis.ipynb'
nb = json.load(open(nb_path, encoding='utf-8'))

new_code = r'''# ── [3c] 45-슬롯 Δ Sharpe 분포 — regime marginal + dimension marginal ──
# §5.3 메인 figure: 슬롯 구성에 따른 LSTM 우위 분포를 한 장으로

# [3] 에서 만든 df_p_sh 재사용 — index=(p_w, prior, q), columns=(period, L/A/Δ)
delta_by_period = {p: df_p_sh[(p, 'Δ')].dropna() for p in period_lbls}
all_d = df_p_sh[('All', 'Δ')]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# ── (1) Regime marginal — 5 박스 (All, R1, R2, R3, R4) ──
ax = axes[0, 0]
data_r = [delta_by_period[p].values for p in period_lbls]
bp = ax.boxplot(data_r, labels=period_lbls, patch_artist=True, showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='black',
                               markeredgecolor='black', markersize=5))
for patch, lbl in zip(bp['boxes'], period_lbls):
    color = '#d62728' if lbl == 'R3' else '#1f77b4'
    patch.set_facecolor(color); patch.set_alpha(0.5)
ax.axhline(0, color='black', lw=0.6, ls='--')
ax.set_title('Regime marginal — 45 슬롯 Δ Sharpe 분포', fontweight='bold')
ax.set_ylabel('Δ Sharpe (LSTM − ANN)')
ax.grid(True, alpha=0.3, axis='y')

# ── (2) Q dimension marginal — 5 박스 (lam, raw, inv, vsp, fpm) — All 기준 ──
ax = axes[0, 1]
data_q = [all_d.xs(q, level='q').dropna().values for q in Q_MODES]
bp = ax.boxplot(data_q, labels=Q_MODES, patch_artist=True, showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='black',
                               markeredgecolor='black', markersize=5))
for patch in bp['boxes']:
    patch.set_facecolor('#2ca02c'); patch.set_alpha(0.5)
ax.axhline(0, color='black', lw=0.6, ls='--')
ax.set_title('Q dimension — All 기간 Δ Sharpe (9 슬롯/box)', fontweight='bold')
ax.set_ylabel('Δ Sharpe')
ax.grid(True, alpha=0.3, axis='y')

# ── (3) P (p_w) dimension marginal — 3 박스 ──
ax = axes[1, 0]
data_p = [all_d.xs(pw, level='p_w').dropna().values for pw in P_WEIGHTS]
bp = ax.boxplot(data_p, labels=P_WEIGHTS, patch_artist=True, showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='black',
                               markeredgecolor='black', markersize=5))
for patch in bp['boxes']:
    patch.set_facecolor('#ff7f0e'); patch.set_alpha(0.5)
ax.axhline(0, color='black', lw=0.6, ls='--')
ax.set_title('P 행렬 (p_w) dimension — All 기간 Δ Sharpe (15 슬롯/box)', fontweight='bold')
ax.set_ylabel('Δ Sharpe')
ax.grid(True, alpha=0.3, axis='y')

# ── (4) Prior dimension marginal — 3 박스 ──
ax = axes[1, 1]
data_pr = [all_d.xs(pr, level='prior').dropna().values for pr in PRIORS]
bp = ax.boxplot(data_pr, labels=PRIORS, patch_artist=True, showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='black',
                               markeredgecolor='black', markersize=5))
for patch in bp['boxes']:
    patch.set_facecolor('#9467bd'); patch.set_alpha(0.5)
ax.axhline(0, color='black', lw=0.6, ls='--')
ax.set_title('Prior dimension — All 기간 Δ Sharpe (15 슬롯/box)', fontweight='bold')
ax.set_ylabel('Δ Sharpe')
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('45-슬롯 LSTM Sharpe 우위 분포 — Regime & Dimension Marginal',
             fontsize=13, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()

# 콘솔 요약: regime별 % LSTM win
print('='*60)
print('% LSTM win (Δ Sharpe > 0) — 45 슬롯 기준')
print('='*60)
for p in period_lbls:
    d = delta_by_period[p]
    n_win = int((d > 0).sum()); n_total = len(d)
    pct = n_win/n_total*100
    bar = '█' * int(pct/5)
    print(f'  {p:<6s}: {n_win:>2d}/{n_total} ({pct:>5.1f}%)  {bar}')

# 차원별 마진 평균
print('\n' + '='*60)
print('Dimension marginal 평균 Δ Sharpe (All 기간)')
print('='*60)
print('Q:')
for q in Q_MODES:
    v = all_d.xs(q, level='q').dropna()
    print(f'  q={q:<4s}: mean Δ = {v.mean():+.3f}  (n={len(v)})')
print('P (p_w):')
for pw in P_WEIGHTS:
    v = all_d.xs(pw, level='p_w').dropna()
    print(f'  p_w={pw:<5s}: mean Δ = {v.mean():+.3f}  (n={len(v)})')
print('Prior:')
for pr in PRIORS:
    v = all_d.xs(pr, level='prior').dropna()
    print(f'  prior={pr:<5s}: mean Δ = {v.mean():+.3f}  (n={len(v)})')
'''

# [3b] 다음에 [3c] 로 삽입
target_idx = None
for i, c in enumerate(nb['cells']):
    if c['cell_type']=='code':
        src = ''.join(c['source'])
        if '[3b]' in src:
            target_idx = i; break
assert target_idx is not None, 'cell [3b] not found'

new_cell = {
    'cell_type': 'code',
    'metadata': {},
    'execution_count': None,
    'outputs': [],
    'source': new_code.splitlines(keepends=True),
}
nb['cells'].insert(target_idx + 1, new_cell)
json.dump(nb, open(nb_path, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(f'Inserted [3c] at index {target_idx + 1}')
