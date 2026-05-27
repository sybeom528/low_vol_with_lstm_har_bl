"""Update [6b] — add rp_rp_lam, eq_eq_fpm slots (total 4 slots for Q comparison)."""
import json, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

nb_path = '99_main_analysis.ipynb'
nb = json.load(open(nb_path, encoding='utf-8'))

new_code = r'''# ── [6b] R3 누적수익률 — LSTM vs ANN vs SPY (4 슬롯, Q 다양화) ─────
# [6] 의 LSTM−ANN 차이가 반등 캡처에서 오는지 / 하락 방어에서 오는지 구분
# Q 가 다르면 (fpm vs lam) 방어 패턴이 달라질 수 있음 → 함께 비교
R3_S = pd.Timestamp('2020-01-31')
R3_E = pd.Timestamp('2023-06-30')

R3_SLOTS = {
    'anchor — mcap/mcap/fpm':           'mat_mcap_mcap_fpm_pap',
    'LSTM-best — mcap/eq/fpm':          'mat_mcap_eq_fpm_pap',
    'rp/rp/lam (Q=lam, sign-fixed)':    'mat_rp_rp_lam_pap',
    'eq/eq/fpm (eq prior + fpm)':       'mat_eq_eq_fpm_pap',
}

def _shifted_ret(name):
    """ pkl ret (construction date) → calendar end date ([6] 과 동일 shift). """
    r = loaded[name]['ret'].dropna()
    r.index = r.index + pd.offsets.MonthEnd(1)
    return r

# SPY 동일 shift
spy_full = loaded['mat_mcap_mcap_fpm_pap']['spy_ret'].dropna()
spy_full.index = spy_full.index + pd.offsets.MonthEnd(1)

fig, axes = plt.subplots(len(R3_SLOTS), 2, figsize=(15, 3.4*len(R3_SLOTS)),
                          gridspec_kw={'width_ratios':[3, 1]})
if len(R3_SLOTS)==1: axes = axes.reshape(1, -1)

print('='*100)
print('R3 (2020-01~2023-06)  —  LSTM vs ANN vs SPY  누적수익률 & SPY 부호별 평균')
print('='*100)
print(f'{"slot":<36s}{"n_up":>6s}{"L_up":>9s}{"A_up":>9s}{"Δ_up":>9s}'
      f'{"n_dn":>6s}{"L_dn":>9s}{"A_dn":>9s}{"Δ_dn":>9s}{"cumL":>8s}{"cumA":>8s}')
print('-'*100)

for i, (label, slot_name) in enumerate(R3_SLOTS.items()):
    if slot_name not in loaded or (slot_name + '_ann') not in loaded:
        print(f'[skip] {slot_name} not loaded'); continue
    rL = _shifted_ret(slot_name); rA = _shifted_ret(slot_name + '_ann')
    common = rL.index.intersection(rA.index).intersection(spy_full.index)
    common_r3 = common[(common>=R3_S) & (common<=R3_E)]
    rL_r3 = rL.loc[common_r3]; rA_r3 = rA.loc[common_r3]; rS_r3 = spy_full.loc[common_r3]

    cumL = (1+rL_r3).cumprod(); cumA = (1+rA_r3).cumprod(); cumS = (1+rS_r3).cumprod()
    diff = rL_r3 - rA_r3

    rL_up = rL_r3[rS_r3>0]; rA_up = rA_r3[rS_r3>0]
    rL_dn = rL_r3[rS_r3<0]; rA_dn = rA_r3[rS_r3<0]
    d_up = (rL_up-rA_up).mean()*100; d_dn = (rL_dn-rA_dn).mean()*100
    print(f'{label:<36s}{len(rL_up):>6d}{rL_up.mean()*100:>+8.2f}%{rA_up.mean()*100:>+8.2f}%{d_up:>+8.3f}%'
          f'{len(rL_dn):>6d}{rL_dn.mean()*100:>+8.2f}%{rA_dn.mean()*100:>+8.2f}%{d_dn:>+8.3f}%'
          f'{cumL.iloc[-1]:>8.3f}{cumA.iloc[-1]:>8.3f}')

    # 좌측: 누적수익률
    ax = axes[i, 0]
    ax.plot(cumL.index, cumL.values, lw=1.8, color='#d62728', label='LSTM', marker='o', markersize=3)
    ax.plot(cumA.index, cumA.values, lw=1.8, color='#1f77b4', label='ANN',  marker='s', markersize=3)
    ax.plot(cumS.index, cumS.values, lw=1.3, color='gray',    label='SPY',  linestyle='--', alpha=0.8)
    ax.axhline(1.0, color='black', lw=0.4)
    for ym, ev in [('2020-03','COVID'),('2022-06','Fed hike'),('2022-09','BoE')]:
        matches = [dt for dt in common_r3 if dt.strftime('%Y-%m')==ym]
        if matches:
            ax.axvline(matches[0], color='gray', ls=':', lw=0.6, alpha=0.6)
            if i==0:
                ax.text(matches[0], ax.get_ylim()[1]*0.96, f' {ev}',
                        fontsize=7, color='dimgray', va='top')
    ax.set_title(f'{label}  —  cumulative return (R3)', fontsize=10, fontweight='bold')
    ax.set_ylabel('cumulative (×)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=15)

    # 우측: SPY sign 별 LSTM−ANN bar
    ax = axes[i, 1]
    colors = ['#2ca02c' if s>0 else '#d62728' for s in rS_r3]
    ax.bar(range(len(diff)), diff.values*100, color=colors, alpha=0.8)
    ax.axhline(0, color='black', lw=0.5)
    ax.axhline(d_up, color='#2ca02c', ls=':', lw=1.2, label=f'mean SPY+: {d_up:+.2f}%')
    ax.axhline(d_dn, color='#d62728', ls=':', lw=1.2, label=f'mean SPY−: {d_dn:+.2f}%')
    ax.set_title('LSTM−ANN by SPY sign', fontsize=9)
    ax.set_ylabel('Δ (%)', fontsize=8)
    ax.set_xticks([]); ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best', fontsize=7)

plt.tight_layout()
plt.show()
'''

# locate [6b]
target_idx = None
for i, c in enumerate(nb['cells']):
    if c['cell_type']=='code':
        src = ''.join(c['source'])
        if '── [6b]' in src:
            target_idx = i; break
assert target_idx is not None, 'cell [6b] not found'

nb['cells'][target_idx]['source'] = new_code.splitlines(keepends=True)
nb['cells'][target_idx]['outputs'] = []
nb['cells'][target_idx]['execution_count'] = None
json.dump(nb, open(nb_path, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(f'Updated [6b] at index {target_idx} — now 4 slots')
