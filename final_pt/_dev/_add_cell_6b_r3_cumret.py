"""Insert [6b] — R3 cumulative return curves (LSTM vs ANN vs SPY) for anchor & LSTM-best slots."""
import json, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

nb_path = '99_main_analysis.ipynb'
nb = json.load(open(nb_path, encoding='utf-8'))

new_code = r'''# ── [6b] R3 누적수익률 — LSTM vs ANN vs SPY (anchor + LSTM-best) ──
# [6] 의 LSTM−ANN 차이가 반등 캡처에서 오는지 / 하락 방어에서 오는지 구분
R3_S = pd.Timestamp('2020-01-31')
R3_E = pd.Timestamp('2023-06-30')

R3_SLOTS = {
    'anchor — mat_mcap_mcap_fpm_pap (Pyo-Lee 2018)':  'mat_mcap_mcap_fpm_pap',
    'LSTM-best — mat_mcap_eq_fpm_pap':                'mat_mcap_eq_fpm_pap',
}

def _shifted_ret(name):
    """ pkl ret (construction date) → calendar end date 정렬 ([6] 과 동일). """
    r = loaded[name]['ret'].dropna()
    r.index = r.index + pd.offsets.MonthEnd(1)
    return r

# SPY 도 동일 shift 로 calendar 정렬
spy_full = loaded['mat_mcap_mcap_fpm_pap']['spy_ret'].dropna()
spy_full.index = spy_full.index + pd.offsets.MonthEnd(1)

fig, axes = plt.subplots(len(R3_SLOTS), 2, figsize=(15, 4.0*len(R3_SLOTS)),
                          gridspec_kw={'width_ratios':[3, 1]})
if len(R3_SLOTS)==1: axes = axes.reshape(1, -1)

print('='*86)
print('R3 (2020-01~2023-06)  —  LSTM vs ANN vs SPY  누적수익률 & 월수익률')
print('='*86)

for i, (label, slot_name) in enumerate(R3_SLOTS.items()):
    rL = _shifted_ret(slot_name)
    rA = _shifted_ret(slot_name + '_ann')
    common = rL.index.intersection(rA.index).intersection(spy_full.index)
    common_r3 = common[(common>=R3_S) & (common<=R3_E)]
    rL_r3 = rL.loc[common_r3]; rA_r3 = rA.loc[common_r3]; rS_r3 = spy_full.loc[common_r3]

    cumL = (1+rL_r3).cumprod(); cumA = (1+rA_r3).cumprod(); cumS = (1+rS_r3).cumprod()
    diff = rL_r3 - rA_r3

    # 통계
    rL_up  = rL_r3[rS_r3>0]; rA_up  = rA_r3[rS_r3>0]
    rL_dn  = rL_r3[rS_r3<0]; rA_dn  = rA_r3[rS_r3<0]
    print(f'\n[{label}]  n_months={len(common_r3)}')
    d_up = (rL_up-rA_up).mean()*100; d_dn = (rL_dn-rA_dn).mean()*100
    print(f'  SPY+ month (n={len(rL_up)}): LSTM={rL_up.mean()*100:+.2f}%, ANN={rA_up.mean()*100:+.2f}%, Δ={d_up:+.3f}%  ← 반등 캡처')
    print(f'  SPY− month (n={len(rL_dn)}): LSTM={rL_dn.mean()*100:+.2f}%, ANN={rA_dn.mean()*100:+.2f}%, Δ={d_dn:+.3f}%  ← 하락 방어')
    print(f'  cumulative end:  LSTM {cumL.iloc[-1]:.3f},  ANN {cumA.iloc[-1]:.3f},  SPY {cumS.iloc[-1]:.3f}')

    # 좌측: 누적수익률
    ax = axes[i, 0]
    ax.plot(cumL.index, cumL.values, lw=2.0, color='#d62728', label='LSTM', marker='o', markersize=3.5)
    ax.plot(cumA.index, cumA.values, lw=2.0, color='#1f77b4', label='ANN',  marker='s', markersize=3.5)
    ax.plot(cumS.index, cumS.values, lw=1.5, color='gray',    label='SPY',  linestyle='--', alpha=0.8)
    ax.axhline(1.0, color='black', lw=0.4)
    # 이벤트
    for ym, ev in [('2020-03','COVID crash'),('2022-06','Fed hike'),('2022-09','BoE')]:
        matches = [dt for dt in common_r3 if dt.strftime('%Y-%m')==ym]
        if matches:
            ax.axvline(matches[0], color='gray', ls=':', lw=0.6, alpha=0.6)
            ax.text(matches[0], ax.get_ylim()[1]*0.98 if i==0 else 1.0,
                    f' {ev}', fontsize=7, color='dimgray', va='top', rotation=0)
    ax.set_title(f'{label}  —  cumulative return (R3)', fontsize=10, fontweight='bold')
    ax.set_ylabel('cumulative (×)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=15)

    # 우측: SPY sign 별 LSTM−ANN bar
    ax = axes[i, 1]
    colors = ['#2ca02c' if s>0 else '#d62728' for s in rS_r3]
    ax.bar(range(len(diff)), diff.values*100, color=colors, alpha=0.8)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_title('LSTM−ANN by SPY sign\n(green=SPY+, red=SPY−)', fontsize=9)
    ax.set_ylabel('Δ (%)', fontsize=9)
    ax.set_xticks([]); ax.grid(True, alpha=0.3, axis='y')
    # 평균선
    avg_up = (rL_up-rA_up).mean()*100; avg_dn = (rL_dn-rA_dn).mean()*100
    ax.axhline(avg_up, color='#2ca02c', ls=':', lw=1.2, label=f'mean SPY+: {avg_up:+.2f}%')
    ax.axhline(avg_dn, color='#d62728', ls=':', lw=1.2, label=f'mean SPY−: {avg_dn:+.2f}%')
    ax.legend(loc='best', fontsize=8)

plt.tight_layout()
plt.show()
'''

# Find [6] index and insert new cell after it
target_idx = None
for i, c in enumerate(nb['cells']):
    if c['cell_type']=='code':
        src = ''.join(c['source'])
        if '── [6]' in src and 'Q × regime' in src:
            target_idx = i
            break

assert target_idx is not None, 'cell [6] not found'

new_cell = {
    'cell_type': 'code',
    'metadata': {},
    'execution_count': None,
    'outputs': [],
    'source': new_code.splitlines(keepends=True),
}
nb['cells'].insert(target_idx+1, new_cell)
json.dump(nb, open(nb_path, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(f'Inserted [6b] at index {target_idx+1}')
