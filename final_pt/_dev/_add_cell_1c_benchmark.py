"""Insert [1c] — Benchmark comparison with two LSTM slots + 6 metrics + cum return chart.

Adds after [1b] (cell index 5). Builds on [1b] infrastructure:
- loaded, rf, ret_spy, ret_1n, ret_rp from [1b]
- Adds LSTM-defensive (eq+lam) alongside LSTM-adaptive (eq+fpm)
- Outputs 6 metric tables (Sharpe/Sortino/MDD/AnnRet/CumRet/AnnVol) × 5 periods
- Plots cumulative return (log scale) + drawdown chart
"""
import json
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

nb_path = '99_main_analysis.ipynb'
nb = json.load(open(nb_path, encoding='utf-8'))

# Find [1b] cell index (should be cell 5 — code cell)
target_idx = None
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        src = ''.join(c['source'])
        if '[1b]' in src and 'Benchmark comparison' in src:
            target_idx = i
            break
assert target_idx is not None, '[1b] cell not found'

# Markdown header cell for [1c]
md_source = """## [1c] Benchmark comparison — 두 LSTM 슬롯 (defensive vs adaptive)

**목적**: §4.4 벤치마크 비교 — 단일 best slot 강요 대신 두 LSTM 옵션의 trade-off 제시

- **외부 baselines** (3): SPY, 1/N, Risk Parity (1/σ_252d)
- **BL portfolios** (4):
  - **ANN-anchor** (`mat_mcap_mcap_fpm_pap_ann`): Pyo-Lee 2018 원안
  - **LSTM-anchor** (`mat_mcap_mcap_fpm_pap`): 같은 slot, σ 모델만 LSTM 교체
  - **LSTM-defensive** (`mat_mcap_eq_lam_pap`): eq P + lam Q — q>0 보장, 저변동 정체성 유지
  - **LSTM-adaptive** (`mat_mcap_eq_fpm_pap`): eq P + fpm Q — peak Sharpe, regime 적응

**출력**: 6개 지표 표 (Sharpe/Sortino/MDD/AnnRet/CumRet/AnnVol) × 5 기간 (All/R1/R2/R3/R4)
+ 누적수익률(log) + Drawdown 2-panel 차트
"""

code_source = """# ── [1c] Benchmark comparison — 두 LSTM 슬롯 + 다지표 표 + 누적수익률 그림 ───
# [1b] 인프라 재사용: loaded, rf, ret_spy, ret_1n, ret_rp

CUTOFF_DT = pd.Timestamp('2025-12-31')

BL_SLOTS_C = {
    'ANN-anchor (Pyo-Lee 2018)':        'mat_mcap_mcap_fpm_pap_ann',
    'LSTM-anchor (σ swap)':             'mat_mcap_mcap_fpm_pap',
    'LSTM-defensive (eq P + lam Q)':    'mat_mcap_eq_lam_pap',
    'LSTM-adaptive (eq P + fpm Q)':     'mat_mcap_eq_fpm_pap',
}

STRATS_C = [
    ('SPY (market)',               ret_spy),
    ('1/N (equal-weight)',         ret_1n),
    ('Risk Parity (1/σ_252d)',     ret_rp),
] + [(lbl, loaded[slot]['ret'].dropna()) for lbl, slot in BL_SLOTS_C.items()]

PERIODS_B = [
    ('All', None, None),
    ('R1', '2010-01-01','2012-06-30'),
    ('R2', '2012-07-01','2019-12-31'),
    ('R3', '2020-01-01','2023-06-30'),
    ('R4', '2023-07-01','2025-12-31'),
]

# 다지표 계산: AnnRet, AnnVol, Sharpe, Sortino, MDD, CumRet
def _full_metrics(s, st, et):
    r = s.dropna()
    if st: r = r[r.index >= st]
    if et: r = r[r.index <= et]
    if len(r) < 6: return None
    rfa = rf.reindex(r.index).fillna(0)
    exc = r - rfa
    ann_ret = (1+r).prod() ** (12/len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sh = float(exc.mean()*12/ann_vol) if ann_vol > 0 else np.nan
    dn = r[r<0].std() * np.sqrt(12)
    so = float(exc.mean()*12/dn) if (dn and dn>0) else np.nan
    cum = (1+r).cumprod()
    mdd = float((cum/cum.cummax() - 1).min())
    cum_ret = float(cum.iloc[-1] - 1)
    return dict(AnnRet=ann_ret, AnnVol=ann_vol, Sharpe=sh, Sortino=so, MDD=mdd, CumRet=cum_ret)

# 전략 × 기간 다지표 dict
metrics_data = {}
for label, s in STRATS_C:
    metrics_data[label] = {}
    for p, st, et in PERIODS_B:
        m = _full_metrics(s, st, et)
        if m: metrics_data[label][p] = m

# ───── 표 출력 (지표별) ─────
def _print_metric_table(title, key, fmt):
    print('\\n' + '='*112)
    print(title)
    print('='*112)
    hdr = f'{"strategy":<38s}' + ''.join(f'{p:>14s}' for p,_,_ in PERIODS_B)
    print(hdr); print('-'*len(hdr))
    for label in metrics_data:
        row = []
        for p,_,_ in PERIODS_B:
            m = metrics_data[label].get(p)
            row.append(fmt(m[key]) if m else '   —   ')
        print(f'{label:<38s}' + ''.join(f'{c:>14s}' for c in row))

_print_metric_table('Table 1: Sharpe ratio × period',         'Sharpe',  lambda v: f'{v:+.3f}')
_print_metric_table('Table 2: Sortino ratio × period',        'Sortino', lambda v: f'{v:+.3f}')
_print_metric_table('Table 3: MDD (max drawdown) × period',   'MDD',     lambda v: f'{v*100:+.2f}%')
_print_metric_table('Table 4: Annualized return × period',    'AnnRet',  lambda v: f'{v*100:+.2f}%')
_print_metric_table('Table 5: Cumulative return × period',    'CumRet',  lambda v: f'{v*100:+.1f}%')
_print_metric_table('Table 6: Annualized volatility × period','AnnVol',  lambda v: f'{v*100:.2f}%')

# ───── 그림: 누적 수익률 (log scale) + Drawdown ─────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                gridspec_kw={'height_ratios': [2.2, 1]})

style_map = {
    'SPY (market)':                  {'color':'#666666', 'ls':'-',  'lw':1.4, 'alpha':0.75},
    '1/N (equal-weight)':            {'color':'#999999', 'ls':'--', 'lw':1.0, 'alpha':0.6},
    'Risk Parity (1/σ_252d)':        {'color':'#bbbbbb', 'ls':':',  'lw':1.0, 'alpha':0.6},
    'ANN-anchor (Pyo-Lee 2018)':     {'color':'#d62728', 'ls':'-',  'lw':1.5, 'alpha':0.85},
    'LSTM-anchor (σ swap)':          {'color':'#ff9896', 'ls':'-',  'lw':1.2, 'alpha':0.7},
    'LSTM-defensive (eq P + lam Q)': {'color':'#1f77b4', 'ls':'-',  'lw':2.0, 'alpha':1.0},
    'LSTM-adaptive (eq P + fpm Q)':  {'color':'#2ca02c', 'ls':'-',  'lw':2.0, 'alpha':1.0},
}

REGIME_BOUNDS = [
    ('R1', pd.Timestamp('2010-01-01'), pd.Timestamp('2012-06-30')),
    ('R2', pd.Timestamp('2012-07-01'), pd.Timestamp('2019-12-31')),
    ('R3', pd.Timestamp('2020-01-01'), pd.Timestamp('2023-06-30')),
    ('R4', pd.Timestamp('2023-07-01'), pd.Timestamp('2025-12-31')),
]

# R3 음영
ax1.axvspan(pd.Timestamp('2020-01-01'), pd.Timestamp('2023-06-30'),
            color='#ffe0e0', alpha=0.5, zorder=0, label='_nolegend_')
ax2.axvspan(pd.Timestamp('2020-01-01'), pd.Timestamp('2023-06-30'),
            color='#ffe0e0', alpha=0.5, zorder=0)

# 라인 플롯
for label, s in STRATS_C:
    s2 = s[s.index <= CUTOFF_DT].dropna()
    if len(s2) < 6: continue
    cum = (1+s2).cumprod()
    style = style_map.get(label, {'color':'black','ls':'-','lw':1.0,'alpha':0.7})
    ax1.plot(cum.index, cum.values, label=label, **style)
    dd = cum / cum.cummax() - 1
    ax2.plot(dd.index, dd.values*100, **style)

# 레짐 경계 vline + 라벨
for lbl, st, et in REGIME_BOUNDS[1:]:
    ax1.axvline(st, color='gray', ls=':', lw=0.5, alpha=0.6)
    ax2.axvline(st, color='gray', ls=':', lw=0.5, alpha=0.6)
for lbl, st, et in REGIME_BOUNDS:
    mid = st + (et - st)/2
    ax1.text(mid, 1.02, lbl, transform=ax1.get_xaxis_transform(),
             ha='center', va='bottom', fontsize=10, fontweight='bold', color='dimgray')

ax1.set_title('Cumulative return (rebased = 1.0 at 2010-01) — R3 위기 구간 음영', fontweight='bold')
ax1.set_ylabel('Cumulative return (log)')
ax1.set_yscale('log')
ax1.legend(loc='upper left', fontsize=9, ncol=2, framealpha=0.9)
ax1.grid(True, alpha=0.3)

ax2.set_title('Drawdown')
ax2.set_ylabel('Drawdown (%)')
ax2.set_xlabel('Date')
ax2.axhline(0, color='black', lw=0.5)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
"""

md_cell = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': md_source.splitlines(keepends=True),
}

code_cell = {
    'cell_type': 'code',
    'metadata': {},
    'execution_count': None,
    'outputs': [],
    'source': code_source.splitlines(keepends=True),
}

# [1b] code cell이 target_idx. 그 바로 다음에 markdown + code 두 셀 삽입
nb['cells'].insert(target_idx + 1, md_cell)
nb['cells'].insert(target_idx + 2, code_cell)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Inserted [1c] markdown + code at indices {target_idx + 1}, {target_idx + 2}')
print(f'Total cells: {len(nb["cells"])} (was {len(nb["cells"])-2})')
