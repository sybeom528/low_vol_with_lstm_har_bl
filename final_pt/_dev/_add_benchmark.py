"""Add [1b] benchmark comparison cell: SPY, 1/N, Risk Parity vs ANN-anchor/LSTM-anchor/LSTM-best."""
import json, uuid, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

nb_path = '99_main_analysis.ipynb'
nb = json.load(open(nb_path, encoding='utf-8'))

md = """## [1b] Benchmark comparison — SPY, 1/N, Risk Parity vs BL portfolios

**목적**: BL framework 의 contribution 을 외부 baseline 대비 측정.

- **외부 baselines**: SPY (passive market), **1/N** equal-weight (DeMiguel et al. 2009), **Risk Parity** (1/σ_realized)
- **BL portfolios (3 단계)**:
  - **ANN-anchor** (`mat_mcap_mcap_fpm_pap_ann`): Pyo-Lee 2018 replication
  - **LSTM-anchor** (`mat_mcap_mcap_fpm_pap`): 같은 slot, σ 모델만 LSTM 으로 교체 → **(a) 모델 효과**
  - **LSTM-best** (`mat_mcap_eq_fpm_pap`): All 기간 best slot, p_w mcap→eq 변경 → **(a) + (b) slot 효과**

- universe: ANN-anchor 의 monthly universe 사용 (BL pkl 의 weights index)
- 1/N, RP 의 max weight = 0.1, tc = 20bp (BL 과 동일 조건)
"""

code = r'''# ── [1b] Benchmark comparison — SPY, 1/N, Risk Parity ─────────
from pathlib import Path

CUTOFF_DT = pd.Timestamp('2025-12-31')

# anchor universe 추출 (월별 valid ticker)
_anchor_w = loaded['mat_mcap_mcap_fpm_pap']['weights']
# panel 에서 vol_252d, fwd_ret_1m 가져오기
_panel = pd.read_csv(DATA_DIR/'monthly_panel.csv', parse_dates=['date'])

def _build_naive_portfolio(weight_mode: str, tc: float = 0.002, max_w: float = 0.10):
    """
    weight_mode: '1N' or 'RP' (risk parity 1/σ_252d)
    universe: each month = anchor weights index 의 non-null ticker
    """
    rets, turns = [], []
    prev_w = None
    for dt in _anchor_w.index:
        if dt > CUTOFF_DT: continue
        # universe = anchor 의 weight 가 nan 이 아닌 종목
        anchor_row = _anchor_w.loc[dt].dropna()
        universe = anchor_row.index.tolist()
        if len(universe) < 20: continue
        # panel 에서 그 월 데이터
        p_row = _panel[(_panel['date']==dt) & (_panel['ticker'].isin(universe))]
        p_row = p_row.dropna(subset=['fwd_ret_1m'])
        if len(p_row) < 20: continue
        valid_tix = p_row['ticker'].tolist()
        ret_vec = p_row.set_index('ticker')['fwd_ret_1m']

        if weight_mode == '1N':
            n = len(valid_tix)
            w = pd.Series(1.0/n, index=valid_tix)
        elif weight_mode == 'RP':
            sig = p_row.set_index('ticker')['vol_252d'].replace(0, np.nan).dropna()
            valid_tix = sig.index.intersection(valid_tix).tolist()
            if len(valid_tix) < 20: continue
            inv = 1.0 / sig.loc[valid_tix]
            w = inv / inv.sum()
            ret_vec = ret_vec.reindex(valid_tix)
        # max weight cap (BL 와 동일)
        w = w.clip(upper=max_w)
        w = w / w.sum()
        # gross return
        ret_vec_aligned = ret_vec.reindex(w.index).fillna(0)
        gross = float((w * ret_vec_aligned).sum())
        # turnover
        if prev_w is not None:
            all_tix = w.index.union(prev_w.index)
            w_a = w.reindex(all_tix).fillna(0)
            pw_a = prev_w.reindex(all_tix).fillna(0)
            turn = float((w_a - pw_a).abs().sum())
        else:
            turn = 0.0
        net = gross - turn*tc
        rets.append((dt, net))
        turns.append((dt, turn))
        prev_w = w
    s = pd.Series({d:r for d,r in rets}).sort_index()
    return s

print('Computing 1/N and Risk Parity portfolios ...')
ret_1n = _build_naive_portfolio('1N')
ret_rp = _build_naive_portfolio('RP')
print(f'  1/N: {len(ret_1n)} months, mean={ret_1n.mean()*100:+.3f}%')
print(f'  RP : {len(ret_rp)} months, mean={ret_rp.mean()*100:+.3f}%')

# SPY return (anchor pkl 의 spy_ret)
ret_spy = loaded['mat_mcap_mcap_fpm_pap']['spy_ret'].dropna()

# BL 슬롯 return
BL_SLOTS = {
    'ANN-anchor (Pyo-Lee 2018)':      'mat_mcap_mcap_fpm_pap_ann',
    'LSTM-anchor (model swap)':       'mat_mcap_mcap_fpm_pap',
    'LSTM-best (mcap-eq-fpm-pap)':   'mat_mcap_eq_fpm_pap',
}
def _bl_ret(name):
    return loaded[name]['ret'].dropna()

# Metric 계산 함수 (rf=0 가정, monthly)
PERIODS_B = [
    ('All', None, None),
    ('R1', '2010-01-01','2012-06-30'),
    ('R2', '2012-07-01','2019-12-31'),
    ('R3', '2020-01-01','2023-06-30'),
    ('R4', '2023-07-01','2025-12-31'),
]
def _metrics(s, st, et):
    r = s.dropna()
    if st: r = r[r.index>=st]
    if et: r = r[r.index<=et]
    if len(r)<6: return None
    rfa = rf.reindex(r.index).fillna(0)
    exc = r - rfa
    vol = r.std()*np.sqrt(12)
    sh  = float(exc.mean()*12/vol) if vol>0 else np.nan
    dn  = r[r<0].std()*np.sqrt(12)
    so  = float(exc.mean()*12/dn) if (dn and dn>0) else np.nan
    cum = (1+r).cumprod()
    mdd = float((cum/cum.cummax()-1).min())
    return dict(Sharpe=sh, Sortino=so, MDD=mdd)

# strategy 목록
STRATS = [
    ('SPY (market)',               ret_spy),
    ('1/N (equal-weight)',         ret_1n),
    ('Risk Parity (1/σ_252d)',     ret_rp),
] + [(lbl, _bl_ret(nm)) for lbl, nm in BL_SLOTS.items()]

# 표 출력
print('\n' + '='*100)
print('Benchmark comparison — Sharpe (Sortino) [MDD] × period')
print('='*100)
hdr = f'{"strategy":<32s}' + ''.join(f'{p:>14s}' for p,_,_ in PERIODS_B)
print(hdr); print('-'*len(hdr))
all_metrics = {}
for label, s in STRATS:
    row = []
    metrics_p = {}
    for p, st, et in PERIODS_B:
        m = _metrics(s, st, et)
        if m:
            row.append(f'{m["Sharpe"]:+.2f}({m["Sortino"]:+.2f})')
            metrics_p[p] = m
        else:
            row.append('       —     ')
    print(f'{label:<32s}' + ''.join(f'{c:>14s}' for c in row))
    all_metrics[label] = metrics_p

# MDD 별도 표
print('\n' + '='*100)
print('MDD (max drawdown) × period')
print('='*100)
print(hdr); print('-'*len(hdr))
for label, s in STRATS:
    row = []
    for p, st, et in PERIODS_B:
        m = all_metrics.get(label, {}).get(p)
        row.append(f'{m["MDD"]*100:+.2f}%' if m else '   —   ')
    print(f'{label:<32s}' + ''.join(f'{c:>14s}' for c in row))

# 누적수익률 + drawdown chart
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
colors_map = {
    'SPY (market)':                'gray',
    '1/N (equal-weight)':          '#888888',
    'Risk Parity (1/σ_252d)':      '#666666',
    'ANN-anchor (Pyo-Lee 2018)':   '#d62728',
    'LSTM-anchor (model swap)':    '#ff7f0e',
    'LSTM-best (mcap-eq-fpm-pap)': '#2ca02c',
}
linestyles = {
    'SPY (market)': '-',
    '1/N (equal-weight)': '--',
    'Risk Parity (1/σ_252d)': ':',
    'ANN-anchor (Pyo-Lee 2018)': '-',
    'LSTM-anchor (model swap)': '-',
    'LSTM-best (mcap-eq-fpm-pap)': '-',
}
for label, s in STRATS:
    s2 = s[s.index<=CUTOFF_DT].dropna()
    if len(s2)<6: continue
    cum = (1+s2).cumprod()
    ax1.plot(cum.index, cum.values, lw=1.8, label=label,
              color=colors_map.get(label,'black'),
              linestyle=linestyles.get(label,'-'))
    dd = cum / cum.cummax() - 1
    ax2.plot(dd.index, dd.values*100, lw=1.2,
              color=colors_map.get(label,'black'),
              linestyle=linestyles.get(label,'-'),
              alpha=0.85)

ax1.set_title('Cumulative return (rebased = 1.0 at 2010-01)')
ax1.set_ylabel('Cumulative return')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)
ax2.set_title('Drawdown')
ax2.set_ylabel('Drawdown (%)')
ax2.grid(True, alpha=0.3)
ax2.axhline(0, color='black', lw=0.5)
plt.tight_layout()
plt.show()
'''

# Insert [1b] right after [1] (σ accuracy). Find [1] code cell (id dfc903b1).
target_id = 'dfc903b1'
insert_at = None
for i, c in enumerate(nb['cells']):
    if c.get('id')==target_id:
        insert_at = i+1
        break
if insert_at is None:
    print('ERROR: [1] code cell not found')
else:
    new_cells = [
        {'cell_type':'markdown','id':uuid.uuid4().hex[:8],'metadata':{},
         'source': md.splitlines(keepends=True)},
        {'cell_type':'code','id':uuid.uuid4().hex[:8],'metadata':{},
         'execution_count':None,'outputs':[],
         'source': code.splitlines(keepends=True)},
    ]
    nb['cells'] = nb['cells'][:insert_at] + new_cells + nb['cells'][insert_at:]
    json.dump(nb, open(nb_path,'w',encoding='utf-8'), indent=1, ensure_ascii=False)
    print(f'Inserted [1b] at position {insert_at}. Total cells: {len(nb["cells"])}')
