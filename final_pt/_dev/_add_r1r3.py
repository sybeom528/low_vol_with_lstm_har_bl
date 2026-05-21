"""Add R1-R3 only version of [3] grid right after [3]."""
import json, uuid, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

nb_path = '99_main_analysis.ipynb'
nb = json.load(open(nb_path, encoding='utf-8'))

md = """## [3b] P 행렬 × 모든 슬롯 grid — R1-R3 sample (R4 제외)

**목적**: [3] 의 동일 분석을 **R4 (AI 랠리, 2023-07~2025-12) 제외** 한 sample (2010-01~2023-06, 162 months) 에서 재확인.

- R4 의 구조적 특수성 (AI 랠리 / Mag7 집중) 이 grid 패턴에 미치는 영향 확인용
- main 본문엔 안 들어감 (sample cutting). discussion / appendix evidence 로만 사용
"""

code = r'''# ── [3b] P 행렬 × 모든 슬롯 grid — R1-R3 sample (R4 제외) ───────
from IPython.display import display

PERIODS_R13 = [
    ('All_R1-R3', '2010-01-01', '2023-06-30'),
    ('R1',        '2010-01-01', '2012-06-30'),
    ('R2',        '2012-07-01', '2019-12-31'),
    ('R3',        '2020-01-01', '2023-06-30'),
]

PRIORS    = ['mcap', 'eq', 'rp']
P_WEIGHTS = ['mcap', 'eq', 'rp']
Q_MODES   = ['lam', 'raw', 'inv', 'vsp', 'fpm']
OMEGA     = 'pap'

slots = [(pw, pr, q)
         for pw in P_WEIGHTS
         for pr in PRIORS
         for q  in Q_MODES]

rows_idx, data_sh, data_so, data_md = [], [], [], []
for pw, pr, q in slots:
    L_name = f'mat_{pr}_{pw}_{q}_{OMEGA}'
    A_name = L_name + '_ann'
    if L_name not in loaded or A_name not in loaded: continue
    rows_idx.append((pw, pr, q))
    row_sh, row_so, row_md = [], [], []
    for plbl, s, e in PERIODS_R13:
        mL = _calc_metrics(L_name, s, e)
        mA = _calc_metrics(A_name, s, e)
        if mL and mA:
            row_sh.extend([mL['Sharpe'],  mA['Sharpe'],  mL['Sharpe']-mA['Sharpe']])
            row_so.extend([mL['Sortino'], mA['Sortino'], mL['Sortino']-mA['Sortino']])
            row_md.extend([mL['MDD'],     mA['MDD'],     mL['MDD']-mA['MDD']])
        else:
            row_sh.extend([np.nan]*3); row_so.extend([np.nan]*3); row_md.extend([np.nan]*3)
    data_sh.append(row_sh); data_so.append(row_so); data_md.append(row_md)

print(f'슬롯 수: {len(rows_idx)} (예상 45)')

cols_p = pd.MultiIndex.from_product(
            [[p for p,_,_ in PERIODS_R13], ['L','A','Δ']], names=['period','col'])
idx_p  = pd.MultiIndex.from_tuples(rows_idx, names=['p_w','prior','q'])
df_p_sh = pd.DataFrame(data_sh, index=idx_p, columns=cols_p)
df_p_so = pd.DataFrame(data_so, index=idx_p, columns=cols_p)
df_p_md = pd.DataFrame(data_md, index=idx_p, columns=cols_p)

period_lbls = [p for p,_,_ in PERIODS_R13]

# (1) p_w × 기간 LSTM 우위 카운트
print('\n' + '='*70)
print('P 행렬 × 기간 LSTM 우위 카운트 (Δ > 0) — R4 제외 sample')
print('='*70)
hdr = f'{"":<14s}' + ''.join(f'{p:>12s}' for p in period_lbls)
for metric_name, df_m in [('Sharpe', df_p_sh), ('Sortino', df_p_so), ('MDD', df_p_md)]:
    print(f'\n[{metric_name}]')
    print(hdr); print('-'*len(hdr))
    for pw in P_WEIGHTS:
        sub = df_m.loc[pw]; n_pair = len(sub)
        cells = [f'{int((sub[(p,"Δ")] > 0).sum())}/{n_pair}' for p in period_lbls]
        print(f'p_w={pw:<10s}' + ''.join(f'{c:>12s}' for c in cells))
    totals = [f'{int((df_m[(p,"Δ")] > 0).sum())}/{len(df_m)}' for p in period_lbls]
    print(f'{"합계":<12s}' + ''.join(f'{c:>12s}' for c in totals))

# (2) p_w × 기간 평균 절대값
print('\n' + '='*86)
print('평균 절대값 (p_w × 기간) — LSTM | ANN — R4 제외')
print('='*86)
for metric_name, df_m, fmt_v in [('Sharpe',  df_p_sh, '5.3f'),
                                  ('Sortino', df_p_so, '5.3f'),
                                  ('MDD',     df_p_md, '6.2%')]:
    print(f'\n[{metric_name}]')
    hdr2 = f'{"":<14s}' + ''.join(f'{p+" L":>10s}{p+" A":>10s}{p+" Δ":>10s}' for p in period_lbls)
    print(hdr2); print('-'*len(hdr2))
    for pw in P_WEIGHTS:
        sub = df_m.loc[pw]
        cells = []
        for p in period_lbls:
            mL_avg = sub[(p,"L")].mean()
            mA_avg = sub[(p,"A")].mean()
            mD_avg = sub[(p,"Δ")].mean()
            cells.append(f'{mL_avg:>+10.3f}{mA_avg:>+10.3f}{mD_avg:>+10.3f}'
                          if metric_name != 'MDD'
                          else f'{mL_avg:>+10.2%}{mA_avg:>+10.2%}{mD_avg:>+10.2%}')
        print(f'p_w={pw:<10s}' + ''.join(cells))

# (3) 표시용 styled table — Sharpe
def _hl(v):
    if not isinstance(v, (int, float)) or pd.isna(v): return ''
    if v > 0: return 'background-color: #b8e0b8; color: black; font-weight: bold'
    if v < 0: return 'background-color: #f4b6b6; color: black; font-weight: bold'
    return ''

delta_cols = [(p,'Δ') for p in period_lbls]
fmt_ratio  = {col: ('{:+.3f}' if col[1]=='Δ' else '{:.3f}') for col in df_p_sh.columns}
fmt_pct    = {col: ('{:+.2%}' if col[1]=='Δ' else '{:.2%}') for col in df_p_md.columns}

print('\n[표 1] Sharpe (LSTM | ANN | Δ) × (All_R1-R3 + R1/R2/R3)')
display(df_p_sh.style.format(fmt_ratio).map(_hl, subset=delta_cols))

print('\n[표 2] Sortino (LSTM | ANN | Δ)')
display(df_p_so.style.format(fmt_ratio).map(_hl, subset=delta_cols))

print('\n[표 3] MDD (LSTM | ANN | Δ)  ← Δ>0 = LSTM 덜 빠짐')
df_p_md.style.format(fmt_pct).map(_hl, subset=delta_cols)
'''

# Find position: insert after cell with id 2d49e28c (the [3] code cell)
target_id = '2d49e28c'
insert_at = None
for i, c in enumerate(nb['cells']):
    if c.get('id') == target_id:
        insert_at = i + 1
        break

if insert_at is None:
    print('ERROR: [3] code cell not found')
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
    print(f'Inserted [3b] at position {insert_at}. Total cells: {len(nb["cells"])}')
    for i,c in enumerate(nb['cells']):
        src = ''.join(c['source'])[:55].replace(chr(10),' | ')
        print(f'{i:3d} {c["cell_type"]:<8s} {c.get("id",""):<12s} {src}')
