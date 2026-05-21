"""Restore deleted cells [4]-[7] in 99_main_analysis.ipynb."""
import json, uuid
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

nb_path = '99_main_analysis.ipynb'
nb = json.load(open(nb_path, encoding='utf-8'))

# ── [4] Low-vol anomaly check ────────────────────────────────────────
md4 = """## [4] Low-vol anomaly per regime check

**목적**: BL framework 의 "long low-vol / short high-vol" view 가 작동하려면 sample 내 low-vol anomaly 가 존재해야 함. SP500 sample 2010-2025 에서 anomaly 가 실제 있는지 확인.

- 매월 cross-sectional 로 σ 상위 30% (high-vol) vs 하위 30% (low-vol) bucket 평균 수익률 비교
- L-H spread (low minus high) 가 양수면 anomaly 작동, null/음수면 부재
"""

code4 = r'''# ── [4] Low-vol anomaly per regime (top/bottom 30%) ──────────────
panel = pd.read_csv(DATA_DIR / 'monthly_panel.csv', parse_dates=['date'])
print(f'panel: {len(panel):,} obs')

def _anomaly_lh(df, q=0.30):
    lows, highs = [], []
    for dt, g in df.groupby('date'):
        g = g.dropna(subset=['vol_252d','fwd_ret_1m'])
        if len(g) < 50: continue
        n = max(1, int(len(g)*q))
        lo = g.nsmallest(n,'vol_252d')['fwd_ret_1m'].mean()
        hi = g.nlargest(n, 'vol_252d')['fwd_ret_1m'].mean()
        lows.append(lo); highs.append(hi)
    if not lows: return np.nan, np.nan, np.nan
    L = np.mean(lows); H = np.mean(highs)
    return L, H, L - H

PERIODS_A = [
    ('All',    None,         None),
    ('R1',     '2010-01-01', '2012-06-30'),
    ('R2',     '2012-07-01', '2019-12-31'),
    ('R3',     '2020-01-01', '2023-06-30'),
    ('R4',     '2023-07-01', '2025-12-31'),
]

print('\n' + '='*70)
print('Low-vol anomaly check — SP500 sample, σ_252d 기준 top/bottom 30%')
print('  L-H spread > 0  → low-vol anomaly 작동')
print('  L-H spread ≈ 0  → null')
print('  L-H spread < 0  → 역 anomaly (high-vol 우위)')
print('='*70)
print(f'{"period":<8s}{"Low avg":>12s}{"High avg":>12s}{"L-H spread":>14s}{"n months":>12s}')
print('-'*60)
for lbl, s, e in PERIODS_A:
    sub = panel
    if s: sub = sub[sub['date'] >= s]
    if e: sub = sub[sub['date'] <= e]
    L, H, spr = _anomaly_lh(sub)
    n_mo = sub['date'].nunique() if len(sub) > 0 else 0
    print(f'{lbl:<8s}{L*100:>+11.3f}%{H*100:>+11.3f}%{spr*100:>+13.3f}%{n_mo:>12d}')

print('\n→ SP500 2010-2025 에서 low-vol anomaly null (또는 negative spread).')
print('→ LSTM 의 portfolio 우위가 anomaly capture 로 인한 것이 아님을 시사 (spurious 가설 배제).')
'''

# ── [5] R1 catastrophic months ───────────────────────────────────────
md5 = """## [5] R1 catastrophic months 분해 (calendar 정렬)

**가설**: R1 LSTM 약세 (특히 mcap p_w) 는 단일 catastrophic month (2011-08 S&P downgrade) 가 dominant.

- pkl 의 ret 는 construction date 기준 indexed → +1 MonthEnd shift 해서 calendar (realization) date 로 정렬
- 45 슬롯 × R1 월별 LSTM-ANN return 차이 분해
- worst 5 months, top 5 worst 제거 시 R1 누적 gap 변화 확인
"""

code5 = r'''# ── [5] R1 catastrophic months 분해 (calendar 정렬) ─────────────
R1_START = pd.Timestamp('2010-01-31')
R1_END   = pd.Timestamp('2012-06-30')

PRIORS=['mcap','eq','rp']; PW=['mcap','eq','rp']; Q=['lam','raw','inv','vsp','fpm']

diff_data = {}
for pw in PW:
    for pr in PRIORS:
        for q in Q:
            L = f'mat_{pr}_{pw}_{q}_pap'; A = L + '_ann'
            if L not in loaded or A not in loaded: continue
            rL = loaded[L]['ret'].dropna()
            rA = loaded[A]['ret'].dropna()
            common = rL.index.intersection(rA.index)
            d = rL.loc[common] - rA.loc[common]
            d.index = d.index + pd.offsets.MonthEnd(1)
            mask = (d.index >= R1_START) & (d.index <= R1_END)
            diff_data[(pw, pr, q)] = d[mask]

df_diff = pd.DataFrame(diff_data)
print(f'R1 (calendar) 월 수: {len(df_diff)}, 슬롯 수: {df_diff.shape[1]}')

pw_avg  = {pw: df_diff[[c for c in df_diff.columns if c[0]==pw]].mean(axis=1) for pw in PW}
all_avg = df_diff.mean(axis=1)

print('\n' + '='*78)
print('R1 월별 LSTM-ANN return 차이 (p_w 별 15-슬롯 평균, calendar 기준)')
print('='*78)
hdr = f'{"month":<10s}' + ''.join(f'{pw:>14s}' for pw in PW) + f'{"All 45":>14s}'
print(hdr); print('-'*len(hdr))
for dt in df_diff.index:
    row = [pw_avg[pw][dt] for pw in PW]
    print(f'{dt.strftime("%Y-%m"):<10s}' + ''.join(f'{v:>+14.4f}' for v in row) + f'{all_avg[dt]:>+14.4f}')

print('\n' + '='*78)
print('p_w 별 LSTM 최악 월 — worst 5 (calendar month, mean d_ret across 15 slots)')
print('='*78)
for pw in PW:
    worst5 = pw_avg[pw].nsmallest(5)
    print(f'\n[p_w={pw}]')
    for dt, v in worst5.items():
        print(f'  {dt.strftime("%Y-%m"):<10s}: {v:+.4f}')

print('\n' + '='*78)
print('R1 cumulative LSTM-ANN gap — catastrophic months 의 기여도')
print('='*78)
print(f'{"제거 대상":<28s}{"mcap":>12s}{"eq":>12s}{"rp":>12s}')
print('-'*64)

scenarios = [
    ('전체 (제거 없음)',           []),
    ('2011-08 (S&P downgrade)',    ['2011-08']),
    ('2011-08 + 2011-04 + 2011-03', ['2011-08','2011-04','2011-03']),
]
for label, excl_ym in scenarios:
    cells = []
    for pw in PW:
        s = pw_avg[pw]
        if not excl_ym:
            total = s.sum()
        else:
            kept = s[~s.index.strftime('%Y-%m').isin(excl_ym)]
            total = kept.sum()
        cells.append(f'{total:+12.4f}')
    print(f'{label:<28s}' + ''.join(cells))

cells = []
for pw in PW:
    s = pw_avg[pw]
    worst5_idx = s.nsmallest(5).index
    kept = s.drop(worst5_idx)
    cells.append(f'{kept.sum():+12.4f}')
print(f'{"top 5 worst (per p_w) 제거":<28s}' + ''.join(cells))

fig, ax = plt.subplots(figsize=(13, 4.5))
for pw, color in zip(PW, ['#d62728', '#2ca02c', '#1f77b4']):
    ax.plot(pw_avg[pw].index, pw_avg[pw].values, label=f'p_w={pw}',
            color=color, lw=1.7, marker='o', markersize=4)
ax.axhline(0, color='black', lw=0.5)

events = [('2010-05','Flash Crash'),('2011-03','Tohoku eq'),('2011-08','S&P downgrade')]
ylim = ax.get_ylim()
for ym, lbl in events:
    matches = [dt for dt in pw_avg['mcap'].index if dt.strftime('%Y-%m') == ym]
    if matches:
        dt = matches[0]
        ax.axvline(dt, color='gray', ls='--', lw=0.8, alpha=0.7)
        ax.annotate(f'{ym}\n{lbl}', xy=(dt, ylim[0]*0.95),
                    fontsize=8, ha='center', color='gray')

ax.set_title('R1 월별 LSTM−ANN return 차이 (p_w 별, calendar 정렬)')
ax.set_ylabel('Monthly d_ret (LSTM − ANN)')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
'''

# ── [6] Q × regime grid ─────────────────────────────────────────────
md6 = """## [6] Q × regime × p_w timeline grid

regime 별로 5 Q 가 세로 stacked, 각 subplot 에 p_w 3 라인. event annotation 포함.

- 5 row (Q: lam/raw/inv/vsp/fpm) × 4 regime (R1/R2/R3/R4) figure
- 각 subplot 의 3 라인 = p_w (mcap/eq/rp) 의 3-prior 평균
- 모든 시간 축은 **calendar 정렬** (+1 MonthEnd shift)
"""

code6 = r'''# ── [6] Q × regime × p_w timeline (레짐별 분리 figure) ────────
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
colors = {'mcap':'#d62728', 'eq':'#2ca02c', 'rp':'#1f77b4'}

def build_series(start, end, q, pw):
    arr = []
    for pr in PRIORS:
        L = f'mat_{pr}_{pw}_{q}_pap'; A = L + '_ann'
        if L not in loaded or A not in loaded: continue
        rL = loaded[L]['ret'].dropna(); rA = loaded[A]['ret'].dropna()
        common = rL.index.intersection(rA.index)
        d = rL.loc[common] - rA.loc[common]
        d.index = d.index + pd.offsets.MonthEnd(1)
        arr.append(d[(d.index>=start)&(d.index<=end)])
    return pd.concat(arr, axis=1).mean(axis=1) if arr else pd.Series(dtype=float)

print('='*86)
print('Q × regime 월평균 LSTM−ANN return  (각 셀: 3 prior × 3 p_w = 9 슬롯 평균)')
print('='*86)
print(f'{"Q":<8s}' + ''.join(f'{r.split()[0]:>16s}' for r in REGIME_PERIODS) + f'{"all (10y)":>14s}')
print('-'*86)
for q in Q:
    cells = []
    for label, (s, e) in REGIME_PERIODS.items():
        vals = [build_series(s,e,q,pw).mean() for pw in PW]
        cells.append(np.mean(vals))
    all_vals = []
    for pw in PW:
        for pr in PRIORS:
            L = f'mat_{pr}_{pw}_{q}_pap'; A = L+'_ann'
            if L not in loaded or A not in loaded: continue
            rL = loaded[L]['ret'].dropna(); rA = loaded[A]['ret'].dropna()
            common = rL.index.intersection(rA.index)
            all_vals.append((rL.loc[common]-rA.loc[common]).mean())
    print(f'{q:<8s}' + ''.join(f'{v:>+16.4f}' for v in cells) + f'{np.mean(all_vals):>+14.4f}')

for rlabel, (s, e) in REGIME_PERIODS.items():
    series_by_q = {q: {pw: build_series(s, e, q, pw) for pw in PW} for q in Q}
    all_vals = [v for q in Q for pw in PW for v in series_by_q[q][pw].values]
    ymin, ymax = min(all_vals)*1.1, max(all_vals)*1.1

    fig, axes = plt.subplots(len(Q), 1, figsize=(13, 11), sharex=True)
    for i, q in enumerate(Q):
        ax = axes[i]
        for pw in PW:
            ser = series_by_q[q][pw]
            ax.plot(ser.index, ser.values, color=colors[pw], lw=1.3,
                    marker='o', markersize=3.5,
                    label=f'p_w={pw}' if i==0 else None)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel(Q_LABEL[q], fontsize=10)
        ax.grid(True, alpha=0.3)
        for ym, ev in REGIME_EVENTS.get(rlabel, []):
            matches = [dt for dt in series_by_q['lam']['mcap'].index if dt.strftime('%Y-%m') == ym]
            if matches:
                dt = matches[0]
                ax.axvline(dt, color='gray', ls='--', lw=0.8, alpha=0.6)
                if i == 0:
                    ax.annotate(f'{ym}\n{ev}', xy=(dt, ymax*0.85),
                                fontsize=8, ha='center', color='dimgray')

    axes[0].legend(loc='upper right', fontsize=9, ncol=3)
    fig.suptitle(f'{rlabel}  —  Monthly LSTM − ANN return  (3-prior 평균, p_w 색상)',
                  fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()
'''

# ── [7] fpm Q sign analysis ────────────────────────────────────────
md7 = """## [7] fpm Q 시계열 분석 — sign behavior

**목적**: σ-direct Q (lam/raw/inv/vsp) 와 fpm 의 본질적 차이를 데이터로 입증.

- σ-direct Q: λ scaling 으로 q 항상 양수 ("long low-vol / short high-vol" view 고정)
- fpm: FF3 OLS 평균으로 q 산출, **sign-flexible** (regime 따라 view 방향 바뀜)

**가설**:
- R4 (AI 랠리) 에서 fpm 우위 = q 가 음수로 → high-vol view 가 AI 종목 강세에 적합
- R3 에서 mcap-mcap-fpm 부진 = 위기에 high-vol view 가 손해

`mat_mcap_mcap_fpm_pap` 의 `meta['Q']` 시계열로 확인.
"""

code7 = r'''# ── [7] fpm Q 시계열 sign analysis ────────────────────────────
fpm_meta = loaded['mat_mcap_mcap_fpm_pap']['meta']
q_fpm = fpm_meta['Q'].copy()
q_fpm_cal = q_fpm.copy()
q_fpm_cal.index = q_fpm_cal.index + pd.offsets.MonthEnd(1)

REGIMES_T = {
    'R1 회복':   (pd.Timestamp('2010-02-28'), pd.Timestamp('2012-07-31')),
    'R2 확장':   (pd.Timestamp('2012-08-31'), pd.Timestamp('2020-01-31')),
    'R3 위기':   (pd.Timestamp('2020-02-29'), pd.Timestamp('2023-07-31')),
    'R4 정상화': (pd.Timestamp('2023-08-31'), pd.Timestamp('2026-01-31')),
}

print('='*80)
print('fpm Q value sign distribution by regime (calendar 정렬)')
print('  q > 0 = long low-vol / short high-vol view')
print('  q < 0 = long high-vol / short low-vol view (sign flip)')
print('='*80)
print(f'{"regime":<14s}{"n_mo":>6s}{"mean q":>12s}{"median":>12s}{"% q<0":>10s}{"min":>12s}{"max":>12s}')
print('-'*78)
all_neg = (q_fpm_cal < 0).mean()*100
print(f'{"All":<14s}{len(q_fpm_cal):>6d}{q_fpm_cal.mean():>+12.5f}{q_fpm_cal.median():>+12.5f}{all_neg:>9.1f}%{q_fpm_cal.min():>+12.5f}{q_fpm_cal.max():>+12.5f}')
print('-'*78)
for lbl, (s, e) in REGIMES_T.items():
    sub = q_fpm_cal[(q_fpm_cal.index>=s)&(q_fpm_cal.index<=e)]
    if len(sub)==0: continue
    neg_pct = (sub<0).mean()*100
    print(f'{lbl:<14s}{len(sub):>6d}{sub.mean():>+12.5f}{sub.median():>+12.5f}{neg_pct:>9.1f}%{sub.min():>+12.5f}{sub.max():>+12.5f}')

lam_meta = loaded['mat_mcap_mcap_lam_pap']['meta']
q_lam = lam_meta['Q'].copy()
q_lam_cal = q_lam.copy(); q_lam_cal.index = q_lam_cal.index + pd.offsets.MonthEnd(1)
print('\n참고: lam Q (σ-direct) 의 음수 비율')
print(f'  All: {(q_lam_cal<0).mean()*100:.1f}%, mean={q_lam_cal.mean():+.5f}')
print(f'  → σ-direct 는 항상 양수, fpm 만 sign-flexible')

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax = axes[0]
regime_colors = {'R1 회복':'#dddddd', 'R2 확장':'#f5f5f5', 'R3 위기':'#ffe0e0', 'R4 정상화':'#e0e8ff'}
ylim_buf = max(abs(q_fpm_cal.min()), abs(q_fpm_cal.max())) * 1.15
for lbl, (s, e) in REGIMES_T.items():
    ax.axvspan(s, e, color=regime_colors[lbl], alpha=0.6, zorder=0)
    mid = s + (e - s)/2
    ax.text(mid, ylim_buf*0.92, lbl, ha='center', fontsize=9, color='dimgray', fontweight='bold')
ax.plot(q_fpm_cal.index, q_fpm_cal.values, color='#1f77b4', lw=1.3, marker='o', markersize=2.5)
ax.axhline(0, color='red', lw=0.8, ls='--', alpha=0.7)
ax.set_ylabel('fpm q value')
ax.set_title('fpm Q timeline — calendar 정렬, regime shaded\n(red dashed = sign flip threshold)')
ax.set_ylim(-ylim_buf, ylim_buf)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(q_lam_cal.index, q_lam_cal.values, color='#2ca02c', lw=1.2, marker='s', markersize=2.5, label='lam q (σ-direct)', alpha=0.8)
ax.plot(q_fpm_cal.index, q_fpm_cal.values, color='#1f77b4', lw=1.2, marker='o', markersize=2.5, label='fpm q (FF3 OLS)', alpha=0.8)
ax.axhline(0, color='red', lw=0.8, ls='--', alpha=0.7)
for lbl, (s, e) in REGIMES_T.items():
    ax.axvspan(s, e, color=regime_colors[lbl], alpha=0.4, zorder=0)
ax.set_ylabel('q value')
ax.set_title('lam Q vs fpm Q timeline — lam 은 양수 유지, fpm 은 sign flip 빈번')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.show()
'''

def mk(md, code):
    return [
        {'cell_type':'markdown','id':uuid.uuid4().hex[:8],'metadata':{},
         'source': md.splitlines(keepends=True)},
        {'cell_type':'code','id':uuid.uuid4().hex[:8],'metadata':{},
         'execution_count':None,'outputs':[],
         'source': code.splitlines(keepends=True)},
    ]

for md, code in [(md4,code4),(md5,code5),(md6,code6),(md7,code7)]:
    nb['cells'].extend(mk(md, code))

json.dump(nb, open(nb_path,'w',encoding='utf-8'), indent=1, ensure_ascii=False)
print(f'Total cells now: {len(nb["cells"])}')
for i,c in enumerate(nb['cells']):
    src = ''.join(c['source'])[:60].replace(chr(10),' | ')
    print(f'{i:3d} {c["cell_type"]:<8s} {c.get("id",""):<12s} {src}')
