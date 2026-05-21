"""Replace [8] with bucket-disagreement based case study."""
import json, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

nb_path = '99_main_analysis.ipynb'
nb = json.load(open(nb_path, encoding='utf-8'))

new_md = """## [8] LSTM vs ANN bucket disagreement case study — 2020-08, 2022-06

**목적**: BL view 는 "long low-vol / short high-vol". 즉 portfolio 효과는 **σ 절대 예측 오차가 아니라 bucket classification (low/high) 의 차이** 에서 발생.

- 대상 month: **2020-08** (post-COVID recovery), **2022-06** (Fed hike) — 세 p_w 모두 LSTM 우위
- construction date: 2020-07-31, 2022-05-31
- 분석: 각 모델의 σ ranking 으로 **bottom 30% (low bucket, long view), top 30% (high bucket, short view)** 분류
- 두 모델이 **다른 bucket 으로 분류한 disagreement 종목** 추출
- 그 종목들의 realized return 으로 누구의 분류가 맞았는지 확인

main 본문 아니고 **discussion / appendix evidence**.
"""

new_code = r'''# ── [8] LSTM vs ANN bucket disagreement case study ─────────────
_lstm_pred = pd.read_csv(DATA_DIR / '03b_lstm' / 'data' / 'ensemble_predictions_stockwise.csv',
                          parse_dates=['date'])[['date','ticker','y_true','y_pred_ensemble']]
_ann_pred  = pd.read_csv(DATA_DIR / 'paper_ann_predictions.csv',
                          parse_dates=['date'])[['date','ticker','y_true','y_pred_ensemble']]
_lstm_pred = _lstm_pred.replace([np.inf,-np.inf], np.nan).dropna()
_ann_pred  = _ann_pred .replace([np.inf,-np.inf], np.nan).dropna()

_lstm_pred['ym'] = _lstm_pred['date'].dt.to_period('M')
_lstm_m = _lstm_pred.sort_values('date').groupby(['ym','ticker'], as_index=False).last()
_ann_pred['ym'] = _ann_pred['date'].dt.to_period('M')

_panel = pd.read_csv(DATA_DIR / 'monthly_panel.csv', parse_dates=['date'])
_panel['ym'] = _panel['date'].dt.to_period('M')
_panel = _panel[['ym','ticker','fwd_ret_1m']].dropna()

_merged = pd.merge(
    _lstm_m[['ym','ticker','y_pred_ensemble','y_true']].rename(columns={'y_pred_ensemble':'sig_lstm','y_true':'sig_real'}),
    _ann_pred[['ym','ticker','y_pred_ensemble']].rename(columns={'y_pred_ensemble':'sig_ann'}),
    on=['ym','ticker'], how='inner',
)
_merged = pd.merge(_merged, _panel, on=['ym','ticker'], how='inner')

CASES = [
    ('2020-08', '2020-07'),
    ('2022-06', '2022-05'),
]
Q_BUCKET = 0.30   # bottom/top 30%

for real_ym, ctor_ym in CASES:
    print('='*92)
    print(f'CASE: realization month = {real_ym}  (construction = {ctor_ym})')
    print('='*92)
    sub = _merged[_merged['ym'] == pd.Period(ctor_ym, freq='M')].copy()
    n = len(sub)
    if n < 50:
        print(f'  n={n} — too few obs, skip'); continue
    n_bucket = max(1, int(n*Q_BUCKET))

    # bucket assignment (rank-based)
    sub['rank_lstm'] = sub['sig_lstm'].rank()
    sub['rank_ann']  = sub['sig_ann'].rank()
    lstm_lo_thr = sub['rank_lstm'].quantile(Q_BUCKET)
    lstm_hi_thr = sub['rank_lstm'].quantile(1-Q_BUCKET)
    ann_lo_thr  = sub['rank_ann'].quantile(Q_BUCKET)
    ann_hi_thr  = sub['rank_ann'].quantile(1-Q_BUCKET)

    sub['lstm_bk'] = np.where(sub['rank_lstm']<=lstm_lo_thr, 'low',
                       np.where(sub['rank_lstm']>=lstm_hi_thr, 'high', 'mid'))
    sub['ann_bk']  = np.where(sub['rank_ann']<=ann_lo_thr, 'low',
                       np.where(sub['rank_ann']>=ann_hi_thr, 'high', 'mid'))

    # 4 disagreement groups
    groups = {
        'LSTM-high & ANN-low  (LSTM short, ANN long)':  (sub['lstm_bk']=='high') & (sub['ann_bk']=='low'),
        'LSTM-low & ANN-high  (LSTM long, ANN short)':  (sub['lstm_bk']=='low')  & (sub['ann_bk']=='high'),
        'both-low (agree, long bucket)':                  (sub['lstm_bk']=='low')  & (sub['ann_bk']=='low'),
        'both-high (agree, short bucket)':                (sub['lstm_bk']=='high') & (sub['ann_bk']=='high'),
    }

    print(f'\n  n_obs={n}, bucket size = {n_bucket} (Q={Q_BUCKET})')
    print(f'\n  Group disagreement / agreement counts & realized fwd return mean:')
    print(f'{"group":<55s}{"n":>5s}{"mean σ_real":>13s}{"mean ret":>12s}')
    print('-'*85)
    for gname, mask in groups.items():
        g = sub[mask]
        if len(g)==0: continue
        print(f'{gname:<55s}{len(g):>5d}{g["sig_real"].mean():>13.4f}{g["fwd_ret_1m"].mean()*100:>+11.2f}%')

    # 핵심: ANN 이 long-bucket 으로 잘못 분류한 종목 (LSTM-high & ANN-low)
    bad = sub[(sub['lstm_bk']=='high') & (sub['ann_bk']=='low')].sort_values('fwd_ret_1m').copy()
    print(f'\n  [핵심] ANN 이 LONG 으로 잘못 분류한 종목 (LSTM-high & ANN-low)')
    print(f'         → ANN portfolio 가 long → 실제 폭락 시 ANN 손실')
    print(f'  {"ticker":<8s}{"σ_LSTM":>10s}{"σ_ANN":>10s}{"σ_real":>10s}{"rank_LSTM":>11s}{"rank_ANN":>11s}{"ret":>10s}')
    print('  '+'-'*72)
    for _, r in bad.head(10).iterrows():
        print(f'  {r["ticker"]:<8s}{r["sig_lstm"]:>10.4f}{r["sig_ann"]:>10.4f}{r["sig_real"]:>10.4f}'
              f'{int(r["rank_lstm"]):>11d}{int(r["rank_ann"]):>11d}{r["fwd_ret_1m"]*100:>+9.2f}%')

    # 반대: ANN 이 short-bucket 으로 잘못 분류한 종목 (LSTM-low & ANN-high)
    bad2 = sub[(sub['lstm_bk']=='low') & (sub['ann_bk']=='high')].sort_values('fwd_ret_1m', ascending=False).copy()
    print(f'\n  [참고] ANN 이 SHORT 으로 잘못 분류한 종목 (LSTM-low & ANN-high)')
    print(f'         → ANN portfolio 가 short → 실제 상승 시 ANN 손실 (또는 기회손실)')
    print(f'  {"ticker":<8s}{"σ_LSTM":>10s}{"σ_ANN":>10s}{"σ_real":>10s}{"rank_LSTM":>11s}{"rank_ANN":>11s}{"ret":>10s}')
    print('  '+'-'*72)
    for _, r in bad2.head(10).iterrows():
        print(f'  {r["ticker"]:<8s}{r["sig_lstm"]:>10.4f}{r["sig_ann"]:>10.4f}{r["sig_real"]:>10.4f}'
              f'{int(r["rank_lstm"]):>11d}{int(r["rank_ann"]):>11d}{r["fwd_ret_1m"]*100:>+9.2f}%')

    # 시각화
    fig, ax = plt.subplots(figsize=(11, 6.5))
    color_map = {'mid':'#cccccc',
                  'L-low_A-low':'#2ca02c',   # both-low (agree, green)
                  'L-high_A-high':'#1f77b4', # both-high (agree, blue)
                  'L-high_A-low':'#d62728',  # disagreement: ANN long but LSTM short (red — ANN bad if crashed)
                  'L-low_A-high':'#ff7f0e'}  # disagreement: ANN short but LSTM long (orange)
    def _grp(r):
        if r['lstm_bk']=='mid' or r['ann_bk']=='mid': return 'mid'
        return f'L-{r["lstm_bk"]}_A-{r["ann_bk"]}'
    sub['grp'] = sub.apply(_grp, axis=1)
    for grp, c in color_map.items():
        g = sub[sub['grp']==grp]
        ax.scatter(g['rank_lstm'], g['rank_ann'], c=c, s=20+abs(g['fwd_ret_1m']*200), alpha=0.6,
                   edgecolor='gray', lw=0.3, label=f'{grp} (n={len(g)})')
    # disagreement 종목 label
    label_set = pd.concat([bad.head(7), bad2.head(7)])
    for _, r in label_set.iterrows():
        ax.annotate(r['ticker'], xy=(r['rank_lstm'], r['rank_ann']), fontsize=7, color='black', alpha=0.85)
    ax.axhline(ann_lo_thr, color='gray', ls='--', lw=0.5)
    ax.axhline(ann_hi_thr, color='gray', ls='--', lw=0.5)
    ax.axvline(lstm_lo_thr, color='gray', ls='--', lw=0.5)
    ax.axvline(lstm_hi_thr, color='gray', ls='--', lw=0.5)
    ax.set_xlabel('rank by σ_LSTM (low → high)')
    ax.set_ylabel('rank by σ_ANN (low → high)')
    ax.set_title(f'LSTM vs ANN σ ranking — {real_ym}  (construction={ctor_ym})\npoint size ∝ |return|, label = top disagreement')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
'''

# Replace [8] (last 2 cells)
# Find ids of the last md+code cells (the [8] we just added)
md_idx = None; code_idx = None
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    if c['cell_type'] == 'markdown' and '## [8]' in src:
        md_idx = i
    elif c['cell_type'] == 'code' and '── [8]' in src:
        code_idx = i

if md_idx is None or code_idx is None:
    print(f'ERROR: cannot find [8] cells (md={md_idx}, code={code_idx})')
else:
    nb['cells'][md_idx]['source'] = new_md.splitlines(keepends=True)
    nb['cells'][code_idx]['source'] = new_code.splitlines(keepends=True)
    nb['cells'][code_idx]['outputs'] = []
    nb['cells'][code_idx]['execution_count'] = None
    json.dump(nb, open(nb_path,'w',encoding='utf-8'), indent=1, ensure_ascii=False)
    print(f'Replaced [8] at md={md_idx}, code={code_idx}')
