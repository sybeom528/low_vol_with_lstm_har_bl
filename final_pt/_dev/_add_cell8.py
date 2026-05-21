"""Add [8] ANN underestimation case study for 2020-08 and 2022-06."""
import json, uuid, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

nb_path = '99_main_analysis.ipynb'
nb = json.load(open(nb_path, encoding='utf-8'))

md = """## [8] ANN σ underestimation case study — 2020-08, 2022-06

**목적**: ANN 이 σ 를 잘못 예측해서 portfolio 손실로 이어진 구체적 종목 사례.

- 대상 month: **2020-08** (post-COVID recovery), **2022-06** (Fed hike) — [6] grid 에서 세 p_w 모두 LSTM 이 크게 우위 보인 month
- construction date: 각각 2020-07-31, 2022-05-31
- 분석: ANN 이 σ 를 underestimate (예측 < 실제) 한 top 10 종목, 해당 종목의 realized return 확인
- LSTM 의 같은 종목 예측 대조 — LSTM 이 제대로 high-vol 로 봤는지

main 본문에 들어가는 결과 아니고 **discussion / appendix evidence** 용.
"""

code = r'''# ── [8] ANN σ underestimation case study ──────────────────────
from pathlib import Path

_lstm_pred = pd.read_csv(DATA_DIR / '03b_lstm' / 'data' / 'ensemble_predictions_stockwise.csv',
                          parse_dates=['date'])[['date','ticker','y_true','y_pred_ensemble']]
_ann_pred  = pd.read_csv(DATA_DIR / 'paper_ann_predictions.csv',
                          parse_dates=['date'])[['date','ticker','y_true','y_pred_ensemble']]
_lstm_pred = _lstm_pred.replace([np.inf,-np.inf], np.nan).dropna()
_ann_pred  = _ann_pred .replace([np.inf,-np.inf], np.nan).dropna()

# LSTM: 월말 (각 (year-month, ticker) 의 last trading day)
_lstm_pred['ym'] = _lstm_pred['date'].dt.to_period('M')
_lstm_m = _lstm_pred.sort_values('date').groupby(['ym','ticker'], as_index=False).last()
_ann_pred['ym'] = _ann_pred['date'].dt.to_period('M')

# monthly panel 에서 realized fwd 1m return 가져오기 (construction date 기준 ret_1m_next)
_panel = pd.read_csv(DATA_DIR / 'monthly_panel.csv', parse_dates=['date'])
_panel['ym'] = _panel['date'].dt.to_period('M')
_panel = _panel[['ym','ticker','fwd_ret_1m']].dropna()

# (date, ticker) merge: LSTM + ANN 예측 + realized fwd ret
_merged = pd.merge(
    _lstm_m[['ym','ticker','y_pred_ensemble','y_true']].rename(columns={'y_pred_ensemble':'sig_lstm','y_true':'sig_real'}),
    _ann_pred[['ym','ticker','y_pred_ensemble']].rename(columns={'y_pred_ensemble':'sig_ann'}),
    on=['ym','ticker'], how='inner',
)
_merged = pd.merge(_merged, _panel, on=['ym','ticker'], how='inner')

CASES = [
    ('2020-08', '2020-07'),   # 2020-07 construction → 2020-08 realization
    ('2022-06', '2022-05'),
]

for real_ym, ctor_ym in CASES:
    print('='*88)
    print(f'CASE: realization month = {real_ym}  (construction = {ctor_ym})')
    print('='*88)
    sub = _merged[_merged['ym'] == pd.Period(ctor_ym, freq='M')].copy()
    if len(sub) < 50:
        print(f'  n={len(sub)} — too few obs, skip')
        continue
    # ANN underestimation error = realized σ − ANN 예측 σ (양수 = ANN 이 너무 낮게 예측)
    sub['ann_err'] = sub['sig_real'] - sub['sig_ann']
    sub['lstm_err'] = sub['sig_real'] - sub['sig_lstm']
    # top 10 ANN underestimation
    top10 = sub.nlargest(10, 'ann_err')[
        ['ticker','sig_lstm','sig_ann','sig_real','ann_err','lstm_err','fwd_ret_1m']
    ]
    print(f'\nTop 10 ANN underestimation (predicted σ << realized σ)')
    print(f'{"ticker":<8s}{"σ_LSTM":>10s}{"σ_ANN":>10s}{"σ_real":>10s}{"ANN_err":>10s}{"LSTM_err":>10s}{"ret_"+real_ym[-2:]:>10s}')
    print('-'*70)
    for _, r in top10.iterrows():
        print(f'{r["ticker"]:<8s}{r["sig_lstm"]:>10.4f}{r["sig_ann"]:>10.4f}{r["sig_real"]:>10.4f}'
              f'{r["ann_err"]:>+10.4f}{r["lstm_err"]:>+10.4f}{r["fwd_ret_1m"]*100:>+9.2f}%')
    print(f'\n  mean ret top10 ANN underest: {top10["fwd_ret_1m"].mean()*100:+.2f}%')
    print(f'  mean ANN err  top10: {top10["ann_err"].mean():+.4f}')
    print(f'  mean LSTM err top10: {top10["lstm_err"].mean():+.4f}')
    print(f'  → ANN 이 더 크게 underestimate 한 종목들에서 LSTM 의 예측 오차가 얼마나 작은지 확인')

    # 시각화: σ_ANN vs σ_real scatter, 점 크기 = |ann_err|, 색 = ret
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    sc = ax.scatter(sub['sig_ann'], sub['sig_real'],
                     c=sub['fwd_ret_1m']*100, cmap='RdYlGn', alpha=0.6,
                     s=20+abs(sub['ann_err'])*200, edgecolor='gray', lw=0.3)
    lim = max(sub['sig_ann'].max(), sub['sig_real'].max())*1.05
    ax.plot([0,lim],[0,lim], 'k--', lw=0.8, alpha=0.5, label='perfect (y=x)')
    for _, r in top10.iterrows():
        ax.annotate(r['ticker'], xy=(r['sig_ann'], r['sig_real']),
                    fontsize=7, color='black', alpha=0.8)
    ax.set_xlabel('σ_ANN (predicted)')
    ax.set_ylabel('σ_real (realized)')
    ax.set_title(f'ANN prediction vs realized σ — {real_ym}\n(label = top 10 underest, color = next 1m return)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label='next 1m return %')

    ax = axes[1]
    sc2 = ax.scatter(sub['sig_lstm'], sub['sig_real'],
                      c=sub['fwd_ret_1m']*100, cmap='RdYlGn', alpha=0.6,
                      s=20+abs(sub['lstm_err'])*200, edgecolor='gray', lw=0.3)
    ax.plot([0,lim],[0,lim], 'k--', lw=0.8, alpha=0.5, label='perfect (y=x)')
    for _, r in top10.iterrows():
        ax.annotate(r['ticker'], xy=(r['sig_lstm'], r['sig_real']),
                    fontsize=7, color='black', alpha=0.8)
    ax.set_xlabel('σ_LSTM (predicted)')
    ax.set_ylabel('σ_real (realized)')
    ax.set_title(f'LSTM prediction vs realized σ — {real_ym}\n(same top 10 tickers labeled)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc2, ax=ax, label='next 1m return %')

    plt.tight_layout()
    plt.show()
'''

new_cells = [
    {'cell_type':'markdown','id':uuid.uuid4().hex[:8],'metadata':{},
     'source': md.splitlines(keepends=True)},
    {'cell_type':'code','id':uuid.uuid4().hex[:8],'metadata':{},
     'execution_count':None,'outputs':[],
     'source': code.splitlines(keepends=True)},
]
nb['cells'].extend(new_cells)
json.dump(nb, open(nb_path,'w',encoding='utf-8'), indent=1, ensure_ascii=False)
print(f'Appended [8]. Total cells: {len(nb["cells"])}')
