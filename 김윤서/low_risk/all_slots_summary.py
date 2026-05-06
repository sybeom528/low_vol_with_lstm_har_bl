"""
4 슬롯(prior, q, p_weight, omega) × 옵션별 효과 종합 정리

- 모든 .pkl 결과 로드 → full + 3 서브기간 Sharpe 매트릭스
- 슬롯별 OFAT(One Factor At A Time) 비교
- 출력:
    김윤서/low_risk/outputs/slots_summary/
      - all_results.csv         : 모든 실험 결과 (config + Sharpe)
      - prior_ofat.csv          : prior 슬롯 OFAT
      - q_ofat.csv              : q 슬롯 OFAT
      - p_weight_ofat.csv       : p_weight 슬롯 OFAT
      - omega_ofat.csv          : omega 슬롯 OFAT
      - heatmap_full.png        : 전체기간 히트맵 (정렬)
      - heatmap_2020_2024.png   : 2020-2024 히트맵
"""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); matplotlib.set_loglevel('error')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / 'final'
OUTDIR = Path(__file__).resolve().parent / 'outputs' / 'slots_summary'
OUTDIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(FINAL))
from bl_config import EXPERIMENTS  # noqa

SUBP = [('full', None, None),
        ('2010-2014', '2010-01-01', '2014-12-31'),
        ('2015-2019', '2015-01-01', '2019-12-31'),
        ('2020-2024', '2020-01-01', '2024-12-31')]

def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 6 or r.std() == 0: return np.nan
    return float(r.mean() / r.std() * np.sqrt(12))

# ── 1. 전체 결과 로드 ──────────────────────────────────────────
rows = []
for cfg in EXPERIMENTS:
    p = FINAL / 'results' / f"{cfg['name']}.pkl"
    if not p.exists(): continue
    with open(p, 'rb') as f: d = pickle.load(f)
    r = d['ret']
    rec = {
        'name'      : cfg['name'],
        'prior'     : cfg.get('prior', 'capm_mcap'),
        'p_mode'    : cfg.get('p_mode', 'trailing_vol21'),
        'p_weight'  : cfg.get('p_weight', 'mcap'),
        'q_mode'    : cfg.get('q_mode', 'fixed'),
        'omega_mode': cfg.get('omega_mode', 'he_litterman'),
        'omega_scale': cfg.get('omega_scale', 1.0),
    }
    for label, s_, e_ in SUBP:
        sub = r if label == 'full' else r.loc[s_:e_]
        rec[f'sharpe_{label}'] = sharpe(sub)
        rec[f'mean_{label}']   = float(sub.mean()) * 12 if len(sub) else np.nan
        rec[f'std_{label}']    = float(sub.std()) * np.sqrt(12) if len(sub) else np.nan
    rows.append(rec)

df = pd.DataFrame(rows)
df.to_csv(OUTDIR / 'all_results.csv', index=False)
print(f'[1] 로드: {len(df)} 실험')

# ── 2. omega_mode 정규화 (scaled_half/double 분리) ────────────
def omega_label(row):
    if row['omega_mode'] == 'scaled':
        return f"scaled_{int(row['omega_scale']*10):d}" if row['omega_scale'] != 1.0 else 'scaled'
    return row['omega_mode']
df['omega_label'] = df.apply(omega_label, axis=1)
df['omega_label'] = df['omega_label'].replace({
    'scaled_5': 'scaled_half', 'scaled_20': 'scaled_double',
    'ff3_paper': 'paper', 'he_litterman': 'he'
})

# ── 3. OFAT: 어떤 슬롯이든 다른 슬롯이 동일하면 비교 가능 ────────
KEYS = ['prior', 'p_mode', 'p_weight', 'q_mode', 'omega_label']

def ofat_table(df, slot, output_path):
    """slot 외 다른 KEYS가 동일한 그룹별로 slot별 Sharpe 변화 추출."""
    other_keys = [k for k in KEYS if k != slot]
    groups = df.groupby(other_keys, dropna=False)
    rows = []
    for grp_key, sub in groups:
        if len(sub) < 2: continue
        rec = dict(zip(other_keys, grp_key))
        for _, r in sub.iterrows():
            for period in ['full', '2010-2014', '2015-2019', '2020-2024']:
                rec[f"{r[slot]}__{period}"] = round(r[f'sharpe_{period}'], 3)
        rec['n_options'] = len(sub)
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)
    return out

ofat_prior = ofat_table(df, 'prior', OUTDIR/'prior_ofat.csv')
ofat_q     = ofat_table(df, 'q_mode', OUTDIR/'q_ofat.csv')
ofat_pw    = ofat_table(df, 'p_weight', OUTDIR/'p_weight_ofat.csv')
ofat_om    = ofat_table(df, 'omega_label', OUTDIR/'omega_ofat.csv')

print(f'[2] OFAT: prior {len(ofat_prior)}, q {len(ofat_q)}, p_weight {len(ofat_pw)}, omega {len(ofat_om)} 그룹')

# ── 4. 슬롯별 평균 효과 (모든 그룹 평균, 모든 기간) ────────────
def avg_by_slot(df, slot):
    return df.groupby(slot)[[f'sharpe_{p}' for p in ['full','2010-2014','2015-2019','2020-2024']]].agg(['mean','std','min','max','count']).round(3)

print('\n=== prior 슬롯 평균 효과 ==='); print(avg_by_slot(df, 'prior'))
print('\n=== q_mode 슬롯 평균 효과 ==='); print(avg_by_slot(df, 'q_mode'))
print('\n=== p_weight 슬롯 평균 효과 ==='); print(avg_by_slot(df, 'p_weight'))
print('\n=== omega_label 슬롯 평균 효과 ==='); print(avg_by_slot(df, 'omega_label'))

# ── 5. Reference base 정의: prior_eq + LSTM + p=mcap + Ω=he ────
# Q 슬롯 비교용
ref_base = (df['prior']=='capm_eq') & (df['p_mode']=='lstm_predicted') & (df['p_weight']=='mcap') & (df['omega_label']=='he')
print('\n=== Q 슬롯 OFAT (prior=eq, LSTM, p=mcap, Ω=he 고정) ===')
sub = df[ref_base][['name','q_mode','sharpe_full','sharpe_2010-2014','sharpe_2015-2019','sharpe_2020-2024']]
sub.to_csv(OUTDIR/'q_at_ref.csv', index=False)
print(sub.to_string(index=False))

# Ω 슬롯 비교용 (q=raw_lam 고정)
ref_om = (df['prior']=='capm_eq') & (df['p_mode']=='lstm_predicted') & (df['p_weight']=='mcap') & (df['q_mode']=='raw_lam')
print('\n=== Ω 슬롯 OFAT (prior=eq, LSTM, p=mcap, q=raw_lam 고정) ===')
sub = df[ref_om][['name','omega_label','sharpe_full','sharpe_2010-2014','sharpe_2015-2019','sharpe_2020-2024']]
sub.to_csv(OUTDIR/'omega_at_ref.csv', index=False)
print(sub.to_string(index=False))

# P_weight 비교용 (prior=eq + LSTM + q=raw_lam + Ω=he 고정)
ref_pw = (df['prior']=='capm_eq') & (df['p_mode']=='lstm_predicted') & (df['q_mode']=='raw_lam') & (df['omega_label']=='he')
print('\n=== P_weight 슬롯 OFAT (prior=eq, LSTM, q=raw_lam, Ω=he 고정) ===')
sub = df[ref_pw][['name','p_weight','sharpe_full','sharpe_2010-2014','sharpe_2015-2019','sharpe_2020-2024']]
sub.to_csv(OUTDIR/'pweight_at_ref.csv', index=False)
print(sub.to_string(index=False))

# Prior 비교용 (LSTM + p=mcap + q=raw_lam + Ω=he 고정)
# 단 prior_rp는 q=raw_lam 조합이 없으면 q=fixed 등 다른 변수도 같아야 함. 가능한 것만 보자.
print('\n=== Prior 슬롯 OFAT (LSTM, p=mcap, q=fixed/raw_lam/lambda 별로) ===')
for q_target in ['fixed', 'raw_lam', 'lambda']:
    ref = (df['p_mode']=='lstm_predicted') & (df['p_weight']=='mcap') & (df['q_mode']==q_target) & (df['omega_label']=='he')
    sub = df[ref][['name','prior','sharpe_full','sharpe_2010-2014','sharpe_2015-2019','sharpe_2020-2024']]
    if len(sub) > 1:
        print(f'\n  [q={q_target}]')
        print(sub.to_string(index=False))

# 추가: trailing vol P + p=mcap + q=fixed + Ω=he 고정 → baseline / prior_eq / prior_rp
print('\n=== Prior 슬롯 OFAT (trailing, p=mcap, q=fixed, Ω=he) ===')
ref = (df['p_mode']=='trailing_vol21') & (df['p_weight']=='mcap') & (df['q_mode']=='fixed') & (df['omega_label']=='he')
sub = df[ref][['name','prior','sharpe_full','sharpe_2010-2014','sharpe_2015-2019','sharpe_2020-2024']]
print(sub.to_string(index=False))

# ── 6. 전체기간 Sharpe 히트맵 (top 30) ─────────────────────────
top = df.sort_values('sharpe_full', ascending=False).head(30)
fig, ax = plt.subplots(figsize=(13, 7))
mat = top[[f'sharpe_{p}' for p in ['2010-2014','2015-2019','2020-2024','full']]].values
im = ax.imshow(mat, cmap='RdYlGn', vmin=0.5, vmax=2.0, aspect='auto')
ax.set_yticks(range(len(top))); ax.set_yticklabels(top['name'], fontsize=8)
ax.set_xticks(range(4)); ax.set_xticklabels(['2010-14','2015-19','2020-24','full'])
for i in range(len(top)):
    for j in range(4):
        v = mat[i,j]
        if pd.notna(v):
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8, color='black' if 0.7<v<1.7 else 'white')
ax.set_title('Top-30 실험 — 서브기간별 Sharpe (full 기준 내림차순)')
plt.colorbar(im); plt.tight_layout()
plt.savefig(OUTDIR/'heatmap_top30.png', dpi=130); plt.close()

# 2020-2024 히트맵
top2024 = df.sort_values('sharpe_2020-2024', ascending=False).head(20)
fig, ax = plt.subplots(figsize=(13, 6))
mat = top2024[[f'sharpe_{p}' for p in ['2010-2014','2015-2019','2020-2024','full']]].values
im = ax.imshow(mat, cmap='RdYlGn', vmin=0.5, vmax=2.0, aspect='auto')
ax.set_yticks(range(len(top2024))); ax.set_yticklabels(top2024['name'], fontsize=8)
ax.set_xticks(range(4)); ax.set_xticklabels(['2010-14','2015-19','2020-24','full'])
for i in range(len(top2024)):
    for j in range(4):
        v = mat[i,j]
        if pd.notna(v):
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8, color='black' if 0.7<v<1.7 else 'white')
ax.set_title('Top-20 by 2020-2024 Sharpe — 강세장 robust 조합')
plt.colorbar(im); plt.tight_layout()
plt.savefig(OUTDIR/'heatmap_top2024.png', dpi=130); plt.close()

print(f'\n✓ 출력: {OUTDIR}')
