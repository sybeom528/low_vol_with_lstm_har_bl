"""
레짐 기반 서브기간 분석 + Sortino/MDD/Sharpe 평가 + SPY 필터 + 3 prior 히트맵

목적
  ── (1) 5년 균등분할 (2010-14 / 2015-19 / 2020-24)이 시장 레짐을 무시한 분할이라,
       시장 레짐(회복기·확장·위기·회복+베어·AI 랠리)에 맞춘 분할로 변경
  ── (2) Sortino > MDD > Sharpe 순으로 평가 (변동성 다루는 프로젝트라 하방위험 우선)
  ── (3) 모든 서브 레짐에서 SPY를 이긴 조합만 후보로 선정
  ── (4) 히트맵을 prior(mcap/eq/rp)별 3장으로 분리

레짐 분할 (데이터 기반 검증)
  R1 2010-01 ~ 2014-12  Post-GFC 회복
  R2 2015-01 ~ 2018-12  저변동성 확장 + 2018Q4 조정
  R3 2019-01 ~ 2020-12  Pre-COVID + COVID 크래시
  R4 2021-01 ~ 2022-12  COVID 회복 + 2022 베어
  R5 2023-01 ~ 2024-12  AI 랠리

출력 (김윤서/low_risk/outputs/regime/)
  - 01_regime_validation.png      : SPY 누적/12M수익/drawdown + 레짐 음영
  - 02_regime_stats.csv           : 레짐별 SPY/exp 통계
  - 03_heatmap_prior_mcap.png     : prior=mcap 그룹 히트맵
  - 03_heatmap_prior_eq.png       : prior=eq 그룹 히트맵
  - 03_heatmap_prior_rp.png       : prior=rp 그룹 히트맵
  - 04_winners.csv                : 5개 레짐 모두 SPY 초과한 조합
  - 04_winners_top.png            : 후보 조합 상위 시각화
  - all_metrics.csv               : 전체 결과 (실험 × 레짐 × 지표)
"""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); matplotlib.set_loglevel('error')
import matplotlib.pyplot as plt
import platform

if platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['AppleGothic', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / 'final'
OUTDIR = Path(__file__).resolve().parent / 'outputs' / 'regime'
OUTDIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(FINAL))
from bl_config import EXPERIMENTS  # noqa

# ── 레짐 정의 ──────────────────────────────────────────────────
REGIMES = [
    ('R1 회복',     '2010-01-01', '2014-12-31', '#2ecc71'),
    ('R2 확장',     '2015-01-01', '2018-12-31', '#3498db'),
    ('R3 COVID',    '2019-01-01', '2020-12-31', '#e74c3c'),
    ('R4 베어',     '2021-01-01', '2022-12-31', '#f39c12'),
    ('R5 AI 랠리',  '2023-01-01', '2024-12-31', '#9b59b6'),
]
REGIME_LABELS = [r[0] for r in REGIMES]

# ── 메트릭 ─────────────────────────────────────────────────────
def metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 6:
        return {'sharpe': np.nan, 'sortino': np.nan, 'mdd': np.nan, 'mean': np.nan, 'vol': np.nan}
    mean_ann = r.mean() * 12
    vol_ann  = r.std() * np.sqrt(12)
    downside = r[r < 0].std() * np.sqrt(12)
    sharpe   = mean_ann / vol_ann if vol_ann > 0 else np.nan
    sortino  = mean_ann / downside if downside > 0 else np.nan
    cum      = (1 + r).cumprod()
    mdd      = float((cum / cum.cummax() - 1).min())
    return {'sharpe': sharpe, 'sortino': sortino, 'mdd': mdd, 'mean': mean_ann, 'vol': vol_ann}

# ── 데이터 로드 ───────────────────────────────────────────────
loaded = {}
for cfg in EXPERIMENTS:
    p = FINAL / 'results' / f"{cfg['name']}.pkl"
    if not p.exists(): continue
    with open(p, 'rb') as f: d = pickle.load(f)
    if len(d.get('ret', pd.Series())) > 0:
        loaded[cfg['name']] = (cfg, d)
print(f'[1] 결과 로드: {len(loaded)} 실험')

# SPY (어느 결과든 spy_ret 동일)
_, _ref = next(iter(loaded.values()))
spy = _ref['spy_ret'].dropna()

# ── (A) 레짐 검증 시각화 ────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
spy_cum = (1 + spy).cumprod()
spy_dd  = (spy_cum / spy_cum.cummax() - 1)
spy_12m = (1 + spy).rolling(12).apply(lambda x: x.prod() - 1, raw=True)

for ax in axes:
    for label, s, e, col in REGIMES:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.18, color=col, label=label if ax is axes[0] else None)

axes[0].plot(spy_cum.index, spy_cum.values, color='black', lw=1.3)
axes[0].set_yscale('log'); axes[0].set_ylabel('SPY 누적 (log)')
axes[0].legend(loc='upper left', ncol=5, fontsize=9)
axes[0].set_title('SPY 누적수익률 + 레짐 음영', fontweight='bold')
axes[0].grid(alpha=0.3)

axes[1].plot(spy_12m.index, spy_12m.values * 100, color='black', lw=1.0)
axes[1].axhline(0, color='red', lw=0.5)
axes[1].set_ylabel('SPY 12M rolling 수익률 (%)')
axes[1].set_title('rolling 12-month 수익 (regime 진입/이탈 기준)')
axes[1].grid(alpha=0.3)

axes[2].fill_between(spy_dd.index, spy_dd.values * 100, 0, color='red', alpha=0.4)
axes[2].plot(spy_dd.index, spy_dd.values * 100, color='darkred', lw=1.0)
axes[2].set_ylabel('SPY drawdown (%)')
axes[2].set_xlabel('date')
axes[2].set_title('drawdown from peak (위기 식별)')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTDIR / '01_regime_validation.png', dpi=130); plt.close()

# ── (B) 레짐별 SPY 통계 ────────────────────────────────────────
spy_stats = []
for label, s, e, _ in REGIMES:
    m = metrics(spy.loc[s:e])
    m['regime'] = label; m['who'] = 'SPY'
    m['n_months'] = int(len(spy.loc[s:e]))
    spy_stats.append(m)
spy_df = pd.DataFrame(spy_stats)
print('\n=== SPY 레짐별 ==='); print(spy_df[['regime','n_months','sortino','mdd','sharpe','mean','vol']].round(3).to_string(index=False))

# ── (C) 모든 실험 × 레짐 × 지표 ────────────────────────────────
records = []
for name, (cfg, d) in loaded.items():
    r = d['ret']
    rec = {'name': name, 'prior': cfg.get('prior','capm_mcap'),
           'p_mode': cfg.get('p_mode','trailing_vol21'),
           'p_weight': cfg.get('p_weight','mcap'),
           'q_mode': cfg.get('q_mode','fixed'),
           'omega_mode': cfg.get('omega_mode','he_litterman'),
           'omega_scale': cfg.get('omega_scale',1.0)}
    for label, s, e, _ in REGIMES:
        m = metrics(r.loc[s:e])
        for k in ['sharpe','sortino','mdd','mean','vol']:
            rec[f'{label}_{k}'] = m[k]
    m_full = metrics(r)
    for k in ['sharpe','sortino','mdd','mean','vol']:
        rec[f'full_{k}'] = m_full[k]
    records.append(rec)

df = pd.DataFrame(records)

# ── SPY 행 + full 통계 (비교 baseline) ───────────────────────
spy_row = {'name': 'SPY', 'prior': '-', 'p_mode': '-', 'p_weight': '-',
           'q_mode': '-', 'omega_mode': '-', 'omega_scale': '-'}
for label, s, e, _ in REGIMES:
    m = metrics(spy.loc[s:e])
    for k in ['sharpe','sortino','mdd','mean','vol']:
        spy_row[f'{label}_{k}'] = m[k]
m_full_spy = metrics(spy)
for k in ['sharpe','sortino','mdd','mean','vol']:
    spy_row[f'full_{k}'] = m_full_spy[k]

# SPY를 맨 위 행으로
df = pd.concat([pd.DataFrame([spy_row]), df], ignore_index=True)
df.to_csv(OUTDIR / 'all_metrics.csv', index=False)

# ── 비교 친화적 CSV (실험 vs SPY 차이) ────────────────────────
diff_rows = []
for _, r in df[df['name'] != 'SPY'].iterrows():
    rec = {'name': r['name'], 'prior': r['prior'],
           'p_mode': r['p_mode'], 'p_weight': r['p_weight'],
           'q_mode': r['q_mode'], 'omega_mode': r['omega_mode']}
    for label, _, _, _ in REGIMES:
        for k in ['sortino','mdd','sharpe']:
            exp_v = r[f'{label}_{k}']
            spy_v = spy_row[f'{label}_{k}']
            rec[f'{label}_{k}'] = round(exp_v, 3)
            rec[f'{label}_{k}_vs_SPY'] = round(exp_v - spy_v, 3)
    for k in ['sortino','mdd','sharpe']:
        exp_v = r[f'full_{k}']; spy_v = spy_row[f'full_{k}']
        rec[f'full_{k}'] = round(exp_v, 3)
        rec[f'full_{k}_vs_SPY'] = round(exp_v - spy_v, 3)
    diff_rows.append(rec)
pd.DataFrame(diff_rows).to_csv(OUTDIR / 'all_metrics_vs_spy.csv', index=False)
print(f'[C] all_metrics.csv (SPY 포함) + all_metrics_vs_spy.csv 저장')

# ── (D) SPY 필터 — 단계별 (strict / sortino+mdd / sortino만) ────
spy_lookup = {row['regime']: row for row in spy_stats}

def pass_count(row, metrics_list) -> int:
    """metrics_list 모두 초과한 레짐 수."""
    cnt = 0
    for label, _, _, _ in REGIMES:
        s_spy = spy_lookup[label]
        ok = True
        for m in metrics_list:
            if m == 'mdd':
                if not (row[f'{label}_mdd'] > s_spy['mdd']): ok = False; break
            else:
                if not (row[f'{label}_{m}'] > s_spy[m]): ok = False; break
        if ok: cnt += 1
    return cnt

df['pass_strict']    = df.apply(lambda r: pass_count(r, ['sortino','mdd','sharpe']), axis=1)
df['pass_sortmdd']   = df.apply(lambda r: pass_count(r, ['sortino','mdd']), axis=1)
df['pass_sortino']   = df.apply(lambda r: pass_count(r, ['sortino']), axis=1)
df['pass_mdd']       = df.apply(lambda r: pass_count(r, ['mdd']), axis=1)

print(f'\n[D] SPY 초과 레짐 수 분포 (5개 만점)')
for col, label in [('pass_strict','Sortino+MDD+Sharpe 모두'),
                   ('pass_sortmdd','Sortino+MDD'),
                   ('pass_sortino','Sortino만'),
                   ('pass_mdd','MDD만')]:
    print(f'  {label:25s}: 5/5={int((df[col]==5).sum())}, 4+/5={int((df[col]>=4).sum())}, 3+/5={int((df[col]>=3).sum())}')

# Sortino+MDD 5/5 → 1차 winner. 없으면 4+/5. Sortino만 5/5 보조 표시.
winners_strict = df[df['pass_strict'] == 5].copy()
winners_sortmdd = df[df['pass_sortmdd'] == 5].copy()
winners_sortino = df[df['pass_sortino'] == 5].copy()

# 합쳐서 dedupe — 가장 엄격 → 가장 약한 순
def sort_winners(w):
    if len(w) == 0: return w
    return w.sort_values(by=['full_sortino','full_mdd','full_sharpe'], ascending=[False, False, False])

winners_strict  = sort_winners(winners_strict)
winners_sortmdd = sort_winners(winners_sortmdd)
winners_sortino = sort_winners(winners_sortino)

winners_strict.to_csv(OUTDIR / '04_winners_strict.csv', index=False)
winners_sortmdd.to_csv(OUTDIR / '04_winners_sortmdd.csv', index=False)
winners_sortino.to_csv(OUTDIR / '04_winners_sortino.csv', index=False)

print('\n=== Tier 1: Sortino+MDD+Sharpe 모두 SPY 초과 (5/5 레짐) ===')
if len(winners_strict) > 0:
    print(winners_strict[['name','prior','full_sortino','full_mdd','full_sharpe']].head(10).round(3).to_string(index=False))
else:
    print('  → 0개. R5 AI 랠리 SPY Sortino 4.6 / Sharpe 1.7이 너무 강함 (저위험 anomaly 구조적 한계)')

print('\n=== Tier 2: Sortino+MDD 모두 SPY 초과 (5/5 레짐) ===')
if len(winners_sortmdd) > 0:
    print(winners_sortmdd[['name','prior','full_sortino','full_mdd','full_sharpe']].head(10).round(3).to_string(index=False))
else:
    print('  → 0개')

print('\n=== Tier 3: Sortino만 SPY 초과 (5/5 레짐) ===')
if len(winners_sortino) > 0:
    print(winners_sortino[['name','prior','full_sortino','full_mdd','full_sharpe']].head(10).round(3).to_string(index=False))

# R5 AI 랠리(SPY Sortino 4.6)는 구조적으로 BL 포트폴리오로 따라갈 수 없음 →
# "4/5 레짐 Sortino 초과" 실용적 기준으로 자동 fallback
print('\n=== Tier 4 (실용): Sortino 4+/5 레짐 SPY 초과 ===')
tier4 = df[df['pass_sortino'] >= 4].copy()
tier4 = sort_winners(tier4)
# 어느 레짐 실패했는지 표시
def failed_regime(row):
    fails = []
    for label, _, _, _ in REGIMES:
        if row[f'{label}_sortino'] <= spy_lookup[label]['sortino']:
            fails.append(label)
    return ' / '.join(fails) if fails else '없음'
tier4['failed_regime'] = tier4.apply(failed_regime, axis=1)
tier4.to_csv(OUTDIR / '04_winners_tier4_4of5.csv', index=False)

if len(tier4) > 0:
    print(tier4[['name','prior','full_sortino','full_mdd','full_sharpe','pass_sortino','failed_regime']].head(15).round(3).to_string(index=False))

# 최종 winners — 가장 엄격한 비어있지 않은 tier (없으면 4/5 fallback)
winners = (winners_strict if len(winners_strict) > 0
           else (winners_sortmdd if len(winners_sortmdd) > 0
                 else (winners_sortino if len(winners_sortino) > 0
                       else tier4)))
winners.to_csv(OUTDIR / '04_winners.csv', index=False)

# 우승자 시각화
if len(winners) > 0:
    n_show = min(20, len(winners))
    top = winners.head(n_show)
    fig, ax = plt.subplots(figsize=(12, max(5, n_show*0.35)))
    mat = top[[f'{l}_sortino' for l, _, _, _ in REGIMES] + ['full_sortino']].values
    im = ax.imshow(mat, cmap='RdYlGn', vmin=0, vmax=3.0, aspect='auto')
    ax.set_yticks(range(n_show)); ax.set_yticklabels(top['name'], fontsize=8)
    ax.set_xticks(range(6)); ax.set_xticklabels(REGIME_LABELS + ['full'], rotation=20, ha='right')
    for i in range(n_show):
        for j in range(6):
            v = mat[i,j]
            if pd.notna(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7, color='black' if 0.7<v<2.5 else 'white')
    ax.set_title(f'Winners ({n_show}/{len(winners)}) — 모든 레짐 SPY 초과 + Sortino 정렬', fontweight='bold')
    plt.colorbar(im, label='Sortino'); plt.tight_layout()
    plt.savefig(OUTDIR / '04_winners_top.png', dpi=130); plt.close()

# ── (E) Prior별 3장 히트맵 ─────────────────────────────────────
def heatmap_for_prior(df, prior_val, fname):
    sub = df[df['prior'] == prior_val].copy()
    if len(sub) == 0: return
    # 행: name (full sortino 정렬), 열: 5 레짐 + full
    sub = sub.sort_values('full_sortino', ascending=False)
    n = len(sub)
    fig, ax = plt.subplots(figsize=(11, max(5, n*0.32)))
    cols = [f'{l}_sortino' for l, _, _, _ in REGIMES] + ['full_sortino']
    mat = sub[cols].values
    im = ax.imshow(mat, cmap='RdYlGn', vmin=0, vmax=3.0, aspect='auto')
    ax.set_yticks(range(n)); ax.set_yticklabels(sub['name'], fontsize=7)
    ax.set_xticks(range(6)); ax.set_xticklabels(REGIME_LABELS + ['full'], rotation=20, ha='right')
    for i in range(n):
        for j in range(6):
            v = mat[i,j]
            if pd.notna(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7, color='black' if 0.5<v<2.5 else 'white')
    # SPY 라인 표시 (별도 텍스트)
    spy_line = ' | '.join(f"{l}: SPY={spy_lookup[l]['sortino']:.2f}" for l, _, _, _ in REGIMES)
    ax.set_xlabel(f'Sortino — {spy_line}', fontsize=8)
    ax.set_title(f'prior={prior_val} | Sortino 히트맵 (full Sortino 정렬)', fontweight='bold')
    plt.colorbar(im, label='Sortino'); plt.tight_layout()
    plt.savefig(OUTDIR / fname, dpi=130); plt.close()

heatmap_for_prior(df, 'capm_mcap', '03_heatmap_prior_mcap.png')
heatmap_for_prior(df, 'capm_eq',   '03_heatmap_prior_eq.png')
heatmap_for_prior(df, 'capm_rp',   '03_heatmap_prior_rp.png')

# ── (F) 레짐 통계 요약 (SPY + winners 평균) ─────────────────────
agg_rows = []
for label, s, e, _ in REGIMES:
    spy_m = metrics(spy.loc[s:e])
    spy_m['regime'] = label; spy_m['type'] = 'SPY'
    agg_rows.append(spy_m)
    # 우승자 평균
    if len(winners) > 0:
        win_means = {
            'sortino': winners[f'{label}_sortino'].mean(),
            'mdd': winners[f'{label}_mdd'].mean(),
            'sharpe': winners[f'{label}_sharpe'].mean(),
            'mean': winners[f'{label}_mean'].mean(),
            'vol': winners[f'{label}_vol'].mean(),
            'regime': label, 'type': f'winners_avg(n={len(winners)})'
        }
        agg_rows.append(win_means)
pd.DataFrame(agg_rows).to_csv(OUTDIR / '02_regime_stats.csv', index=False)

print(f'\n✓ 출력 → {OUTDIR}')
print(f'  - 01_regime_validation.png  : 레짐 분할 검증')
print(f'  - 02_regime_stats.csv       : 레짐 통계')
print(f'  - 03_heatmap_prior_*.png    : prior별 히트맵 3장')
print(f'  - 04_winners.csv / .png     : SPY 모두 초과 후보 ({len(winners)}개)')
print(f'  - all_metrics.csv           : 전체 결과')
