"""
다중 지표 기반 레짐 판별 EDA

지표
  ── 변동성 ──
  - VIX (level, 12M rolling z-score)
  - SPY realized vol (12M rolling annualized)
  - 시장 cross-sectional 평균 vol_21d
  - 시장 cross-sectional 평균 vol_60d
  - 시장 cross-sectional vol dispersion (std)
  ── 가격/수익 ──
  - SPY 누적
  - SPY 12M rolling return
  - SPY drawdown from peak
  ── 매크로 ──
  - 10Y-2Y yield curve spread (음수 = 역전)
  - HY spread (BAA10Y, fred_data 있을 시)

판별 기준 (학계/실무 표준)
  - VIX < 15        : 저변동 정상
  - 15 ≤ VIX < 25   : 정상
  - 25 ≤ VIX < 35   : 스트레스
  - VIX ≥ 35        : 위기
  - SPY DD > 10%    : 조정/베어
  - T10Y2Y < 0      : 수익률 곡선 역전 (경기침체 선행)

출력 (김윤서/low_risk/outputs/regime_detection/)
  - 01_indicators_full.png   : 전 지표 패널 시각화
  - 02_regime_breakpoints.png: 제안 레짐 + 음영
  - 03_vix_regime_dist.png   : VIX 분포 + 레짐별 평균
  - regime_indicators.csv    : 월별 모든 지표
  - regime_proposal.md       : 각 breakpoint 정당화 표
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); matplotlib.set_loglevel('error')
import matplotlib.pyplot as plt
import platform

if platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['AppleGothic', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = Path(__file__).resolve().parent / 'outputs' / 'regime_detection'
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── 데이터 로드 ──────────────────────────────────────────────
panel = pd.read_csv(ROOT/'final/data/monthly_panel.csv', parse_dates=['date'])
panel = panel.set_index(['date','ticker']).sort_index()
spy = panel['spy_ret'].groupby(level='date').first().dropna()

macro = pd.read_csv(ROOT/'data/macro_daily.csv', parse_dates=['date'])
macro = macro.set_index('date').sort_index()
# 월말 리샘플
macro_m = macro.resample('ME').last()

# fred 추가 (HY spread BAA10Y가 있는지 확인)
fred_path = ROOT/'김윤서/data/fred_data.csv'
if fred_path.exists():
    fred = pd.read_csv(fred_path, parse_dates=['Date']).rename(columns={'Date':'date'}).set_index('date').sort_index()
    fred_m = fred.resample('ME').last()
else:
    fred_m = None

# ── 월별 지표 계산 ────────────────────────────────────────────
START, END = '2010-01-31', '2024-12-31'

# SPY 기반
spy_idx = spy.loc[START:END]
ind = pd.DataFrame(index=spy_idx.index)
ind['spy_ret']    = spy_idx
ind['spy_cum']    = (1 + spy_idx).cumprod()
ind['spy_dd']     = ind['spy_cum']/ind['spy_cum'].cummax() - 1
ind['spy_6m_ret'] = (1 + spy_idx).rolling(6).apply(lambda x: x.prod()-1, raw=True)
ind['spy_12m_ret']= (1 + spy_idx).rolling(12).apply(lambda x: x.prod()-1, raw=True)
ind['spy_12m_vol']= spy_idx.rolling(12).std() * np.sqrt(12)

# macro에서 VIX, t10y2y
mm = macro_m.reindex(ind.index, method='ffill')
ind['vix']       = mm['vix']
ind['vix_chg']   = ind['vix'].diff()
ind['vix_z12']   = (ind['vix'] - ind['vix'].rolling(12).mean()) / ind['vix'].rolling(12).std()
ind['t10y2y']    = mm['t10y2y']
ind['skew']      = mm['skew_idx']

# fred BAA10Y (HY spread)
if fred_m is not None and 'BAA10Y' in fred_m.columns:
    fm = fred_m.reindex(ind.index, method='ffill')
    ind['hy_spread'] = fm['BAA10Y']

# 패널 cross-sectional vol
xsec = panel.groupby(level='date').agg(
    vol21_avg=('vol_21d','mean'),
    vol21_std=('vol_21d','std'),
    vol60_avg=('vol_60d','mean'),
    vol60_std=('vol_60d','std'),
)
xsec = xsec.reindex(ind.index)
ind['vol21_avg'] = xsec['vol21_avg']
ind['vol21_std'] = xsec['vol21_std']
ind['vol60_avg'] = xsec['vol60_avg']
ind['vol60_std'] = xsec['vol60_std']

ind.to_csv(OUTDIR/'regime_indicators.csv')
print(f'[1] 지표 계산: {len(ind)}개월, 컬럼 {len(ind.columns)}개')

# ── (A) 다지표 패널 시각화 ─────────────────────────────────────
fig, axes = plt.subplots(7, 1, figsize=(14, 16), sharex=True)

# 1. SPY 누적 + drawdown
ax = axes[0]
ax2 = ax.twinx()
ax.plot(ind.index, ind['spy_cum'], color='black', lw=1.3, label='SPY 누적')
ax2.fill_between(ind.index, ind['spy_dd']*100, 0, color='red', alpha=0.3, label='Drawdown')
ax.set_yscale('log'); ax.set_ylabel('SPY 누적 (log)')
ax2.set_ylabel('DD (%)', color='darkred'); ax2.set_ylim(-50, 5)
ax.set_title('① SPY 누적 + Drawdown', fontweight='bold'); ax.grid(alpha=0.3)

# 2. SPY 6M / 12M rolling return
ax = axes[1]
ax.plot(ind.index, ind['spy_6m_ret']*100, label='6M return', alpha=0.7)
ax.plot(ind.index, ind['spy_12m_ret']*100, label='12M return', alpha=0.9, lw=1.3)
ax.axhline(0, color='red', lw=0.5)
ax.set_ylabel('수익률 (%)'); ax.legend(); ax.grid(alpha=0.3)
ax.set_title('② SPY rolling 수익률', fontweight='bold')

# 3. VIX
ax = axes[2]
ax.plot(ind.index, ind['vix'], color='purple', lw=1.2)
ax.axhline(15, color='green', ls='--', lw=0.8, label='15 (저변동)')
ax.axhline(25, color='orange', ls='--', lw=0.8, label='25 (스트레스)')
ax.axhline(35, color='red', ls='--', lw=0.8, label='35 (위기)')
ax.set_ylabel('VIX'); ax.legend(loc='upper right'); ax.grid(alpha=0.3)
ax.set_title('③ VIX (CBOE Volatility Index)', fontweight='bold')

# 4. VIX 12M z-score
ax = axes[3]
ax.fill_between(ind.index, ind['vix_z12'], 0,
                where=(ind['vix_z12']>0), color='red', alpha=0.4, label='평균 이상')
ax.fill_between(ind.index, ind['vix_z12'], 0,
                where=(ind['vix_z12']<=0), color='green', alpha=0.4, label='평균 이하')
ax.axhline(0, color='black', lw=0.5)
ax.set_ylabel('VIX 12M z-score'); ax.legend(); ax.grid(alpha=0.3)
ax.set_title('④ VIX 정상화 (12M rolling z-score)', fontweight='bold')

# 5. SPY realized vol + cross-sectional avg vol
ax = axes[4]
ax.plot(ind.index, ind['spy_12m_vol']*100, color='black', label='SPY 12M vol', lw=1.2)
ax.plot(ind.index, ind['vol21_avg']*100*np.sqrt(252), color='steelblue',
        alpha=0.7, label='시장 평균 vol_21d (annualized)')
ax.plot(ind.index, ind['vol60_avg']*100*np.sqrt(252), color='navy',
        alpha=0.7, label='시장 평균 vol_60d (annualized)')
ax.set_ylabel('변동성 (%)'); ax.legend(); ax.grid(alpha=0.3)
ax.set_title('⑤ 실현 변동성 (top-down + bottom-up)', fontweight='bold')

# 6. Cross-sectional vol dispersion
ax = axes[5]
ax.plot(ind.index, ind['vol21_std']*100*np.sqrt(252), color='teal', label='vol_21d std')
ax.plot(ind.index, ind['vol60_std']*100*np.sqrt(252), color='darkblue', label='vol_60d std')
ax.set_ylabel('Cross-sec std (%)'); ax.legend(); ax.grid(alpha=0.3)
ax.set_title('⑥ 종목 간 변동성 분산 (vol of vol)', fontweight='bold')

# 7. Yield curve
ax = axes[6]
ax.fill_between(ind.index, ind['t10y2y'], 0,
                where=(ind['t10y2y']<0), color='red', alpha=0.4, label='역전')
ax.fill_between(ind.index, ind['t10y2y'], 0,
                where=(ind['t10y2y']>=0), color='green', alpha=0.3, label='정상')
ax.plot(ind.index, ind['t10y2y'], color='black', lw=1.0)
ax.axhline(0, color='red', lw=0.6)
ax.set_ylabel('T10Y - T2Y (%)'); ax.set_xlabel('date')
ax.legend(); ax.grid(alpha=0.3)
ax.set_title('⑦ 수익률 곡선 (10Y - 2Y) — 음수 = 침체 선행', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTDIR/'01_indicators_full.png', dpi=130); plt.close()
print('[2] 7-panel 지표 시각화 저장')

# ── (B) 자동 regime breakpoint 탐지 ──────────────────────────
# 규칙 기반: VIX > 25 (3개월+ 지속) OR SPY DD > 15% → 스트레스
ind['stress'] = ((ind['vix'] > 25) | (ind['spy_dd'] < -0.15)).astype(int)
# 3개월 rolling — 진입/이탈 노이즈 제거
ind['stress_smooth'] = ind['stress'].rolling(3).mean() > 0.5

# Bull market: drawdown < 5% AND 12M return > 5%
ind['bull'] = ((ind['spy_dd'] > -0.05) & (ind['spy_12m_ret'] > 0.05)).astype(int)

# 자동 breakpoint: stress 진입/이탈 시점
state = ind['stress_smooth'].astype(int)
transitions = state.diff().fillna(0)
breakpoints = ind.index[transitions != 0].tolist()
print(f'\n[3] 자동 탐지 breakpoint ({len(breakpoints)}개):')
for d in breakpoints:
    transition = '진입' if transitions.loc[d] > 0 else '이탈'
    print(f'  {d.strftime("%Y-%m")} 스트레스 {transition} (VIX={ind.loc[d,"vix"]:.1f}, DD={ind.loc[d,"spy_dd"]*100:.1f}%)')

# ── (C) 데이터 기반 6 레짐 제안 ────────────────────────────────
# 검증된 분기점 (학계/실무 통설 + 데이터 일치):
PROPOSED = [
    ('R1 회복',     '2010-01-01', '2011-04-30', '#2ecc71', 'Post-GFC 회복기, VIX 평균 22'),
    ('R2 유럽위기', '2011-05-01', '2012-12-31', '#e67e22', 'PIIGS 부채위기, VIX 25+ 스트레스'),
    ('R3 저변동 확장','2013-01-01','2014-12-31', '#3498db', '미국 회복 본격화, VIX 14 평균'),
    ('R4 다중충격', '2015-01-01', '2016-06-30', '#f39c12', '유가폭락 + 中위안 평가절하 + Brexit'),
    ('R5 골디락스', '2016-07-01', '2018-01-31', '#1abc9c', '동시성장, VIX 사상 최저'),
    ('R6 변동성 회귀','2018-02-01','2019-12-31', '#95a5a6', 'Vol-mageddon + 18Q4 + 19 Fed dovish'),
    ('R7 COVID',    '2020-01-01', '2020-12-31', '#e74c3c', 'COVID 크래시 + 회복'),
    ('R8 인플레/베어','2021-01-01','2022-12-31', '#c0392b', '재개+버블, 22년 Fed hike + 베어'),
    ('R9 AI 랠리',  '2023-01-01', '2024-12-31', '#9b59b6', 'Mag 7 주도, T10Y2Y 역전'),
]

# 5 레짐 제안 (단순화 버전 — 너무 잘게 나누면 통계력 부족)
SIMPLE_5 = [
    ('R1 회복',     '2010-01-01', '2012-12-31', '#2ecc71', 'Post-GFC 회복 + 11유럽위기 (VIX avg 22)'),
    ('R2 저변동 확장','2013-01-01','2014-12-31', '#3498db', '미국 본격회복 (VIX avg 14)'),
    ('R3 다중충격', '2015-01-01', '2019-12-31', '#f39c12', '유가/中/Vol-mageddon/18Q4 (VIX avg 16)'),
    ('R4 COVID+베어','2020-01-01','2022-12-31', '#e74c3c', 'COVID + 재개버블 + 22 베어 (VIX avg 23)'),
    ('R5 AI 랠리',  '2023-01-01', '2024-12-31', '#9b59b6', 'Mag 7 강세 (VIX avg 15)'),
]

# 9 / 5 레짐 비교 시각화
def plot_proposal(regimes, title, save_name):
    fig, ax = plt.subplots(figsize=(14, 5))
    for label, s, e, col, _ in regimes:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.25, color=col, label=label)
    ax.plot(ind.index, ind['spy_cum'], color='black', lw=1.3)
    ax.set_yscale('log')
    ax.set_ylabel('SPY 누적 (log)')
    ax.set_title(title, fontweight='bold', fontsize=13)
    ax.legend(loc='upper left', ncol=3 if len(regimes) > 5 else 5, fontsize=8)
    ax.grid(alpha=0.3)
    # 각 레짐 평균 VIX/DD 텍스트
    for label, s, e, col, _ in regimes:
        s_ts, e_ts = pd.Timestamp(s), pd.Timestamp(e)
        sub = ind.loc[s_ts:e_ts]
        if len(sub) > 0:
            mid = s_ts + (e_ts - s_ts) / 2
            txt = f"VIX{sub['vix'].mean():.0f}\nDD{sub['spy_dd'].min()*100:.0f}%"
            ax.text(mid, ind['spy_cum'].max() * 1.15, txt,
                    ha='center', va='bottom', fontsize=8, alpha=0.7)
    plt.tight_layout()
    plt.savefig(OUTDIR/save_name, dpi=130); plt.close()

plot_proposal(SIMPLE_5,  '제안 5-레짐 (현재 사용 중) + SPY 누적', '02a_5regime.png')
plot_proposal(PROPOSED,  '데이터 기반 9-레짐 (세분화) + SPY 누적', '02b_9regime.png')

# ── (D) 레짐별 통계 테이블 ─────────────────────────────────────
def regime_stats(regimes):
    rows = []
    for label, s, e, _, desc in regimes:
        sub = ind.loc[s:e]
        rows.append({
            'regime': label, 'desc': desc,
            'months': len(sub),
            'vix_avg': sub['vix'].mean(),
            'vix_max': sub['vix'].max(),
            'spy_ret_ann': sub['spy_ret'].mean()*12,
            'spy_dd_max': sub['spy_dd'].min(),
            'spy_vol_ann': sub['spy_ret'].std()*np.sqrt(12),
            'vol21_avg_ann': sub['vol21_avg'].mean()*np.sqrt(252),
            't10y2y_avg': sub['t10y2y'].mean(),
            't10y2y_min': sub['t10y2y'].min(),
        })
    return pd.DataFrame(rows)

stats_5 = regime_stats(SIMPLE_5)
stats_9 = regime_stats(PROPOSED)

print('\n=== 5-레짐 통계 ==='); print(stats_5.round(3).to_string(index=False))
print('\n=== 9-레짐 통계 ==='); print(stats_9.round(3).to_string(index=False))
stats_5.to_csv(OUTDIR/'stats_5regime.csv', index=False)
stats_9.to_csv(OUTDIR/'stats_9regime.csv', index=False)

# ── (E) 5/9 레짐 분리력 측정: 레짐 간 VIX 분포 ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, regimes, title in [(axes[0], SIMPLE_5, '5-레짐'), (axes[1], PROPOSED, '9-레짐')]:
    data = []
    labels = []
    for label, s, e, col, _ in regimes:
        sub = ind.loc[s:e, 'vix'].dropna()
        if len(sub) > 0:
            data.append(sub.values); labels.append(label)
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
    cols = [r[3] for r in regimes if len(ind.loc[r[1]:r[2], 'vix'].dropna()) > 0]
    for patch, col in zip(bp['boxes'], cols): patch.set_facecolor(col); patch.set_alpha(0.5)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.axhline(15, color='green', ls='--', lw=0.5)
    ax.axhline(25, color='orange', ls='--', lw=0.5)
    ax.axhline(35, color='red', ls='--', lw=0.5)
    ax.set_ylabel('VIX'); ax.set_title(f'{title} VIX 분포 (분리력 측정)', fontweight='bold')
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTDIR/'03_vix_regime_dist.png', dpi=130); plt.close()

# ── (F) 정당화 마크다운 ────────────────────────────────────────
md = ['# 레짐 분할 정당화\n',
      '## 사용 지표', '- VIX (level + 12M z-score)', '- SPY 누적/drawdown/12M rolling', '- 시장 cross-sectional vol_21d/vol_60d (mean + std)', '- T10Y-2Y yield curve', '',
      '## 5-레짐 (운용용, 통계력 우선)', '']
md.append('| 레짐 | 기간 | 개월 | VIX avg | VIX max | SPY 연수익 | DD max | T10Y2Y avg | 근거 |')
md.append('|---|---|---:|---:|---:|---:|---:|---:|---|')
for _, r in stats_5.iterrows():
    md.append(f"| **{r['regime']}** | — | {int(r['months'])} | {r['vix_avg']:.1f} | {r['vix_max']:.1f} | "
              f"{r['spy_ret_ann']*100:.1f}% | {r['spy_dd_max']*100:.1f}% | {r['t10y2y_avg']:.2f} | {r['desc']} |")

md.append('\n## 9-레짐 (정밀 분석용, 세분화)\n')
md.append('| 레짐 | 기간 | 개월 | VIX avg | VIX max | SPY 연수익 | DD max | T10Y2Y avg | 근거 |')
md.append('|---|---|---:|---:|---:|---:|---:|---:|---|')
for _, r in stats_9.iterrows():
    md.append(f"| **{r['regime']}** | — | {int(r['months'])} | {r['vix_avg']:.1f} | {r['vix_max']:.1f} | "
              f"{r['spy_ret_ann']*100:.1f}% | {r['spy_dd_max']*100:.1f}% | {r['t10y2y_avg']:.2f} | {r['desc']} |")

(OUTDIR/'regime_proposal.md').write_text('\n'.join(md))

print(f'\n✓ 출력 → {OUTDIR}')
print('  - 01_indicators_full.png   (7-panel 지표)')
print('  - 02a_5regime.png / 02b_9regime.png (제안 시각화)')
print('  - 03_vix_regime_dist.png   (분리력 측정)')
print('  - regime_indicators.csv    (월별 모든 지표)')
print('  - stats_5regime.csv / stats_9regime.csv (통계)')
print('  - regime_proposal.md       (정당화 표)')
