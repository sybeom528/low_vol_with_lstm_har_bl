"""
master_table.py — 149개 실험을 한 DataFrame으로 통합.

results/*.pkl 전부 로드 → config dict 파싱 → 표준 토큰 + 성과지표.
파일명 그대로 유지하고, 분석은 canonical 컬럼 기준으로 통일.

사용:
    from master_table import build_master_table, PERIODS_DEFAULT
    mt = build_master_table(RESULTS_DIR, rf, spy_ret, periods=PERIODS_DEFAULT)
    mt.sort_values('sharpe', ascending=False).head(20)

    # 슬롯 필터:
    mt[(mt.prior_s=='eq') & (mt.q_s=='lam')]

    # 매트릭스 셀 조회:
    mt[mt.canonical=='eq_ls_rp_lam_he']
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

from bl_functions import compute_metrics


# ── 슬롯 → 약어 매핑 (canonical 토큰) ─────────────────────────────────
PRIOR_SHORT = {
    'capm_mcap': 'mcap',
    'capm_eq'  : 'eq',
    'capm_rp'  : 'rp',
}
PMODE_SHORT = {
    'trailing_vol21' : 'tr',
    'trailing_vol252': 'tr252',
    'lstm_predicted' : 'ls',
}
PWEIGHT_SHORT = {
    'mcap'    : 'mcap',
    'eq'      : 'eq',
    'rp'      : 'rp',
    'vol_mcap': 'volm',
}
QMODE_SHORT = {
    'fixed'           : 'fix',
    'lambda'          : 'lam',
    'inv_lambda'      : 'inv',
    'raw_lam'         : 'raw',
    'vol_spread'      : 'vsp',
    'ff3_paper'       : 'ff3',
    'ff3_regression'  : 'ff3reg',
    'realized_spread' : 'rsp',
    'regime'          : 'reg',
    'hrp'             : 'hrp',
    'capm'            : 'capm',
    'none'            : 'none',
}
OMEGA_SHORT = {
    'he_litterman': 'he',
    'rmse'        : 'rms',
    'ff3_paper'   : 'pap',
    # 'scaled'는 _omega_short()에서 scale 값 보고 sh/sd 부여
}


def _omega_short(omega_mode: str, omega_scale: float) -> str:
    """omega_mode + omega_scale → 약어."""
    if omega_mode == 'scaled':
        if abs(omega_scale - 0.5) < 1e-9:
            return 'sh'
        if abs(omega_scale - 2.0) < 1e-9:
            return 'sd'
        return f'sc{omega_scale}'
    return OMEGA_SHORT.get(omega_mode, omega_mode)


def parse_config(cfg: dict) -> dict:
    """config dict → 슬롯 + 약어 + canonical name."""
    prior   = cfg.get('prior',       'capm_mcap')
    p_mode  = cfg.get('p_mode',      'trailing_vol21')
    p_wt    = cfg.get('p_weight',    'mcap')
    q_mode  = cfg.get('q_mode',      'fixed')
    om_mode = cfg.get('omega_mode',  'he_litterman')
    om_scl  = cfg.get('omega_scale', 1.0)

    pr_s = PRIOR_SHORT.get(prior,   prior)
    pm_s = PMODE_SHORT.get(p_mode,  p_mode)
    pw_s = PWEIGHT_SHORT.get(p_wt,  p_wt)
    q_s  = QMODE_SHORT.get(q_mode,  q_mode)
    om_s = _omega_short(om_mode, om_scl)

    return {
        # 원본 슬롯
        'prior'      : prior,
        'p_mode'     : p_mode,
        'p_weight'   : p_wt,
        'q_mode'     : q_mode,
        'omega_mode' : om_mode,
        'omega_scale': om_scl,
        'tc'         : cfg.get('tc',         0.001),
        'max_weight' : cfg.get('max_weight', 0.10),
        # 약어 토큰
        'prior_s'    : pr_s,
        'p_s'        : pm_s,
        'pw_s'       : pw_s,
        'q_s'        : q_s,
        'om_s'       : om_s,
        # canonical: prior_pmode_pweight_q_omega
        'canonical'  : f'{pr_s}_{pm_s}_{pw_s}_{q_s}_{om_s}',
    }


# ── 5-레짐 정의 (김윤서/low_risk/regime_analysis.py 기준) ──────────────
# 본 프로젝트 단일 기간 시스템. mt와 rt 모두 동일 레짐 사용.
# Sortino/MDD/Sharpe 변동성 평가 — 거시환경 차이가 뚜렷한 시장 국면.
REGIMES_5 = [
    ('R1_회복',    '2010-01-01', '2014-12-31'),  # Post-GFC + 유럽위기
    ('R2_확장',    '2015-01-01', '2018-12-31'),  # 미국 회복 + 다중충격
    ('R3_COVID',   '2019-01-01', '2020-12-31'),  # 코로나 크래시 + 재개
    ('R4_베어',    '2021-01-01', '2022-12-31'),  # 인플레 + 22년 베어
    ('R5_AI랠리',  '2023-01-01', '2024-12-31'),  # Mag7 강세
]

# 호환용 별칭 — 옛 PERIODS_DEFAULT 자리에 REGIMES_5 dict 형식 노출
PERIODS_DEFAULT = {label: (s, e) for label, s, e in REGIMES_5}


def _sharpe_subperiod(ret: pd.Series, rf: pd.Series, start: str, end: str) -> float:
    """서브기간 Sharpe 계산. 6개월 미만이면 NaN."""
    sub = ret[(ret.index >= start) & (ret.index <= end)]
    if len(sub) < 6:
        return np.nan
    rf_sub = rf.reindex(sub.index).fillna(0)
    exc    = sub - rf_sub
    vol    = sub.std() * np.sqrt(12)
    return float(exc.mean() * 12 / vol) if vol > 0 else np.nan


def build_master_table(
    results_dir,
    rf: pd.Series,
    spy_ret: pd.Series = None,
    periods: dict = None,
    min_months: int = 12,
) -> pd.DataFrame:
    """
    모든 results/*.pkl → DataFrame.

    Returns
    -------
    DataFrame columns:
      이름/메타  : name, canonical, prior_s, p_s, pw_s, q_s, om_s,
                   prior, p_mode, p_weight, q_mode, omega_mode, omega_scale, tc, max_weight
      성과(전체) : sharpe, sortino, cagr, vol, mdd, calmar, cvar_5, win_rate,
                   beta, alpha, mdd_duration
      포트폴리오 : turnover_avg, eff_n_avg, n_months
      서브기간   : sharpe_<period> (periods 인자 있을 때)
    """
    results_dir = Path(results_dir)
    rows = []

    for pkl in sorted(results_dir.glob('*.pkl')):
        try:
            with open(pkl, 'rb') as f:
                res = pickle.load(f)
        except Exception as e:
            print(f'[skip] {pkl.name}: 로드 실패 {e}')
            continue

        ret = res.get('ret', pd.Series(dtype=float))
        if not isinstance(ret, pd.Series):
            continue
        ret = ret.dropna()
        if len(ret) < min_months:
            print(f'[skip] {pkl.name}: 데이터 부족 ({len(ret)}개월)')
            continue

        cfg = res.get('config', {})
        row = parse_config(cfg)
        row['name']     = pkl.stem
        row['n_months'] = len(ret)

        # 포트폴리오 통계 (turnover, 유효종목수)
        comp = res.get('comp', pd.DataFrame())
        row['turnover_avg'] = float(comp['turnover'].mean()) if (isinstance(comp, pd.DataFrame) and 'turnover' in comp.columns) else np.nan
        row['eff_n_avg']    = float(comp['eff_n'].mean())    if (isinstance(comp, pd.DataFrame) and 'eff_n'    in comp.columns) else np.nan

        # 전체기간 성과
        m = compute_metrics(ret, rf, label=pkl.stem, mkt_ret=spy_ret)
        for k in ('sharpe', 'sortino', 'cagr', 'vol', 'mdd', 'calmar',
                  'cvar_5', 'win_rate', 'beta', 'alpha', 'mdd_duration'):
            row[k] = m.get(k, np.nan)

        # 서브기간 Sharpe
        if periods:
            for label, (s, e) in periods.items():
                row[f'sharpe_{label}'] = _sharpe_subperiod(ret, rf, s, e)

        rows.append(row)

    df = pd.DataFrame(rows)

    # 컬럼 순서 정돈
    front = ['name', 'canonical', 'prior_s', 'p_s', 'pw_s', 'q_s', 'om_s',
             'sharpe', 'sortino', 'cagr', 'vol', 'mdd', 'calmar',
             'beta', 'alpha', 'turnover_avg', 'eff_n_avg', 'n_months']
    rest = [c for c in df.columns if c not in front]
    return df[[c for c in front if c in df.columns] + rest]


def regime_metrics(ret: pd.Series, rf: pd.Series, start: str, end: str) -> dict:
    """단일 레짐 구간의 sharpe/sortino/mdd."""
    sub = ret[(ret.index >= start) & (ret.index <= end)]
    if len(sub) < 6:
        return dict(sharpe=np.nan, sortino=np.nan, mdd=np.nan, n_months=len(sub))
    rf_sub  = rf.reindex(sub.index).fillna(0)
    exc     = sub - rf_sub
    vol     = sub.std() * np.sqrt(12)
    sharpe  = float(exc.mean() * 12 / vol) if vol > 0 else np.nan
    down    = sub[sub < 0].std() * np.sqrt(12)
    sortino = float(exc.mean() * 12 / down) if (down and down > 0) else np.nan
    cum     = (1 + sub).cumprod()
    mdd     = float((cum / cum.cummax() - 1).min())
    return dict(sharpe=sharpe, sortino=sortino, mdd=mdd, n_months=len(sub))


def build_regime_table(
    mt: pd.DataFrame,
    results_dir,
    rf: pd.Series,
    regimes: list = None,
) -> pd.DataFrame:
    """
    각 실험 × 5 레짐별 sharpe/sortino/mdd + 레짐 간 변동성 지표.

    핵심 컬럼:
      sortino_R1..R5   : 레짐별 Sortino
      sharpe_R1..R5    : 레짐별 Sharpe
      mdd_R1..R5       : 레짐별 MDD
      sortino_mean     : 5 레짐 평균
      sortino_std      : 5 레짐 표준편차 (작을수록 안정)
      sortino_min      : 5 레짐 최저 (worst case)
      sortino_cv       : sortino_std / |sortino_mean| (변동계수)
      sharpe_std       : 5 레짐 Sharpe std
      mdd_worst        : 5 레짐 중 가장 나쁜 MDD
      stability_score  : sortino_mean − sortino_std (안정성 + 평균 종합)

    높은 stability_score = 모든 레짐에서 일관되게 좋은 후보.
    """
    if regimes is None:
        regimes = REGIMES_5

    results_dir = Path(results_dir)
    rows = []

    for _, row in mt.iterrows():
        name = row['name']
        p = results_dir / f'{name}.pkl'
        if not p.exists():
            continue
        with open(p, 'rb') as f:
            ret = pickle.load(f).get('ret', pd.Series(dtype=float))
        ret = ret.dropna() if isinstance(ret, pd.Series) else pd.Series(dtype=float)
        if len(ret) < 12:
            continue

        rec = row.to_dict()
        sortinos, sharpes, mdds = [], [], []

        for label, s, e in regimes:
            m = regime_metrics(ret, rf, s, e)
            rec[f'sortino_{label}'] = round(m['sortino'], 3) if not np.isnan(m['sortino']) else np.nan
            rec[f'sharpe_{label}']  = round(m['sharpe'],  3) if not np.isnan(m['sharpe'])  else np.nan
            rec[f'mdd_{label}']     = round(m['mdd'],     3) if not np.isnan(m['mdd'])     else np.nan
            if not np.isnan(m['sortino']): sortinos.append(m['sortino'])
            if not np.isnan(m['sharpe']):  sharpes.append(m['sharpe'])
            if not np.isnan(m['mdd']):     mdds.append(m['mdd'])

        # 레짐 간 변동성 / 안정성 지표
        if sortinos:
            rec['sortino_mean'] = round(np.mean(sortinos), 3)
            rec['sortino_std']  = round(np.std(sortinos),  3)
            rec['sortino_min']  = round(np.min(sortinos),  3)
            rec['sortino_cv']   = round(np.std(sortinos) / abs(np.mean(sortinos)), 3) if np.mean(sortinos) != 0 else np.nan
            rec['stability_score'] = round(np.mean(sortinos) - np.std(sortinos), 3)
        if sharpes:
            rec['sharpe_mean']  = round(np.mean(sharpes), 3)
            rec['sharpe_std']   = round(np.std(sharpes),  3)
            rec['sharpe_min']   = round(np.min(sharpes),  3)
        if mdds:
            rec['mdd_worst']    = round(np.min(mdds),     3)
            rec['mdd_std']      = round(np.std(mdds),     3)

        rows.append(rec)

    df = pd.DataFrame(rows)

    # Sortino IR (mean / std) — 변동 대비 평균, 스케일 무관
    df['sortino_ir'] = (df['sortino_mean'] / df['sortino_std'].replace(0, np.nan)).round(2)

    # Sharpe IR (mean / std) — 동일 개념 sharpe 버전
    if {'sharpe_mean','sharpe_std'}.issubset(df.columns):
        df['sharpe_ir'] = (df['sharpe_mean'] / df['sharpe_std'].replace(0, np.nan)).round(2)

    return df


def slot_summary(mt: pd.DataFrame, slot: str, metric: str = 'sharpe') -> pd.DataFrame:
    """
    슬롯값별 성과 분포 요약 (Marginal Effect 1단계).

    Example
    -------
    slot_summary(mt, 'q_s', 'sharpe')
        → q_mode 약어별 sharpe의 mean/median/std/count
    """
    g = mt.groupby(slot)[metric]
    out = pd.DataFrame({
        'count' : g.count(),
        'mean'  : g.mean().round(3),
        'median': g.median().round(3),
        'std'   : g.std().round(3),
        'min'   : g.min().round(3),
        'max'   : g.max().round(3),
    })
    return out.sort_values('mean', ascending=False)


def matrix_pivot(mt: pd.DataFrame, metric: str = 'sharpe',
                 row_keys=('prior_s', 'pw_s'),
                 col_keys=('q_s', 'om_s'),
                 only_lstm: bool = True) -> pd.DataFrame:
    """
    행=prior×p_weight, 열=q×Ω 매트릭스 피벗 (히트맵 입력용).

    only_lstm=True 면 p_s='ls' 만 (108-cell 매트릭스).
    """
    sub = mt[mt.p_s == 'ls'] if only_lstm else mt
    sub = sub.copy()
    sub['_row'] = sub[list(row_keys)].agg('_'.join, axis=1)
    sub['_col'] = sub[list(col_keys)].agg('_'.join, axis=1)
    return sub.pivot_table(index='_row', columns='_col', values=metric, aggfunc='mean')
