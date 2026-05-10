"""
lib/metric_calculators.py - 메트릭 계산 함수

★★★ Ground truth: final/bl_functions.py:compute_metrics + master_table.py 헬퍼 ★★★
모든 결과가 final 폴더의 펀드 backtest 결과와 정확히 일치하도록 구현.

핵심 일관성 원칙:
- ANN = sqrt(12) (월별 → 연환산)
- Sortino: excess.mean() * 12 / (excess[excess<0].std() * sqrt(12))
  → final 정의 그대로 (downside 의 일반 std, RMS 아님)
- Beta: cov(excess, mkt_excess) / var(mkt_excess) — excess 기준
- Alpha: (excess.mean() - β × mkt_excess.mean()) × 12
- CAGR: cum.iloc[-1] ** (12/n) - 1
- MDD: (cum - cum.cummax()) / cum.cummax() 의 min
- CVaR: ret[ret <= ret.quantile(α)].mean() (pandas quantile)
- Skewness: pd.Series.skew()
- Excess Kurtosis: pd.Series.kurt() (pandas default = fisher=True)

참조: final/bl_functions.py compute_metrics (line 446-548)
       final/master_table.py _{sharpe/sortino/cagr/mdd}_subperiod (line 127-164)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# === Annualization 상수 (final 과 동일) ===============================
ANN = np.sqrt(12)
PERIODS_PER_YEAR = 12


# ======================================================================
# Helper — rf 정규화 (Series / scalar 둘 다 처리)
# ======================================================================

def _normalize_rf(ret: pd.Series, rf) -> pd.Series:
    """
    rf 를 ret 의 인덱스에 맞춰 월별 시리즈로 변환.

    - Series: reindex + fillna(0) (final compute_metrics 패턴)
    - scalar: 연환산이라 가정 → 월별 (rf / 12) 로 broadcast
    """
    if isinstance(rf, pd.Series):
        return rf.reindex(ret.index).fillna(0)
    return pd.Series([rf / PERIODS_PER_YEAR] * len(ret), index=ret.index)


# ======================================================================
# Pool-1 수익성
# ======================================================================

def calc_cagr(ret: pd.Series, periods_per_year: int = 12) -> float:
    """
    CAGR — final compute_metrics 정의 정확 재현:
      cagr = cum.iloc[-1] ** (12/n_months) - 1.0
    """
    r = pd.Series(ret).dropna()
    if len(r) == 0:
        return np.nan
    n_months = len(r)
    cum_last = float((1 + r).cumprod().iloc[-1])
    if n_months <= 0 or cum_last <= 0:
        return np.nan
    return float(cum_last ** (periods_per_year / n_months) - 1.0)


def calc_arithmetic_mean(ret: pd.Series, annualize: bool = True, periods_per_year: int = 12) -> float:
    """단순 평균 (월별 → 연환산 기본)."""
    r = pd.Series(ret).dropna()
    if len(r) == 0:
        return np.nan
    return float(r.mean() * periods_per_year) if annualize else float(r.mean())


def calc_cumulative_return(ret: pd.Series) -> float:
    """final compute_metrics 의 cum_ret = cum.iloc[-1] - 1."""
    r = pd.Series(ret).dropna()
    if len(r) == 0:
        return np.nan
    return float((1 + r).cumprod().iloc[-1] - 1)


# ======================================================================
# Pool-3 위험 (Sharpe / Sortino 분모로도 사용)
# ======================================================================

def calc_volatility(ret: pd.Series, annualize: bool = True, periods_per_year: int = 12) -> float:
    """final 의 vol = ret.std() * sqrt(12)."""
    r = pd.Series(ret).dropna()
    if len(r) == 0:
        return np.nan
    std = r.std()
    if pd.isna(std):
        return np.nan
    return float(std * np.sqrt(periods_per_year)) if annualize else float(std)


def calc_mdd(ret: pd.Series) -> float:
    """final 의 mdd = (cum - cum.cummax()) / cum.cummax() 의 min."""
    r = pd.Series(ret).dropna()
    if len(r) == 0:
        return np.nan
    cum = (1 + r).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    return float(dd.min())


def calc_mdd_duration(ret: pd.Series) -> int:
    """
    final 의 mdd_duration: 연속 underwater (dd<0) 월 최대.
    """
    r = pd.Series(ret).dropna()
    if len(r) == 0:
        return 0
    cum = (1 + r).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    underwater = (dd < 0).astype(int)
    max_dur, cur = 0, 0
    for v in underwater:
        cur = cur + 1 if v else 0
        if cur > max_dur:
            max_dur = cur
    return int(max_dur)


def calc_downside_deviation(
    ret: pd.Series, mar: float = 0.0, annualize: bool = True, periods_per_year: int = 12
) -> float:
    """
    final Sortino 의 분모와 동일 정의:
      downside = ret[ret < MAR]; downside.std() * sqrt(12)

    NOTE: pandas .std() 는 ddof=1 (sample std). final 도 동일.
    """
    r = pd.Series(ret).dropna()
    if len(r) == 0:
        return np.nan
    downside = r[r < mar]
    if len(downside) <= 1:
        return np.nan
    dd_std = downside.std()
    if pd.isna(dd_std) or dd_std == 0:
        return 0.0
    return float(dd_std * np.sqrt(periods_per_year)) if annualize else float(dd_std)


def calc_var(ret: pd.Series, alpha: float = 0.05) -> float:
    """Historical VaR — pandas quantile (final 과 일관)."""
    r = pd.Series(ret).dropna()
    if len(r) == 0:
        return np.nan
    return float(r.quantile(alpha))


def calc_cvar(ret: pd.Series, alpha: float = 0.05) -> float:
    """
    final compute_metrics 의 cvar_5 정의 정확 재현:
      cutoff = ret.quantile(0.05); cvar = ret[ret <= cutoff].mean()
    """
    r = pd.Series(ret).dropna()
    if len(r) == 0:
        return np.nan
    cutoff = r.quantile(alpha)
    tail = r[r <= cutoff]
    if len(tail) == 0:
        return np.nan
    return float(tail.mean())


# ======================================================================
# Pool-2 위험조정 수익
# ======================================================================

def calc_sharpe(ret: pd.Series, rf=0.0, periods_per_year: int = 12) -> float:
    """
    final compute_metrics 정의:
      sharpe = excess.mean() / excess.std() * sqrt(12)
    """
    r = pd.Series(ret).dropna()
    if len(r) == 0:
        return np.nan
    rf_a = _normalize_rf(r, rf)
    excess = r - rf_a
    if excess.std() == 0 or pd.isna(excess.std()):
        return np.nan
    return float(excess.mean() / excess.std() * np.sqrt(periods_per_year))


def calc_sortino(ret: pd.Series, rf=0.0, periods_per_year: int = 12) -> float:
    """
    final compute_metrics 정의 정확 재현:
      excess  = ret - rf_a
      downside = excess[excess < 0]
      down_std = downside.std() * sqrt(12)        # 일반 std (RMS 아님)
      sortino  = excess.mean() * 12 / down_std
    """
    r = pd.Series(ret).dropna()
    if len(r) == 0:
        return np.nan
    rf_a = _normalize_rf(r, rf)
    excess = r - rf_a
    downside = excess[excess < 0]
    if len(downside) <= 1:
        return np.nan
    down_std = downside.std() * np.sqrt(periods_per_year)
    if down_std is None or down_std == 0 or pd.isna(down_std):
        return np.nan
    return float(excess.mean() * periods_per_year / down_std)


def calc_calmar(ret: pd.Series, periods_per_year: int = 12) -> float:
    """final 의 calmar = cagr / |mdd|."""
    cagr = calc_cagr(ret, periods_per_year)
    mdd = calc_mdd(ret)
    if pd.isna(cagr) or pd.isna(mdd) or mdd == 0:
        return np.nan
    return float(cagr / abs(mdd))


def calc_ir(fund_ret: pd.Series, bench_ret: pd.Series, periods_per_year: int = 12) -> float:
    """
    Information Ratio = mean(Fund - Bench) / std(Fund - Bench) × sqrt(12).
    """
    df = pd.DataFrame({"f": fund_ret, "b": bench_ret}).dropna()
    if len(df) == 0:
        return np.nan
    active = df["f"] - df["b"]
    te = active.std()
    if pd.isna(te) or te == 0:
        return np.nan
    return float(active.mean() / te * np.sqrt(periods_per_year))


# ======================================================================
# Pool-5 시장 비교 (Beta / Alpha — final 정확 재현)
# ======================================================================

def calc_beta(fund_ret: pd.Series, mkt_ret: pd.Series, rf=0.0, periods_per_year: int = 12) -> float:
    """
    final compute_metrics 정의 정확 재현:
      excess     = fund - rf
      mkt_excess = mkt - rf
      cov_mat    = np.cov(excess, mkt_excess)
      beta       = cov_mat[0,1] / cov_mat[1,1]   (var_m > 0 일 때)

    NOTE: final 은 "valid.sum() >= 12" 조건 (12개월 이상 정렬된 데이터 필요).
    """
    r = pd.Series(fund_ret)
    m = pd.Series(mkt_ret)
    rf_a = _normalize_rf(r, rf) if isinstance(rf, pd.Series) or rf != 0 else pd.Series(0.0, index=r.index)
    df = pd.DataFrame({"r": r, "m": m, "rf": rf_a}).dropna()
    if len(df) < 12:
        return np.nan
    excess = (df["r"] - df["rf"]).values
    mkt_excess = (df["m"] - df["rf"]).values
    cov_mat = np.cov(excess, mkt_excess)
    var_m = cov_mat[1, 1]
    if var_m <= 0 or pd.isna(var_m):
        return np.nan
    return float(cov_mat[0, 1] / var_m)


def calc_alpha(fund_ret: pd.Series, mkt_ret: pd.Series, rf=0.0, periods_per_year: int = 12) -> float:
    """
    final compute_metrics 정의 정확 재현:
      alpha = (excess.mean() - beta × mkt_excess.mean()) × 12
    """
    r = pd.Series(fund_ret)
    m = pd.Series(mkt_ret)
    rf_a = _normalize_rf(r, rf) if isinstance(rf, pd.Series) or rf != 0 else pd.Series(0.0, index=r.index)
    df = pd.DataFrame({"r": r, "m": m, "rf": rf_a}).dropna()
    if len(df) < 12:
        return np.nan
    excess = (df["r"] - df["rf"]).values
    mkt_excess = (df["m"] - df["rf"]).values
    cov_mat = np.cov(excess, mkt_excess)
    var_m = cov_mat[1, 1]
    if var_m <= 0 or pd.isna(var_m):
        return np.nan
    beta = cov_mat[0, 1] / var_m
    return float((excess.mean() - beta * mkt_excess.mean()) * periods_per_year)


def calc_tracking_error(
    fund_ret: pd.Series, bench_ret: pd.Series, annualize: bool = True, periods_per_year: int = 12
) -> float:
    """추적오차 = std(Fund - Bench) × sqrt(12)."""
    df = pd.DataFrame({"f": fund_ret, "b": bench_ret}).dropna()
    if len(df) == 0:
        return np.nan
    te = (df["f"] - df["b"]).std()
    if pd.isna(te):
        return np.nan
    return float(te * np.sqrt(periods_per_year)) if annualize else float(te)


def calc_active_return(fund_ret: pd.Series, bench_ret: pd.Series, periods_per_year: int = 12) -> float:
    """Annualized Active Return = (mean_fund - mean_bench) × 12."""
    df = pd.DataFrame({"f": fund_ret, "b": bench_ret}).dropna()
    if len(df) == 0:
        return np.nan
    return float((df["f"].mean() - df["b"].mean()) * periods_per_year)


def calc_win_rate(ret: pd.Series) -> float:
    r = pd.Series(ret).dropna()
    if len(r) == 0:
        return np.nan
    return float((r > 0).mean())


def calc_avg_win(ret: pd.Series) -> float:
    """final 의 avg_win = wins.mean() (월별)."""
    r = pd.Series(ret).dropna()
    wins = r[r > 0]
    if len(wins) == 0:
        return np.nan
    return float(wins.mean())


def calc_avg_loss(ret: pd.Series) -> float:
    """final 의 avg_loss = losses.mean() (월별, 음수)."""
    r = pd.Series(ret).dropna()
    losses = r[r < 0]
    if len(losses) == 0:
        return np.nan
    return float(losses.mean())


def calc_up_capture(fund_ret: pd.Series, mkt_ret: pd.Series) -> float:
    df = pd.DataFrame({"f": fund_ret, "m": mkt_ret}).dropna()
    up = df[df["m"] > 0]
    if len(up) == 0:
        return np.nan
    fund_mean = up["f"].mean()
    mkt_mean = up["m"].mean()
    if pd.isna(mkt_mean) or mkt_mean == 0:
        return np.nan
    return float(fund_mean / mkt_mean)


def calc_down_capture(fund_ret: pd.Series, mkt_ret: pd.Series) -> float:
    df = pd.DataFrame({"f": fund_ret, "m": mkt_ret}).dropna()
    down = df[df["m"] < 0]
    if len(down) == 0:
        return np.nan
    fund_mean = down["f"].mean()
    mkt_mean = down["m"].mean()
    if pd.isna(mkt_mean) or mkt_mean == 0:
        return np.nan
    return float(fund_mean / mkt_mean)


# ======================================================================
# Pool-9 분포 통계 (Performance 영역 8)
# ======================================================================

def calc_skewness(ret: pd.Series) -> float:
    """final 의 skewness = ret.skew() (pandas default)."""
    r = pd.Series(ret).dropna()
    if len(r) < 3:
        return np.nan
    return float(r.skew())


def calc_excess_kurtosis(ret: pd.Series) -> float:
    """
    Excess Kurtosis = pd.Series.kurt() (pandas default = Fisher's, 즉 -3 적용된 excess).
    정규분포 대비 첨도 — 양수 = fat tail.
    """
    r = pd.Series(ret).dropna()
    if len(r) < 4:
        return np.nan
    return float(r.kurt())


def calc_tail_ratio(ret: pd.Series, q_high: float = 0.95, q_low: float = 0.05) -> float:
    """
    Tail Ratio = q_high 분위 / |q_low 분위|.
    > 1 = 우측 꼬리 비대칭 (좋음).
    """
    r = pd.Series(ret).dropna()
    if len(r) == 0:
        return np.nan
    high = r.quantile(q_high)
    low = abs(r.quantile(q_low))
    if low == 0 or pd.isna(low):
        return np.nan
    return float(high / low)


# ======================================================================
# Pool-4 운용 효율성 (집중도)
# ======================================================================

def calc_hhi(weights) -> float:
    """Herfindahl-Hirschman Index = Σw² (weights sum=1 가정)."""
    w = pd.Series(weights).dropna()
    if len(w) == 0:
        return np.nan
    return float((w ** 2).sum())


def calc_effective_n(weights) -> float:
    """Effective N = 1 / HHI."""
    hhi = calc_hhi(weights)
    if pd.isna(hhi) or hhi == 0:
        return np.nan
    return float(1 / hhi)


# ======================================================================
# 서브기간 헬퍼 (final master_table 정확 재현)
# ======================================================================

def calc_sharpe_subperiod(
    ret: pd.Series, rf, start: str, end: str, periods_per_year: int = 12
) -> float:
    """
    final master_table._sharpe_subperiod 정확 재현 (line 127-135).

    NaN 처리: final 은 sub.std() / exc.mean() 등이 pandas default skipna=True
              로 NaN 자동 무시 → 우리도 동일 동작 (명시적 dropna 불필요).
    """
    r = pd.Series(ret)
    sub_raw = r[(r.index >= start) & (r.index <= end)]
    sub = sub_raw.dropna()
    if len(sub) < 6:
        return np.nan
    rf_sub = pd.Series(rf).reindex(sub.index).fillna(0) if isinstance(rf, pd.Series) else pd.Series(rf / periods_per_year, index=sub.index)
    exc = sub - rf_sub
    vol = sub.std() * np.sqrt(periods_per_year)
    if vol <= 0 or pd.isna(vol):
        return np.nan
    return float(exc.mean() * periods_per_year / vol)


def calc_sortino_subperiod(
    ret: pd.Series, rf, start: str, end: str, periods_per_year: int = 12
) -> float:
    """
    final master_table._sortino_subperiod 정확 재현 (line 138-146):
      down = sub[sub < 0].std() * sqrt(12)   # ret < 0 의 std (excess 아님!)
      return exc.mean() * 12 / down
    """
    r = pd.Series(ret)
    sub = r[(r.index >= start) & (r.index <= end)].dropna()
    if len(sub) < 6:
        return np.nan
    rf_sub = pd.Series(rf).reindex(sub.index).fillna(0) if isinstance(rf, pd.Series) else pd.Series(rf / periods_per_year, index=sub.index)
    exc = sub - rf_sub
    down_series = sub[sub < 0]
    if len(down_series) <= 1:
        return np.nan
    down = down_series.std() * np.sqrt(periods_per_year)
    if down is None or down <= 0 or pd.isna(down):
        return np.nan
    return float(exc.mean() * periods_per_year / down)


def calc_cagr_subperiod(ret: pd.Series, start: str, end: str, periods_per_year: int = 12) -> float:
    """
    final master_table._cagr_subperiod 정확 재현 (line 149-155) + NaN-safe.

    final 은 NaN 처리를 안 하지만 (1+NaN).cumprod() = NaN 전염 → CAGR 산출 실패.
    fund.ret 자체는 NaN 없어 final 결과와 동일. 벤치마크 (e.g., fund.spy_ret 의
    마지막 월 NaN) 산출 시에만 dropna 효과 — final 정합 유지.
    """
    r = pd.Series(ret)
    sub = r[(r.index >= start) & (r.index <= end)].dropna()
    if len(sub) < 6:
        return np.nan
    cum = (1 + sub).cumprod().iloc[-1]
    if cum <= 0 or pd.isna(cum):
        return np.nan
    return float(cum ** (periods_per_year / len(sub)) - 1)


def calc_mdd_subperiod(ret: pd.Series, start: str, end: str) -> float:
    """
    final master_table._mdd_subperiod 정확 재현 (line 158-164) + NaN-safe.

    cumprod NaN 전염 회피 위해 dropna 적용 (fund.ret 영향 X — NaN 없음).
    """
    r = pd.Series(ret)
    sub = r[(r.index >= start) & (r.index <= end)].dropna()
    if len(sub) < 6:
        return np.nan
    cum = (1 + sub).cumprod()
    return float((cum / cum.cummax() - 1).min())


# ======================================================================
# 통합 메트릭 dict (final compute_metrics 정확 재현)
# ======================================================================

def compute_all_metrics(
    ret: pd.Series,
    rf: pd.Series,
    label: str = "",
    mkt_ret: pd.Series | None = None,
) -> dict:
    """
    final/bl_functions.py:compute_metrics 와 동일한 인터페이스 + 결과.

    rf 필수 (월별 시리즈, fund_results 기간과 일치해야 함).
    mkt_ret None 이면 Beta/Alpha NaN.

    Returns: dict (label, sharpe, sortino, cagr, vol, mdd, calmar, cum_ret,
                   skewness, win_rate, avg_win, avg_loss, cvar_5, mdd_duration,
                   beta, alpha)

    NOTE: final 은 round() 적용. 우리는 raw float 반환 (호출 측에서 포맷팅).
    """
    rf_a = pd.Series(rf).reindex(ret.index).fillna(0)
    excess = ret - rf_a

    return {
        "label": label,
        "sharpe": calc_sharpe(ret, rf_a),
        "sortino": calc_sortino(ret, rf_a),
        "cagr": calc_cagr(ret),
        "vol": calc_volatility(ret),
        "mdd": calc_mdd(ret),
        "calmar": calc_calmar(ret),
        "cum_ret": calc_cumulative_return(ret),
        "skewness": calc_skewness(ret),
        "win_rate": calc_win_rate(ret),
        "avg_win": calc_avg_win(ret),
        "avg_loss": calc_avg_loss(ret),
        "cvar_5": calc_cvar(ret, alpha=0.05),
        "mdd_duration": calc_mdd_duration(ret),
        "beta": calc_beta(ret, mkt_ret, rf_a) if mkt_ret is not None else np.nan,
        "alpha": calc_alpha(ret, mkt_ret, rf_a) if mkt_ret is not None else np.nan,
    }
