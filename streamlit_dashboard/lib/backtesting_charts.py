"""
lib/backtesting_charts.py — Risk Metrics 페이지 영역 5/6 차트 함수

영역:
  5. render_regime_detail_table (12 메트릭 × 5 Regime + Sortino 막대)
  6. render_sub_events (4 위기 Timeline + 표)

설계 원칙:
  - 각 영역 독립 함수 (영역 추가/제거 시 페이지 entry 만 변경)
  - 메트릭 / 위기 정의는 metric_calculators.py 의 외부 dict (REGIME_METRICS / SUB_EVENTS)
  - 모든 메트릭 final 정합 또는 학술 출처 명시

이력:
  - 2026-05-11: 원래 Backtesting 페이지 (8 영역) 전용 모듈이었으나 페이지 통합 삭제 후
    두 함수만 Risk Metrics 페이지 영역 5/6 로 이전됨.
  - 2026-05-12: deprecated 함수 3개 + 미사용 헬퍼 5개 cleanup (567 → ~165 라인).
    제거된 함수: render_backtest_kpi / render_cumulative_comparison / render_sensitivity_test
    제거된 헬퍼: _fmt_pct / _fmt_ratio / _color_by_threshold /
                _compute_all_config_metrics / _compute_all_config_cumulative
    복원 필요 시 git history (`git log -p lib/backtesting_charts.py`) 에서 복구.

참조: docs/decisionlog/04_risk_metrics.md, docs/decisionlog/08_backtesting.md (이력)
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import metric_calculators as mc
from lib.colors import COLORS


# ======================================================================
# 영역 5: Regime 메트릭 자세한 비교 (12 메트릭)
# ======================================================================

def render_regime_detail_table(
    ret: pd.Series,
    spy: pd.Series,
    rf: pd.Series,
) -> None:
    """12 메트릭 × 5 Regime 표 + Sortino 막대 + Best/Worst 강조."""
    df = mc.calc_regime_full_metrics(ret, spy, rf)
    if len(df) == 0:
        st.warning("Regime 데이터 산출 불가.")
        return

    # 표 표시 (포맷 적용)
    df_display = df.copy()
    pct_cols = ["CAGR", "MDD", "Volatility", "Active_Return", "Win_Rate", "VaR_5", "CVaR_5"]
    ratio_cols = ["Sortino", "Sharpe", "IR", "Calmar", "Beta"]

    st.dataframe(
        df_display.style.format(
            {c: "{:+.2%}" for c in pct_cols} | {c: "{:+.3f}" for c in ratio_cols}
        ),
        use_container_width=True,
        height=240,
    )

    # Best/Worst Regime (Sortino 기준)
    if "Sortino" in df.columns and df["Sortino"].notna().any():
        best = df["Sortino"].idxmax()
        worst = df["Sortino"].idxmin()
        c1, c2 = st.columns(2)
        with c1:
            st.success(f"⭐ **Best Regime**: {best} (Sortino {df.loc[best, 'Sortino']:.2f})")
        with c2:
            st.error(f"🔴 **Worst Regime**: {worst} (Sortino {df.loc[worst, 'Sortino']:.2f})")

    # Sortino 막대
    fig = go.Figure(go.Bar(
        x=df.index, y=df["Sortino"],
        marker_color=[
            COLORS["accent_green"] if v == df["Sortino"].max()
            else (COLORS["accent_red"] if v == df["Sortino"].min() else COLORS["primary"])
            for v in df["Sortino"]
        ],
        text=[f"{v:.2f}" for v in df["Sortino"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Sortino: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["background"],
        font_color=COLORS["text"],
        yaxis_title="Sortino",
        height=320, margin=dict(t=20, l=0, r=0, b=0),
        title=dict(text="Regime별 Sortino 비교", font=dict(size=13)),
    )
    st.plotly_chart(fig, use_container_width=True)

    # CSV 다운로드
    csv = df.to_csv().encode("utf-8-sig")
    st.download_button(
        label="⬇ CSV 다운로드 (Regime × 12 메트릭)",
        data=csv,
        file_name="regime_metrics.csv",
        mime="text/csv",
        key="bt_regime_csv",
    )


# ======================================================================
# 영역 6: Sub-events 분석 (4 위기)
# ======================================================================

def render_sub_events(
    ret: pd.Series,
    spy: pd.Series,
    rf: pd.Series,
) -> None:
    """4 위기 Timeline + 표 + Best/Worst 강조."""
    df = mc.calc_sub_event_metrics(ret, spy, rf)
    if len(df) == 0:
        st.warning("Sub-events 데이터 산출 불가.")
        return

    # 표 표시
    df_display = df.copy()
    df_display["Recovery_months"] = df_display["Recovery_months"].apply(
        lambda v: f"{int(v)}m" if v is not None and not pd.isna(v) else "미회복"
    )
    st.dataframe(
        df_display.style.format(
            {"Fund": "{:+.2%}", "SPY": "{:+.2%}", "Active": "{:+.2%}", "MDD": "{:.2%}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Best/Worst (Active Return 기준)
    if df["Active"].notna().any():
        best_idx = df["Active"].idxmax()
        worst_idx = df["Active"].idxmin()
        c1, c2 = st.columns(2)
        with c1:
            ev = df.loc[best_idx, "Event"]
            v = df.loc[best_idx, "Active"]
            st.success(f"⭐ **Best 방어**: {ev} (Active {v:+.2%})")
        with c2:
            ev = df.loc[worst_idx, "Event"]
            v = df.loc[worst_idx, "Active"]
            star = " ★" if "2024" in ev else ""
            st.error(f"🔴 **Worst**: {ev}{star} (Active {v:+.2%})")

    # Active Return 막대 (위기별)
    fig = go.Figure(go.Bar(
        x=df["Event"], y=df["Active"],
        marker_color=[
            COLORS["accent_green"] if v > 0 else COLORS["accent_red"]
            for v in df["Active"]
        ],
        text=[f"{v:+.2%}" for v in df["Active"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Active: %{y:+.3%}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["background"],
        font_color=COLORS["text"],
        yaxis=dict(tickformat=".1%", title="Active Return (Fund − SPY)"),
        height=320, margin=dict(t=30, l=0, r=0, b=0),
        title=dict(text="위기별 Active Return — Fund vs SPY", font=dict(size=13)),
    )
    fig.add_hline(y=0, line_color=COLORS["text"], line_width=1)
    st.plotly_chart(fig, use_container_width=True)
