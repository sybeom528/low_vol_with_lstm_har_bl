"""
Phase 1.5 핵심 그림 생성 스크립트
==================================

본 통합 리포트 (REPORT_Phase1_5_to_3.md) 의 Phase 1.5 섹션에 인라인 참조되는
그림 5장을 result CSV 들로부터 생성합니다.

Run from project root:
  cd "시계열_Test"
  python REPORT_assets/phase1_5/_generate_figures.py

생성 그림:
  fig1_phase15_evolution.png      — v1~v8 진화 (RMSE)
  fig2_multi_asset_rmse_heatmap.png — 7종목 × 5모델 RMSE Heatmap (v5)
  fig3_ensemble_comparison.png    — v8 앙상블 6 모델 비교 (7종목 × 6모델)
  fig4_v6_external_indicators.png — v4 vs v6 외부지표 추가 폭증 악화
  fig5_v7_ablation.png            — v7 ablation 외부지표 noise 순위
"""

from __future__ import annotations

import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 한글 폰트 설정 (CLAUDE.md 전역 지침 준수)
# ---------------------------------------------------------------------------
if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
elif platform.system() == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
else:
    import koreanize_matplotlib  # noqa: F401  (pip install koreanize-matplotlib --break-system-packages)
    plt.rcParams["font.family"] = "NanumGothic"

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent              # REPORT_assets/phase1_5/
REPO_ROOT = HERE.parent.parent                       # 시계열_Test/
RESULTS_DIR = REPO_ROOT / "Phase1_5_Volatility" / "results"
OUT_DIR = HERE
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 색상 팔레트
COLOR_LSTM = "#2E86AB"
COLOR_HAR = "#E63946"
COLOR_ENSEMBLE = "#06A77D"
COLOR_NEUTRAL = "#888888"


# ---------------------------------------------------------------------------
# Figure 1 — v1 ~ v8 진화 (RMSE)
# ---------------------------------------------------------------------------
def fig1_phase15_evolution() -> None:
    """v1~v8 진화 — REPORT.md §23.2 의 정리 결과 기반.

    숫자는 REPORT.md / 각 보조 보고서에서 추출 (90 fold avg).
    v8 의 simple/ivw/performance/asset_specific 4 변형 중 performance 사용.
    v6/v7 은 SPY/QQQ avg 만 표시 (다른 종목은 noise 패턴 동일).
    """
    versions = ["v1", "v2", "v3 best", "v4 best", "v5*", "v6 (8ch)", "v7 (-TNX)", "v8 ensemble"]
    # SPY/QQQ avg RMSE (90 fold mean) — 출처: REPORT.md §23.2
    rmse = [0.4509, 0.4592, 0.4001, 0.3107, 0.3107, 0.6692, 0.3608, 0.3005]
    notes = [
        "1ch / IS=504",
        "3ch / IS=504",
        "Optuna IS=750",
        "3ch_vix / IS=1250 ⭐",
        "7종목 일반화",
        "외부지표 9ch ❌",
        "SKEW 잔존 nope",
        "Performance ⭐⭐⭐",
    ]
    HAR_AVG = 0.3477  # SPY 0.3646 + QQQ 0.3308 의 평균

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(range(len(versions)), rmse, color=[
        "#B0BEC5", "#90A4AE", "#78909C",  # v1, v2, v3
        COLOR_LSTM,                          # v4 best (LSTM 색)
        COLOR_LSTM,                          # v5
        COLOR_HAR,                           # v6 (악화 빨강)
        "#FFB300",                           # v7 (호박)
        COLOR_ENSEMBLE,                      # v8 ensemble
    ], width=0.65)

    ax.axhline(HAR_AVG, color=COLOR_HAR, linestyle="--", linewidth=1.5,
               label=f"HAR-RV baseline (RMSE {HAR_AVG:.4f})")

    for i, (b, r, n) in enumerate(zip(bars, rmse, notes)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                f"{r:.4f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, -0.038, n,
                ha="center", va="top", fontsize=8.5, color="#333", style="italic")

    ax.set_xticks(range(len(versions)))
    ax.set_xticklabels(versions, fontsize=11)
    ax.set_ylabel("RMSE (log-RV, 90 fold avg of SPY+QQQ)", fontsize=11)
    ax.set_ylim(0, 0.78)
    ax.set_title("Phase 1.5 v1 → v8 진화 — RMSE 추이", fontsize=13, fontweight="bold", pad=14)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Phase 1 reference line  (134 샘플 baseline)
    ax.axhline(0.50, color="#666", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(7.4, 0.50, "Phase 1 baseline\n(134 샘플)", ha="right", va="center", fontsize=8, color="#666")

    fig.subplots_adjust(bottom=0.18, top=0.90)
    out = OUT_DIR / "fig1_phase15_evolution.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] {out.name}")


# ---------------------------------------------------------------------------
# Figure 2 — 7 종목 × 5 모델 RMSE Heatmap (v5 multi_asset)
# ---------------------------------------------------------------------------
def fig2_multi_asset_rmse_heatmap() -> None:
    """multi_asset/multi_asset_comparison.csv 기반 (5 model × 7 ticker)."""
    csv = RESULTS_DIR / "multi_asset" / "multi_asset_comparison.csv"
    df = pd.read_csv(csv)

    pivot = df.pivot(index="ticker", columns="model", values="rmse_mean")
    # 모델 순서: lstm_v4 / har / ewma / naive / train_mean
    pivot = pivot[["lstm_v4", "har", "ewma", "naive", "train_mean"]]
    # 종목 순서: 카테고리별
    pivot = pivot.reindex(["SPY", "QQQ", "DIA", "XLF", "EEM", "GOOGL", "WMT"])

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto", vmin=0.24, vmax=0.42)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(["LSTM v4", "HAR", "EWMA", "Naive", "Train-Mean"], fontsize=10.5)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=11)

    # 각 셀에 값 + best 표시
    for i in range(len(pivot.index)):
        row = pivot.iloc[i].values
        best_j = int(np.argmin(row))
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            txt = f"{val:.4f}"
            color = "white" if val > 0.36 else "black"
            if j == best_j:
                ax.text(j, i, f"★\n{txt}", ha="center", va="center",
                        fontsize=9.5, color=color, fontweight="bold")
            else:
                ax.text(j, i, txt, ha="center", va="center", fontsize=9.5, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("RMSE", fontsize=10)

    ax.set_title("Phase 1.5 v5 — 7 종목 × 5 모델 RMSE Heatmap\n"
                 "(★ = 종목별 best — LSTM v4 우위 5/7, HAR 우위 2/7)",
                 fontsize=12, fontweight="bold", pad=12)

    fig.subplots_adjust(top=0.85, left=0.12, right=0.98, bottom=0.10)
    out = OUT_DIR / "fig2_multi_asset_rmse_heatmap.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] {out.name}")


# ---------------------------------------------------------------------------
# Figure 3 — v8 Ensemble: 7 ticker × 6 model RMSE
# ---------------------------------------------------------------------------
def fig3_ensemble_comparison() -> None:
    """lstm_ensemble/ensemble_comparison.csv 기반 — 6 model × 7 ticker."""
    csv = RESULTS_DIR / "lstm_ensemble" / "ensemble_comparison.csv"
    df = pd.read_csv(csv)

    pivot = df.pivot(index="ticker", columns="model", values="rmse")
    # 모델 순서
    cols = ["lstm_v4", "har", "simple", "ivw", "performance", "asset_specific"]
    pivot = pivot[cols]
    pivot = pivot.reindex(["SPY", "QQQ", "DIA", "EEM", "XLF", "GOOGL", "WMT"])

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    n_tickers = len(pivot.index)
    n_models = len(cols)
    x = np.arange(n_tickers)
    bar_w = 0.135

    palette = {
        "lstm_v4": COLOR_LSTM,
        "har": COLOR_HAR,
        "simple": "#FFB300",
        "ivw": "#FF8800",
        "performance": COLOR_ENSEMBLE,   # 강조색
        "asset_specific": "#9B5DE5",
    }
    labels = {
        "lstm_v4": "LSTM v4",
        "har": "HAR-RV",
        "simple": "simple (50/50)",
        "ivw": "IVW",
        "performance": "performance ⭐",
        "asset_specific": "asset-specific",
    }

    for i, m in enumerate(cols):
        offsets = (i - (n_models - 1) / 2) * bar_w
        edge = "black" if m == "performance" else "none"
        ax.bar(x + offsets, pivot[m].values, bar_w, label=labels[m],
               color=palette[m], edgecolor=edge, linewidth=1.2 if m == "performance" else 0)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, fontsize=11)
    ax.set_ylabel("RMSE (log-RV, 55 fold avg, IS=1250)", fontsize=11)
    ax.set_title("Phase 1.5 v8 — Performance-Weighted Ensemble 비교 (7 종목 × 6 모델)\n"
                 "Performance avg RMSE 0.2934 / QLIKE 0.2582 (둘 다 1위, 5/7 종목 best)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="upper right", fontsize=9, ncol=3)
    ax.set_ylim(0.23, 0.40)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.subplots_adjust(top=0.85)
    out = OUT_DIR / "fig3_ensemble_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] {out.name}")


# ---------------------------------------------------------------------------
# Figure 4 — v6 외부지표 9채널 — v4 대비 RMSE 폭증
# ---------------------------------------------------------------------------
def fig4_v6_external_indicators() -> None:
    """lstm_v6_9ch/v6_comparison.csv 기반."""
    csv = RESULTS_DIR / "lstm_v6_9ch" / "v6_comparison.csv"
    df = pd.read_csv(csv)
    df = df.set_index("ticker").reindex(["SPY", "QQQ", "DIA", "EEM", "XLF", "GOOGL", "WMT"])

    fig, ax = plt.subplots(figsize=(10, 5))
    n = len(df.index)
    x = np.arange(n)
    bar_w = 0.36

    ax.bar(x - bar_w / 2, df["v4_rmse"].values, bar_w, label="v4 (4ch: HAR + VIX)", color=COLOR_LSTM)
    ax.bar(x + bar_w / 2, df["v6_rmse"].values, bar_w, label="v6 (8ch: + VVIX/SKEW/TNX/DXY)", color=COLOR_HAR)

    # Δ RMSE 라벨 (각 종목 위)
    for i, t in enumerate(df.index):
        delta = df.loc[t, "rmse_diff"]
        pct = delta / df.loc[t, "v4_rmse"] * 100
        y = max(df.loc[t, "v4_rmse"], df.loc[t, "v6_rmse"]) + 0.03
        ax.text(x[i], y, f"+{pct:.0f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=COLOR_HAR)

    ax.set_xticks(x)
    ax.set_xticklabels(df.index, fontsize=11)
    ax.set_ylabel("RMSE (log-RV, 55 fold avg)", fontsize=11)
    ax.set_title("Phase 1.5 v6 — 외부지표 4종 추가 시 모든 종목 폭증 악화 (7/7)\n"
                 "본 환경에서 단순 추가는 noise 첨가 — VIX 만 효과 있는 외부지표",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_ylim(0, 0.85)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.subplots_adjust(top=0.85)
    out = OUT_DIR / "fig4_v6_external_indicators.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] {out.name}")


# ---------------------------------------------------------------------------
# Figure 5 — v7 Ablation: 외부지표 noise 순위
# ---------------------------------------------------------------------------
def fig5_v7_ablation() -> None:
    """v7_ablation_report.md §2 의 표 기반 — SPY/QQQ avg Δ RMSE."""
    indicators = ["TNX", "SKEW", "DXY", "VVIX"]
    # avg Δ RMSE (vs v6, 음수일수록 noise) — v7 보고서 §2
    delta_avg = [-0.3083, -0.2714, -0.2468, -0.2141]
    spy = [-0.2910, -0.2402, -0.2199, -0.1121]
    qqq = [-0.3256, -0.3026, -0.2738, -0.3162]

    fig, ax = plt.subplots(figsize=(9, 5))
    n = len(indicators)
    x = np.arange(n)
    bar_w = 0.28

    ax.bar(x - bar_w, spy, bar_w, label="SPY", color=COLOR_LSTM)
    ax.bar(x, qqq, bar_w, label="QQQ", color="#FF6B35")
    ax.bar(x + bar_w, delta_avg, bar_w, label="평균", color=COLOR_HAR, edgecolor="black", linewidth=0.8)

    for i, v in enumerate(delta_avg):
        ax.text(x[i] + bar_w, v - 0.013, f"{v:+.3f}", ha="center", va="top",
                fontsize=9.5, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"-{ind}\n({rank})" for ind, rank in zip(indicators, ["1위 noise", "2위", "3위", "4위"])],
                       fontsize=10.5)
    ax.set_ylabel("Δ RMSE (vs v6, 음수=제거 시 개선=noise)", fontsize=11)
    ax.set_title("Phase 1.5 v7 — 외부지표 Ablation Study\n"
                 "TNX > SKEW > DXY > VVIX 순으로 noise (모두 음수 = 제거 시 개선)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_ylim(-0.36, 0.05)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.subplots_adjust(top=0.85, bottom=0.18)
    out = OUT_DIR / "fig5_v7_ablation.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] {out.name}")


def main() -> None:
    print(f"Phase 1.5 그림 생성 시작 — 출력: {OUT_DIR}")
    fig1_phase15_evolution()
    fig2_multi_asset_rmse_heatmap()
    fig3_ensemble_comparison()
    fig4_v6_external_indicators()
    fig5_v7_ablation()
    print("\n완료. REPORT_Phase1_5_to_3.md 에서 다음 경로로 인라인 참조 가능합니다:")
    for p in sorted(OUT_DIR.glob("*.png")):
        print(f"  ![](REPORT_assets/phase1_5/{p.name})")


if __name__ == "__main__":
    main()
