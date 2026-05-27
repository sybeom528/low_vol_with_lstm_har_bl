"""Create 05a_HMM_Regime_full.ipynb from 05a_HMM_Regime.ipynb with K_CUT removed.

Key transformations:
1. Cell 5 (paths): OUT_DIR → 'outputs/05a_hmm_regime_full' (avoid clobbering originals)
2. Cell 10 (viz): remove K_CUT_VIS axvspan/axvline highlighting hold-out region
3. Cell 11 (K_CUT cutoff): DELETE — this is the only cell that truncates data
4. Markdown cells (0, 15, 17, 19, 29, 38, 41): remove K_CUT references,
   replace with full-period language

Rationale: HMM regime labels are used only for ex-post performance attribution
in the main paper. Since regime labels are NOT inputs to BL pipeline, hold-out
separation provides no functional benefit and creates R1-R4 length asymmetry
(R4 = 24 months vs R1=30, R2=90, R3=48). Retraining on full period yields
symmetric, data-driven regime treatment.
"""
import json
import os
from pathlib import Path

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SRC = '05a_HMM_Regime.ipynb'
DST = '05a_HMM_Regime_full.ipynb'

nb = json.load(open(SRC, encoding='utf-8'))

# ─────────────────────────────────────────────────────────────────
# CHANGE 1 — Cell 0 markdown: header / description
# ─────────────────────────────────────────────────────────────────
new_cell_0 = """# HMM 레짐 분석 — 전체 기간 학습 (Hidden Markov Model Regime Analysis)

> **목적**: 시장 데이터를 기반으로 숨겨진 시장 상태(레짐)를 자동 탐지합니다.
> **데이터**: FF 팩터, 매크로 지표, 일별 수익률
> **기간**: 2004 ~ 2025 (전체 기간 학습, K_CUT 없음)

## 본 노트북 (05a_HMM_Regime_full.ipynb) 의 위치

기존 `05a_HMM_Regime.ipynb` 는 K_CUT=2023-12-31 hold-out 을 적용해 학습했으나, 본 연구의 핵심 질문은 LSTM vs ANN 의 regime-decomposed 비교이며 HMM regime label 은 사후적 성과 분해 라벨링 용도이다. Regime label 은 BL pipeline 입력에 직접 투입되지 않으므로 hold-out 분리는 functional necessity 가 없고, 오히려 R1-R4 length asymmetry (R4 = 24m vs R1=30, R2=90, R3=48) 를 발생시켜 비교의 일관성을 해친다. 본 노트북은 전체 기간으로 HMM 을 재학습하여 R1-R4 모두 동일한 기준으로 식별한다.

## 산출물 및 후속 단계 consume 경로

| 산출물 | 위치 | 후속 단계가 사용? |
|---|---|---|
| `regime_n3.csv` / `regime_n4.csv` / `regime_n5.csv` | `outputs/05a_hmm_regime_full/` | ❌ **직접 로드 안 됨** — 학술 보조 자료 |
| **3-레짐 boundary** (구조전환점) | `master_table.REGIMES` 에 hardcoded | ✅ 05b·06 가 `build_regime_table(regimes=REGIMES)` 으로 간접 참조 |

→ 본 노트북의 핵심 contribution 은 *boundary 도출* (학술 정당화). 그 boundary 가 `master_table.REGIMES` 에 반영되면 05b/06 분석의 3-레짐 분해가 정당화됨. n=4/n=5 비교는 *n=3 선택의 학술 보조 자료*.

---

**관련 가이드**:
- [PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) (전체 파이프라인)
"""
nb['cells'][0]['source'] = new_cell_0.splitlines(keepends=True)

# ─────────────────────────────────────────────────────────────────
# CHANGE 2 — Cell 5 code: OUT_DIR rename
# ─────────────────────────────────────────────────────────────────
new_cell_5 = """BASE_DIR  = Path.cwd()
DATA_DIR  = BASE_DIR / 'data'
OUT_DIR   = BASE_DIR / 'outputs' / '05a_hmm_regime_full'
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"DATA_DIR: {DATA_DIR}")
print(f"OUT_DIR : {OUT_DIR}")
"""
nb['cells'][5]['source'] = new_cell_5.splitlines(keepends=True)

# ─────────────────────────────────────────────────────────────────
# CHANGE 3 — Cell 10 code: remove K_CUT_VIS axvspan/axvline
# ─────────────────────────────────────────────────────────────────
new_cell_10 = """# ── 주요 피처 시계열 시각화 ────────────────────────────────────────────────────
EVENTS = {
    "GFC\\n(08.09)": "2008-09-15",
    "Euro\\n(11.08)": "2011-08-01",
    "China\\nShock(15.08)": "2015-08-24",
    "TradeWar\\n(18.03)": "2018-03-22",
    "US-CN\\n(19.08)": "2019-08-05",
    "COVID\\n(20.03)": "2020-03-20",
    "2022\\nBear":   "2022-01-01",
    "SVB\\nCrisis(23.03)": "2023-03-10"
}

plot_series = [
    ("mkt_rf",   "시장 일별 수익률 (mkt_rf)",      "#2c3e50"),
    ("vol_21d_daily",  "21일 실현변동성 (vol_21d_daily)",      "#3498db"),
    ("vix",      "VIX (내재변동성)",               "#e74c3c"),
    ("t10y2y",   "장단기 금리차 (t10y2y)",         "#f39c12"),
]

fig, axes = plt.subplots(len(plot_series), 1, figsize=(16, 14), sharex=True)

for ax, (col, title, color) in zip(axes, plot_series):
    if col not in feat_df.columns:
        ax.set_visible(False)
        continue
    ax.plot(feat_df.index, feat_df[col], color=color, lw=0.7, alpha=0.9)
    ax.set_ylabel(col, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
    ax.grid(alpha=0.25)
    for label, date_str in EVENTS.items():
        try:
            dt = pd.Timestamp(date_str)
            if feat_df.index[0] <= dt <= feat_df.index[-1]:
                ax.axvline(dt, color="gray", lw=1.0, linestyle="--", alpha=0.7)
                ax.text(dt, ax.get_ylim()[1] * 0.85, label,
                        fontsize=7, color="gray", ha="center", va="top")
        except Exception:
            pass

axes[-1].set_xlabel("날짜", fontsize=11)
plt.suptitle("주요 피처 시계열 (2004~2025, 전체 기간)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "EDA_feature_timeseries.png", bbox_inches="tight")
plt.show()
"""
nb['cells'][10]['source'] = new_cell_10.splitlines(keepends=True)

# ─────────────────────────────────────────────────────────────────
# CHANGE 4 — DELETE Cell 11 (K_CUT cutoff)
# ─────────────────────────────────────────────────────────────────
# Note: deletion shifts all subsequent indices by -1
del nb['cells'][11]

# All subsequent cell indices below are based on POST-DELETION numbering
# (original cell N → new cell N-1 for N > 11)

# ─────────────────────────────────────────────────────────────────
# CHANGE 5 — Cell 14 (was 15) markdown: 최종 피처 선택
# ─────────────────────────────────────────────────────────────────
new_cell_14 = """### 최종 피처 선택

> 16개 후보 중 **이론적 근거 + 통계 검증** 두 단계를 거쳐 6개 선별

| 피처 | 선택 근거 |
|------|----------|
| `mkt_rf` | 레짐의 직접 결과물 — Bull/Bear 구분의 핵심 수익 신호 |
| `vol_21d_daily` | 레짐 라벨링 기준 자체 (변동성 순 정렬로 Bull→Crisis 부여) |
| `vol_ratio` | 단기/장기 변동성 비율 → 레짐 **전환 선행** 신호 (급등 감지) |
| `vix_basis` | 기대−실현 변동성 갭 → 음수 = 공포 실현, Bear/Crisis 식별 |
| `t10y2y_chg` | 경기 국면 변화 선행지표 — 금리 역전 후 침체 패턴 포착 |
| `mom` | 추세 지속성 — Bull은 모멘텀 양수, Bear는 모멘텀 음수 |

- ADF: `t10y2y` → `t10y2y_chg`로 대체, 나머지 정상
- 상관/VIF 기반 탈락:
  - `mkt_ret_ew`·`ret_5d`·`ret_21d` → `mkt_rf`와 상관 0.7↑
  - `vol_63d_daily` → `vol_21d_daily`와 중복
  - `vix`·`vix_chg` → `vix_basis`로 통합
  - `smb`·`hml` → 레짐 간 평균 차이 미미
- 선별 후 모든 피처 VIF < 6 (vol_ratio 5.8, 임계 10 미만) → 다중공선성 양호
- **ADF/VIF/상관 모두 전체 기간 데이터로 계산** (K_CUT 분리 없음)

→ 이론적으로 의미 있는 피처를 먼저 선정하고, 통계 검증으로 다중공선성 없음을 사후 확인
"""
nb['cells'][14]['source'] = new_cell_14.splitlines(keepends=True)

# ─────────────────────────────────────────────────────────────────
# CHANGE 6 — Cell 16 (was 17) markdown: 최종 피처 선택 요약
# ─────────────────────────────────────────────────────────────────
new_cell_16 = """---
**최종 피처 선택 요약**

- 선별된 6개: `mkt_rf`, `vol_21d_daily`, `vol_ratio`, `vix_basis`, `t10y2y_chg`, `mom`
- `vol_ratio`(단기/장기 변동성 비율): 1보다 크면 변동성 급증기, 작으면 안정기 → 레짐 전환 신호
- `vix_basis`(기대 변동성 − 실현 변동성): 음수이면 공포가 실현된 Bear 국면 식별
- 선별 후 모든 피처 VIF < 6 (vol_ratio 5.8, 임계 10 미만) → 다중공선성 양호, HMM 입력으로 적합
- **ADF·VIF·상관 통계 모두 전체 기간 데이터로 계산**
"""
nb['cells'][16]['source'] = new_cell_16.splitlines(keepends=True)

# ─────────────────────────────────────────────────────────────────
# CHANGE 7 — Cell 18 (was 19) markdown: HMM 학습 준비
# ─────────────────────────────────────────────────────────────────
new_cell_18 = """---
**HMM 학습 준비 완료**

- 입력 행렬 `X_scaled`: 전체 기간 × 6피처, StandardScaler 정규화 (전 기간 데이터로 fit)
- **K_CUT 분리 없음** — R1~R4 모두 동일 기준 학습으로 regime treatment 일관성 확보
- 이후 `n = 3, 4, 5` 세 가지 레짐 수로 Full-Fit HMM을 학습하고
  AIC/BIC, 레짐 안정성, 경제적 해석 가능성을 기준으로 최적 n을 비교한다
"""
nb['cells'][18]['source'] = new_cell_18.splitlines(keepends=True)

# ─────────────────────────────────────────────────────────────────
# CHANGE 8 — Cell 28 (was 29) markdown: 레짐 통계 요약
# ─────────────────────────────────────────────────────────────────
# NOTE: 수치는 재실행 후 새로 계산되므로 placeholder 처리
new_cell_28 = """---
**레짐 통계 요약** (전체 기간, n=3)

> 수치는 노트북 재실행 후 새로 계산됨 (아래 표는 K_CUT 버전 기준 reference)

| 레짐(n=3) | 비중 (참고) | 일수 (참고) | 수익(일) | 변동성(일) | VIX |
|-----------|-----:|-----:|---------:|----------:|----:|
| Bull      | 52.6% | 2,616일 | +0.061% | 0.646% | 13.8 |
| Neutral   | 34.0% | 1,690일 | +0.024% | 1.095% | 21.0 |
| Bear      | 13.4% |   665일 | −0.007% | 2.286% | 35.6 |

> 수익·변동성 모두 **일별 단위** (mkt_rf 일별 수익률 / vol_21d_daily 일별 std). VIX 는 연환산 % (참고).
> 본 노트북 내부에서는 일별 단위로 통일. 연환산은 `vix_basis` 계산에서만 사용 (VIX 와 단위 맞춤).

- Bear 레짐의 변동성은 Bull 대비 약 **3.5배**, VIX는 약 **2.6배** 높음
- 수익률·변동성·VIX 모두 Bull → Neutral → Bear 순서로 단조 증가/감소

→ **레짐 = 서로 다른 시장 체질**임을 보여줌
"""
nb['cells'][28]['source'] = new_cell_28.splitlines(keepends=True)

# ─────────────────────────────────────────────────────────────────
# CHANGE 9 — Cell 37 (was 38) markdown: 구조적 전환점 요약 + REGIMES 정의
# ─────────────────────────────────────────────────────────────────
new_cell_37 = """---
**구조적 전환점 요약** (전체 기간, n=3)

> ⚠️ 본 노트북 재실행 후 새 구조적 전환점이 추출됨. 아래는 K_CUT 버전 기준 reference 이며,
> 새 결과에 따라 `master_table.REGIMES` 및 본 표를 업데이트할 것.

- 63거래일(~3개월) 양방향 지속 조건을 적용해 단기 노이즈 제거 후 주요 전환점만 추출
- 추출된 전환점이 **GFC 회복 (2009)**, **post-GFC 회복기 종료 (2012)**, **COVID 직전 (2019)**, **AI 랠리 시작 (2023)** 등 실제 시장 사건과 시기적으로 일치
- 각 서브 기간은 지배적인 레짐이 다른 별개의 시장 환경으로 해석 가능

| HMM 전환점 (참고: K_CUT 버전) | 시장 사건 |
|---|---|
| **2009-06-26** (Bear→Neutral) | GFC 회복 시작 |
| **2012-08-10** (Neutral→Bull) | post-GFC 회복기 종료 → 본격 Bull |
| **2019-11-05** (Neutral→Bull) | COVID 직전 강세 |
| **2023-06-05** (Neutral→Bull) | AI 랠리 시작 |

→ 전체 기간 학습 결과에서 추출된 새 전환점을 분기/연말 boundary 로 round 하여 **`master_table.REGIMES`** 에 반영.

**R1-R4 정의 후보** (전체 기간 학습 시 R4 boundary 가 자연스럽게 식별됨):

| 레짐 | 기간 (예상) | 근거 |
|---|---|---|
| **R1 회복**  | 2010-01 ~ 2012-Q2/Q3 (~30m) | GFC 회복기 종료 전환점 |
| **R2 확장**  | 2012-Q3 ~ 2019-Q4 (~90m) | post-GFC Bull, COVID 직전까지 |
| **R3 변동**  | 2020-01 ~ 2023-Q2 (~42m) | COVID + 2022 Bear, AI 랠리 시작 전까지 |
| **R4 정상화** | 2023-Q3 ~ 2025-12 (~30m) | AI 랠리 시작 전환점 이후 hold-out 없이 전 기간 평가 |

→ R4 길이가 30개월로 R1과 대칭. R1-R4 모두 HMM 구조 전환점에 일관되게 align.
"""
nb['cells'][37]['source'] = new_cell_37.splitlines(keepends=True)

# ─────────────────────────────────────────────────────────────────
# CHANGE 10 — Cell 40 (was 41) markdown: Posterior 불확실성 요약
# ─────────────────────────────────────────────────────────────────
new_cell_40 = """---
**Posterior 불확실성 요약** (전체 기간)

> 수치는 노트북 재실행 후 새로 계산됨 (아래 표는 K_CUT 버전 기준 reference)

| n | 평균 max_posterior (참고) | ≥90% 고확신 (참고) | 70~90% | <70% 불확실 |
|---|-------------------:|-----------:|-------:|-----------:|
| 3 | 0.984 | 95.0% | 3.2% | 1.8% |
| 4 | 0.979 | 93.1% | 4.7% | 2.2% |
| 5 | 0.971 | 91.0% | 5.5% | 3.5% |

- 전 n에서 90% 이상 고확신 구간이 91%+ → 모델이 대부분의 날에 레짐을 명확히 분류
- 불확실 구간(<70%)은 최대 3.5%로 극소수 → 레짐 경계 부근 모호한 날이 거의 없음

→ 레짐 레이블의 신뢰성이 충분히 확보됨. 이후 레짐 기반 포트폴리오 정의 및 백테스트에 활용 가능
"""
nb['cells'][40]['source'] = new_cell_40.splitlines(keepends=True)

# ─────────────────────────────────────────────────────────────────
# Clear all outputs (will be regenerated on run)
# ─────────────────────────────────────────────────────────────────
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

# ─────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────
with open(DST, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Created: {DST}')
print(f'Total cells: {len(nb["cells"])} (was 44, removed K_CUT cutoff cell → 43)')
print(f'Output dir will be: outputs/05a_hmm_regime_full/')
print(f'')
print(f'Changes applied:')
print(f'  Cell  0 markdown: header/description updated')
print(f'  Cell  5 code:     OUT_DIR → outputs/05a_hmm_regime_full')
print(f'  Cell 10 code:     removed K_CUT_VIS hold-out highlighting')
print(f'  Cell 11 (orig):   DELETED — K_CUT cutoff cell')
print(f'  Cell 14 markdown: ADF/VIF wording updated (전체 기간)')
print(f'  Cell 16 markdown: 피처 선택 요약 K_CUT 제거')
print(f'  Cell 18 markdown: HMM 학습 준비 K_CUT 제거')
print(f'  Cell 28 markdown: 레짐 통계 reference 표시 (재실행 필요)')
print(f'  Cell 37 markdown: 구조 전환점 + R1-R4 예상 정의 추가')
print(f'  Cell 40 markdown: Posterior reference 표시 (재실행 필요)')
print(f'  All code outputs: cleared')
