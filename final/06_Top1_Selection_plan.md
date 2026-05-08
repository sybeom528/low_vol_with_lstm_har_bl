# 06_Top1_Selection.ipynb — 구축 plan

> **목적**: 객관적 다중 메트릭 + lexicographic 우선순위 + sensitivity test 절차로 BL 156 cfg 중 Top 1 모델을 재산출.
>
> **배경**: 잠정 Top 1 (`mat_eq_eq_lam_pap`) 의 HOLD_OUT 24m sortino 0.685 (Top 134/153) 부진 → 학습편향 가능성 → 재산출 필요.
>
> **작성일**: 2026-05-08

---

## 1. 의사결정 5개 (사용자 합의)

| 항목 | 결정 | 근거 |
|---|---|---|
| 노트북 위치 | `final/06_Top1_Selection.ipynb` | 정식 분석 산출물, 발표·시연 가능 |
| 1순위 메트릭 | **Sortino_TEST** | 학습 168m 위험조정수익 (downside risk) |
| 2순위 메트릭 | **MDD** (TEST + HOLD_OUT 별도) | 최대 낙폭 — 위험관리 핵심 |
| 3순위 메트릭 | **sortino_ir** | 3-regime mean / std (안정성) |
| 보조 분석 | 모든 다른 지표 포함 | Sortino_HO/FULL, Sharpe, CAGR, Calmar, VaR, turnover, eff_n, sector HHI, alpha, **beta**, IR |
| Universe | TEST 기준 **Sortino_TEST top 50 ∩ sortino_ir top 50** | 성과·안정성 동시 우수 cfg 만 |
| 점수화 | **Lexicographic** (1순위 동순위 → 2순위 → 3순위) | 우선순위 명확, 가중치 임의성 회피 |
| Hard filter | 교집합 분포 확인 후 결정 (§3) | 데이터 보고 결정 (사용자 합의) |

### Lexicographic 정렬 정의 (정밀)

```
1차: Sortino_TEST 내림차순 정렬
  └─ tied 정의: |s1 - s2| < ε (ε = 0.10)
2차: tied 그룹 내 → (rank_MDD_TEST + rank_MDD_HO) / 2 평균 rank 오름차순
  └─ tied 정의: rank 차이 ≤ 1
3차: 그래도 tied 시 → sortino_ir 내림차순
```

**ε = 0.10 선정 근거**
- main 153 cfg 의 sortino_TEST 표준편차 ≈ 0.30
- ε = 0.10 → top 후보군 내 통계적 동순위 그룹 자연스럽게 형성 (1/3σ)
- ε 변경 sensitivity 는 §10 에서 검증 (0.05 / 0.10 / 0.20)

---

## 2. 평가 메트릭 16개 (5 카테고리)

### 2-1. 성과 (Performance) — 7개

| 메트릭 | 의미 | master_table 컬럼 |
|---|---|---|
| Sortino_TEST | 학습 168m downside risk-adjusted | `sortino_TEST` ★ 1순위 |
| Sortino_HOLD_OUT | 미래 24m downside risk-adjusted | `sortino_HOLD_OUT` |
| Sortino_FULL | 전 192m | `sortino_FULL` |
| Sharpe_TEST/HO/FULL | 변동성 기반 risk-adjusted | `sharpe_TEST/HOLD_OUT/FULL` |
| CAGR | 연복리수익률 | `cagr_FULL` |

### 2-2. 위험 (Risk) — 4개

| 메트릭 | 의미 | master_table 컬럼 |
|---|---|---|
| MDD_TEST | 학습 168m 최대 낙폭 | `max_dd_TEST` ★ 2순위-A |
| MDD_HO | 미래 24m 최대 낙폭 | `max_dd_HOLD_OUT` ★ 2순위-B |
| MDD_FULL | 전 기간 최대 낙폭 | `max_dd_FULL` |
| Calmar | CAGR / abs(MDD) | 계산 |

### 2-3. 안정성 (Stability) — 2개

| 메트릭 | 의미 | 출처 |
|---|---|---|
| sortino_ir | 3-regime sortino mean / std | `sortino_ir` ★ 3순위 |
| TEST vs HO 격차 | `abs(s_TEST - s_HO) / s_TEST` | 계산 (overfitting 진단) |

### 2-4. 견고성 (Robustness) — 3개

| 메트릭 | 의미 | 출처 |
|---|---|---|
| turnover_avg | 평균 회전율 | `turnover_avg` |
| eff_n_avg | 유효 보유 종목 수 | `eff_n_avg` |
| sector HHI | 섹터 집중도 | 계산 (weights × sector mapping) |

### 2-5. Alpha (Market) — 3개

| 메트릭 | 의미 | 출처 |
|---|---|---|
| alpha (CAPM) | SPY 대비 초과수익 | bl_functions.compute_alpha |
| beta | SPY 민감도 | bl_functions.compute_alpha |
| Information Ratio | active return / tracking error | 계산 |

### VaR / CVaR (참고용)

VaR_95 / CVaR_95 — 꼬리위험 (5% 분위 손실)

---

## 3. 노트북 §0 ~ §10 구조

### §0. 환경 설정 + 데이터 로딩

**의도**: 153 main cfg + 보조 데이터 메모리 로드, 한글 폰트 표준 적용.

**작업**:
- imports (numpy, pandas, matplotlib, seaborn)
- **한글 폰트 설정** (CLAUDE.md OS 분기 처리 표준)
- master_table.build_master_table → 153 main cfg
- monthly_panel (rf_1m, spy_ret, sector) + daily_returns
- 156 pkl 캐시
- assert: `len(mt_main) == 153`

**출력**: 메모리 dict — `mt`, `mt_main`, `panel`, `daily`, `pkls`

---

### §1. 평가 기준 정의 + narrative

**의도**: 본 plan §2 의 16개 메트릭 + lexicographic 우선순위를 노트북 안에서 명시.

**작업**:
- 메트릭 카테고리 표 출력
- 1/2/3 순위 narrative (왜 sortino_TEST / MDD / sortino_ir?)
- Lexicographic ε=0.10 정의 + 시각화 예시

---

### §2. Universe 정의 — 교집합 추출

**의도**: 153 cfg → top 50 ∩ top 50 = N (예상 25~35) 핵심 후보군 추출.

**작업**:
1. `list_A = mt_main.nlargest(50, 'sortino_TEST')['name']`
2. `list_B = mt_main.nlargest(50, 'sortino_ir')['name']`
3. `intersection = set(list_A) & set(list_B)` → N개
4. cfg 별 메트릭 표 출력 (이름 + sortino_TEST + sortino_ir + 두 rank)

**시각화** (3장):
- `fig01_universe_scatter.png` — 153 cfg sortino_TEST × sortino_ir 산점도, 교집합 강조
- `fig02_intersection_metric_dist.png` — N cfg 의 4핵심 메트릭 box/violin
- `fig03_test_vs_ho_universe.png` — TEST vs HOLD_OUT 산점도 (universe 전체)

**산출**: `outputs/06_top1/intersection_summary.csv`

---

### §3. Hard filter 결정 + 적용 (사용자 합의 단계)

**의도**: 교집합 N cfg 의 분포 보고 추가 필터링 결정.

**작업**:
1. N cfg 의 `max_dd_TEST`, `max_dd_HO`, `eff_n_avg`, `sortino_HO` 분포 통계 출력
2. 추천 필터 후보 평가:

| 후보 filter | 의미 | 예상 탈락 |
|---|---|---|
| `max_dd_TEST > -0.40` | 40% 초과 손실 cfg 제외 | ~3-5개 |
| `max_dd_HO > -0.30` | HO 30% 초과 손실 cfg 제외 | ~5-10개 |
| `eff_n_avg ≥ 30` | 분산 부족 cfg 제외 | ~2-5개 |
| `sortino_HO > 0` | HO 음수 (학습편향 의심) 제외 | ~10-15개 |

3. **사용자 합의 후** filter 조합 적용 → filtered M cfg
4. 시각화 — `fig04_filter_distribution.png` (필터 전후 분포 비교)

**산출**: `outputs/06_top1/filtered_M_summary.csv`

---

### §4. Lexicographic 종합 점수

**의도**: M cfg 에 1/2/3 순위 적용 → 최종 정렬.

**작업**:
1. 1단계: Sortino_TEST 내림차순 → 동순위 (ε=0.10) 그룹화
2. 2단계: 그룹 내 (rank_MDD_TEST + rank_MDD_HO) / 2 → 동순위 (Δ≤1) 그룹화
3. 3단계: sortino_ir 내림차순 → 최종 결정
4. 최종 점수표 (M cfg, 정렬됨)

**시각화**:
- `fig05_lexicographic_heatmap.png` — M cfg × 4 메트릭 rank heatmap

---

### §5. Top 10 후보 정밀 분석

**의도**: §4 의 상위 10개 cfg 에 대해 16개 메트릭 종합 표 + heatmap.

**작업**:
1. Top 10 cfg 추출
2. 16개 메트릭 종합 (z-score 정규화)
3. heatmap (Top 10 × 16 metric)

**시각화**:
- `fig06_top10_metric_heatmap.png` — z-score heatmap
- `fig07_top10_metric_radar.png` — radar chart (Top 5)

**산출**: `outputs/06_top1/top10_metrics.csv`

---

### §6. 안정성 + 견고성 검증

**의도**: Top 10 의 regime / overfitting / time-series 행동 검증.

**작업**:
- regime sortino bar chart (R1/R2/R3, Top 10)
- TEST vs HOLD_OUT 산점도 (Top 10)
- 누적수익 곡선 (Top 5 + SPY)
- 월별 수익 분포 violin

**시각화** (4장):
- `fig08_regime_sortino_top10.png`
- `fig09_test_vs_ho_top10.png`
- `fig10_cumulative_return_top5.png`
- `fig11_monthly_return_dist_top5.png`

---

### §7. 위험 분석

**의도**: drawdown + sector concentration + crisis behavior.

**작업**:
- drawdown 곡선 (Top 5)
- sector HHI 비교 + 2024-12 sector rotation 영향
- 회전율 / eff_n 안정성

**시각화** (3장):
- `fig12_drawdown_top5.png`
- `fig13_sector_hhi_top5.png`
- `fig14_2024_12_sector_loss.png`

---

### §8. baseline 우월성 검증

**의도**: SPY / equal_weight / naive_lowvol / mvp_60m 대비 alpha + IR.

**작업**:
- baseline 4개 정의
- alpha (CAPM) + beta + IR (Top 5)
- bootstrap 95% CI (alpha) — 1000회 resampling

**시각화**:
- `fig15_alpha_with_ci.png`

---

### §9. Top 5 → Top 1 결정

**의도**: 결정 matrix 점수 합산 → Top 1 narrative.

**작업**:
1. 결정 matrix (Top 5 × 5 차원)

| 차원 | 가중치 | 메트릭 |
|---|---|---|
| 성과 | 30% | Sortino_TEST + Sortino_HO |
| 위험 | 25% | MDD_TEST + MDD_HO + Calmar |
| 안정성 | 20% | sortino_ir + TEST/HO 격차 |
| 견고성 | 15% | eff_n + HHI + turnover |
| Alpha | 10% | alpha + beta proximity to 0.7 (defensive) |

2. 점수 합산 → Top 1 선정
3. narrative 작성 (선정 이유 + 한계점)

**산출**:
- `outputs/06_top1/top5_decision_matrix.csv`
- `final/_top1_decision_2026_05_08.md` (의사결정 보고서)

---

### §10. Sensitivity test (견고성 추가 검증)

**의도**: Top 1 결정이 평가 기준 변경에 robust 한지 검증.

**작업**:
1. **lexicographic ε 변경**: 0.05 / 0.10 / 0.20 → Top 1 변경 여부
2. **우선순위 변경**: sortino_HO 1순위 시 → Top 1 변경 여부
3. **sub-period 검증**: 2015~ TEST / 2018~ TEST / 2020~ TEST → Top 1 변경 여부
4. **quantile variants 비교**: q55 / q64 / q70 시 Top 1 유사도

**결론 분류**:
- ✓ robust — 모든 변경 시 Top 1 유지
- △ 조건부 — 일부 변경 시 Top 1 변경 (조건부 권고)
- ✗ unstable — 변경 민감, 추가 검증 필요

**산출**: `outputs/06_top1/sensitivity_summary.csv`

---

## 4. 산출물 명세

| 경로 | 형식 | 단계 |
|---|---|---|
| `final/06_Top1_Selection.ipynb` | ipynb | §0~§10 |
| `final/outputs/06_top1/intersection_summary.csv` | csv | §2 |
| `final/outputs/06_top1/filtered_M_summary.csv` | csv | §3 |
| `final/outputs/06_top1/top10_metrics.csv` | csv | §5 |
| `final/outputs/06_top1/top5_decision_matrix.csv` | csv | §9 |
| `final/outputs/06_top1/sensitivity_summary.csv` | csv | §10 |
| `final/outputs/06_top1/figures/fig01~fig15.png` | png (15장) | §2~§10 |
| `final/_top1_decision_2026_05_08.md` | md | §9 후 |

---

## 5. 실행 순서

1. **Step 1** (현 단계): `final/06_Top1_Selection_plan.md` 작성 ✓
2. **Step 2**: 노트북 신규 + §0~§2 — Universe 교집합 추출
3. **Step 3**: §3 분포 시각화 → **사용자 합의** (hard filter 결정)
4. **Step 4**: §4~§5 — Lexicographic + Top 10 정밀
5. **Step 5**: §6~§8 — 안정성 + 위험 + baseline
6. **Step 6**: §9 — Top 5 → Top 1 → **사용자 검증**
7. **Step 7**: §10 — sensitivity test
8. **Step 8**: 의사결정 보고서 → **사용자 최종 승인**
9. **Step 9** (선택, 사용자 승인 후): commit + push

각 Step 완료 시 사용자 검증 후 다음 단계 진행 (CLAUDE.md 대화형 진행 원칙).

---

## 6. 명시적 비포함 (스코프 외)

- ❌ Streamlit 갱신/재구축 — 별도 plan
- ❌ PROJECT_OVERVIEW.md 갱신 — Top 1 결정 후 별도 turn
- ❌ 새 BL 실험 정의 (`bl_config.py` 무수정)
- ❌ 추가 LSTM 재학습
- ❌ commit/push — 모든 작업 완료 + 사용자 승인 후만

---

## 7. 다음 단계 (본 plan 완료 후)

1. **`narrative.py` + `recommendations.py` 갱신** — Streamlit 재구축 시 활용
2. **Streamlit 단계적 재구축** — 별도 plan
3. **`PROJECT_OVERVIEW.md` 본문 갱신** — Top 후보 명시 + 발표 narrative
