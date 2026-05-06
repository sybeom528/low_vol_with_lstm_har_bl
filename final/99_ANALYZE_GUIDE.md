# 99_analyze.ipynb 셀별 해설 가이드

> **최종 갱신: 2026-05-06** — 24-cell 구조, K2 통합 대시보드(Top 20 × 5 레짐 단일 figure), composite_rank/scatter 제거, Q 민감도 추가 반영. 현재 EXPERIMENTS **214**개.

본 문서는 [`99_analyze.ipynb`](99_analyze.ipynb)의 각 셀이 *무엇을 하고 / 어떻게 해석할지 / 결과의 의미*를 설명합니다.

## 사전 준비

- `final/results/*.pkl` 백테스트 결과 214개 (Q 민감도 39 포함)
- `master_table.py` (mt/rt 빌더), `analyze_plots.py` (시각화)
- 5-레짐 정의: [master_table.REGIMES_5](master_table.py)

---

## 명명 체계 두 가지

| 컬럼 | 형식 | 예 |
|---|---|---|
| `name` | 디스크 파일명 (mat_안: 4-토큰 / 밖: semantic) | `mat_eq_mcap_lam_he` / `baseline` |
| `canonical` | 분석용 5-토큰 `{prior}_{p_mode}_{p_weight}_{q}_{omega}` | `eq_ls_mcap_lam_he` |

분석 표·차트는 `canonical` 사용.

| 토큰 | 가능 값 |
|---|---|
| prior | `mcap` / `eq` / `rp` |
| p_mode | `tr` (trailing) / `ls` (LSTM) |
| p_weight | `mcap` / `eq` / `rp` / `volm` |
| q | `fix` / `lam` / `raw` / `inv` / `vsp` / `none` / `capm` |
| omega | `he` / `pap` / `rms` / `sh` (scaled_half) / `sd` (scaled_double) |

---

# 셀별 해설 (24 cells)

## [Cell 0] 머리말 (markdown)
노트북 4-단계 구조 요약: ① 빠른 진단 → ② 거래비용 → ③ 섹터 구성 → ④ J/K Master Table 분석.

## [Cell 1] import + 데이터 로드
- `loaded` dict: 모든 pkl 로드
- `rf` Series: 월별 무위험 수익률
- `spy_ret`: SPY 월별
- `calc(name)`, `calc_spy()`: Sharpe/CAGR/MDD/Sortino 헬퍼

> ⚠️ 이후 셀들은 모두 `loaded`, `rf`, `spy_ret`, `RESULTS_DIR`, `OUT_DIR`을 참조. **Cell 1 실패 시 모두 실패.**

---

## [Cell 2-3] 1. 빠른 진단

**Cell 2 (markdown)**: 섹션 안내.

**Cell 3 (code)**: 214개 실험을 Sharpe 내림차순 → **Top 10 / Bottom 5** + 핵심 비교군(SPY, baseline, capm_no_bl, naive_lowvol).

**해석법**:
- Top 10 패턴(슬롯 조합) 직관 파악
- baseline 등수 → 표준 BL 위치
- naive_lowvol·capm_no_bl 등수 → BL 가치
- SPY 비교 → 액티브 펀드 가치

---

## [Cell 4-6] 2. 거래비용 분석

**Cell 4 (markdown)**: TC 정의 + 본 백테스트 가정 (TC_RATE=0.001=10bp 편도).

**Cell 5 (code, I1+I2)**: 슬롯별 Turnover & TC 집계. 5개 슬롯(prior, p_mode, p_weight, q, omega) × 평균 turnover + 연간 TC.

**산출**: 슬롯별 표 5개 + boxplot 5장 + 평균 TC Top 15 + Turnover 최저/최고 5.

**해석 인사이트** (현 데이터):
- **q=ff3 폭망**: turnover 1.39, 연 TC 1.67% (다른 q ~0.5)
- **omega=pap turnover 0.81 vs rms 0.37** → omega=pap는 sharpe 우위지만 TC 부담 ~2배
- **trailing이 LSTM보다 더 잦음** (예상 반대) — trailing은 직접 vol에 반응, LSTM은 부드러운 예측
- **prior, p_weight 차이 미미**

> ⚠️ TC는 mt/rt의 sharpe/sortino에 **이미 차감 반영**됨 (`net_ret = gross - turnover×tc`). 본 셀은 turnover 자체 관찰용.

**Cell 6 (code, I3)**: 4 페어(`baseline-mat_mcap_mcap_fix_he` 등)의 trailing vs LSTM Turnover 직접 비교 막대.

---

## [Cell 7-8] 3. Trailing vs LSTM 섹터 구성 차이

**Cell 7 (markdown)**: 섹터 구성 비교 의의 — 같은 슬롯에서 trailing/LSTM이 *어느 섹터에 비중 다르게 주는가*. 성과(J/K)와 별개의 portfolio *구성* 측면.

**Cell 8 (code, F)**: GICS 11 섹터 × 4 페어 비중 비교 stacked bar.

**해석법**:
- LSTM이 IT/통신 등 변동성 큰 섹터에 비중 높으면 → "고변동 추정" 편향
- 헬스케어/필수소비재 비중 높으면 → 저변동 anomaly 충실
- 차이 크면 → 둘은 *완전 다른 portfolio* 생성

---

## [Cell 9-15] 4. J. Master Table 통합 파이프라인

### Cell 9 (markdown) — J 섹션 개요
파이프라인: results/*.pkl → mt(214 × ~32 col) → 슬롯/매트릭스/Top-N/위기/IR 분석.

### Cell 10 (J1) — Master Table 빌드 + Top 20

**산출**: `mt` (DataFrame, ~214행 × 32+ 컬럼), Top 20 표 (sharpe).

**해석법**: Top 20에서 자주 보이는 *canonical* 패턴 → robust 후보군.

### Cell 11 (J2) — 슬롯별 marginal effect

**무엇을**:
- **J2-A**: 슬롯 5개 × Sharpe 분포 (boxplot)
- **J2-B**: 슬롯 5개 × Sortino 분포 (boxplot)

**해석법**:
- 박스 분산 큼 → 슬롯 효과 큼 (선택 시 신경)
- 박스 차이 작음 → noise (baseline 값 고정 가능)
- Sharpe vs Sortino 1위 다르면 → 평균 vs 하방 위험 차이

**현재 핵심**:
- **q_mode 격차**: Sharpe 0.47 < Sortino 0.96 → Sortino 측면에서 q가 결정적
- omega: Sharpe pap 1위, Sortino sh 1위
- **prior·p_mode 거의 무시 가능**

### Cell 12 (J3) — 108-cell 매트릭스 히트맵

**무엇을**:
- 매트릭스 = LSTM 고정 + 4 슬롯 정형 조합 (prior 3 × p_weight 3 × q 4 × omega 3) = 108
- **행 9개**: `prior_p_weight` 조합
- **열 12개**: `q_mode_omega` 조합
- **셀**: LSTM 백테스트 평균 sharpe (녹=좋, 적=나쁨)

**해석 단계**:
1. 가장 진한 녹 셀 → sweet spot
2. 행 평균 → 어느 prior×pweight 우세?
3. 열 평균 → 어느 q×omega 우세?
4. 한 행 전체 녹 → 그 prior×pweight가 q/omega 무관히 강함
5. 행/열 평균은 좋은데 한 셀만 적 → 그 q×omega 부적합

**한계**: TC 미반영(명목 sharpe). pap 열은 turnover ↑ → 실수익 차감 후엔 격차 줄어듦.

### Cell 13 (J4) — Top-N 정밀 분석

**무엇을**: sortino_ir / sharpe_ir 두 정렬로 Top 5의 Equity Curve / Rolling Sharpe / Drawdown 3-panel.

**해석**:
- Equity (log scale): SPY 점선 대비 누적수익
- Rolling Sharpe (12m): 시간 일관성, 위기 시 음수 빠지는지
- Drawdown: 깊이 + 회복 시간

### Cell 14 (J5) — 위기구간 행동 비교

**무엇을**: 2018Q4 변동성 쇼크 / 2020 COVID / 2022 인플레 3 위기 구간에서 Top 5의 cum_ret · MDD · 최악월.

**해석**: 2020 COVID +수익 = 강력 방어. 2022 인플레에서 SPY(-8%) 대비 우위 = 본 프로젝트 핵심 가치.

### Cell 15 (J6) — 벤치마크 IR 표

**무엇을**: 후보별 baseline / capm_no_bl / naive_lowvol / SPY 대비 IR (Information Ratio).

**해석**:
- IR > 0.4 = 의미 있는 부가가치
- vs `capm_no_bl`: BL 효과
- vs `naive_lowvol`: 모델(BL+ML) 가치
- vs SPY: 액티브 펀드 가치

---

## [Cell 16-23] 5. K. 레짐 안정성 분석

### Cell 16 (markdown) — K 섹션 개요
**왜 Sortino**: 저변동 anomaly 컨셉 → 하방 변동성만이 진짜 위험. **왜 5-레짐**: 거시환경 차이가 큰 시장 국면(R1 회복 / R2 확장 / R3 COVID / R4 베어 / R5 AI랠리).

### Cell 17 (K1) — 레짐 테이블 빌드 + 4-view + 비교군

**산출**:
- `rt`: build_regime_table 결과 (~214 × 50+ 컬럼) — 레짐별 sortino/sharpe/mdd + 안정성 지표
- 비교군 4종(SPY/baseline/capm_no_bl/naive_lowvol) 5-레짐 종합 메트릭 표
- 비교군 레짐별 Sortino/Sharpe/MDD 표 3개
- 4-view 정렬 Top 5 × 4

**4-view**:
| view | 정렬 키 | 의미 |
|---|---|---|
| ① | `sortino_min` | 최악 레짐 보호 (worst case) |
| ② | `sortino_mean` | 평균 (outlier 위험) |
| ③ | `stability_score` | mean − std (균형, std 5표본 noisy) |
| ④ | `sortino_ir` | mean / std (변동 대비 평균, 스케일 무관) |

**현재 비교군 핵심**:
- baseline R3 COVID Sortino 2.86 (SPY 1.27의 2배+) ← **위기 방어 명확**
- baseline R5 AI랠리 0.66 (SPY 3.57의 1/5) ← **메가캡 강세장 약점**
- capm_no_bl R1 회복 MDD -22% (SPY -16%) ← **BL 없으면 회복기 위험**

### Cell 18 (K2) — Sortino IR Top 20 통합 대시보드

**산출**: `plot_styled_regime_dashboard(rt, rank_by='sortino_ir', top_n=20)` — 단일 figure에 SORTINO/SHARPE/MDD 3 panel × 5 레짐, Top 20 행. 비교군 우위 카운트(`n_beat_SPY/baseline/capm_no_bl/naive_lowvol`) 함께 출력.

**해석법**:
- 좌→우: Sortino / Sharpe / MDD 순. 녹=좋음, 적=나쁨
- 행 라벨: `1. mcap_ls_rp_vsp_rms` 형식 (rank + canonical)
- 모든 레짐에서 일관 녹색 = 진짜 robust

> ⚠️ K2-A(sortino_min 정렬)·K2-C(composite_rank)·K2-D(교집합)·K2-E(beat 표)·K2-F(MDD)·K2-G(Sharpe) 모두 **단일 K2 통합 대시보드로 흡수됨**(2026-05-06 리팩토링).

### Cell 19 (K2-H markdown)
Sharpe IR(=sharpe_mean/sharpe_std) 정렬의 정당화 — sortino_ir와 다른 측면 보강.

### Cell 20 (K2-H code) — Sharpe IR Top 20 대시보드 + IR 교집합

**산출**: 동일 dashboard 함수, `rank_by='sharpe_ir'`. + `sortino_ir Top 20 ∩ sharpe_ir Top 20` 텍스트 출력.

**해석**: 두 IR 모두에 등장 = 진짜 robust 후보 풀.

**현재**: 교집합 ~16개 (80% 중복). RP 가중 7 + EQ+pap 5 등.

### Cell 21 (K3) — MDD 5-레짐 히트맵

**정렬**: `mdd_worst` (5 레짐 중 가장 깊은 MDD가 얕은 순). 비교군 MDD per 레짐 함께.

**해석**: 어느 레짐에서도 -10% 이내 = 안정형 1순위.

### Cell 22 (K5) — 레짐별 위너 + 종합 위너

**무엇을**: 각 레짐 Sortino Top 5 + 종합 안정성(stability_score) Top 5 → 6개 표.

**해석**: 같은 후보가 여러 레짐 등장 = robust. 한 레짐만 등장 = 운.

### Cell 23 (K6) — 위험성향별 최종 매핑

**무엇을**: 3 페르소나(공격/균형/안정) × 후보 추천.

| 페르소나 | 정렬 키 | 해석 |
|---|---|---|
| 공격형 | `sortino_mean` 최고 | 평균 우선, 변동 감수 |
| 균형형 | `stability_score` 최고 | 평균과 안정성 균형 |
| 안정형 | `sortino_min > 0` 필터 + `mdd_worst` 가장 얕음 | 최악 보호 |

**해석**: 각 페르소나 1순위 = 펀드 상품 후보. K1 4-view 패턴과 일관되어야 신뢰.

---

# 분석 흐름 — 권장 순서

```
[Cell 1 import]
   ↓
[Cell 3 빠른 진단]      ← 1분: 전체 윤곽
   ↓
[Cell 10 J1]           ← Master Table 빌드 (mt 생성)
   ↓
[Cell 11 J2]           ← 슬롯 효과
   ↓
[Cell 12 J3]           ← 108-cell 매트릭스
   ↓
[Cell 17 K1]           ← 레짐 테이블 (rt 생성) + 4-view + 비교군 상세
   ↓
[Cell 18 K2]           ← Sortino IR Top 20 통합 대시보드
   ↓
[Cell 20 K2-H]         ← Sharpe IR Top 20 + IR 교집합
   ↓
[Cell 23 K6]           ← 위험성향 매핑

(부가)
[Cell 5,6 Turnover]    ← 거래비용 검증
[Cell 8 F]             ← 섹터 구성
[Cell 13 J4]           ← Top 5 정밀 곡선
[Cell 14,15 J5,J6]     ← 위기 + 벤치마크
[Cell 21 K3, 22 K5]    ← MDD / 레짐 위너
```

---

# 산출 파일 정리

```
final/outputs/99_analyze/
├── J2A_marginal_sharpe.png       슬롯 boxplot (Sharpe)
├── J2B_marginal_sortino.png      슬롯 boxplot (Sortino)
├── J3_matrix_heatmap.png         108-cell 매트릭스
├── J4A_top5_sortino_ir.png       Top 5 정밀 (sortino_ir 기준)
├── J4B_top5_sharpe_ir.png        Top 5 정밀 (sharpe_ir 기준)
├── K2A_sortino_ir_dashboard.png  K2 — Sortino IR Top 20 × 3 metric × 5 레짐
├── K2B_sharpe_ir_dashboard.png   K2-H — Sharpe IR Top 20
├── K3_regime_mdd_heatmap.png     K3 — MDD 레짐별
└── (기타 비교/위기 차트)
```

대표 차트:
- 슬롯 효과: `J2A` + `J2B`
- 매트릭스: `J3`
- 레짐 안정성: `K2A` + `K2B`
- 위험성향 매핑: K6 텍스트 출력

---

# 의존성 및 모듈

- `master_table.py`: `build_master_table`, `build_regime_table`, `parse_config`, `slot_summary`, `matrix_pivot`, `regime_metrics`, `REGIMES_5`, `PERIODS_DEFAULT`
- `analyze_plots.py`: `plot_marginal_effects`, `plot_matrix_heatmap`, `plot_top_n_analysis`, `plot_styled_regime_dashboard`, `crisis_comparison`, `benchmark_table`, `regime_winners_table`
- `bl_functions.py` (간접): `compute_metrics`, `compute_turnover`, `apply_tc`

새 실험 추가 후 (예: Q 민감도 39 완료):
```python
mt = build_master_table(...)   # J1 다시 실행 → 자동 갱신
rt = build_regime_table(...)   # K1 다시 실행
```

---

# 참고 문서

- [BL_EXPERIMENT_GUIDE.md](BL_EXPERIMENT_GUIDE.md): 실험 정의/슬롯 가이드
- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md): 프로젝트 개요
- [../김윤서/low_risk/04_Prior_Universe_Analysis.ipynb](../김윤서/low_risk/04_Prior_Universe_Analysis.ipynb): 백테스트 *전* 사전 EDA
- [../김윤서/low_risk/regime_analysis.py](../김윤서/low_risk/regime_analysis.py): 레짐 정의 원본
