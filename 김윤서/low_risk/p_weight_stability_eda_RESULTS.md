# P 가중치 4종 안정성 EDA — 결과

> 분석 스크립트: [p_weight_stability_eda.py](p_weight_stability_eda.py)
> 출력: [outputs/pweight_eda/](outputs/pweight_eda/)
> 비교 대상: `prior_eq + LSTM`을 고정한 채 P 가중치만 4종으로 바꾼 8개 실험
> (q_lambda × {mcap, eq, rp, vol_mcap}, q_raw_lam × {mcap, eq, rp, vol_mcap})

---

## 0. TL;DR

| 질문 | 결론 |
|---|---|
| 가장 평탄(서브기간 std 최소) | **rp** (q_lambda 0.287, q_raw_lam 0.302) — mcap이 아님 |
| 모든 서브기간 0.9+ 유지 + 전체 Sharpe 최고 | **mcap** (full 1.140 / 1.148, 2020-2024 0.90) |
| 2020-2024에 무너진 스킴 | **vol_mcap** (Sharpe 0.526) — 메가캡 short 노출이 원인 |
| eq의 약점 | 30% 컷 안의 소형주가 동일가중으로 소음 유입, full Sharpe 1.05로 떨어짐 |

핵심: "mcap이 가장 평탄"은 정확하지 않습니다. **rp가 평탄도(std)는 가장 낮지만, mcap은 어떤 서브기간에서도 무너지지 않으면서(2020-2024 단독 0.9+) 전체 Sharpe도 최고**라는 게 정답입니다.

---

## 1. 4개 P 가중치의 개념적 차이

`build_P()`는 변동성(여기선 LSTM 예측 vol)으로 종목을 정렬한 뒤 4가지 방식 중 하나로 가중치를 부여합니다. [bl_functions.py:71-121](../../final/bl_functions.py#L71-L121)

| 스킴 | 유니버스 | 가중치 공식 | 의미 |
|---|---|---|---|
| **mcap** | 변동성 하위/상위 30%만 | long: m_i / Σm_low,  short: −m_i / Σm_high | 30% 컷 + 시총 비례. 큰 종목(메가캡) 위주, 그룹 안에서만 시총 가중 |
| **eq** | 변동성 하위/상위 30%만 | long: 1/n,  short: −1/n | 30% 컷 + 동일가중. 컷 안의 소형주도 메가캡과 동일 비중 |
| **rp** | 변동성 하위/상위 30%만 | long: (1/σ_i) / Σ(1/σ_low),  short: −σ_i / Σσ_high | 30% 컷 + 역변동성/변동성. 가장 안정한/불안정한 종목에 더 큰 베팅 |
| **vol_mcap** | **전체 유니버스** | (1−r_i)·m_i 정규화 long, r_i·m_i 정규화 short (r=vol pct rank) | 컷 없는 연속 함수. 모든 종목이 long+short 양쪽에 vol rank 비례로 분포 |

가장 큰 차이는:
- **mcap/eq/rp**는 30% 컷이 있어 종목이 long 또는 short 한쪽에만 들어감 (또는 0)
- **vol_mcap**은 컷이 없어 한 종목이 양쪽에 모두 들어감(net으로만 long/short)

---

## 2. P-벡터 진단 (180개월 평균)

`outputs/pweight_eda/pvec_summary.csv`

| 스킴 | eff_n (1/HHI) | max\|w\| | top10 share | turnover | n_nonzero |
|---|---:|---:|---:|---:|---:|
| eq | **257.3** | 0.008 | 0.039 | 0.532 | 257.3 |
| rp | 85.9 | 0.133 | 0.258 | 0.651 | 257.3 |
| vol_mcap | 65.7 | 0.078 | 0.311 | **0.289** | 430.4 |
| mcap | 52.8 | 0.186 | 0.332 | 0.740 | 257.3 |

[01_pweight_concentration.png](outputs/pweight_eda/01_pweight_concentration.png), [02_pweight_turnover.png](outputs/pweight_eda/02_pweight_turnover.png)

해석
- **eq는 분산도(eff_n=257) 최고지만 max\|w\|가 0.008로 시그널이 거의 균등화됨** — 컷 안 작은 종목까지 동일 비중이라 BL 사후 mu에 던져도 view 강도가 약하고 노이즈 비중↑
- **rp는 30% 컷 안에서 1/σ 가중**이라 가장 안정한 종목 1~2개에 베팅이 집중됨. top10 share 26%, max\|w\| 0.133. 시그널 자체는 강하지만 한 종목의 σ 변화에 민감
- **mcap은 max\|w\| 0.186, top10 33%**로 가장 집중 — 하지만 그 집중이 **시총 큰 종목**으로 가는 거라 거래/추정 노이즈 측면에서는 가장 안정한 곳에 집중
- **vol_mcap은 turnover 0.289로 압도적으로 낮음**(연속 함수 + 전체 유니버스이므로 vol rank 살짝 변해도 많은 종목으로 분산), 하지만 long/short이 같은 종목에 동시에 들어가면서 **시그널 희석**

---

## 3. 서브기간 Sharpe (q_lambda)

`outputs/pweight_eda/sharpe_q_lambda.csv`, [05_subperiod_sharpe.png](outputs/pweight_eda/05_subperiod_sharpe.png)

| 기간 | mcap | eq | rp | vol_mcap |
|---|---:|---:|---:|---:|
| 2010-2014 | 1.707 | **1.795** | 1.225 | 1.794 |
| 2015-2019 | 0.927 | 0.892 | **1.153** | 0.883 |
| 2020-2024 | **0.902** | 0.708 | 0.696 | 0.526 |
| **full** | **1.140** | 1.053 | 1.011 | 1.015 |
| **서브 std** | 0.458 | 0.582 | **0.287** | 0.654 |

q_raw_lam도 거의 동일 패턴:

| 기간 | mcap | eq | rp | vol_mcap |
|---|---:|---:|---:|---:|
| 2010-2014 | 1.715 | 1.793 | 1.226 | **1.804** |
| 2015-2019 | 0.972 | 0.932 | **1.258** | 0.943 |
| 2020-2024 | **0.903** | 0.716 | 0.720 | 0.526 |
| **full** | **1.148** | 1.069 | 1.052 | 1.039 |
| **서브 std** | 0.450 | 0.570 | **0.302** | 0.652 |

읽는 법
- "어떤 환경에서도 지지 않는 것"을 원한다면 → **mcap** (모든 칸 0.9 이상)
- "Sharpe 변동성 자체가 작은 것"을 원한다면 → **rp**
- 두 기준 다 만족시키지는 못함 — rp는 2010-2014에서 1.22로 떨어지고, mcap은 2010-2014에서 1.71로 튀어 std가 커짐

[06_cumulative_returns.png](outputs/pweight_eda/06_cumulative_returns.png) 누적수익도 mcap이 q_lambda·q_raw_lam 양쪽 모두 1위.

---

## 4. 왜 이런 차이가 나오는가 — 메커니즘 근거

### 4-1. eq가 2020-2024에 깨진 이유 (Sharpe 0.708)
30% 컷 안에서 동일가중 → low-vol 30%에 들어간 **소형 ETF/방어주**가 메가캡과 같은 1/n 비중. 2020-2024는 megacap 차별화가 극심한 시기라 동일가중은 메가캡 익스포저를 희석해 손해.

### 4-2. rp가 2010-2014에 약했던 이유 (Sharpe 1.22)
1/σ 가중 → 그 시기 **low-vol 채권형/방어주(KO, JNJ 같은 σ 0.01-0.015)**에 비중 집중. 회복장(2010-2014)에서 risk-on 자산이 더 강했기 때문에 risk-off로 치우친 가중이 mcap/eq/vol_mcap 대비 ~0.5 sharpe 손실.

### 4-3. **vol_mcap이 2020-2024에 무너진 이유 (Sharpe 0.526)** ← 가장 결정적
vol_mcap은 **컷 없이** 모든 종목을 두 basket(long: (1−r)·m, short: r·m)에 양쪽 다 넣고, BL에 입력되는 P는 두 basket의 차이값입니다. 2020-2024 메가캡이 high-vol 구간에 잡히면서 net P가 큰 음수 → 실질 short 노출이 컸습니다.

월말 스냅샷 — **net P 기준 실질 short top3** (P[i] = long_w[i] − short_w[i], 음수일수록 short):

| 날짜 | net P short top1 | top2 | top3 |
|---|---|---|---|
| 2020-12-31 | **AAPL net −10.7%** | DIS −2.7% | GE −2.6% |
| 2022-12-31 | **AAPL net −6.0%** | AMZN −4.0% | TSLA −2.7% |
| 2024-12-31 | **NVDA net −8.8%, GOOGL net −6.7%** | TSLA −4.7% | AMZN −4.3% |

> 참고: short basket 비중(`r·m`/Σ만으로 본 값)은 더 큼 (AAPL 2020: 14.4%, NVDA 2024: 10.7%). BL에 들어가는 건 long basket을 차감한 net 값이라 위 표가 정확한 실효 노출.

→ AAPL은 2020-2024에 +330%, NVDA는 2024년만 +170%. vol_mcap은 이걸 net 한 자릿수 후반 ~ 두 자릿수 % short으로 들고 있어 직격탄. mcap의 30% 컷 방식은 메가캡이 high-vol 30% 안에 들어가는 경우만 short(컷 밖이면 P=0)이라 short 노출이 훨씬 분산됨.

### 4-4. mcap이 2020-2024에 살아남은 이유 (Sharpe 0.90)
2020-12 / 2024-12에 메가캡(MSFT, AAPL, BRK-B)이 **low-vol 30% 안**에 들어왔습니다.

| 날짜 | low-30 안의 top mcap |
|---|---|
| 2020-12-31 | MSFT 1.6T (low30 안에 위치) |
| 2022-12-31 | BRK-B 0.7T |
| 2024-12-31 | AAPL 3.8T |

mcap은 30% 컷 + 시총 비례라 이 메가캡들에 long 비중을 크게 주고, high-vol 30%에는 메가캡이 적게 들어가 short 노출이 분산됨. → 2020-2024 메가캡 장세에 **자연스럽게 정렬**.

### 4-5. 왜 mcap이 평탄해 "보이는가"
서브기간 std로는 rp가 더 낮은 게 사실인데, **두 q_mode(q_lambda·q_raw_lam) 양쪽에서 모든 서브기간 Sharpe가 0.9 이상으로 유지되는 건 mcap뿐**입니다. 히트맵을 색감으로 볼 때 빨간(저성과) 칸이 mcap에는 없고 다른 스킴들엔 한두 칸씩 있어서 "평탄해 보이는" 인상을 줍니다. 통계량 std로는 측정 안 되는 "최저 Sharpe(min) 가장 높음" 효과:

| 스킴 | min(서브 Sharpe) | max(서브 Sharpe) |
|---|---:|---:|
| mcap | **0.902** | 1.715 |
| rp | 0.696 | 1.258 |
| eq | 0.708 | 1.795 |
| vol_mcap | 0.526 | 1.804 |

→ **mcap의 min Sharpe(0.90)가 다른 모든 스킴의 min(0.53–0.71)보다 압도적으로 높음**. 이게 "평탄"의 진짜 의미.

---

## 5. P 시그널 상관

[04_pweight_signal_corr.png](outputs/pweight_eda/04_pweight_signal_corr.png)

대각 1 외 주요 쌍의 Pearson 상관:
- **mcap ↔ eq ≈ 높음** (같은 30% 종목, 가중치만 다름)
- **mcap ↔ rp ≈ 중간** (같은 30% 종목, 시총 vs 1/σ)
- **eq ↔ vol_mcap, rp ↔ vol_mcap ≈ 낮음~음수** (vol_mcap만 컷 없는 연속 함수라 메가캡 short 비중 다름)

상관이 가장 갈라지는 게 vol_mcap — 다른 셋과 다른 시그널을 만든다는 뜻이고, 그 다른 시그널이 2020-2024에 부정적으로 발현됨.

---

## 6. 결론 — 발표용 한 줄

> "mcap+30%컷"이 다른 스킴 대비 Sharpe **변동성**(std)이 가장 작은 건 아니지만(그건 rp), **모든 서브기간 0.9+ 유지하면서 전체 Sharpe도 최고**로 나오는 건 mcap뿐이다. 이유는 (1) 30% 컷이 short 사이드를 분산시켜 vol_mcap처럼 메가캡을 대량 short하지 않고, (2) 그룹 안 시총가중이 eq처럼 소형주 노이즈를 끌어들이지 않으며, (3) rp처럼 한 종목의 σ 추정 오차에 좌우되지도 않기 때문이다.

발표 시 보여줄 자료: [05_subperiod_sharpe.png](outputs/pweight_eda/05_subperiod_sharpe.png) (히트맵) + [06_cumulative_returns.png](outputs/pweight_eda/06_cumulative_returns.png) (누적수익) 두 장이 핵심.
