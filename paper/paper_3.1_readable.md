# §3.1 변동성 예측 모델 — 가독용 마크다운 사본

> **본 문서는 `paper/paper.md`의 §3.1 변동성 예측 모델 부분을 LaTeX/Overleaf 문법 없이 순수 마크다운으로 옮긴 사본이다.**
> 인용은 본문 내 (저자, 연도) 형식, 수식은 KaTeX/MathJax 호환 `$...$` 표기를 사용한다.
> 식·그림 번호는 본 문서 내에서만 유효하다(원본 `paper.md`의 LaTeX 라벨과 다름).

---

본 절은 저변동성 포트폴리오 구성을 위한 변동성 예측 모델을 정의한다. 주가 변동성의 군집성과 장기 자기상관 특성(volatility clustering and persistence, Cont, 2001)을 포착하기 위해, 본 연구는 장단기 메모리(Long-Short Term Memory, LSTM, Hochreiter & Schmidhuber, 1997)를 기반 모델로 채택한다. 또한 LSTM을 단독으로 사용하기보다는, 금융 변동성 예측의 학술적 표준인 Heterogeneous Autoregressive model of Realized Volatility(HAR-RV, Müller et al., 1997; Corsi, 2009)와 앙상블하여 비선형 구조와 선형 구조의 장점을 모두 반영한다.

본 절은 다음 세 항목으로 구성된다. 먼저 종속변수·입력변수와 블랙-리터만(Black-Litterman) 입력 변환을 정의하고, 이어서 LSTM, HAR-RV, 두 모델의 앙상블, 비교 baseline인 ANN(Pyo & Lee, 2018)의 구조를 기술한다. 마지막으로 모든 모델의 학습·평가에 공통 적용되는 Walk-Forward 구조를 설명한다.

---

## §3.1.1 종속변수, 입력변수, 블랙-리터만 입력 변환

본 절은 변동성 예측 모델의 공통 입출력 구조와, 예측치를 블랙-리터만 view로 연결하는 입력 변환을 정의한다.

### 종속변수

본 연구는 시점 $t$ 기준으로 향후 21영업일(약 1개월)의 실현변동성을 종속변수로 사용한다. 21일 horizon은 본 연구의 포트폴리오 리밸런싱 주기(월 1회)와 동기화되며, 일별 log-return $r[\cdot]$로부터 다음과 같이 정의된다.

$$
y[t] = \log(\text{std}(r[t+1], \ldots, r[t+21])) \qquad (1)
$$

log 변환은 다음 세 가지 요건을 동시에 충족한다.

1. **학습 안정성** — log-RV가 근사 정규분포를 따라 평균제곱오차(Mean Squared Error, MSE) 손실의 가우시안 가정과 정합.
2. **음수 변동성 예측 방지** — 예측치 $\hat{y}$를 지수 변환하면 항상 양수.
3. **HAR-RV의 학술적 관행과의 일관성** — Corsi (2009).

평가 단계에서는 단위 해석이 용이한 평균제곱근오차(Root Mean Squared Error, RMSE)를 주된 비교 지표로 사용한다.

### 입력변수

본 연구는 시장 참여자가 단기·중기·장기의 서로 다른 시간 해상도에서 의사결정을 내린다는 이질적 시장 가정(Heterogeneous Market Hypothesis, Müller et al., 1997)에 따라, 1일·5일·22일의 세 시간 척도로부터 실현분산을 산출한다.

$$
RV_d[t] = r[t]^2 \qquad (2a)
$$

$$
RV_w[t] = \frac{1}{5}\sum_{s=0}^{4} r[t-s]^2 \qquad (2b)
$$

$$
RV_m[t] = \frac{1}{22}\sum_{s=0}^{21} r[t-s]^2 \qquad (2c)
$$

본 연구의 분산 proxy $RV_d, RV_w, RV_m$는 일별 종가 기반의 squared close-to-close return으로 구성된다. 이는 Corsi (2009)의 원 HAR-RV가 사용하는 고빈도(intraday) 데이터 기반 realized variance(Andersen et al., 2003)의 일별 데이터 근사에 해당하며, 본 연구는 분석 대상의 광범위성(전 종목 약 617종, 2010–2025년)에 따른 데이터 가용성을 고려하여 일별 proxy를 채택한다.

본 연구의 투자 가능 종목군은 분석 기간(2010–2025) 동안 S&P500 지수에 편입된 적이 있는 617 종목이며(상세는 §4 도입부 참조), 이는 생존편향(survivorship bias)을 회피하기 위한 구성이다. HAR-RV는 위 세 항만을 입력으로 사용한다. 반면 LSTM은 세 항의 로그 변환에 더해, 시장의 미래 변동성 기대를 반영하는 변동성 지수(VIX, Whaley, 2009)의 로그 변환을 추가하여 다음과 같이 4채널 입력 벡터를 구성한다. 로그 변환은 다른 입력 변수(log $RV_d$/$RV_w$/$RV_m$)와의 스케일 통일을 위한 것이다.

$$
\mathbf{x}[t] = \bigl[\log RV_d[t],\ \log RV_w[t],\ \log RV_m[t],\ \log \text{VIX}[t]\bigr]^\top \qquad (3)
$$

LSTM에만 VIX를 도입한 이유는 사전 실험의 결과에 근거한다. VIX 외의 외부지표(VVIX, SKEW, 10년 국채금리, 달러지수) 4종을 추가한 8채널 모델은 검증한 모든 종목에서 예측 성능이 저하되었으며, 각 지표별 ablation 결과 4종 모두 noise로 입증되어 VIX만을 최종 채택하였다. HAR-RV는 학술적 표준 사양의 보존을 위해 외부지표를 추가하지 않는다. 본 투자 가능 종목군은 S&P500 구성 종목을 포함하고 VIX는 S&P500 옵션의 implied volatility로 산출되므로, VIX의 사용은 옵션 시장의 미래 변동성 기대가 실현변동성에 정보를 제공한다는 가정에 기반한다.

### 블랙-리터만 입력 변환

모델 출력 $\hat{y}[t]$는 식 (1)에 따라 log-daily 단위로 표현되므로, 블랙-리터만 view에 주입하기 위해 지수 역변환 후 연환산을 수행한다.

$$
\hat{\sigma}[t] = \exp(\hat{y}[t]) \times \sqrt{252} \qquad (4)
$$

연환산 계수 $\sqrt{252}$는 일간 표준편차를 연간 표준편차로 환산하는 표준 처리이며, 단위 일관성은 운영 단계에서 예측치 중앙값이 합리적 범위(예: 0.05 이상) 내에 들도록 검증한다.

---

## §3.1.2 모델

본 절은 변동성 예측에 사용되는 네 모델(LSTM, HAR-RV, 두 모델의 Performance-Weighted 앙상블, 비교 baseline ANN)의 구조를 정의한다. 네 모델은 §3.1.1에서 정의한 입력 변수와 종속변수를 공유하며, 출력은 모두 log-daily 변동성 공간에서 비교 가능하도록 통일한다.

### LSTM

본 연구의 기반 예측 모델은 1-layer LSTM이다. LSTM의 셀 상태(cell state)와 망각 게이트(forget gate, Gers et al., 2000)는 변동성의 군집성으로부터 발생하는 장기 의존 구조를 가변적으로 보존·갱신하기에 적합하다.

최종 사양은 다음과 같다.

| 항목 | 값 |
|---|---|
| 은닉 차원 (hidden size) | 32 |
| 드롭아웃 | 0.3 |
| 시퀀스 길이 | 63영업일 (§3.1.3에서 식별된 변동성 자기상관 안정화 길이와 동일) |
| 옵티마이저 | AdamW |
| 학습률 | $10^{-3}$ |
| 가중치 감쇠 (weight decay) | $10^{-3}$ |
| 손실 함수 | MSE |
| 최대 에폭 | 50 |
| 조기 종료 patience | 10 |
| 배치 크기 | 64 |

LSTM의 예측은 입력 시퀀스 $\mathbf{X}_{t-L+1}, \ldots, \mathbf{X}_t$ (여기서 $L = 63$이고 $\mathbf{X}_s$는 식 (3)의 입력 벡터)로부터 다음과 같이 산출된다.

$$
\hat{y}^{\text{LSTM}}[t] = f_{\text{LSTM}}(\mathbf{X}_{t-L+1}, \ldots, \mathbf{X}_t) \qquad (5)
$$

본 연구의 LSTM은 Hochreiter & Schmidhuber (1997)이 제안한 셀 상태 기반 구조에 Gers et al. (2000)이 도입한 망각 게이트를 결합한 4-게이트 표준 구조를 그대로 사용하며, 구현 디테일은 Yu et al. (2019)의 현대적 정리를 따른다.

### HAR-RV

HAR-RV는 변동성 예측의 학술적 표준 모델로(Müller et al., 1997; Corsi, 2009), 단·중·장기 실현분산의 선형 결합으로 미래 변동성을 설명한다.

$$
\log RV_h[t+h] = \beta_0 + \beta_d \log RV_d[t] + \beta_w \log RV_w[t] + \beta_m \log RV_m[t] \qquad (6)
$$

회귀계수 $(\beta_0, \beta_d, \beta_w, \beta_m)$는 IS 구간(§3.1.3)에서 보통최소제곱법(Ordinary Least Squares, OLS)으로 적합하며, OOS 예측은 적합된 계수와 OOS 시점의 입력을 결합하여 산출한다.

$$
\hat{y}^{\text{HAR}}[t] = \beta_0 + \beta_d \log RV_d[t] + \beta_w \log RV_w[t] + \beta_m \log RV_m[t] \qquad (7)
$$

HAR-RV는 LSTM 대비 모델 구조가 단순하고 해석 가능성이 높아, 본 연구에서는 LSTM의 비선형 예측을 보완하는 선형 기준 모델로 활용한다.

### 앙상블 (Performance-Weighted)

본 연구는 LSTM과 HAR-RV의 예측을 시점별로 가변하는 가중치로 결합하는 Performance-Weighted 앙상블(Diebold & Pauly, 1987)을 채택한다. 폴드 $k$의 가중치는 직전 폴드 $k-1$의 OOS RMSE 역수에 비례하도록 설정한다.

$$
w_k^{\text{LSTM}} = \frac{1/\text{RMSE}_{k-1}^{\text{LSTM}}}{1/\text{RMSE}_{k-1}^{\text{LSTM}} + 1/\text{RMSE}_{k-1}^{\text{HAR}}} \qquad (8)
$$

$$
w_k^{\text{HAR}} = 1 - w_k^{\text{LSTM}} \qquad (9)
$$

폴드 $k=0$에서는 사전 성능 정보가 없으므로 두 모델에 동일 가중치(각 0.5)를 부여한다. 폴드 $k$의 최종 예측은 두 모델 예측의 가중 평균이다.

$$
\hat{y}^{\text{ens}}[t] = w_k^{\text{LSTM}} \cdot \hat{y}^{\text{LSTM}}[t] + w_k^{\text{HAR}} \cdot \hat{y}^{\text{HAR}}[t] \qquad (10)
$$

이 구조는 시장 상황에 따라 우세한 모델의 비중이 자동으로 증가하는 동적 적응 특성을 가진다. 본 연구의 예비 평가에서 Performance-Weighted 앙상블은 평가 대상 다수 종목에서 단일 LSTM 및 단일 HAR-RV 대비 RMSE와 QLIKE 모두에서 우위를 보였으며, 이는 두 모델의 오차 구조가 부분적으로 보완적임을 시사한다.

**[그림 1] Performance-Weighted 앙상블의 예측 흐름**

```
입력                          모델              가중 결합            블랙-리터만 입력
──────────────────────────────────────────────────────────────────────────────
                          ┌──────────┐
   x[t] (4채널)  ────────→ │   LSTM   │ ──┐
                          │ f_LSTM   │   │
                          └──────────┘   │      ┌─────────────┐
                                          ├───→ │ ŷ^ens[t]    │
                                          │      │ (log-daily) │
                          ┌──────────┐   │      └─────────────┘
   RV_d, RV_w, RV_m ───→ │  HAR-RV  │ ──┘             │
                          │   OLS    │                  │
                          └──────────┘                  │ exp(·) × √252
                                                         ▼
                                                  ┌─────────────┐
                                                  │ σ̂[t]        │
                                                  │ (연환산,     │
                                                  │  BL view)   │
                                                  └─────────────┘
```

LSTM과 HAR-RV는 각각의 입력으로 log-daily 변동성을 예측하고, 폴드 단위 RMSE 가중치(식 8, 9)로 결합된 후 식 (4)의 변환($\exp(\cdot) \times \sqrt{252}$)을 거쳐 블랙-리터만 입력 단위(연환산 $\hat{\sigma}$)로 변환된다.

### ANN (비교 baseline)

본 연구는 변동성 예측 모델의 비교 baseline으로 Pyo & Lee (2018)가 블랙-리터만 view 산출에 사용한 인공신경망(Artificial Neural Network, ANN) 구조를 채택한다. 해당 ANN은 종목 $i$의 직전 10개월 변동성을 입력으로 받아 다음 1개월 변동성을 출력한다.

$$
\hat{\sigma}_{i,t} = f(\sigma_{i,t-1}, \sigma_{i,t-2}, \dots, \sigma_{i,t-10}), \quad i \in \{1, \dots, n\} \qquad (11)
$$

ANN의 내부 연산은 다음과 같이 정의된다. 은닉층 노드 $j$의 가중합과 ReLU 활성화 출력은 각각

$$
\text{net}_j = \sum_{i} w_{ij} x_i, \qquad o_j = f(\text{net}_j), \qquad f(S) = \max(0, S) \qquad (12)
$$

이며, 학습은 손실 함수 $\mathcal{L}$에 대한 역전파(backpropagation)로 가중치를 갱신한다.

$$
w_{ij}^{(t+1)} = w_{ij}^{(t)} - \eta \frac{\partial \mathcal{L}}{\partial w_{ij}} \qquad (13)
$$

본 연구의 구현은 Pyo & Lee (2018)의 사양에 따라 단일 은닉층 4개 뉴런 구조(학습 가능한 파라미터 약 49개)를 사용하며, 활성화 함수는 ReLU, 옵티마이저는 Adam이다. 과적합 방지를 위해 L2 정규화($\alpha = 0.01$)를 적용하고, 학습은 60개월 rolling window에서 종목별로 수행한다. 종목·시점별 학습 표본이 50개 내외에 머무르는 작은 데이터 규모는 Pyo & Lee (2018)의 원 사양 보존을 위한 선택이며, 모델 출력 공간은 LSTM·HAR-RV·앙상블과 동일한 log-daily 변동성으로 통일하여 §3.1.3의 Walk-Forward 평가에서 공정한 직접 비교가 가능하도록 한다.

---

## §3.1.3 Walk-Forward 구조

본 절은 변동성 예측 모델의 학습·평가에 공통 적용되는 Walk-Forward 구조를 정의한다. 시계열 데이터의 정보 누수(look-ahead bias)와 시변 분포(non-stationarity) 문제를 동시에 다루기 위해, 본 연구는 López de Prado (2018)가 제안한 Purge-Embargo 표준 절차를 따르는 Walk-Forward 교차검증을 채택한다.

### 폴드 구성

각 폴드는 그림 2와 같이 In-Sample(IS), Purge, Embargo, Out-of-Sample(OOS)의 4개 구간으로 구성된다. IS 구간에서 모델을 학습하고, IS 종료점 직후 Purge 구간을 두어 IS의 마지막 시점이 가지는 forward 타깃(식 1)이 OOS 시점과 시간적으로 겹치지 않도록 격리한다. Purge 이후 Embargo 구간은 변동성의 장기 자기상관이 충분히 감쇠할 때까지 OOS 시작점을 지연시키는 추가 안전구간이다. 마지막으로 OOS 21영업일에서 모델 예측 성능을 평가하며, 각 폴드는 step = 21영업일 단위로 이동하여 다음 폴드를 형성한다.

**[그림 2] Walk-Forward 폴드 구조 (step=21영업일). 폴드 $k$, $k+1$, $k+2$의 day 인덱스 예시.**

| 폴드 | In-Sample (1,250) | Purge (21) | Embargo (63) | OOS (21) |
|:---:|:---:|:---:|:---:|:---:|
| $k$ | day 1~1,250 | day 1,251~1,271 | day 1,272~1,334 | day 1,335~1,355 |
| $k+1$ | day 22~1,271 | day 1,272~1,292 | day 1,293~1,355 | day 1,356~1,376 |
| $k+2$ | day 43~1,292 | day 1,293~1,313 | day 1,314~1,376 | day 1,377~1,397 |

각 폴드는 직전 폴드 대비 step = 21영업일 우측으로 이동한다. 한 폴드 내 IS 학습 → Purge로 타깃 horizon 격리 → Embargo로 잔여 자기상관 차단 → OOS 평가의 순서로 진행되며, 폴드 $k+1$은 폴드 $k$의 모든 구간을 21영업일 우측으로 이동시킨 형태이다.

### 파라미터 근거

네 구간의 길이는 다음 근거로 결정한다.

| 구간 | 값 | 근거 |
|---|---|---|
| In-Sample (IS) | 1,250영업일 | 약 5영업년에 해당. LSTM 학습에 충분한 표본 크기를 확보하면서 동시에 단일 시장 체제(market regime)에 과적합되지 않도록 한 범위. |
| Purge | 21영업일 | 종속변수(식 1)의 forward horizon과 정확히 일치 → IS 마지막 시점의 타깃이 OOS 시점과 중첩되지 않도록 보장. |
| Embargo | 63영업일 | 본 연구의 예비 분석에서 SPY와 QQQ 일별 실현변동성의 자기상관함수(autocorrelation function, ACF)가 Bartlett 95% 신뢰구간 내로 진입하여 안정화되는 시점으로 확인된 값. 종목별 ACF 재추정에 따른 임의성을 배제하기 위해 평가 대상 전 종목의 Walk-Forward에 동일하게 일괄 적용. 종목별 ACF 분포 민감도 분석은 향후 연구 과제. |
| Out-of-Sample (OOS) / step | 21영업일 / 21영업일 | 포트폴리오 월별 리밸런싱 주기와 동기화 → 매 폴드의 OOS 예측이 한 달치 블랙-리터만 입력으로 직접 활용. |

### 비교 baseline ANN의 학습 구조

식 (11)에서 정의한 ANN은 Pyo & Lee (2018)의 원 사양에 따라 **월별** 데이터를 사용하므로, 일별 데이터를 사용하는 LSTM 및 HAR-RV와 학습·평가 단위가 다르다. 시점 $t$ 기준 직전 60개월을 학습 윈도우로 종목별 모델을 적합하고, 동일 시점에서 직전 10개월의 변동성을 입력으로 다음 한 달의 변동성을 1회 예측한다. 학습 윈도우는 매월 1개월씩 이동하여, ANN의 OOS 1개월 단위는 LSTM·HAR-RV·앙상블의 OOS 21영업일 단위와 정확히 일치한다. 본 연구는 ANN의 출력을 log-daily 변동성 공간으로 통일하여, 모든 모델이 동일 시점·동일 단위에서 RMSE 및 QLIKE로 직접 비교 가능하도록 한다.

| ANN 학습 파라미터 | 값 |
|---|---|
| 데이터 단위 | 월별 (LSTM/HAR-RV는 일별) |
| 입력 윈도우 | 직전 10개월 변동성 |
| 학습 윈도우 | 60개월 rolling |
| 예측 horizon | 1개월 |
| OOS 이동 단위 | 1개월 (= LSTM/HAR-RV 21영업일과 동기) |

> **§3.1의 한계**는 §5 Conclusion & Limitation 절에서 통합 정리되며, 본 절의 핵심 한계 — LSTM 단독 성능이 HAR-RV 대비 우월하지 않음 — 는 `TEAM_DISCUSSION_LIMITATIONS.md` §6에 정리됨.

---

## 부록 §A.2. 슬롯 분석 결과 전체 표 — 도입부 (paper4.md 미러)

> **위치**: paper4.md `\section{슬롯 분석 결과 전체 표}` (`\label{app:result_table}` + 보조 라벨 `\label{app:full_grid}`).
> §3.1 본문에는 속하지 않으며, paper4.md 추가 변경(부록 §A.2 도입 문단)의 동기화용 미러이다.

§4.3절(슬롯 구성, `\label{slot_config}`)의 평균 결과(표 `tab:p_matrix_avg`)를 45 슬롯 raw로 확장하여, 뷰 신뢰도 $\Omega \in \{\mathrm{err},\ \mathrm{he}\}$와 평가 지표 $\in \{\mathrm{Sharpe},\ \mathrm{Sortino},\ \mathrm{MDD}\}$의 모든 조합 6개 표를 첨부한다. 표 `tab:app_grid_sharpe_err`~`tab:app_grid_mdd_err`은 $\Omega=$ err 기준, 표 `tab:app_grid_sharpe_he`~`tab:app_grid_mdd_he`은 $\Omega=$ he 기준이다.

평가 지표는 다음 세 가지를 사용한다.

- **Sharpe 비율**: 평균 초과수익을 표준편차로 나눈 위험 조정 수익률. 클수록 우위.
- **Sortino 비율**: Sharpe의 하방 변동성 버전으로, 분모를 하방 표준편차로 대체한다(`sortino1994`). 하락 변동성만 페널티로 반영하며 클수록 우위.
- **최대낙폭 (MDD)**: 누적 고점 대비 최대 손실 폭(%). 음수가 손실이므로 클수록(0에 가까울수록) 우위.

따라서 모든 표의 $\Delta > 0$ 셀(녹색)은 LSTM이 ANN 대비 위험 조정 수익 또는 손실 폭 측면에서 우위인 경우이다.

> **보조 라벨 추가의 의의**: paper4.md §4.3 본문(L520, L554)에서 부록을 `\ref{app:full_grid}`로 인용하나, 부록 절에는 `\label{app:result_table}`만 정의되어 있어 깨진 참조 상태였다. 동일 절에 `\label{app:full_grid}`를 보조로 추가하여 본문 인용을 무수정으로 받아낸다.

---

## 부록. 인용 문헌 (본 문서에서 사용)

| 본문 표기 | 전체 서지 정보 (references.bib 등록 키) |
|---|---|
| Andersen et al. (2003) | Andersen, T. G., Bollerslev, T., Diebold, F. X., and Labys, P. Modeling and forecasting realized volatility. *Econometrica*, 71(2), 579–625, 2003. (`andersen2003`) |
| Cont (2001) | Cont, R. Empirical properties of asset returns: stylized facts and statistical issues. *Quantitative Finance*, 1(2), 223–236, 2001. (`cont2001`) |
| Corsi (2009) | Corsi, F. A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174–196, 2009. (`corsi2009`) |
| Diebold & Pauly (1987) | Diebold, F. X. and Pauly, P. Structural change and the combination of forecasts. *Journal of Forecasting*, 6(1), 21–40, 1987. (`diebold1987`) |
| Gers et al. (2000) | Gers, F. A., Schmidhuber, J., and Cummins, F. Learning to forget: Continual prediction with LSTM. *Neural Computation*, 12(10), 2451–2471, 2000. (`gers2000`) |
| Hochreiter & Schmidhuber (1997) | Hochreiter, S. and Schmidhuber, J. Long short-term memory. *Neural Computation*, 9(8), 1735–1780, 1997. (`hochreiter1997`) |
| López de Prado (2018) | López de Prado, M. *Advances in Financial Machine Learning*. John Wiley & Sons, 2018. (`lopezdeprado2018`) |
| Müller et al. (1997) | Müller, U. A. et al. Volatilities of different time resolutions—Analyzing the dynamics of market components. *Journal of Empirical Finance*, 4(2-3), 213–239, 1997. (`muller1997`) |
| Patton (2011) | Patton, A. J. Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*, 160(1), 246–256, 2011. (`patton2011`) |
| Pyo & Lee (2018) | Pyo, S. and Lee, J. Exploiting the low-risk anomaly using machine learning to enhance the Black-Litterman framework. *Pacific-Basin Finance Journal*, 51, 1–12, 2018. (`pyo2018`) |
| Whaley (2009) | Whaley, R. E. Understanding the VIX. *The Journal of Portfolio Management*, 35(3), 98–105, 2009. (`whaley2009`) |
| Yu et al. (2019) | Yu, Y. et al. A review of recurrent neural networks: LSTM cells and network architectures. *Neural Computation*, 31(7), 1235–1270, 2019. (`yu2019review`) |

---

## 본 문서 ↔ paper.md 매핑

| 본 문서 식·그림 | paper.md (LaTeX) 라벨 |
|---|---|
| 식 (1) — 종속변수 | `eq:lstm_target` |
| 식 (2a/b/c) — RV 입력 | (paper.md는 `align` 환경, 별도 라벨 없음) |
| 식 (3) — LSTM 입력 벡터 | `eq:lstm_input` |
| 식 (4) — BL 입력 변환 | `eq:bl_input` |
| 식 (5) — LSTM 예측 | `eq:lstm_pred` |
| 식 (6) — HAR-RV 전체 | `eq:har_full` |
| 식 (7) — HAR-RV 예측 | `eq:har_pred` |
| 식 (8) — LSTM 앙상블 가중치 | `eq:ens_weight_lstm` |
| 식 (9) — HAR 앙상블 가중치 | `eq:ens_weight_har` |
| 식 (10) — 앙상블 최종 예측 | `eq:ens_pred` |
| 식 (11) — ANN 예측 | `eq:ann_pred` |
| 식 (12) — ANN 가중합·ReLU·출력 | `eq:ann_forward` |
| 식 (13) — ANN 역전파 가중치 갱신 | `eq:ann_backprop` |
| 그림 1 — 앙상블 흐름 | `fig:ensemble_flow` |
| 그림 2 — Walk-Forward 폴드 | `fig:walk_forward` |
| §3.1 | `subsec:vol_pred` |
| §3.1.1 | `subsec:vol_io` |
| §3.1.2 | `subsec:vol_model` |
| §3.1.3 | `subsec:vol_wf` |
