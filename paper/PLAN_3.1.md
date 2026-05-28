# §3.1 변동성 예측 모델 — 상세 작성 플랜

> **목적**: 사용자(김재천) 담당 파트 §3.1 작성을 위한 단락 구성, 수식, 인용, 참고 자료를 사전 확정한다.
> **준수 규약**: `claude.md`(최상위 규약), `paper/Notation_Guide.md`(수식 표기).
> **작성 시점**: 본 플랜 확정 후 다음 작업 세션에서 착수.

---

## 0. 전체 구성 · 분량 · 작성 순서

| 절 | 예상 분량 | 도식 | 작성 우선순위 |
|---|---|---|---|
| §3.1 도입 (현 paper.md 31~33줄 보강) | +3~5줄 | 없음 | 1 |
| §3.1.1 종속변수·입력변수·BL 입력 변환 | 35~45줄 (≈1.2p) | 없음 | 2 |
| §3.1.3 Walk-Forward 구조 | 20~30줄 (≈0.7p) | 1개 (시퀀스 다이어그램) | 3 |
| §3.1.2 모델 (LSTM/HAR/앙상블/ANN) | 60~80줄 (≈2.2p) | 1~2개 (앙상블·LSTM 구조도) | 4 |
| §4.1 표기 정렬(점검) | 1~3줄 미세 수정 | — | 5 (마무리) |

**총 분량 목표**: §3.1 전체 약 120~160줄 ≈ 3.5~4페이지.

**순서 근거**: §3.1.1은 코드+Notation_Guide 매핑이 가장 명확. §3.1.3은 정량값 확정 완료. §3.1.2는 분량이 크고 도식 결정이 필요해 마지막.

---

## 1. §3.1 도입 보강

### 현재 상태 (paper.md 29~33줄)

```latex
\subsection{변동성 예측 모델}

본 연구에서 변동성 예측을 활용해 저변동성 포트폴리오를 구성하는 전략의 핵심적인 부분이다. 
주가의 변동성은 보통 클러스터링 효과로 장기적인 구조를 담는데 효과적인 방법이기에, 
Long-Short Term Memory(LSTM, \cite{hochreiter1997})기반으로 변동성을 예측했다.

또한, LSTM을 단독으로 사용하기 보단, 금융에서 변동성을 예측할 때 표준으로 사용하는 
Heterogeneous Autoregressive model of Realized Volatility(HAR-RV, \cite{muller1997})를 
앙상블 하여 각 모델의 선형 구조와 비선형 구조에 대한 장점을 모두 반영하였다.
```

### 보강 방향
- 첫 문장 끝에 본 절의 구성을 한 문장으로 안내(§3.1.1 입출력, §3.1.2 모델, §3.1.3 평가 구조).
- 톤은 기존 두 문단 그대로 유지.
- "본 연구" 표현으로 일관화 (현재 "본 연구에서 ..."로 시작하나 일부 어색).

### 참고 자료
- `paper/architecture.md` §3.1
- `paper/detail.md` 항목 2

---

## 2. §3.1.1 종속변수 · 입력변수 · BL 입력 변환

### 2.1 단락 구성 (4단락)

#### 단락 1 — 도입 (1~2문장)
- 메시지: 본 절은 모델 공통의 입출력 구조를 정의한다.
- 인용: 없음.

#### 단락 2 — 종속변수 (타깃)
- 메시지: 21일 forward log-realized volatility를 채택한다. log 변환의 3가지 근거 — (a) MSE 학습 안정성, (b) 음수 변동성 예측 방지, (c) 가우시안 가정 정합.
- 수식 (Notation_Guide LSTM 절):
  ```latex
  y[t] = \log(\text{std}(r[t+1], \ldots, r[t+21]))
  ```
- 인용: `\cite{corsi2009}`(Log-RV 학술 표준), `\cite{patton2011}`(QLIKE 정합성)

#### 단락 3 — 입력변수
- 메시지: 단·중·장기 다중 시간 해상도 관점으로 1·5·22일 변동성을 사용. HAR-RV는 세 시점만, LSTM에는 VIX를 추가해 외부 시장 기대를 반영. ablation 결과(`v7_ablation_report.md`)에서 VIX만 유효함이 입증되어 단독 채택.
- 수식 (Notation_Guide):
  ```latex
  % HAR-RV 입력 변수
  RV_d[t] = r[t]^2
  RV_w[t] = \tfrac{1}{5}\sum_{s=0}^{4} r[t-s]^2
  RV_m[t] = \tfrac{1}{22}\sum_{s=0}^{21} r[t-s]^2
  % LSTM 입력 벡터
  \mathbf{x}[t] = [\log RV_d[t],\ \log RV_w[t],\ \log RV_m[t],\ \log \text{VIX}[t]]^\top
  ```
- 인용: `\cite{muller1997}`(시간 이질성), `\cite{corsi2009}`, `\cite{whaley2009}`(VIX 정의·해석)

#### 단락 4 — BL 입력 변환
- 메시지: 모델 출력 ŷ는 log-daily-RV이므로, BL view에 주입하기 위해 지수 역변환 후 √252로 연환산한다. 단위 가드(median < 0.05 검출)를 운영상 보완으로 한 문장 언급.
- 수식 (Notation_Guide BL 입력):
  ```latex
  \hat{\sigma}[t] = \exp(\hat{y}[t]) \times \sqrt{252}
  ```
- 인용: 없음(본 연구 구현).

### 2.2 참고 자료

| 항목 | 1차 참고 |
|---|---|
| Log-RV 타깃 정의 | `시계열_Test/Phase1_5_Volatility/PLAN.md` (타깃 정의 §) |
| MSE 손실 채택 근거 | `재천_WORKLOG.md` 2026-04-26 "핵심 의사결정 요약" |
| 단·중·장기 입력 | `Phase1_5_Volatility/scripts/dataset.py`, `targets_volatility.py` |
| VIX 추가 (LSTM 한정) | `results/v6_external_indicators_report.md`, `v7_ablation_report.md` |
| BL 입력 변환 | `final_pt/bl_runner.py:106` (`np.exp(...)*np.sqrt(252)`) |

---

## 3. §3.1.3 Walk-Forward 구조

### 3.1 단락 구성 (3단락)

#### 단락 1 — 동기
- 메시지: 시계열 데이터에서 정보 누수(look-ahead bias)를 방지하고 월별 OOS 평가를 수행하기 위해 Walk-Forward를 채택. López de Prado(2018)의 Purge·Embargo 표준 절차를 준용.
- 인용: `\cite{lopezdeprado2018}`

#### 단락 2 — 구조 (도식 포함)
- 다이어그램: `[← IS 1250 →][Purge 21][Embargo 63][← OOS 21 →]`, step=21로 fold 이동.
- 도식 출처: `finance project.pdf` 추출 1순위, 실패 시 TikZ 또는 텍스트 박스로 대체.
- 인용: `\cite{lopezdeprado2018}`

#### 단락 3 — 파라미터 근거
- IS 1250 ≈ 5영업년 (LSTM 학습량 확보)
- Purge 21 = 타깃 horizon (forward 21일과 IS 종료점의 시간 중첩 차단)
- Embargo 63 = ACF 안정화 지점 (SPY·QQQ 변동성 ACF 분석에서 약 63영업일에서 수렴)
- OOS 21 = 월별 리밸런싱 주기와 동기화
- 인용: 없음(내부 ACF 분석 근거)

### 3.2 참고 자료

| 항목 | 1차 참고 |
|---|---|
| In-sample 1250, OOS 21 | `final_pt/lstm_pipeline.py:68-87` (V4_BEST_CONFIG) |
| Purge 21·Embargo 63 | `final_pt/timeseries_lib.py:114-158` (`walk_forward_folds`) |
| ACF 근거 (SPY, QQQ → 63일) | `final_pt/02a_EDA_Returns_Volatility.ipynb` |
| 설계 결정 기록 | `시계열_Test/Phase1_5_Volatility/재천_WORKLOG.md` |

---

## 4. §3.1.2 모델 (LSTM, HAR, 앙상블, ANN)

### 4.1 단락 구성 (5단락)

#### 단락 1 — 도입
- 메시지: 본 절은 4개 모델의 역할 분담을 정의 — LSTM(비선형·장기기억), HAR-RV(선형·학술표준), 두 모델의 Performance-Weighted 앙상블(상호보완), ANN(Pyo & Lee 2018 비교 baseline).

#### 단락 2 — LSTM
- 동기: 변동성 클러스터링(`cont2001` 인용 고려) → 장기 의존성 포착. LSTM의 망각 게이트가 변동성 체제 변화에 적응.
- 구조·하이퍼파라미터: 1-layer LSTM, hidden=32, dropout=0.3, seq_len=63. 학습은 MSE + AdamW (lr=1e-3, weight_decay=1e-3, max_epochs=50, EarlyStop patience=10, batch=64).
- 수식 (Notation_Guide):
  ```latex
  \hat{y}^{\text{LSTM}}[t] = f_{\text{LSTM}}(\mathbf{X}_{t-L+1}, \ldots, \mathbf{X}_t)
  ```
- 인용: `\cite{hochreiter1997}`, `\cite{gers2000}`, `\cite{yu2019review}`, 옵션 `\cite{cont2001}`(클러스터링)

#### 단락 3 — HAR-RV
- 동기: 금융 변동성 예측의 학술 표준. 선형 구조로 해석 가능성 확보. OLS 적합.
- 수식 (Notation_Guide):
  ```latex
  \log RV_h[t+h] = \beta_0 + \beta_d \log RV_d[t] + \beta_w \log RV_w[t] + \beta_m \log RV_m[t]
  \hat{y}^{\text{HAR}}[t] = \beta_0 + \beta_d \log RV_d[t] + \beta_w \log RV_w[t] + \beta_m \log RV_m[t]
  ```
- 인용: `\cite{muller1997}`, `\cite{corsi2009}`

#### 단락 4 — 앙상블 (Performance-Weighted)
- 동기: 비선형(LSTM)과 선형(HAR)의 보완. fold별 직전 RMSE 역수 가중치로 시점 동적 적응. fold 0은 정보가 없으므로 0.5/0.5 초기화.
- 실증 근거: `REPORT.md` v8 마감 — 7종목 중 5종목 best, avg RMSE 0.2934, DM 검정 6/7종목에서 LSTM 단독 대비 5% 유의 우위.
- 수식 (Notation_Guide):
  ```latex
  w_k^{\text{LSTM}} = \frac{1/\text{RMSE}_{k-1}^{\text{LSTM}}}{1/\text{RMSE}_{k-1}^{\text{LSTM}} + 1/\text{RMSE}_{k-1}^{\text{HAR}}}
  w_k^{\text{HAR}} = 1 - w_k^{\text{LSTM}}
  \hat{y}^{\text{ens}}[t] = w_k^{\text{LSTM}} \cdot \hat{y}^{\text{LSTM}}[t] + w_k^{\text{HAR}} \cdot \hat{y}^{\text{HAR}}[t]
  ```
- 인용: `\cite{diebold1987}`

#### 단락 5 — ANN (Pyo & Lee 2018 baseline)
- 메시지: 비교 모델로서, 동일 universe·동일 평가 구조에서 변동성 예측을 수행. 직전 10개월 변동성 → 다음 1개월 변동성. Hidden (4,), ReLU 활성화, Adam 옵티마이저, 60개월 rolling 학습. 출력 공간은 LSTM과 동일한 log-daily-σ로 통일해 비교 가능성을 확보.
- 수식 (Notation_Guide ANN 절):
  ```latex
  \hat{\sigma}_{i,t} = f(\sigma_{i,t-1}, \sigma_{i,t-2}, \ldots, \sigma_{i,t-10})
  ```
- 인용: `\cite{pyo2018}`

### 4.2 도식
- **앙상블 구조도 1개** 권장: 입력(공통/LSTM 전용 VIX) → LSTM/HAR 병렬 → 직전 RMSE 가중치 → 최종 ŷ → exp×√252 → BL view.
- 1순위: `finance project.pdf` 해당 슬라이드 추출 → 학술 톤 재가공.
- 2순위: 추출 어려울 시 TikZ 또는 텍스트 박스로 대체.

### 4.3 참고 자료

| 항목 | 1차 참고 | 보조 참고 |
|---|---|---|
| LSTM 구조 | `final_pt/03b_Volatility_Forecasting.ipynb`, `lstm_pipeline.py` (V4_BEST_CONFIG) | `Phase1_5_Volatility/02_v4_final_evaluation.ipynb` |
| HAR-RV 사양 | `final_pt/timeseries_lib.py:315-371` (`fit_har_rv`) | `Phase1_5_Volatility/results/har_rv_diagnostics.md` |
| 앙상블 설계 | `Phase1_5_Volatility/results/ensemble_report.md`, `08_ensemble_evaluation.ipynb` | `finance project.pdf` 앙상블 슬라이드, `REPORT.md` v8 |
| ANN 구조 | `final_pt/_dev/_train_paper_ann.py` | `final_pt/99_analyze_ann.ipynb`, `bl_config_ann.py` |

---

## 5. §4.1 표기 정렬 (마무리 점검)

- §4.1 본문(`paper.md` 51~81줄)은 "LSTM", "ANN" 명칭만 사용하므로 표기 충돌 가능성 낮음.
- 점검 항목:
  1. 표 \ref{tab:sigma_pred}의 평가지표(RMSE/Spearman/Hit Rate)가 §3.1.1에서 정의한 평가 맥락과 일관.
  2. §4.1에서 σ나 ŷ 표기가 추가로 필요해질 경우 §3.1과 통일.
  3. 표 캡션 영어 ↔ 한국어 혼용 여부 정렬.
- 예상 수정량: 1~3줄 미세 수정 또는 무수정.

---

## 6. 작성 시 준수 사항 (요약)

작성 매 단계에서 `claude.md` §8 작업 워크플로우 7단계를 따른다.

| 단계 | 행동 |
|---|---|
| 1 | `paper/paper.md` Read — 현재 상태 파악 |
| 2 | `architecture.md` + `detail.md` Read — 흐름·의도 재확인 |
| 3 | 본 PLAN의 §2.2 / §3.2 / §4.3 참고 자료 핀포인트 Read |
| 4 | `paper/Notation_Guide.md`의 해당 분야 절 Read — 수식 표기 확정 |
| 5 | `paper.md`에 본문 추가 |
| 6 | 셀프 점검 — LaTeX 짝, 인용 키 존재, 수식 글자 단위 일치, 인접 절 일관성, contribution 정합 |
| 7 | 사용자 보고 — 수정 절 / 근거 자료 / 점검 결과 / 확인 요청 |

---

## 7. 보완 필요 항목 (작성 도중 처리)

### 7.1 신규 인용 — `whaley2009` 등록 완료
- 본 PLAN 확정 시점에 `references.bib`에 등록.
- §3.1.1 단락 3 "VIX 추가" 근거 문장에 사용.

### 7.2 PDF 도식 추출 — §3.1.2 / §3.1.3 작성 시점에 시도
- 대상: LSTM 앙상블 구조도, Walk-Forward 시퀀스 다이어그램.
- 시도 → 실패 시 TikZ 또는 텍스트 박스로 대체. 본 PLAN 사전 작업 불필요.

### 7.3 §3.2 인터페이스 — 비동기 합의
- 사용자가 §3.2 담당자(서윤범)와 BL 입력 σ̂ 형식(연환산 σ, 소수) 호환 확인.
- 현재 코드 + Notation_Guide 둘 다 동일 단위라 충돌 가능성 낮음.

### 7.4 ANN 학습 상세 (`_dev/_train_paper_ann.py`) 정밀 점검 — §3.1.2 단락 5 작성 직전
- LOOKBACK=10, TRAIN_WINDOW=60, HIDDEN=(4,), ALPHA=0.01, MAX_ITER=50 확인 완료.
- 추가로 ReLU 외 다른 activation 옵션이 있는지, 입력 스케일링 방식이 본문에 언급할 만한지 작성 직전 재확인.

---

## 부록. 인용 키 현황 (작성 시 즉시 참조)

| cite key | 등록 상태 | 본 PLAN 사용 위치 |
|---|---|---|
| `hochreiter1997` | ✓ 등록 (paper.md 사용 중) | §3.1 도입, §3.1.2 LSTM |
| `gers2000` | ✓ 등록 | §3.1.2 LSTM |
| `yu2019review` | ✓ 등록 | §3.1.2 LSTM |
| `muller1997` | ✓ 등록 (paper.md 사용 중) | §3.1 도입, §3.1.1 입력, §3.1.2 HAR |
| `corsi2009` | ✓ 등록 | §3.1.1 종속변수·입력, §3.1.2 HAR |
| `diebold1987` | ✓ 등록 | §3.1.2 앙상블 |
| `lopezdeprado2018` | ✓ 등록 | §3.1.3 Walk-Forward |
| `pyo2018` | ✓ 등록 (paper.md 사용 중) | §3.1.2 ANN |
| `patton2011` | ✓ 등록 | §3.1.1 종속변수 |
| `whaley2009` | ✓ **신규 등록 (본 PLAN 확정 시점)** | §3.1.1 VIX 추가 |
| `cont2001` | ✓ 등록 (옵션) | §3.1.2 LSTM 도입 |

---

## 변경 이력

| 일자 | 변경 | 작성 |
|---|---|---|
| 2026-05-25 | 초판 — 사용자 결정(파일 저장, whaley2009 추가, 다음 세션 작성) 반영 | Claude (사용자 검토 후 확정) |
