\documentclass{article}
\usepackage{graphicx} % Required for inserting images
\usepackage{kotex}
\usepackage{natbib} % Required for only writing author(exclude date)
\usepackage{amsmath} % Writing equation
\usepackage{appendix} % Use Appendix

\usepackage{booktabs}   % toprule/midrule/bottomrule
\usepackage{multirow}
\usepackage[table]{xcolor}   % for cell coloring (optional)
\definecolor{posgreen}{HTML}{D5E8D4}
\definecolor{negred}{HTML}{F8CECC}

\title{변동성 예측을 결합한 블랙-리터만 저변동성 포트폴리오 전략}
\author{서윤범, 김재천, 김윤서, 김하연, 서정욱}
\date{May 2026}

\begin{document}

\maketitle

\section{서론}

마코위츠의 평균-분산 최적화(Mean-Variance Optimazation, MVO, \citep{markowitz1952}) 이래로 포트폴리오 구성 이론은 기대 수익률과 위험의 절충을 다루는 방법론을 다수 발전시켜왔다. 그러나 MVO는 입력값의 추정 오차에 극도로 민감하여, 최적해가 표본 외에서 불안정하고 입력값의 작은 변화에도 극단적인 가중치를 산출한다는 한계가 지적되어 왔다. 이러한 한계에 대응하기 위해 shrinkage 추정\citep{ledoitwolf2004}, 베이지안 접근 등 다양한 후속 연구가 제안되어 왔다. 그 중 \citet{black1992}이 제안한 블랙-리터만 모델(Black-Litterman, BL)은 시장 균형 수익률을 사전 분포로 두고 투자자의 견해(뷰)를 결합해 사후 기대 수익률을 도출하는 베이지안 방법론으로, 추정 오차를 완화하면서도 투자 전문가의 견해를 반영해 포트폴리오의 가중치를 설정할 수 있어 폭 넓게 연구되어왔다. 그러나, 전문가라 할지라도 기대수익률 뷰에 사람의 의견이 반영되어 주관적이라는 한계를 가지고 있다.

앞선 한계를 해결하기 위해, 투자자의 뷰를 머신러닝 기반으로 예측하는 다수의 연구가 진행되어 왔다. BL 프레임워크의 핵심은 포트폴리오 뷰 행렬 $\mathbf{P}$와 기대 수익률 뷰인 $\mathbf{q}$의 설정에 있으며, 머신러닝 기반 BL 응용 연구는 주로 $\mathbf{q}$를 트리 기반, 신경망, LLM 등의 모형으로 예측하여 뷰로 주입하는 방식을 취해왔다\citep{barua2023using, su2026objective, lee2025llm}. 이들은 머신러닝 기반 수익률 예측이 BL 프레임워크의 뷰 정량화에 효과적으로 활용될 수 있음을 보였다.

한편 BL의 뷰는 반드시 기대 수익률에 국한될 필요는 없다. 자산의 변동성 또한 포트폴리오 구성에서 중요한 요소이며, \citet{baker2011}, \citet{frazzini2014}, \citet{blitz2007} 등의 연구의 저위험 



본 연구는 다음과 같은 순서로 구성된다. 제 2장에서는 저변동 포트폴리오를 구성하는 방식과 블랙-리터만을 활용한 선행 연구들을 소개하고, 제 3장에서는 변동성을 예측하는 모델과 이를 뷰로 연결하는 블랙-리터만 구조를 설명한다. 제 4장에서는 블랙-리터만 구조의 실험 방식, 변동성 예측과 뷰를 연결하기 위해 제안하는 슬롯, 이를 평가하는 방법을 소개하고, 제 5장에서는 실험 결과를 기술한다. 마지막 장에서는 연구 내용을 요약하고 한계를 검토한다.

\section{선행연구}



\section{방법론}

\subsection{변동성 예측 모델}
\label{subsec:vol_pred}

본 절은 저변동성 포트폴리오 구성을 위한 변동성 예측 모델을 정의한다. 주가 변동성의 군집성과 장기 자기상관 특성(volatility clustering and persistence, \cite{cont2001})을 포착하기 위해, 본 연구는 장단기 메모리(Long-Short Term Memory, LSTM, \cite{hochreiter1997})를 기반 모델로 채택한다. 또한 LSTM을 단독으로 사용하기보다는, 금융 변동성 예측의 학술적 표준인 Heterogeneous Autoregressive model of Realized Volatility(HAR-RV, \cite{muller1997, corsi2009})와 앙상블하여 비선형 구조와 선형 구조의 장점을 모두 반영한다.

본 절은 다음 세 항목으로 구성된다. 먼저 종속변수·입력변수와 블랙-리터만(Black-Litterman) 입력 변환을 정의하고, 이어서 LSTM, HAR-RV, 두 모델의 앙상블, \cite{pyo2018}가 제안한 모델 ANN(\cite{pyo2018})의 구조를 기술한다. 마지막으로 모든 모델의 학습·평가에 공통 적용되는 Walk-Forward 구조를 설명한다.

\subsubsection{종속변수, 입력변수, 블랙-리터만 입력 변환}
\label{subsec:vol_io}

본 절은 변동성 예측 모델의 공통 입출력 구조와, 예측치를 블랙-리터만 뷰로 연결하는 입력 변환을 정의한다.

\paragraph{종속변수.} 본 연구는 시점 $t$ 기준으로 향후 21영업일(약 1개월)의 실현변동성을 종속변수로 사용한다. 21일 horizon은 본 연구의 포트폴리오 리밸런싱 주기(월 1회)와 동기화되며, 일별 log-return $r[\cdot]$로부터 다음과 같이 정의된다.
\begin{equation}
y[t] = \log(\text{std}(r[t+1], \ldots, r[t+21])) \label{eq:lstm_target}
\end{equation}
log 변환은 (i) 학습 안정성(log-RV가 근사 정규분포를 따라 평균제곱오차(Mean Squared Error, MSE) 손실의 가우시안 가정과 정합), (ii) 음수 변동성 예측 방지(예측치 $\hat{y}$를 지수 변환하면 항상 양수), (iii) HAR-RV의 학술적 관행(\cite{corsi2009})과의 일관성이라는 세 가지 요건을 동시에 충족한다. 평가 단계에서는 단위 해석이 용이한 평균제곱근오차(Root Mean Squared Error, RMSE)를 주된 비교 지표로 사용한다.

\paragraph{입력변수.} 본 연구는 시장 참여자가 단기·중기·장기의 서로 다른 시간 해상도에서 의사결정을 내린다는 이질적 시장 가정(Heterogeneous Market Hypothesis, \cite{muller1997})에 따라, 1일·5일·22일의 세 시간 척도로부터 실현분산을 산출한다.
\begin{align}
RV_d[t] &= r[t]^2,\\
RV_w[t] &= \frac{1}{5}\sum_{s=0}^{4} r[t-s]^2,\\
RV_m[t] &= \frac{1}{22}\sum_{s=0}^{21} r[t-s]^2.
\end{align}
본 연구의 분산 대리변수 $RV_d, RV_w, RV_m$는 일별 종가수익률 제곱으로 구성된다. 이는 \cite{corsi2009}의 원 HAR-RV가 사용하는 고빈도(intraday) 데이터 기반 실현 변동성(\cite{andersen2003modeling})의 일별 데이터 근사에 해당하며, 본 연구는 분석 대상의 광범위성(전 종목 약 617종, 2010--2025년)에 따른 데이터 가용성을 고려하였다.

본 연구의 투자 가능 종목군은 분석 기간(2010--2025) 동안 S\&P500 지수에 편입된 적이 있는 617 종목이며(상세는 §4 도입부 참조), 이는 생존편향(survivorship bias)을 회피하기 위한 구성이다. HAR-RV는 위 세 항만을 입력으로 사용한다. 반면 LSTM은 세 항의 로그 변환에 더해, 시장의 미래 변동성 기대를 반영하는 변동성 지수(CBOE Volatility Index, VIX, \cite{whaley2009understanding})의 로그 변환을 추가하여 다음과 같이 4채널 입력 벡터를 구성한다. 로그 변환은 다른 입력 변수(log $RV_d$/$RV_w$/$RV_m$)와의 스케일 통일을 위한 것이다.
\begin{equation}
\mathbf{x}[t] = \bigl[\log RV_d[t],\ \log RV_w[t],\ \log RV_m[t],\ \log \text{VIX}[t]\bigr]^\top \label{eq:lstm_input}
\end{equation}
LSTM에만 VIX를 도입한 이유는 사전 실험의 결과에 근거한다. VIX 외의 외부지표(VVIX, SKEW, 10년 국채금리, 달러지수) 4종을 추가한 8채널 모델은 검증한 모든 종목에서 예측 성능이 저하되었으며, 각 지표별 ablation 결과 4종 모두 오차로 입증되어 VIX만을 최종 채택하였다. HAR-RV는 학술적 표준 사양의 보존을 위해 외부지표를 추가하지 않는다. 본 투자 가능 종목군은 S\&P500 구성 종목을 포함하고 VIX는 S\&P500 옵션의 내재 변동성으로 산출되므로, VIX의 사용은 옵션 시장의 미래 변동성 기대가 실현변동성에 정보를 제공한다는 가정에 기반한다.

\paragraph{블랙-리터만 입력 변환.} 모델 출력 $\hat{y}[t]$는 식~(\ref{eq:lstm_target})에 따라 log-daily 단위로 표현되므로, 블랙-리터만 view에 주입하기 위해 지수 역변환 후 연환산을 수행한다.
\begin{equation}
\hat{\sigma}[t] = \exp(\hat{y}[t]) \times \sqrt{252} \label{eq:bl_input}
\end{equation}
연환산 계수 $\sqrt{252}$는 일간 표준편차를 연간 표준편차로 환산하는 표준 처리이며, 단위 일관성은 운영 단계에서 예측치 중앙값이 합리적 범위(예: 0.05 이상) 내에 들도록 검증한다.

\subsubsection{모델}
\label{subsec:vol_model}

본 절은 변동성 예측에 사용되는 네 모델(LSTM, HAR-RV, 두 모델의 Performance-Weighted 앙상블, 기준 모델 ANN)의 구조를 정의한다. 네 모델은 \ref{subsec:vol_io}에서 정의한 입력 변수와 종속변수를 공유하며, 출력은 모두 log-daily 변동성 공간에서 비교 가능하도록 통일한다.

\paragraph{LSTM.} 본 연구의 기반 예측 모델은 1-layer LSTM이다. LSTM의 셀 상태(cell state)와 망각 게이트(forget gate, \cite{gers2000})는 변동성의 군집성으로부터 발생하는 장기 의존 구조를 가변적으로 보존·갱신하기에 적합하다.

최종 사양은 은닉 차원 32, 드롭아웃 0.3, 시퀀스 길이 63영업일을 사용하며(시퀀스 길이는 \ref{subsec:vol_wf}에서 식별된 변동성 자기상관 안정화 길이와 동일하게 설정), 학습은 AdamW 옵티마이저(학습률 $10^{-3}$, 가중치 감쇠 $10^{-3}$), MSE 손실, 최대 50 에폭, 조기 종료(early stopping) patience 10, 배치 크기 64로 수행한다. LSTM의 예측은 입력 시퀀스 $\mathbf{X}_{t-L+1}, \ldots, \mathbf{X}_t$(여기서 $L = 63$이고 $\mathbf{X}_s$는 식~\ref{eq:lstm_input}의 입력 벡터)로부터 다음과 같이 산출된다.
\begin{equation}
\hat{y}^{\text{LSTM}}[t] = f_{\text{LSTM}}(\mathbf{X}_{t-L+1}, \ldots, \mathbf{X}_t) \label{eq:lstm_pred}
\end{equation}
본 연구의 LSTM은 \cite{hochreiter1997}이 제안한 셀 상태 기반 구조에 \cite{gers2000}이 도입한 망각 게이트를 결합한 4-게이트 표준 구조를 그대로 사용하며, 구현 디테일은 \cite{yu2019review}의 현대적 정리를 따른다.

\paragraph{HAR-RV.} HAR-RV는 변동성 예측의 학술적 표준 모델로(\cite{muller1997, corsi2009}), 단·중·장기 실현분산의 선형 결합으로 미래 변동성을 설명한다.
\begin{equation}
\log RV_h[t+h] = \beta_0 + \beta_d \log RV_d[t] + \beta_w \log RV_w[t] + \beta_m \log RV_m[t] \label{eq:har_full}
\end{equation}
회귀계수 $(\beta_0, \beta_d, \beta_w, \beta_m)$는 IS 구간(\ref{subsec:vol_wf})에서 보통최소제곱법(Ordinary Least Squares, OLS)으로 적합하며, OOS 예측은 적합된 계수와 OOS 시점의 입력을 결합하여 산출한다.
\begin{equation}
\hat{y}^{\text{HAR}}[t] = \beta_0 + \beta_d \log RV_d[t] + \beta_w \log RV_w[t] + \beta_m \log RV_m[t] \label{eq:har_pred}
\end{equation}
HAR-RV는 LSTM 대비 모델 구조가 단순하고 해석 가능성이 높아, 본 연구에서는 LSTM의 비선형 예측을 보완하는 선형 기준 모델로 활용한다.

\paragraph{앙상블 (Performance-Weighted).} 본 연구는 LSTM과 HAR-RV의 예측을 시점별로 가변하는 가중치로 결합하는 Performance-Weighted 앙상블(\cite{diebold1987})을 채택한다. 폴드 $k$의 가중치는 직전 폴드 $k-1$의 OOS RMSE 역수에 비례하도록 설정한다.
\begin{align}
w_k^{\text{LSTM}} &= \frac{1/\text{RMSE}_{k-1}^{\text{LSTM}}}{1/\text{RMSE}_{k-1}^{\text{LSTM}} + 1/\text{RMSE}_{k-1}^{\text{HAR}}}, \label{eq:ens_weight_lstm}\\
w_k^{\text{HAR}} &= 1 - w_k^{\text{LSTM}}. \label{eq:ens_weight_har}
\end{align}
폴드 $k=0$에서는 사전 성능 정보가 없으므로 두 모델에 동일 가중치(각 0.5)를 부여한다. 폴드 $k$의 최종 예측은 두 모델 예측의 가중 평균이다.
\begin{equation}
\hat{y}^{\text{ens}}[t] = w_k^{\text{LSTM}} \cdot \hat{y}^{\text{LSTM}}[t] + w_k^{\text{HAR}} \cdot \hat{y}^{\text{HAR}}[t] \label{eq:ens_pred}
\end{equation}
이 구조는 시장 상황에 따라 우세한 모델의 비중이 자동으로 증가하는 동적 적응 특성을 가진다. 본 연구의 예비 평가에서 Performance-Weighted 앙상블은 평가 대상 다수 종목에서 단일 LSTM 및 단일 HAR-RV 대비 RMSE와 QLIKE 모두에서 우위를 보였으며, 이는 두 모델의 오차 구조가 부분적으로 보완적임을 시사한다. 그림~\ref{fig:ensemble_flow}은 입력에서 블랙-리터만 입력까지의 전체 흐름을 요약한다.

\begin{figure}[ht]
\centering
\renewcommand{\arraystretch}{1.4}
\begin{tabular}{c c c c c c c}
\textbf{입력} & & \textbf{모델} & & \textbf{가중 결합} & & \textbf{블랙-리터만 입력} \\
\hline
$\mathbf{x}[t]$ (4채널) & $\longrightarrow$ & LSTM & $\searrow$ & & & \\
& & & & $\hat{y}^{\text{ens}}[t]$ & $\longrightarrow$ & $\hat{\sigma}[t]$ \\
$RV_d, RV_w, RV_m$ & $\longrightarrow$ & HAR-RV & $\nearrow$ & & & \\
\end{tabular}
\caption{Performance-Weighted 앙상블의 예측 흐름. LSTM과 HAR-RV는 각각의 입력으로 log-daily 변동성을 예측하고, 폴드 단위 RMSE 가중치(식~\ref{eq:ens_weight_lstm})로 결합된 후 식~\ref{eq:bl_input}의 변환($\exp(\cdot) \times \sqrt{252}$)을 거쳐 블랙-리터만 입력 단위(연환산 $\hat{\sigma}$)로 변환된다.}
\label{fig:ensemble_flow}
\end{figure}

\paragraph{ANN (비교 baseline).} 본 연구는 변동성 예측 모델의 비교 baseline으로 \cite{pyo2018}이 블랙-리터만 view 산출에 사용한 인공신경망(Artificial Neural Network, ANN, ) 구조를 채택한다. 해당 ANN은 종목 $i$의 직전 10개월 변동성을 입력으로 받아 다음 1개월 변동성을 출력한다.
\begin{equation}
\hat{\sigma}_{i,t} = f(\sigma_{i,t-1}, \sigma_{i,t-2}, \dots, \sigma_{i,t-10}), \quad i \in \{1, \dots, n\} \label{eq:ann_pred}
\end{equation}
ANN의 내부 연산은 다음과 같이 정의된다. 은닉층 노드 $j$의 가중합과 ReLU 활성화 출력은 각각
\begin{equation}
\text{net}_j = \sum_{i} w_{ij} x_i, \qquad o_j = f(\text{net}_j), \qquad f(S) = \max(0, S), \label{eq:ann_forward}
\end{equation}
이며, 학습은 손실 함수 $\mathcal{L}$에 대한 역전파(backpropagation)로 가중치를 갱신한다.
\begin{equation}
w_{ij}^{(t+1)} = w_{ij}^{(t)} - \eta \frac{\partial \mathcal{L}}{\partial w_{ij}} \label{eq:ann_backprop}
\end{equation}
본 연구의 구현은 \cite{pyo2018}의 사양에 따라 단일 은닉층 4개 뉴런 구조(학습 가능한 파라미터 약 49개)를 사용하며, 활성화 함수는 ReLU, 옵티마이저는 Adam이다. 과적합 방지를 위해 L2 정규화($\alpha = 0.01$)를 적용하고, 학습은 60개월 롤링 윈도우에서 종목별로 수행한다. 종목·시점별 학습 표본이 50개 내외에 머무르는 작은 데이터 규모는 \cite{pyo2018}의 원 사양 보존을 위한 선택이며, 모델 출력 공간은 LSTM·HAR-RV·앙상블과 동일한 로그 일별 종가 변동성으로 통일하여 \ref{subsec:vol_wf}의 Walk-Forward 평가에서 공정한 직접 비교가 가능하도록 한다.

\subsubsection{Walk-Forward 구조}
\label{subsec:vol_wf}

본 절은 변동성 예측 모델의 학습·평가에 공통 적용되는 Walk-Forward 구조를 정의한다. 시계열 데이터의 정보 누수(look-ahead bias)와 시변 분포(non-stationarity) 문제를 동시에 다루기 위해, 본 연구는 \cite{lopezdeprado2018}이 제안한 Purge-Embargo 표준 절차를 따르는 Walk-Forward 교차검증을 채택한다.

\paragraph{폴드 구성.} 각 폴드는 그림~\ref{fig:walk_forward}와 같이 In-Sample(IS), Purge, Embargo, Out-of-Sample(OOS)의 4개 구간으로 구성된다. IS 구간에서 모델을 학습하고, IS 종료점 직후 Purge 구간을 두어 IS의 마지막 시점이 가지는 forward 타깃(식~\ref{eq:lstm_target})이 OOS 시점과 시간적으로 겹치지 않도록 격리한다. Purge 이후 Embargo 구간은 변동성의 장기 자기상관이 충분히 감쇠할 때까지 OOS 시작점을 지연시키는 추가 안전구간이다. 마지막으로 OOS 21영업일에서 모델 예측 성능을 평가하며, 각 폴드는 step $= 21$영업일 단위로 이동하여 다음 폴드를 형성한다.

% LSTM 성능이 ANN 대비해선 좋은게 맞는데, LSTM 자체가 HAR-RV보다 좋지 않다!
% -> 통계 모델은 해석이 가능하다! 근데 LSTM이 딥러닝이고 블랙박스인데 통계 모델보다 성능이 좋지 않다면, 사실 딥러닝을 쓸 이유가 없다!
% -> 사실 입력 변수나 채널의 수를 더 늘린다던지 -> 이런 식으로 LSTM의 성능을 더 개선하는 게 향후 연구 과제이다


\subsection{블랙-리터만 구조}

블랙-리터만(Black–Litterman, BL, \cite{black1992}) 모델은 시장 균형 수익률과 투자자의 뷰를 베이지안 방식으로 결합하는 포트폴리오 최적화 프레임워크로, Markowitz 평균-분산 최적화가 갖는 추정 오차 민감성을 완화하는 데 효과적이다. 모델은 시장 균형 초과수익률 벡터 $\boldsymbol{\pi}$에서 출발하며, 이는 이차 효용 최대화 문제의 해로 도출된다.

\begin{align}\label{1st}
    & \boldsymbol{\pi} = \lambda\Sigma\mathbf{w_{mkt}}
\end{align}

여기서 $\boldsymbol{\pi} \in R^{n \times 1}$은 균형 초과수익률 벡터, $\boldsymbol{\Sigma}_{n \times n}$는 초과수익률 공분산 행렬, $\mathbf{w_{mkt}} \in R^{n \times 1}$은 시장 비중 벡터다. 위험회피계수 $\lambda$는 본 연구에서 $\lambda = 2.5$로 고정하였으며, 이는 BL 문헌의 일반적 관행을 따른 것이다(\cite{helitterman1999}). 공분산 행렬 $\boldsymbol{\Sigma}$는 직전 60개월(약 1,250 거래일) 일별 수익률에 Ledoit-Wolf 수축 추정량을 적용하여 산출하며, 이는 고차원 추정 시 발생하는 표본 오차를 완화하기 위함이다(\cite{ledoitwolf2004}).

사전 분포(prior)는 다음과 같이 설정된다.

\begin{align}\label{2st}
    \boldsymbol{\mu} \sim N(\boldsymbol{\pi}, \tau\boldsymbol{\Sigma})
\end{align}

$\tau$는 사전 분포의 불확실성을 조절하며, 본 연구에서는 $\tau = 0.1$로 설정하였다. 투자자의 뷰는 다음과 같이 표현된다.

\begin{align}\label{3rd}
    q|\boldsymbol{\mu} \sim N(\mathbf{p}\boldsymbol{\mu}, \omega)
\end{align}

여기서 $\mathbf{p} \in R^{1 \times n}$은 뷰 포트폴리오 가중치 벡터, $q \in R$은 단일 상대 뷰의 기대 수익률 스칼라, $\omega$는 뷰의 불확실성을 나타내는 스칼라다. 사전 분포와 뷰 우도를 결합하면 사후 기대수익률과 공분산은 다음과 같이 도출된다.

\begin{align}\label{4th}
    & \boldsymbol{\Sigma}_{BL} = [\tau\boldsymbol{\Sigma}^{-1} + \mathbf{p}^T\omega^{-1}\mathbf{p}]^{-1}\\
    \label{5th}
    & \boldsymbol{\mu}_{BL} = [\tau\boldsymbol{\Sigma}^{-1} + \mathbf{p}^T\omega^{-1}\mathbf{p}]^{-1}[(\tau\boldsymbol{\Sigma})^{-1}\boldsymbol{\pi}+\mathbf{p}^T\omega^{-1}q]
\end{align}

최적 포트폴리오 비중은 다음을 풀어 산출한다.

\begin{align}
    min\frac{1}{2}\lambda\mathbf{w}^T\boldsymbol{\Sigma}_{BL}\mathbf{w}-\mathbf{w}^T\boldsymbol{\mu}_{BL} \\
    \text{s.t.} \quad \sum_{i=1}^{n}{\mathbf{w_i}} = 1, \mathbf{w_i} \leq 0.1
\end{align}

단일 종목 비중 상한 10\%는 극단적 집중 투자를 방지하기 위한 제약이다.

\subsubsection{제안 슬롯}\label{bl:proposed_slot}

본 연구의 핵심은 \ref{subsec:vol_pred}절의 변동성 예측 결과를 BL 프레임워크의 각 슬롯에 체계적으로 연결하는 것이다. 예측 변동성이 $\mathbf{p}$ 벡터를 통해 자산을 분류하고, 이를 기반으로 q와 $\omega$가 결합되어 사후 기대수익률 $\boldsymbol{\mu}_{BL}$이 산출된다. 본 절에서 각 슬롯의 제안 설정을 기술한다.

\paragraph{사전 분포(Prior)}

균형 수익률 계산에 필요한 시장 비중 $\mathbf{w}_{mkt}$를 세 가지 방식으로 구성한다.

\begin{itemize}
    \item $\mathbf{w}^{\mathrm{mcap}}$ : 시가총액 비중 가중. \cite{pyo2018}의 원안이며, CAPM의 표준적 해석이다.
    \item $\mathbf{w}^{\mathrm{eq}}$ : 동일가중. 유니버스 전체에 균등한 사전 정보를 부여한다.
    \item $\mathbf{w}^{\mathrm{rp}}$ : 21일 실현 변동성의 역수($1/\sigma_{\mathrm{realized}}$) 비중 가중치. 저변동 종목에 더 큰 가중치를 배분하는 리스크 패리티(\cite{maillard2010}) 철학에 기반하였다. 단, 사용하는 변동성은 LSTM 예측값이 아닌 전월의 실현 변동성이다.
\end{itemize} 

\paragraph{뷰 포트폴리오($\textbf{p}$)}

$\mathbf{p}$ 벡터는 저변동 자산이 고변동 자산을 초과 성과할 것이라는 뷰를 인코딩한다.
각 리밸런싱 시점 $t$에서 유니버스 내 종목을 LSTM 예측 변동성 $\hat{\sigma}_{i,t}$ 기준으로 
오름차순 정렬 후, 하위 30\%(저변동 그룹 $L$)에 양($+$)의 가중치, 상위 30\%(고변동 그룹 $H$)에 
음($-$)의 가중치를 부여하며, 나머지 40\%는 0으로 처리한다.

$\mathbf{p}$ 벡터의 가중 방식으로 세 가지를 제안한다.

\begin{itemize}
  \item $\mathbf{p}^{\mathrm{mcap}}$: 시가총액 비례 가중. \citet{pyo2018}의 원안으로, 
        각 그룹 내 시가총액 합계로 정규화한다.
        \begin{equation}
          p_i^{\mathrm{mcap}} =
          \begin{cases}
            +\dfrac{\mathrm{mcap}_i}{\sum_{j \in L} \mathrm{mcap}_j}, & i \in L, \\[8pt]
            -\dfrac{\mathrm{mcap}_i}{\sum_{j \in H} \mathrm{mcap}_j}, & i \in H, \\[6pt]
            \phantom{+}0, & \text{otherwise},
          \end{cases}
          \label{eq:p_mcap}
        \end{equation}
        여기서 $\mathrm{mcap}_i$는 종목 $i$의 시가총액이다.

  \item $\mathbf{p}^{\mathrm{eq}}$: 동일가중. 각 그룹 내 모든 종목에 $+1/|L|$(롱 측) 
        또는 $-1/|H|$(숏 측)을 배분하여, 시가총액 집중에 따른 편향을 제거하고 변동성 
        분류 신호를 균등하게 반영한다.

  \item $\mathbf{p}^{\mathrm{rp}}$: 역변동성 가중. 롱 측($L$)에서는 예측 변동성의 역수에 
        비례하여 저변동 종목에 더 큰 비중을 배분하고, 숏 측($H$)에서는 예측 변동성 자체에 
        비례하여 고변동 종목에 더 큰 숏 포지션을 부여한다.
        \begin{equation}
          p_i^{\mathrm{rp}} =
          \begin{cases}
            +\dfrac{\mathrm{1/\hat{\sigma}}_i}{\sum_{j \in L} \mathrm{1/\hat{\sigma}}_j}, & i \in L, \\[8pt]
            -\dfrac{\mathrm{1/\hat{\sigma}}_i}{\sum_{j \in H} \mathrm{1/\hat{\sigma}}_j}, & i \in H, \\[6pt]
            \phantom{+}0, & \text{otherwise},
          \end{cases}
          \label{eq:p_rp}
        \end{equation}
        분류(상하위 30\% 선택) 단계에서는 LSTM 예측값의 순위 정보를, 가중(그룹 내 배분) 단계에서는 크기 정보를 각각 활용한다는 점에서, $\mathbf{p}^{\mathrm{rp}}$는 LSTM 예측값이 이중으로 활용되는 구조다.
\end{itemize}

\paragraph{뷰 기대수익률($q$)}

$q$는 저변동-고변동 그룹 간 기대수익률 스프레드로, 본 연구는 다섯 가지 슬롯을 비교한다. 
다섯 슬롯은 $q$를 결정하는 정보 채널에 따라 다음 세 범주로 구분된다.

\begin{itemize}
  \item \textbf{시장 위험회피도 ($\lambda_q$) 기반} : $q^{\mathrm{lam}}$, 
        $q^{\mathrm{raw}}$, $q^{\mathrm{inv}}$ — SPY 초과수익과 시장 분산으로 추정한 
        매크로 위험회피도를 신호로 사용한다.
  \item \textbf{LSTM 예측 변동성 ($\Delta\hat{\sigma}$) 기반} : $q^{\mathrm{vsp}}$ — 
        고/저변동 그룹의 예측 변동성 격차를 $q$에 직접 반영한다.
  \item \textbf{Fama-French 3팩터 모형 기반} : $q^{\mathrm{fpm}}$ — \citet{pyo2018}의 원안에 
        해당하는 본 연구의 앵커 슬롯.
\end{itemize}

앞의 두 분류(총 4개 슬롯)는 공통 기준값 $q_{\mathrm{base}} = 0.003$를 시장 신호로 동적 조정하는 반면, $q^{\mathrm{ff3}}$는 종목별 팩터 회귀 결과를 $\mathbf{p}$ 벡터와 내적하여 $q$를 직접 산출한다.

\vspace{1em}

\textbf{기준값 $q^{\mathrm{base}}$ 설정.}

4개 동적 조정 슬롯의 기준값으로 사용하는 $q^{\mathrm{base}} = 0.003$(월 0.3\%)은 
저위험 이상현상 실증 문헌의 보고치를 보수적으로 반영한 값이다. \citet{frazzini2014}는 
미국 주식 표본에서 BAB(Betting Against Beta) 팩터의 월 평균 초과수익을 약 0.55--0.70\% 
수준으로 보고하며, \citet{baker2011}, \citet{blitz2007} 역시 저변동 long-short 전략의 
월 수익이 0.4--0.7\% 범위임을 확인한다. 본 연구는 예측 모델의 오판 가능성이 있어 실증치의 약 절반(0.3\%)을 기준값으로 채택한다.

\vspace{1em}

\textbf{$\lambda_q$ 기반 슬롯.} 

\begin{itemize}
  \item $q^{\mathrm{lam}}$: $\lambda_q$ 비례 조정. 시장이 안정적($\lambda_q$ 상승)일수록 
        $q$를 강화하고, 불안정($\lambda_q$ 하락)할수록 약화한다.
        \begin{equation}
          q^{\mathrm{lam}} = q^{\mathrm{base}} \times 
          \mathrm{clip}(\lambda_q/\bar{\lambda},\ 0.1,\ 3.0).
        \end{equation}

  \item $q^{\mathrm{raw}}$: clip 적용 이전의 $\lambda_q$ 사용. SPY 하락으로 
        $\lambda_q^{\mathrm{raw}}<0$이면 $q$가 0으로 수렴하여, 시장 방향에 따라 뷰가 
        자연 소멸된다.
        \begin{equation}
          q^{\mathrm{raw}} = \max\!\left(0,\ q^{\mathrm{base}} \times 
          \lambda_q^{\mathrm{raw}}/\bar{\lambda}\right).
        \end{equation}

  \item $q^{\mathrm{inv}}$: $\lambda_q$ 역수 조정. 불안정 국면에서 $q$를 강화하여 
        저변동 방어 전략을 키우는, $q^{\mathrm{lam}}$과 반대 방향으로 작동한다.
        \begin{equation}
          q^{\mathrm{inv}} = q^{\mathrm{base}} \times 
          \mathrm{clip}(\bar{\lambda}/\lambda_q,\ 0.1,\ 3.0).
        \end{equation}
\end{itemize}

위 수식에서 동적 조정에 사용하는 시장 위험회피도 $\lambda_q$는 $\boldsymbol{\pi}$ 계산용 고정 $\lambda=2.5$와 별개로, 매 시점 다음과 같이 추정한다.
\begin{equation}
  \lambda_q = \mathrm{clip}\!\left(\bar{r}_{\mathrm{SPY}}^{\mathrm{ex}}/\sigma_{\mathrm{mkt}}^2,\ 0.5,\ 10.0\right).
\end{equation}
$\bar{\lambda}$는 훈련 기간 내 $\lambda_q$의 평균이다.
% clip 그대로 써도 될지? r_spy같은 것들도 수정해야 함

\vspace{1em}

\textbf{예측 변동성 기반 슬롯.}
$q^{\mathrm{vsp}}$는 LSTM 예측 변동성 격차를 훈련 기간 중앙값으로 정규화하여 $q$의 크기를 조정한다. $\lambda_q$ 분류가 매크로 시장 신호를 사용하는 반면, $q^{\mathrm{vsp}}$는 LSTM 예측 변동성을 $q$에 직접 반영한다.
\begin{equation}
  q^{\mathrm{vsp}} = q^{\mathrm{base}} \times 
  \mathrm{clip}\!\left(\Delta\hat{\sigma}_t/\mathrm{median}(\Delta\hat{\sigma}_{\mathrm{train}}),\ 
  0.1,\ 3.0\right),
\end{equation}
여기서 $\Delta\hat{\sigma}_t = \mathrm{mean}(\hat{\sigma}_{i\in H}) - 
\mathrm{mean}(\hat{\sigma}_{i\in L})$이며, $H$, $L$의 정의에 의해 $\Delta\hat{\sigma}_t>0$이 보장된다.

\vspace{1em}

\textbf{팩터 모형 기반 슬롯.}
$q^{\mathrm{fpm}}$은 FF3 훈련 윈도우(60개월) 내 팩터 평균을 다음 달 기대 팩터 프리미엄으로 사용하는 방식이다 \citep{fama1973risk, cochrane2005}. 단기 팩터 프리미엄 추정의 불안정성을 감안하여 장기 평균을 기대치로 대입함으로써 추정 안정성을 높인다. 종목별 FF3 롤링 회귀로 추정한 기대수익률 $\hat{r}_i$와 $\mathbf{p}$ 벡터의 내적으로 $q$를 산출하며, 훈련 기간 내 데이터만 사용하므로 look-ahead bias가 없다. \citet{pyo2018}의 FF3 기반 $q$  산출 방식을 장기 평균 팩터 프리미엄 관점으로 재해석한 것으로, 본 연구의 앵커(anchor) 슬롯에 해당한다.
\begin{equation}
  q^{\mathrm{fpm}} = \mathbf{p}^{\top}\hat{\mathbf{r}}, \qquad
  \hat{r}_i = \hat{\alpha}_i + \hat{\beta}_i^{\mathrm{mkt}}\bar{f}_{\mathrm{mkt}} 
  + \hat{\beta}_i^{\mathrm{SMB}}\bar{f}_{\mathrm{SMB}} 
  + \hat{\beta}_i^{\mathrm{HML}}\bar{f}_{\mathrm{HML}},
\end{equation}
여기서 $\bar{f}_k$는 훈련 윈도우 내 팩터 $k$의 평균값이다.

\paragraph{뷰 신뢰도($\omega$)}

$\omega$는 사전 균형 수익률 대비 투자자 뷰의 신뢰도를 결정하며, 본 연구는 두 가지 설정을 비교한다.

\begin{itemize}
  \item \textbf{$\omega^{\mathrm{he}}$ (He-Litterman 표준)}: 뷰 포트폴리오의 사전 분산에 비례하여 불확실성을 설정하는 BL 문헌의 표준 방식이다 
        \citep{helitterman1999}.
        \begin{equation}
          \omega^{\mathrm{he}} = \tau \cdot \mathbf{p}\,\boldsymbol{\Sigma}\,\mathbf{p}^{\top}.
        \end{equation}

  \item \textbf{$\omega^{\mathrm{err}}$ (전월 예측 오차 기반)}: 각 시점 $t$에서 전월($t-1$)의 뷰 포트폴리오 예측 수익률과 실현 수익률 간 오차 제곱으로 불확실성을 추정하며, \citet{pyo2018}의 방식을 따른다.
        \begin{equation}
          \omega^{\mathrm{err}} = \left(q_{t-1} - \mathbf{p}_{t-1}^{\top}\mathbf{r}_{t-1}\right)^{2},
        \end{equation}
        여기서 $\mathbf{r}_{t-1}$은 $t-1$ 시점의 실현 수익률 벡터다. 전월 뷰 예측이 크게 빗나간 시점에서 불확실성이 자동으로 증가하는 자기교정적 구조이며, 첫 달은 $\omega^{\mathrm{he}}$로 초기화한다.
\end{itemize}

\subsubsection{Walk-Forward 구조}
\label{subsec:bl_walkforward}

포트폴리오 비중은 \ref{subsec:vol_wf}절의 변동성 예측과 동일한 원칙을 따르는 Walk-Forward 구조로 산출한다. 각 리밸런싱 시점 $t$(월 1회)에서 직전 60개월 일별 수익률로 공분산 행렬 $\boldsymbol{\Sigma}_t$를 추정하고, 해당 시점의 시가총액 비중 
$w_{\mathrm{mkt},t}$와 위험회피도 $\lambda = 2.5$로 균형 수익률 $\pi_t = \lambda \boldsymbol{\Sigma}_t w_{\mathrm{mkt},t}$를 계산한다. 이후 LSTM(또는 ANN 베이스라인)의 
예측 변동성 $\hat{\sigma}_{i,t}$로 $\mathbf{p}_t$를 구성하고, 설정에 따라 $q_t$와 
$\omega_t$를 산출한 뒤 BL 사후 분포의 평균 $\mu_{\mathrm{BL},t}$와 공분산 
$\boldsymbol{\Sigma}_{\mathrm{BL},t}$를 계산하여 포트폴리오 비중 최적화를 수행한다.

각 리밸런싱 시점에서 훈련 윈도우(60개월) 내 일별 수익률 데이터가 90\% 이상 존재하는 
종목만 유효 유니버스로 선별한다. 거래비용은 편측 20bp를 적용하며, 월별 실제 비중 변화량 $\sum_i |\Delta w_{i,t}|$에 
비례하여 차감한다. 이는 S\&P 500 구성 종목의 실제 매매 스프레드 대비 보수적인 설정으로, $\omega^{\mathrm{err}}$ 슬롯의 뷰 교정이 고변동 구간에서 회전율을 높일 수 있다는 점을 고려한 것이다. 무위험 수익률은 미국 3개월 국채 수익률을 사용하며, Sharpe 비율 산출 시 초과수익률 계산에 적용한다.
% 평균 N종목 들어가면 좋을 듯?
% \Delta w_{i,t}에 비례 차감은 확인해봐야 함

\subsubsection{평가 방법론}
\label{subsec:bl_eval}

전체 분석 기간은 2010년 1월부터 2025년 12월까지(192개월)이며, 시장 국면에 따라 성과가 크게 달라질 수 있으므로 전체 기간 평가와 레짐 별 성과 분해를 병행한다.

레짐 분류는 \ref{app:hmm}에 상세히 기술하며, Hidden Markov Model(HMM, \cite{hamilton1989new}) 기반 통계적 구조전환점 탐지와 금융 이벤트 도메인 지식을 결합하여 네 개의 비중첩 구간을 식별하였다(표 \ref{tab:regimes}).

각 레짐 및 전체 기간에 대해 Sharpe 비율, Sortino 비율, MDD(Maximum Drawdown)를 보고한다. 레짐별 분해를 통해 LSTM 기반 BL 포트폴리오의 ANN 대비 우위가 특정 시장 국면에 집중된 것인지, 전반적으로 일관된 것인지를 검증한다. 이를 위해 LSTM과 ANN 베이스라인은 동일한 90개 슬롯 조합에 대해 1:1 대응 비교하며, 슬롯 구성은 두 모델 간 완전히 동일하게 유지한다.

\begin{table}[ht]
\centering
\caption{분석 레짐(regime) 구분 및 시기별 특성}
\label{tab:regimes}
\begin{tabular}{llcl}
\toprule
레짐 & 기간 & 기간(개월) & 특성 \\
\midrule
R1 (회복기)  & 2010.01--2012.06 & 30 & Post-GFC 회복, 유럽 재정위기 \\
R2 (확장기)  & 2012.07--2019.12 & 90 & 장기 Bull Market, 저변동 \\
R3 (위기기)  & 2020.01--2023.06 & 42 & COVID-19, 2022 약세장 \\
R4 (정상화)  & 2023.07--2025.12 & 30 & AI 주도 랠리 \\
\midrule
\multicolumn{1}{l}{전체} & 2010.01--2025.12 & 192 & --- \\
\bottomrule
\end{tabular}
\end{table}

\section{실증 분석}\label{result}

본 연구에서 활용한 야후 파이낸스의 주가 데이터는 2010년부터 2025년까지 S\&P500에 한 번이라도 포함된 총 617 종목을 대상으로 한다. 포트폴리오는 매월 말 리밸런싱하며(총 192개월) 거래비용은 편측 20bp를 적용하고, 무위험 수익률은 1개월 미국 국채 수익률을 사용한다.

\subsection{Volatility Forecast Performance}

표 \ref{tab:sigma_pred}는 LSTM과 ANN의 변동성 예측 성능을 전체 기간과 4개 레짐 국면 별로 비교한 결과이다. LSTM은 RMSE, Spearman 순위상관, 상\/하위 30\% Hit rate 지표 모두에서 ANN을 일관되게 상회한다. 이는 시장 국면과 무관하게 LSTM의 변동성 예측이 ANN 대비 우위에 있음을 시사한다.

특히, R3에서 다른 구간 대비 격차가 가장 크게 나타나며, 특히 Spearman 순위 상관 격차가 +0.326으로 가장 두드러지는데, 이는 LSTM이 종목 간 변동성의 상대적 순위를 금융 위기 구간에서 보다 잘 보존함을 알 수 있다.

Hit Rate 또한 동일한 패턴을 보여 R3에서 하위 30\% 식별 정확도가 +0.047, 상위 30\% 식별 정확도가 +0.037 향상되는데, 이는 LSTM이 저변동성 종목을 정확히 선별하는 능력뿐 아니라 고변동성 종목을 정확히 배제하는 양방향 선별력에서도 ANN을 능가함을 보여준다. 본 연구의 포트폴리오는 저변동성 종목의 비중 확대와 고변동성 종목의 비중 축소가 모두 작동해야 하므로, Hit Rate 양 끝단에서의 우위는 포트폴리오 단계 성과로 직접 전이될 가능성이 높다. 이러한 예측 성능 차이가 포트폴리오 단계에서 어떻게 전이되는지는 \ref{baseline_slot}절에서 살펴본다.

\begin{table}[ht]
\centering
\caption{변동성 예측 성능: LSTM vs ANN ($\Delta = $LSTM$-$ANN)}
\label{tab:sigma_pred}
\resizebox{\textwidth}{!}{%
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.15}
\small
\begin{tabular}{lccc ccc ccc ccc}
\toprule
        & \multicolumn{3}{c}{RMSE} & \multicolumn{3}{c}{Spearman} & \multicolumn{3}{c}{Hit Rate(low) (30\%)} & \multicolumn{3}{c}{Hit Rate(high) (30\%)} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10} \cmidrule(lr){11-13}
Period & LSTM & ANN & $\Delta$ & LSTM & ANN & $\Delta$ & LSTM & ANN & $\Delta$ & LSTM & ANN & $\Delta$ \\
\midrule
All       & 0.3685 & 0.4521 & \textbf{-0.0837} & 0.6805 & 0.5305 & \textbf{+0.1499} & 0.6319 & 0.6023 & \textbf{+0.0295} & 0.6612 & 0.6390 & \textbf{+0.0222} \\
R1 (회복)   & 0.3534 & 0.5168 & \textbf{-0.1633} & 0.7218 & 0.5446 & \textbf{+0.1771} & 0.7192 & 0.6932 & \textbf{+0.0261} & 0.6986 & 0.6663 & \textbf{+0.0323} \\
R2 (확장)   & 0.3668 & 0.4019 & \textbf{-0.0351} & 0.5928 & 0.5071 & \textbf{+0.0856} & 0.6070 & 0.5827 & \textbf{+0.0244} & 0.6376 & 0.6248 & \textbf{+0.0128} \\
R3 (위기)   & 0.3855 & 0.5298 & \textbf{-0.1443} & 0.6996 & 0.3739 & \textbf{+0.3257} & 0.6309 & 0.5836 & \textbf{+0.0473} & 0.6895 & 0.6525 & \textbf{+0.0370} \\
R4 (AI랠리) & 0.3610 & 0.4039 & \textbf{-0.0430} & 0.6465 & 0.5400 & \textbf{+0.1065} & 0.6202 & 0.5967 & \textbf{+0.0235} & 0.6546 & 0.6353 & \textbf{+0.0193} \\
\bottomrule
\end{tabular}
}
\end{table}

\subsection{Baseline Slot}\label{baseline_slot}

본 절에서는 \citet{pyo2018}이 제안한 슬롯을 기준 슬롯(mcap\_mcap\_fpm\_err, 표 \ref{tab:1step_variation}에서 $\star$로 표기)으로 두고, $q$, $\mathbf{p}$, prior, $\omega$ 네 차원을 하나씩 변경하며 LSTM과 ANN의 Sharpe 비율 차이를 비교한다.

\begin{table}[ht]
\centering
\caption{\cite{pyo2018}의 기준 슬롯(mat\_mcap\_mcap\_fpm\_err, $\star$로 표기)에서 한 차원씩 변경한 BL 슬롯의 Sharpe 비율. LSTM(L)과 ANN(A)의 Sharpe 비율 및 그 차이 $\Delta=\mathrm{L}-\mathrm{A}$를 전체 표본과 4개 구간(R1 회복, R2 확장, R3 위기, R4 AI랠리)에 대해 비교한다. 양의 $\Delta$는 LSTM 우위를 의미한다.}
\label{tab:1step_variation}
\setlength{\tabcolsep}{2.5pt}
\renewcommand{\arraystretch}{1.1}
\footnotesize
\resizebox{\textwidth}{!}{%
\begin{tabular}{ll ccc ccc ccc ccc ccc}
\toprule
   &        & \multicolumn{3}{c}{All} & \multicolumn{3}{c}{R1} & \multicolumn{3}{c}{R2} & \multicolumn{3}{c}{R3} & \multicolumn{3}{c}{R4} \\
\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11} \cmidrule(lr){12-14} \cmidrule(lr){15-17}
차원 & 슬롯 & L & A & $\Delta$ & L & A & $\Delta$ & L & A & $\Delta$ & L & A & $\Delta$ & L & A & $\Delta$ \\
\midrule
\multirow{5}{*}{\shortstack[l]{SET 1\\(q)}}
  & q=lam   & 0.913 & 0.952 & \cellcolor{negred!50}$-0.038$  & 0.821 & 1.041 & \cellcolor{negred!50}$-0.220$ & 0.953 & 1.213 & \cellcolor{negred!50}$-0.260$ & 0.991 & 0.638 & \cellcolor{posgreen!50}$+0.353$ & 0.864 & 0.833 & \cellcolor{posgreen!50}$+0.030$ \\
  & q=raw   & 0.976 & 0.988 & \cellcolor{negred!50}$-0.012$  & 0.818 & 1.044 & \cellcolor{negred!50}$-0.226$ & 1.115 & 1.332 & \cellcolor{negred!50}$-0.217$ & 0.999 & 0.643 & \cellcolor{posgreen!50}$+0.357$ & 0.864 & 0.833 & \cellcolor{posgreen!50}$+0.030$ \\
  & q=inv   & 0.995 & 0.982 & \cellcolor{posgreen!50}$+0.014$ & 1.061 & 1.244 & \cellcolor{negred!50}$-0.183$ & 1.100 & 1.084 & \cellcolor{posgreen!50}$+0.016$ & 0.894 & 0.738 & \cellcolor{posgreen!50}$+0.156$ & 0.915 & 0.963 & \cellcolor{negred!50}$-0.048$ \\
  & q=vsp   & 0.949 & 0.948 & $+0.000$                       & 0.894 & 1.105 & \cellcolor{negred!50}$-0.211$ & 1.030 & 1.067 & \cellcolor{negred!50}$-0.037$ & 0.943 & 0.720 & \cellcolor{posgreen!50}$+0.223$ & 0.891 & 0.958 & \cellcolor{negred!50}$-0.067$ \\
  & q=fpm $\star$ & 0.946 & 0.945 & $+0.000$                 & 0.738 & 1.059 & \cellcolor{negred!50}$-0.321$ & 1.287 & 1.076 & \cellcolor{posgreen!50}$+0.212$ & 0.702 & 0.850 & \cellcolor{negred!50}$-0.148$ & 0.940 & 0.712 & \cellcolor{posgreen!50}$+0.228$ \\
\midrule
\multirow{3}{*}{\shortstack[l]{SET 2\\($p$)}}
  & p=mcap $\star$ & 0.946 & 0.945 & $+0.000$              & 0.738 & 1.059 & \cellcolor{negred!50}$-0.321$ & 1.287 & 1.076 & \cellcolor{posgreen!50}$+0.212$ & 0.702 & 0.850 & \cellcolor{negred!50}$-0.148$ & 0.940 & 0.712 & \cellcolor{posgreen!50}$+0.228$ \\
  & p=eq          & 1.099 & 1.049 & \cellcolor{posgreen!50}$+0.050$ & 0.872 & 1.024 & \cellcolor{negred!50}$-0.152$ & 1.421 & 1.408 & \cellcolor{posgreen!50}$+0.013$ & 0.888 & 0.705 & \cellcolor{posgreen!50}$+0.183$ & 1.066 & 0.868 & \cellcolor{posgreen!50}$+0.198$ \\
  & p=rp          & 1.054 & 0.970 & \cellcolor{posgreen!50}$+0.084$ & 0.806 & 0.809 & \cellcolor{negred!50}$-0.003$ & 1.344 & 1.311 & \cellcolor{posgreen!50}$+0.033$ & 0.870 & 0.714 & \cellcolor{posgreen!50}$+0.155$ & 1.102 & 0.900 & \cellcolor{posgreen!50}$+0.202$ \\
\midrule
\multirow{3}{*}{\shortstack[l]{SET 3\\(prior)}}
  & prior=mcap $\star$ & 0.946 & 0.945 & $+0.000$            & 0.738 & 1.059 & \cellcolor{negred!50}$-0.321$ & 1.287 & 1.076 & \cellcolor{posgreen!50}$+0.212$ & 0.702 & 0.850 & \cellcolor{negred!50}$-0.148$ & 0.940 & 0.712 & \cellcolor{posgreen!50}$+0.228$ \\
  & prior=eq           & 0.894 & 0.889 & \cellcolor{posgreen!50}$+0.006$ & 0.919 & 1.182 & \cellcolor{negred!50}$-0.262$ & 1.276 & 1.007 & \cellcolor{posgreen!50}$+0.269$ & 0.673 & 0.862 & \cellcolor{negred!50}$-0.189$ & 0.623 & 0.420 & \cellcolor{posgreen!50}$+0.202$ \\
  & prior=rp           & 0.918 & 0.904 & \cellcolor{posgreen!50}$+0.014$ & 1.032 & 1.299 & \cellcolor{negred!50}$-0.267$ & 1.298 & 1.038 & \cellcolor{posgreen!50}$+0.260$ & 0.638 & 0.829 & \cellcolor{negred!50}$-0.191$ & 0.651 & 0.394 & \cellcolor{posgreen!50}$+0.257$ \\
\midrule
\multirow{2}{*}{\shortstack[l]{SET 4\\($\omega$)}}
  & omega=err $\star$ & 0.946 & 0.945 & $+0.000$             & 0.738 & 1.059 & \cellcolor{negred!50}$-0.321$ & 1.287 & 1.076 & \cellcolor{posgreen!50}$+0.212$ & 0.702 & 0.850 & \cellcolor{negred!50}$-0.148$ & 0.940 & 0.712 & \cellcolor{posgreen!50}$+0.228$ \\
  & omega=he          & 0.830 & 0.871 & \cellcolor{negred!50}$-0.041$  & 0.882 & 1.133 & \cellcolor{negred!50}$-0.251$ & 1.178 & 1.043 & \cellcolor{posgreen!50}$+0.135$ & 0.580 & 0.680 & \cellcolor{negred!50}$-0.100$ & 0.599 & 0.637 & \cellcolor{negred!50}$-0.038$ \\
\bottomrule
\end{tabular}%
}
\end{table}

전체 기간의 결과를 참고했을 때, LSTM은 대부분의 슬롯에서 ANN 대비 우위를 보이나 기준 슬롯 자체에서의 격차는 $\Delta = +0.000$으로 미미하다. 이는 변동성 예측 모델의 우위가 포트폴리오 성과로 전이되기 위해서는 슬롯 구성 차원의 설계가 함께 동반되어야 함을 시사한다. 아래에서는 4개 차원별로 모델 간 차이의 패턴을 분리해 분석한다.

표 \ref{tab:1step_variation}의 \textbf{q} 변경에서 기존 $\textbf{q}$는 파마-프랜치 3 팩터 회귀 기반으로 산출되어 부호가 유동적인 반면, 본 연구에서 제안한 방식은 항상 $\textbf{q}>0$이 보장되는 구조이다. 이 방식으로 변경하는 경우, R3 위기 구간에서 LSTM 우위가 $\Delta = +0.156 \sim +0.357$로 강하게 나타난다. 반면 기준 슬롯의 $q_{\mathrm{fpm}}$은 R3에서 오히려 ANN이 우위를 보인다($\Delta = -0.148$). 이는 위기 구간에서의 모델 차이가 $q$ 부호의 안정성에 의존하며, 부호가 유동적일 경우 위기 구간에서 LSTM의 변동성 순위 식별력이 포트폴리오 단계에서 희석됨을 시사한다.

\textbf{p} 벡터을 기준 슬롯의 시가총액 가중(mcap)에서 동일 가중(eq) 또는 역변동성 가중(rp)으로 변경하는 경우, 전체 기간 Sharpe 비율이 LSTM은 0.946에서 1.099, 1.054로, ANN은 0.945에서 1.049, 0.970으로 양 모델 모두 상승한다. \textbf{P} 벡터는 개별 종목의 $\sigma$ 예측을 포트폴리오 뷰로 종합하는 가중치를 결정하는데, 동일 가중과 역변동성 가중치는 대형주 비중 집중을 완화하여 예측 우위가 포트폴리오 성과로 보다 효과적으로 전이된다. 모델 자체보다 슬롯 구조의 영향이 더 결정적임을 보여주는 결과이며, 이후 \ref{subsec:slot_config}에서 전체 슬롯에 걸친 분포로 강건성을 확인한다.

사전 분포(prior)를 시가총액 비중에서 동일 가중, 역변동성 가중으로 변경하더라도 $\Delta$의 부호 패턴은 기준 슬롯과 동일하게 유지된다. 즉, R1, R3에서 ANN 우위, R2, R4에서 LSTM 우위 구조가 prior 변경과 무관하게 보존되며, 격차의 크기만 조정된다. 사전 분포는 모델 간 차이의 방향이 아닌 크기에만 영향을 미치는 차원으로 해석된다.

$\Omega$ 변경에서 전월 대비 오차(err)의 경우 예측이 크게 빗나간 직후에는 포트폴리오 뷰에 대한 신뢰도를 자동으로 낮춰 사전 분포에 더 의존하므로, 예측력이 시점에 따라 변동하는 환경에서 자동으로 손실을 통제하는 효과를 갖는다. 기준 슬롯이 전월 대비 오차를 채택한 효과가 양 모델 모두에서 공통적으로 나타난다.

\subsection{Slot Configuration}\label{slot_config}

본 절에서는 \S 4.2의 한 차원 변경 결과가 전체 슬롯 조합에 대해서도 일관되게 나타나는지 확인하고, 각 슬롯 차원의 한계 효과(marginal effect)를 분포로 시각화한다. 전체 90 슬롯의 상세 성과는 Appendix \ref{app:result_table}에 첨부하였다.

\begin{figure}[hbt!]
    \centering
    \includegraphics[width=15cm]{Figure/marginal_mean.png}
    \caption{Marginal Effects of Black-Litterman Slots}
    \label{fig:marginal_boxplot}
\end{figure}

\begin{table}[ht]
\centering
\caption{\textbf{p} 벡터 변경에 따른 평균 Sharpe 비율 ($Q$ 5개 $\times$ prior 3개 = 15 슬롯 평균, $\Omega=$err 고정). LSTM(L)과 ANN(A)의 평균 Sharpe 비율 및 차이 $\Delta = \mathrm{L} - \mathrm{A}$를 전체 표본과 4개 구간(R1 회복, R2 확장, R3 위기, R4 AI랠리)에 대해 비교한다.}
\label{tab:p_matrix_avg}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.15}
\footnotesize
\resizebox{\textwidth}{!}{%
\begin{tabular}{l ccc ccc ccc ccc ccc}
\toprule
  & \multicolumn{3}{c}{All} & \multicolumn{3}{c}{R1} & \multicolumn{3}{c}{R2} & \multicolumn{3}{c}{R3} & \multicolumn{3}{c}{R4} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10} \cmidrule(lr){11-13} \cmidrule(lr){14-16}
$p$ & L & A & $\Delta$ & L & A & $\Delta$ & L & A & $\Delta$ & L & A & $\Delta$ & L & A & $\Delta$ \\
\midrule
mcap & 0.929 & 0.948 & \cellcolor{negred!50}$-0.019$ & 1.008 & 1.222 & \cellcolor{negred!50}$-0.214$ & 1.104 & 1.157 & \cellcolor{negred!50}$-0.053$ & 0.869 & 0.735 & \cellcolor{posgreen!50}$+0.134$ & 0.633 & 0.614 & \cellcolor{posgreen!50}$+0.019$ \\
eq & 1.038 & 0.970 & \cellcolor{posgreen!50}$+0.068$ & 1.184 & 1.230 & \cellcolor{negred!50}$-0.046$ & 1.303 & 1.314 & \cellcolor{negred!50}$-0.012$ & 0.923 & 0.669 & \cellcolor{posgreen!50}$+0.253$ & 0.492 & 0.415 & \cellcolor{posgreen!50}$+0.076$ \\
rp & 1.008 & 0.920 & \cellcolor{posgreen!50}$+0.088$ & 1.114 & 1.034 & \cellcolor{posgreen!50}$+0.081$ & 1.248 & 1.238 & \cellcolor{posgreen!50}$+0.010$ & 0.896 & 0.685 & \cellcolor{posgreen!50}$+0.212$ & 0.555 & 0.479 & \cellcolor{posgreen!50}$+0.076$ \\
\bottomrule
\end{tabular}%
}
\end{table}

표 \ref{tab:p_matrix_avg}는 $\omega$를 $\omega_{err}$로 고정한 상태에서 $\textbf{P}$ 벡터 가중치를 mcap/eq/rp 세 가지로 변경하고, 각 조합에 대해 $q$ 5개와 prior 3개의 15개 슬롯을 평균낸 결과이다. 전체 표본 기준으로 mcap 가중 시 LSTM의 변동성 예측 우위에도 불구하고 포트폴리오 Sharpe는 ANN이 $\Delta = -0.019$로 소폭 우위를 보인다. 반면, eq, rp로 가중치를 변경하면 LSTM의 Sharpe 비율은 0.929에서 각각 1.038과 1.008로 상승하며, ANN도 mcap의 0.948에서 eq의 0.970으로 상승한다 (rp는 0.920으로 소폭 감소). 이는 변동성 예측의 우위가 포트폴리오 성과로 전이되기 위해서는 $\mathbf{p}$ 벡터을 대형주 비중 집중이 적은 가중치로 구성해야 함을 시사한다. 그림 \ref{fig:marginal_boxplot}에서도 eq, rp 구성 시 LSTM과 ANN의 분포 차이가 가장 두드러진다.


위기 구간(R3)에서는 $\textbf{P}$ 벡터 구성과 무관하게 LSTM이 일관된 우위를 보이며 ($\Delta = +0.134, +0.253, +0.212$), 표 \ref{tab:p_matrix_win}에서도 거의 모든 슬롯에서 LSTM이 ANN을 상회한다 (mcap 12/15, eq 15/15, rp 15/15). 변동성 분포가 가장 분산되는 위기 시점에 종목 순위 식별의 가치가 극대화되며, LSTM의 변동성 예측 우위가 포트폴리오 성과로 가장 효과적으로 전이된다. 이러한 R3 우위는 Sortino 비율과 MDD에서도 유사한 패턴으로 관찰되며, 상세 슬롯별 지표는 부록 \ref{app:full_grid}에 첨부하였다.

반면 회복기(R1)와 확장기(R2)에서는 mcap과 eq 가중 모두 LSTM이 ANN에 패배하는 슬롯이 더 많지만 (R1: mcap 0/15, eq 1/15; R2: mcap·eq 모두 5/15), 손실 규모 측면에서는 두 가중치가 크게 다르다. R1에서 mcap의 평균 손실 $\Delta = -0.214$는 2011년 8월 미국 국가신용등급 강등을 포함한 소수 이벤트 월에 손실이 집중되고, 이것이 대형주 비중 집중이 강한 mcap에서 증폭되어 나타난 결과이다. 반면 eq에서는 균등 배분으로 충격의 영향이 작아져 손실이 $-0.046$으로 통제된다. 즉 eq 가중은 회복기에 LSTM 우위를 잃는 빈도는 mcap과 유사하나, 손실 규모는 mcap의 대형주 비중 집중이 야기하는 대규모 손실 위험을 효과적으로 완화한다. 자세한 분석은 Appendix \ref{app:r1}에서 확인할 수 있다.
% eq, rp, mcap 일단 막 적고 용어 통일은 한 번 해야할 것 같음. 너무 무지성이네

그림 \ref{fig:marginal_boxplot}의 $q$ 차원을 보면, R3 위기 구간에서 $q^{\mathrm{lam}}$과 $q^{\mathrm{raw}}$처럼 항상 $q>0$이 보장되어 저변동성 종목의 비중을 일관되게 확대하는 방식이 LSTM과 ANN의 성과 차이를 가장 크게 만든다. 부호가 유동적인 $q_{\mathrm{fpm}}$은 R3에서 오히려 ANN 우위로 역전되는데, 이는 LSTM이 식별한 저변동성 종목의 정확한 순위가 $q$ 부호의 안정성을 통해서만 포트폴리오 가중치로 보존됨을 시사한다.

본 절에서 확인한 슬롯 구성이 단순 벤치마크 대비 실제로 의미 있는 성과를 갖는지는 \ref{res:benchmark}절에서 확인한다.

\begin{table}[ht]
\centering
\caption{$\textbf{p}$ 벡터 변경에 따른 LSTM 우위 슬롯 수 ($Q$ 5개 $\times$ prior 3개 = 15 슬롯 중 $\Delta > 0$인 슬롯 수, $\omega=$err 고정).}
\label{tab:p_matrix_win}
\centering
\renewcommand{\arraystretch}{1.15}
\begin{tabular}{l ccccc}
\toprule
$p$ & All & R1 & R2 & R3 & R4 \\
\midrule
mcap & 5/15  & \cellcolor{negred!50}0/15  & 5/15 & \cellcolor{posgreen!50}12/15 & 7/15 \\
eq   & \cellcolor{posgreen!50}15/15 & \cellcolor{negred!50}1/15  & 5/15 & \cellcolor{posgreen!50}15/15 & \cellcolor{posgreen!50}15/15 \\
rp   & \cellcolor{posgreen!50}15/15 & \cellcolor{posgreen!50}14/15 & 9/15 & \cellcolor{posgreen!50}15/15 & \cellcolor{posgreen!50}15/15 \\
\bottomrule
\end{tabular}
\end{table}



\subsection{벤치마크 성능 비교}\label{res:benchmark}

본 절은 \S\ref{result}에서 확인한 변동성 예측 향상 및 슬롯 설계 효과가 표준 벤치마크 대비 실증 우위로 이어지는지 확인한다. 비교 대상은 세 외부 벤치마크(SPY, $1/N$ 동일 가중, Risk Parity)와 \citet{pyo2018}의 ANN 앵커 슬롯, 비교 위해 모델만 LSTM으로 교체한 앵커 슬롯, 그리고 본 연구의 LSTM 기반 두 옵션(각각 $\mathbf{p}$ 벡터 $\mathbf{p}^{\mathrm{eq}}$와 $\mathbf{p}^{\mathrm{rp}}$ 두 변형 — 총 4개 슬롯)이다.

본 연구는 단일 슬롯을 권장하지 않으며, 운영 목표에 따라 선택 가능한 두 옵션을 함께 보고한다. $q$가 항상 양수로 모든 시장 국면에서 저변동 뷰를 일관되게 유지하는 \textbf{LSTM-defensive}($q^{\mathrm{lam}}$)와, \citet{pyo2018}의 FF3 회귀 기반 $q^{\mathrm{fpm}}$을 유지하여 시장 국면에 따라 뷰가 적응적으로 조정되는 \textbf{LSTM-adaptive}($q^{\mathrm{fpm}}$)이다. 각 옵션은 $\mathbf{p}$ 벡터을 $\mathbf{p}^{\mathrm{eq}}$ 또는 $\mathbf{p}^{\mathrm{rp}}$로 선택할 수 있으며, 두 변형은 후술하듯 거의 동등한 성과를 보인다. (R4에서의 적응 메커니즘 상세는 부록 \ref{app:r4_mechanism} 참조.)

\begin{table}[ht]
\centering
\caption{벤치마크 성능 비교 — Sharpe 비율 $\times$ 5 기간. LSTM 슬롯은 모두 prior=mcap 고정, $\mathbf{p}$에서 EQ와 RP 두 변형을 비교한다. MDD, Sortino 등 보조 지표는 부록 \ref{app:full_grid} 참조.}
\label{tab:benchmark_sharpe}
\setlength{\tabcolsep}{6pt}
\renewcommand{\arraystretch}{1.15}
\small
\begin{tabular}{lccccc}
\toprule
Strategy & All & R1 & R2 & R3 & R4 \\
\midrule
SPY (market)                       & 0.93 & 0.81 & 1.23 & 0.61 & 1.19 \\
1/N (equal-weight)                 & 0.86 & 0.78 & 1.23 & 0.70 & 0.47 \\
Risk Parity ($1/\sigma$)           & 0.89 & 0.91 & 1.28 & 0.67 & 0.45 \\
ANN-anchor \citep{pyo2018}         & 0.95 & 1.06 & 1.08 & 0.85 & 0.71 \\
LSTM-anchor ($\sigma$ swap)        & 0.95 & 0.74 & 1.29 & 0.70 & 0.94 \\
\midrule
\textbf{LSTM-defensive-EQ} (eq P + $q^{\mathrm{lam}}$)
                                   & 1.07 & \textbf{1.08} & 1.32 & \textbf{0.96} & 0.64 \\
\textbf{LSTM-defensive-RP} (rp P + $q^{\mathrm{lam}}$)
                                   & 1.04 & 0.98 & 1.28 & 0.95 & 0.71 \\
\textbf{LSTM-adaptive-EQ}  (eq P + $q^{\mathrm{fpm}}$)
                                   & \textbf{1.10} & 0.87 & \textbf{1.42} & 0.89 & 1.07 \\
\textbf{LSTM-adaptive-RP}  (rp P + $q^{\mathrm{fpm}}$)
                                   & 1.05 & 0.81 & 1.34 & 0.87 & \textbf{1.10} \\
\bottomrule
\end{tabular}
\end{table}

표 \ref{tab:benchmark_sharpe}에서 네 LSTM 슬롯은 모두 외부 벤치마크와 BL 앵커를 상회한다. 전체 기간 Sharpe는 LSTM-defensive($\mathbf{p}^{\mathrm{eq}}$ 1.07, $\mathbf{p}^{\mathrm{rp}}$ 1.04)와 LSTM-adaptive($\mathbf{p}^{\mathrm{eq}}$ 1.10, $\mathbf{p}^{\mathrm{rp}}$ 1.05)로, $\mathbf{p}$ 벡터 변형과 무관하게 SPY (0.93), 1/N (0.86), Risk Parity (0.89), ANN-anchor (0.95)를 모두 능가한다. MDD 측면에서도 네 LSTM 슬롯이 $-14.0\% \sim -19.2\%$ 범위로 SPY ($-23.9\%$)와 ANN-anchor ($-18.4\%$) 대비 더 얕은 손실을 기록하며, 그중 LSTM-defensive 두 변형($\mathbf{p}^{\mathrm{eq}}$ $-14.7\%$, $\mathbf{p}^{\mathrm{rp}}$ $-14.0\%$)이 가장 우수하다.

LSTM-anchor의 전체 기간 Sharpe 0.95는 ANN-anchor와 동률이다. 즉 anchor 슬롯에서 $\sigma$ 예측만 LSTM으로 교체했을 때 포트폴리오 성과 우위가 거의 발생하지 않는다. $\mathbf{p}$ 가중치를 mcap에서 $\mathbf{p}^{\mathrm{eq}}$ 또는 $\mathbf{p}^{\mathrm{rp}}$로 변경하면 두 LSTM 옵션 모두 $+0.09 \sim +0.15$의 Sharpe 개선이 일관되게 나타나, 변동성 예측 우위(\S\ref{tab:sigma_pred})가 포트폴리오로 전이되는 결정적 통로가 슬롯 설계, 특히 $\mathbf{p}$ 가중치의 대형주 비중 집중 완화에 있으며, $\mathbf{p}^{\mathrm{eq}}$와 $\mathbf{p}^{\mathrm{rp}}$가 거의 동등한 효과를 제공함을 보여준다.

또한, 두 LSTM 옵션은 시장 국면에 따라 보완적인 강점을 보인다. \textbf{LSTM-defensive}는 회복(R1)과 위기(R3) 등 변동성 상승 구간에서 강점을 보이며 ($\mathbf{p}^{\mathrm{eq}}$ 기준 Sharpe 1.08, 0.96), 양수 제약 $q^{\mathrm{lam}}$이 모든 시점에서 저변동 뷰를 유지하기 때문이다. 반면 \textbf{LSTM-adaptive}는 확장(R2)과 정상화(R4) 강세장에서 강점을 보이며 ($\mathbf{p}^{\mathrm{eq}}$ 기준 Sharpe 1.42, 1.07), $q^{\mathrm{fpm}}$의 FF3 회귀가 강세장에서 뷰를 적응적으로 재조정하여 상승장 포착 능력을 제공하기 때문이다. R3 위기 구간에서는 두 옵션의 네 변형이 모두 ANN-anchor (0.85)를 상회하며 ($0.87 \sim 0.96$), R4 강세장에서는 LSTM-adaptive 두 변형이 ANN-anchor (0.71)를 크게 상회한다 (1.07, 1.10). 한편 $\mathbf{p}$ 변형의 효과는 국면에 따라 비대칭적으로 작동하여 R1에서는 $\mathbf{p}^{\mathrm{eq}}$가 $\mathbf{p}^{\mathrm{rp}}$보다 약 0.10 우위이나 R4에서는 $\mathbf{p}^{\mathrm{rp}}$가 두 옵션 모두에서 $\mathbf{p}^{\mathrm{eq}}$를 약간 상회한다 (LSTM-defensive 0.71 vs 0.64, LSTM-adaptive 1.10 vs 1.07).

% 누적 수익률 그림
\begin{figure}[ht]
\centering
\includegraphics[width=\textwidth]{Figure/benchmark_plot.png}
\caption{전 분석 기간(2010-01~2025-12) 누적수익률(상단, log scale)과 drawdown(하단). R3 위기 구간 음영. LSTM-defensive(파랑)와 LSTM-adaptive(초록)는 전체 기간에 걸쳐 거의 동일한 누적 궤적을 보이며, R3 구간에서 SPY 대비 drawdown이 현저히 얕다. R4 종반에서 LSTM-adaptive가 LSTM-defensive를 다소 상회한다.}
\label{fig:benchmark_cumret}
\end{figure}

LSTM-defensive와 LSTM-adaptive는 전체 기간 Sharpe와 누적 수익률이 거의 동등하지만 (그림 \ref{fig:benchmark_cumret}), 레짐별 성과 구성이 상이하여 운영 목표에 따라 선택할 수 있다. 저변동 정체성을 모든 시장 국면에서 엄격히 유지하려는 보수적 운용에 적합한 LSTM-defensive와, 강세장 추가 상승 수익을 일정 부분 수용하려는 적응적 운용에 적합한 LSTM-adaptive 모두 표준 벤치마크 대비 의미 있는 우위를 제공하며, $\mathbf{p}^{\mathrm{eq}}$와 $\mathbf{p}^{\mathrm{rp}}$ 어느 변형으로도 일관된 성과를 보여 $\mathbf{p}$ 벡터 선택에 견고하다. $q^{\mathrm{fpm}}$이 강세장에서 뷰를 적응적으로 재조정하는 구체적 메커니즘 및 $\mathbf{p}$ 가중치의 흡수 효과는 부록 \ref{app:r4_mechanism}에서 분석한다.

\section{결론 및 한계}

본 연구는 \citet{pyo2018}의 KOSPI 200 BL 저변동 framework를 S\&P 500으로 확장하면서, 변동성 예측 모델과 BL 슬롯 구성을 재검토하고 2010--2025년 192개월을 4개 시장 국면(R1 회복, R2 확장, R3 위기, R4 정상화)으로 분해하여 평가하였다.

LSTM-HAR 앙상블은 ANN 대비 변동성 예측 정확도(RMSE, Spearman, hit rate)에서 전 구간 일관된 우위를 보였으며, 특히 위기 구간(R3)에서 격차가 전체 기간 대비 약 1.6--2.2배 확대되었다. 이 예측 우위가 포트폴리오 성과로 전이되기 위해서는 슬롯 설계, 특히 $\mathbf{p}$ 벡터의 대형주 비중 집중 완화가 결정적이다. $\sigma$ 모델만 LSTM으로 교체한 LSTM-anchor는 ANN-anchor와 Sharpe 동률(0.95)에 머무르나, $\mathbf{p}$ 가중치를 $\mathbf{p}^{\mathrm{eq}}$ 또는 $\mathbf{p}^{\mathrm{rp}}$로 변경하면 두 LSTM 옵션 모두 +0.09~+0.15의 Sharpe 개선이 일관되게 나타난다.

본 연구는 단일 권장 슬롯을 단정하지 않으며, 운영 목표에 따라 선택 가능한 두 옵션을 제시한다. 모든 시장 국면에서 저변동 뷰를 일관되게 유지하는 \textbf{LSTM-defensive} ($q^{\mathrm{lam}}$, Sharpe 1.04--1.07)와 강세장에서 추가 상승 수익을 수용하는 \textbf{LSTM-adaptive} ($q^{\mathrm{fpm}}$, Sharpe 1.05--1.10)는 모두 $\mathbf{p}^{\mathrm{eq}}$와 $\mathbf{p}^{\mathrm{rp}}$ 어느 변형으로도 SPY (0.93), 1/N (0.86), Risk Parity (0.89), \citet{pyo2018} ANN-anchor (0.95)를 상회한다.

하지만, 본 연구는 다음과 같은 한계를 가진다. 첫째, S\&P 500의 전체 기간에서 한 번이라도 포함된 종목은 총 833개이지만, 데이터를 수집한 Yahoo Finance는 상장 폐지되거나 인수 합병된 종목에 대해서는 주가 정보를 제공하지 않아 최종 617종목을 대상으로 분석하였다. 멤버십 필터링은 해당 시점에 실제로 S\&P 500에 편입되어 있던 종목만 분석에 포함되도록 통제하였으나, 데이터 가용성 한계로 인해 가장 극단적으로 실패한 종목들이 분석에서 빠졌을 가능성을 완전히 배제할 수 없다.

둘째, LSTM 앙상블이 ANN 대비 통계적·실증적 우위를 보였으나 hit-rate 기준 60--66\% 수준은 실무 활용에 충분한 수준이라 보기 어렵고, 본 연구의 우위가 LSTM 단일 모델이 아닌 HAR과의 앙상블 효과에도 일부 기인한다는 점도 짚어둘 필요가 있다. 본 연구가 사용한 LSTM은 입력 피처가 4채널($\mathrm{RV}_{d/w/m}$, $\log(\mathrm{VIX})$)에 단일 은닉층으로 구성되어 있어, 거시 변수·뉴스 감성 정보·옵션 시장 정보 등을 추가하고 다층·어텐션 구조로 고도화할 경우 추가 개선 여지가 충분히 존재한다. 향후 연구에서는 모델 고도화 후 ANN 대비 비교를 재수행할 필요가 있다.

셋째, R1 구간에서 LSTM-vs-ANN 우열은 $\mathbf{p}$ 벡터 선택에 따라 정반대로 갈리며($p_w \in {\mathrm{mcap, eq}}$에서 ANN 우위, $p_w = \mathrm{rp}$에서 LSTM 우위; §\ref{sec:p_matrix} 참조), 누적 손실의 약 76\%가 2011년 3·4·8월의 시장 충격 시점에 집중되고 2011-08 단일 월이 약 40\%를 견인한다(부록 \ref{app:r1}). R1 결과를 ``LSTM 기반 BL은 회복 구간에서 일관되게 약하다''와 같은 일반화로 확장하는 것은 표본 크기와 분포의 집중성을 고려할 때 신중해야 한다.

마지막으로, 본 연구의 모든 비교는 상관적·실증적 우위에 머무른다. R3 구간에서 LSTM 기반 구성이 강한 우위를 보였으나, LSTM이 위기 시 변동성 레짐 변화를 더 잘 포착하기 때문이라는 해석은 직관적으로 합리적이나 본 연구의 분석 범위로는 입증되지 않는다. 또한 본 연구는 \citet{pyo2018}가 사용한 Fama-French 회귀 기반 $q^{\mathrm{fpm}}$이 시장 상황에 따라 부호가 뒤집히면서 저변동 전략의 본질이 약화될 수 있다는 점을 부록 \ref{app:r4_mechanism}에서 사후적으로 관찰하였으나, 이것이 어떤 시장 메커니즘에서 비롯되는지에 대한 인과적 규명은 본 연구의 범위를 벗어난다. 변동성 예측 모델의 내부 메커니즘과 시장 미시구조 특성 간의 상호작용에 대한 후속 연구가 필요하다.

\bibliographystyle{apalike}
\bibliography{references}

% --------------- appendix 시작 ---------------
\appendix
\renewcommand{\thesection}{부록 \Alph{section}}
\renewcommand{\thefigure}{A.\arabic{figure}}
\renewcommand{\thetable}{A.\arabic{table}}
\setcounter{figure}{0}
\setcounter{table}{0}

\section{HMM 기반 시장 레짐 분류}\label{app:hmm}

요정 1호 여기부터 적을 것!

\section{슬롯 분석 결과 전체 표}\label{app:result_table}\label{app:full_grid}

\S\ref{slot_config}절의 평균 결과(표~\ref{tab:p_matrix_avg})를 45 슬롯 raw로 확장하여, 뷰 신뢰도 $\Omega \in \{\mathrm{err},\ \mathrm{he}\}$와 평가 지표 $\in \{\text{Sharpe},\ \text{Sortino},\ \text{MDD}\}$의 모든 조합 6개 표를 첨부한다. 표~\ref{tab:app_grid_sharpe_err}--\ref{tab:app_grid_mdd_err}은 $\Omega=$ err 기준, 표~\ref{tab:app_grid_sharpe_he}--\ref{tab:app_grid_mdd_he}은 $\Omega=$ he 기준이다.

평가 지표는 다음 세 가지를 사용한다.
\begin{itemize}
  \item \textbf{Sharpe 비율}: 평균 초과수익을 표준편차로 나눈 위험 조정 수익률. 클수록 우위.
  \item \textbf{Sortino 비율}: Sharpe의 하방 변동성 버전으로, 분모를 하방 표준편차로 대체한다(\cite{sortino1994}). 하락 변동성만 페널티로 반영하며 클수록 우위.
  \item \textbf{최대낙폭 (MDD)}: 누적 고점 대비 최대 손실 폭(\%). 음수가 손실이므로 클수록(0에 가까울수록) 우위.
\end{itemize}
따라서 모든 표의 $\Delta > 0$ 셀(녹색)은 LSTM이 ANN 대비 위험 조정 수익 또는 손실 폭 측면에서 우위인 경우이다.

% NOTE: 표가 페이지 폭을 넘으면 paper3.md 프리앰블에 \usepackage{pdflscape}를 추가하고
%       \begin{landscape} ... \end{landscape}로 표 전체를 감싸 가로 회전하시기 바랍니다.
\begin{table}[!htbp]
\centering
\caption{전체 45 슬롯 Sharpe 비율 결과 ($\Omega=$err 고정). $p_w \times \text{prior} \times q$ = $3 \times 3 \times 5$ = 45 슬롯 각각의 LSTM(L)·ANN(A) Sharpe 비율 및 차이 $\Delta = \mathrm{L} - \mathrm{A}$를 5개 구간(All, R1 회복, R2 확장, R3 위기, R4 AI 랠리)에 대해 보고한다.}
\label{tab:app_grid_sharpe_err}
\setlength{\tabcolsep}{2pt}
\renewcommand{\arraystretch}{1.05}
\scriptsize
\resizebox{\textwidth}{!}{%
\begin{tabular}{lll ccc ccc ccc ccc ccc}
\toprule
 & & & \multicolumn{3}{c}{All} & \multicolumn{3}{c}{R1} & \multicolumn{3}{c}{R2} & \multicolumn{3}{c}{R3} & \multicolumn{3}{c}{R4} \\
\cmidrule(lr){4-6} \cmidrule(lr){7-9} \cmidrule(lr){10-12} \cmidrule(lr){13-15} \cmidrule(lr){16-18}
$p_w$ & prior & $q$ & L & A & $\Delta$ & L & A & $\Delta$ & L & A & $\Delta$ & L & A & $\Delta$ & L & A & $\Delta$ \\
\midrule
mcap & mcap & lam & $0.913$ & $0.952$ & \cellcolor{negred!50}$-0.038$ & $0.821$ & $1.041$ & \cellcolor{negred!50}$-0.220$ & $0.953$ & $1.213$ & \cellcolor{negred!50}$-0.260$ & $0.991$ & $0.638$ & \cellcolor{posgreen!50}$+0.353$ & $0.864$ & $0.833$ & \cellcolor{posgreen!50}$+0.030$ \\
 &  & raw & $0.976$ & $0.988$ & \cellcolor{negred!50}$-0.012$ & $0.818$ & $1.044$ & \cellcolor{negred!50}$-0.226$ & $1.115$ & $1.332$ & \cellcolor{negred!50}$-0.217$ & $0.999$ & $0.643$ & \cellcolor{posgreen!50}$+0.357$ & $0.864$ & $0.833$ & \cellcolor{posgreen!50}$+0.030$ \\
 &  & inv & $0.995$ & $0.982$ & \cellcolor{posgreen!50}$+0.014$ & $1.061$ & $1.244$ & \cellcolor{negred!50}$-0.183$ & $1.100$ & $1.084$ & \cellcolor{posgreen!50}$+0.016$ & $0.894$ & $0.738$ & \cellcolor{posgreen!50}$+0.156$ & $0.915$ & $0.963$ & \cellcolor{negred!50}$-0.048$ \\
 &  & vsp & $0.949$ & $0.948$ & $+0.000$ & $0.894$ & $1.105$ & \cellcolor{negred!50}$-0.211$ & $1.030$ & $1.067$ & \cellcolor{negred!50}$-0.037$ & $0.943$ & $0.720$ & \cellcolor{posgreen!50}$+0.223$ & $0.891$ & $0.958$ & \cellcolor{negred!50}$-0.067$ \\
 &  & fpm & $0.946$ & $0.945$ & $+0.000$ & $0.738$ & $1.059$ & \cellcolor{negred!50}$-0.321$ & $1.287$ & $1.076$ & \cellcolor{posgreen!50}$+0.212$ & $0.702$ & $0.850$ & \cellcolor{negred!50}$-0.148$ & $0.940$ & $0.712$ & \cellcolor{posgreen!50}$+0.228$ \\
\cmidrule(lr){2-18}
 & eq & lam & $0.895$ & $0.947$ & \cellcolor{negred!50}$-0.052$ & $0.979$ & $1.168$ & \cellcolor{negred!50}$-0.189$ & $0.997$ & $1.245$ & \cellcolor{negred!50}$-0.248$ & $0.956$ & $0.711$ & \cellcolor{posgreen!50}$+0.244$ & $0.505$ & $0.518$ & \cellcolor{negred!50}$-0.014$ \\
 &  & raw & $0.961$ & $0.984$ & \cellcolor{negred!50}$-0.023$ & $0.988$ & $1.163$ & \cellcolor{negred!50}$-0.176$ & $1.175$ & $1.382$ & \cellcolor{negred!50}$-0.207$ & $0.965$ & $0.717$ & \cellcolor{posgreen!50}$+0.248$ & $0.505$ & $0.518$ & \cellcolor{negred!50}$-0.014$ \\
 &  & inv & $0.905$ & $0.940$ & \cellcolor{negred!50}$-0.035$ & $1.164$ & $1.381$ & \cellcolor{negred!50}$-0.217$ & $1.075$ & $1.072$ & \cellcolor{posgreen!50}$+0.003$ & $0.823$ & $0.756$ & \cellcolor{posgreen!50}$+0.066$ & $0.455$ & $0.551$ & \cellcolor{negred!50}$-0.096$ \\
 &  & vsp & $0.888$ & $0.927$ & \cellcolor{negred!50}$-0.038$ & $1.042$ & $1.226$ & \cellcolor{negred!50}$-0.184$ & $1.018$ & $1.077$ & \cellcolor{negred!50}$-0.059$ & $0.905$ & $0.780$ & \cellcolor{posgreen!50}$+0.125$ & $0.437$ & $0.538$ & \cellcolor{negred!50}$-0.101$ \\
 &  & fpm & $0.894$ & $0.889$ & \cellcolor{posgreen!50}$+0.006$ & $0.919$ & $1.182$ & \cellcolor{negred!50}$-0.262$ & $1.276$ & $1.007$ & \cellcolor{posgreen!50}$+0.269$ & $0.673$ & $0.862$ & \cellcolor{negred!50}$-0.189$ & $0.623$ & $0.420$ & \cellcolor{posgreen!50}$+0.202$ \\
\cmidrule(lr){2-18}
 & rp & lam & $0.905$ & $0.946$ & \cellcolor{negred!50}$-0.041$ & $1.106$ & $1.295$ & \cellcolor{negred!50}$-0.189$ & $0.977$ & $1.229$ & \cellcolor{negred!50}$-0.252$ & $0.928$ & $0.653$ & \cellcolor{posgreen!50}$+0.276$ & $0.502$ & $0.478$ & \cellcolor{posgreen!50}$+0.024$ \\
 &  & raw & $0.966$ & $0.983$ & \cellcolor{negred!50}$-0.017$ & $1.110$ & $1.296$ & \cellcolor{negred!50}$-0.185$ & $1.133$ & $1.353$ & \cellcolor{negred!50}$-0.220$ & $0.942$ & $0.659$ & \cellcolor{posgreen!50}$+0.283$ & $0.502$ & $0.478$ & \cellcolor{posgreen!50}$+0.024$ \\
 &  & inv & $0.916$ & $0.947$ & \cellcolor{negred!50}$-0.031$ & $1.274$ & $1.478$ & \cellcolor{negred!50}$-0.204$ & $1.090$ & $1.098$ & \cellcolor{negred!50}$-0.008$ & $0.793$ & $0.720$ & \cellcolor{posgreen!50}$+0.073$ & $0.431$ & $0.512$ & \cellcolor{negred!50}$-0.081$ \\
 &  & vsp & $0.909$ & $0.935$ & \cellcolor{negred!50}$-0.027$ & $1.167$ & $1.346$ & \cellcolor{negred!50}$-0.179$ & $1.039$ & $1.086$ & \cellcolor{negred!50}$-0.046$ & $0.881$ & $0.742$ & \cellcolor{posgreen!50}$+0.138$ & $0.409$ & $0.506$ & \cellcolor{negred!50}$-0.097$ \\
 &  & fpm & $0.918$ & $0.904$ & \cellcolor{posgreen!50}$+0.014$ & $1.032$ & $1.299$ & \cellcolor{negred!50}$-0.267$ & $1.298$ & $1.038$ & \cellcolor{posgreen!50}$+0.260$ & $0.638$ & $0.829$ & \cellcolor{negred!50}$-0.191$ & $0.651$ & $0.394$ & \cellcolor{posgreen!50}$+0.257$ \\
\midrule
eq & mcap & lam & $1.067$ & $0.973$ & \cellcolor{posgreen!50}$+0.094$ & $1.076$ & $1.120$ & \cellcolor{negred!50}$-0.045$ & $1.320$ & $1.335$ & \cellcolor{negred!50}$-0.016$ & $0.956$ & $0.583$ & \cellcolor{posgreen!50}$+0.373$ & $0.638$ & $0.578$ & \cellcolor{posgreen!50}$+0.059$ \\
 &  & raw & $1.060$ & $0.976$ & \cellcolor{posgreen!50}$+0.084$ & $1.085$ & $1.125$ & \cellcolor{negred!50}$-0.040$ & $1.311$ & $1.345$ & \cellcolor{negred!50}$-0.034$ & $0.953$ & $0.583$ & \cellcolor{posgreen!50}$+0.370$ & $0.638$ & $0.578$ & \cellcolor{posgreen!50}$+0.059$ \\
 &  & inv & $1.047$ & $0.988$ & \cellcolor{posgreen!50}$+0.059$ & $1.240$ & $1.270$ & \cellcolor{negred!50}$-0.030$ & $1.195$ & $1.205$ & \cellcolor{negred!50}$-0.010$ & $0.878$ & $0.681$ & \cellcolor{posgreen!50}$+0.197$ & $0.816$ & $0.720$ & \cellcolor{posgreen!50}$+0.097$ \\
 &  & vsp & $1.040$ & $0.963$ & \cellcolor{posgreen!50}$+0.076$ & $1.131$ & $1.158$ & \cellcolor{negred!50}$-0.028$ & $1.205$ & $1.245$ & \cellcolor{negred!50}$-0.040$ & $0.914$ & $0.624$ & \cellcolor{posgreen!50}$+0.291$ & $0.772$ & $0.688$ & \cellcolor{posgreen!50}$+0.084$ \\
 &  & fpm & $1.099$ & $1.049$ & \cellcolor{posgreen!50}$+0.050$ & $0.872$ & $1.024$ & \cellcolor{negred!50}$-0.152$ & $1.421$ & $1.408$ & \cellcolor{posgreen!50}$+0.013$ & $0.888$ & $0.705$ & \cellcolor{posgreen!50}$+0.183$ & $1.066$ & $0.868$ & \cellcolor{posgreen!50}$+0.198$ \\
\cmidrule(lr){2-18}
 & eq & lam & $1.022$ & $0.935$ & \cellcolor{posgreen!50}$+0.086$ & $1.163$ & $1.203$ & \cellcolor{negred!50}$-0.040$ & $1.339$ & $1.350$ & \cellcolor{negred!50}$-0.011$ & $0.955$ & $0.621$ & \cellcolor{posgreen!50}$+0.334$ & $0.277$ & $0.227$ & \cellcolor{posgreen!50}$+0.050$ \\
 &  & raw & $1.016$ & $0.932$ & \cellcolor{posgreen!50}$+0.084$ & $1.164$ & $1.210$ & \cellcolor{negred!50}$-0.046$ & $1.339$ & $1.339$ & $-0.000$ & $0.953$ & $0.622$ & \cellcolor{posgreen!50}$+0.331$ & $0.277$ & $0.227$ & \cellcolor{posgreen!50}$+0.050$ \\
 &  & inv & $1.008$ & $0.947$ & \cellcolor{posgreen!50}$+0.061$ & $1.329$ & $1.337$ & \cellcolor{negred!50}$-0.007$ & $1.196$ & $1.194$ & \cellcolor{posgreen!50}$+0.002$ & $0.941$ & $0.748$ & \cellcolor{posgreen!50}$+0.193$ & $0.397$ & $0.349$ & \cellcolor{posgreen!50}$+0.048$ \\
 &  & vsp & $1.012$ & $0.956$ & \cellcolor{posgreen!50}$+0.057$ & $1.221$ & $1.247$ & \cellcolor{negred!50}$-0.026$ & $1.239$ & $1.285$ & \cellcolor{negred!50}$-0.046$ & $0.965$ & $0.723$ & \cellcolor{posgreen!50}$+0.243$ & $0.354$ & $0.317$ & \cellcolor{posgreen!50}$+0.038$ \\
 &  & fpm & $1.068$ & $1.031$ & \cellcolor{posgreen!50}$+0.037$ & $1.020$ & $1.123$ & \cellcolor{negred!50}$-0.103$ & $1.436$ & $1.461$ & \cellcolor{negred!50}$-0.025$ & $0.956$ & $0.802$ & \cellcolor{posgreen!50}$+0.154$ & $0.522$ & $0.405$ & \cellcolor{posgreen!50}$+0.117$ \\
\cmidrule(lr){2-18}
 & rp & lam & $1.013$ & $0.923$ & \cellcolor{posgreen!50}$+0.090$ & $1.268$ & $1.307$ & \cellcolor{negred!50}$-0.039$ & $1.317$ & $1.312$ & \cellcolor{posgreen!50}$+0.004$ & $0.891$ & $0.580$ & \cellcolor{posgreen!50}$+0.311$ & $0.230$ & $0.164$ & \cellcolor{posgreen!50}$+0.066$ \\
 &  & raw & $1.008$ & $0.922$ & \cellcolor{posgreen!50}$+0.085$ & $1.272$ & $1.313$ & \cellcolor{negred!50}$-0.041$ & $1.314$ & $1.307$ & \cellcolor{posgreen!50}$+0.007$ & $0.888$ & $0.581$ & \cellcolor{posgreen!50}$+0.307$ & $0.230$ & $0.164$ & \cellcolor{posgreen!50}$+0.066$ \\
 &  & inv & $1.012$ & $0.948$ & \cellcolor{posgreen!50}$+0.064$ & $1.462$ & $1.428$ & \cellcolor{posgreen!50}$+0.034$ & $1.218$ & $1.199$ & \cellcolor{posgreen!50}$+0.019$ & $0.884$ & $0.723$ & \cellcolor{posgreen!50}$+0.160$ & $0.350$ & $0.296$ & \cellcolor{posgreen!50}$+0.053$ \\
 &  & vsp & $1.014$ & $0.957$ & \cellcolor{posgreen!50}$+0.056$ & $1.339$ & $1.362$ & \cellcolor{negred!50}$-0.023$ & $1.238$ & $1.274$ & \cellcolor{negred!50}$-0.036$ & $0.908$ & $0.688$ & \cellcolor{posgreen!50}$+0.220$ & $0.326$ & $0.274$ & \cellcolor{posgreen!50}$+0.052$ \\
 &  & fpm & $1.081$ & $1.043$ & \cellcolor{posgreen!50}$+0.038$ & $1.116$ & $1.225$ & \cellcolor{negred!50}$-0.109$ & $1.453$ & $1.454$ & $-0.000$ & $0.908$ & $0.776$ & \cellcolor{posgreen!50}$+0.132$ & $0.484$ & $0.374$ & \cellcolor{posgreen!50}$+0.110$ \\
\midrule
rp & mcap & lam & $1.042$ & $0.940$ & \cellcolor{posgreen!50}$+0.101$ & $0.977$ & $0.956$ & \cellcolor{posgreen!50}$+0.021$ & $1.275$ & $1.285$ & \cellcolor{negred!50}$-0.010$ & $0.953$ & $0.620$ & \cellcolor{posgreen!50}$+0.333$ & $0.709$ & $0.647$ & \cellcolor{posgreen!50}$+0.061$ \\
 &  & raw & $1.036$ & $0.956$ & \cellcolor{posgreen!50}$+0.080$ & $0.986$ & $0.955$ & \cellcolor{posgreen!50}$+0.031$ & $1.270$ & $1.330$ & \cellcolor{negred!50}$-0.060$ & $0.951$ & $0.621$ & \cellcolor{posgreen!50}$+0.329$ & $0.709$ & $0.647$ & \cellcolor{posgreen!50}$+0.061$ \\
 &  & inv & $1.020$ & $0.922$ & \cellcolor{posgreen!50}$+0.098$ & $1.137$ & $1.001$ & \cellcolor{posgreen!50}$+0.136$ & $1.151$ & $1.129$ & \cellcolor{posgreen!50}$+0.023$ & $0.852$ & $0.679$ & \cellcolor{posgreen!50}$+0.172$ & $0.926$ & $0.808$ & \cellcolor{posgreen!50}$+0.117$ \\
 &  & vsp & $1.009$ & $0.906$ & \cellcolor{posgreen!50}$+0.103$ & $1.055$ & $0.977$ & \cellcolor{posgreen!50}$+0.078$ & $1.150$ & $1.151$ & \cellcolor{negred!50}$-0.001$ & $0.884$ & $0.631$ & \cellcolor{posgreen!50}$+0.254$ & $0.871$ & $0.769$ & \cellcolor{posgreen!50}$+0.102$ \\
 &  & fpm & $1.054$ & $0.970$ & \cellcolor{posgreen!50}$+0.084$ & $0.806$ & $0.809$ & \cellcolor{negred!50}$-0.003$ & $1.344$ & $1.311$ & \cellcolor{posgreen!50}$+0.033$ & $0.870$ & $0.714$ & \cellcolor{posgreen!50}$+0.155$ & $1.102$ & $0.900$ & \cellcolor{posgreen!50}$+0.202$ \\
\cmidrule(lr){2-18}
 & eq & lam & $1.003$ & $0.908$ & \cellcolor{posgreen!50}$+0.095$ & $1.098$ & $1.038$ & \cellcolor{posgreen!50}$+0.061$ & $1.289$ & $1.302$ & \cellcolor{negred!50}$-0.013$ & $0.941$ & $0.654$ & \cellcolor{posgreen!50}$+0.287$ & $0.363$ & $0.297$ & \cellcolor{posgreen!50}$+0.066$ \\
 &  & raw & $1.005$ & $0.909$ & \cellcolor{posgreen!50}$+0.096$ & $1.114$ & $1.042$ & \cellcolor{posgreen!50}$+0.072$ & $1.306$ & $1.307$ & \cellcolor{negred!50}$-0.001$ & $0.938$ & $0.654$ & \cellcolor{posgreen!50}$+0.284$ & $0.363$ & $0.297$ & \cellcolor{posgreen!50}$+0.066$ \\
 &  & inv & $0.962$ & $0.882$ & \cellcolor{posgreen!50}$+0.080$ & $1.223$ & $1.079$ & \cellcolor{posgreen!50}$+0.144$ & $1.133$ & $1.104$ & \cellcolor{posgreen!50}$+0.029$ & $0.890$ & $0.740$ & \cellcolor{posgreen!50}$+0.149$ & $0.447$ & $0.416$ & \cellcolor{posgreen!50}$+0.030$ \\
 &  & vsp & $0.971$ & $0.887$ & \cellcolor{posgreen!50}$+0.084$ & $1.140$ & $1.061$ & \cellcolor{posgreen!50}$+0.079$ & $1.181$ & $1.155$ & \cellcolor{posgreen!50}$+0.026$ & $0.911$ & $0.717$ & \cellcolor{posgreen!50}$+0.194$ & $0.415$ & $0.387$ & \cellcolor{posgreen!50}$+0.028$ \\
 &  & fpm & $1.020$ & $0.953$ & \cellcolor{posgreen!50}$+0.067$ & $0.997$ & $0.937$ & \cellcolor{posgreen!50}$+0.059$ & $1.339$ & $1.331$ & \cellcolor{posgreen!50}$+0.009$ & $0.910$ & $0.804$ & \cellcolor{posgreen!50}$+0.106$ & $0.535$ & $0.434$ & \cellcolor{posgreen!50}$+0.101$ \\
\cmidrule(lr){2-18}
 & rp & lam & $1.006$ & $0.901$ & \cellcolor{posgreen!50}$+0.105$ & $1.221$ & $1.141$ & \cellcolor{posgreen!50}$+0.080$ & $1.283$ & $1.265$ & \cellcolor{posgreen!50}$+0.018$ & $0.893$ & $0.615$ & \cellcolor{posgreen!50}$+0.278$ & $0.309$ & $0.243$ & \cellcolor{posgreen!50}$+0.066$ \\
 &  & raw & $1.003$ & $0.908$ & \cellcolor{posgreen!50}$+0.095$ & $1.237$ & $1.146$ & \cellcolor{posgreen!50}$+0.092$ & $1.283$ & $1.283$ & $-0.000$ & $0.891$ & $0.618$ & \cellcolor{posgreen!50}$+0.273$ & $0.309$ & $0.243$ & \cellcolor{posgreen!50}$+0.066$ \\
 &  & inv & $0.968$ & $0.896$ & \cellcolor{posgreen!50}$+0.072$ & $1.352$ & $1.175$ & \cellcolor{posgreen!50}$+0.177$ & $1.162$ & $1.124$ & \cellcolor{posgreen!50}$+0.037$ & $0.828$ & $0.727$ & \cellcolor{posgreen!50}$+0.101$ & $0.406$ & $0.365$ & \cellcolor{posgreen!50}$+0.042$ \\
 &  & vsp & $0.976$ & $0.893$ & \cellcolor{posgreen!50}$+0.083$ & $1.269$ & $1.150$ & \cellcolor{posgreen!50}$+0.119$ & $1.189$ & $1.157$ & \cellcolor{posgreen!50}$+0.032$ & $0.858$ & $0.692$ & \cellcolor{posgreen!50}$+0.166$ & $0.371$ & $0.340$ & \cellcolor{posgreen!50}$+0.031$ \\
 &  & fpm & $1.041$ & $0.971$ & \cellcolor{posgreen!50}$+0.070$ & $1.103$ & $1.038$ & \cellcolor{posgreen!50}$+0.065$ & $1.369$ & $1.338$ & \cellcolor{posgreen!50}$+0.031$ & $0.872$ & $0.781$ & \cellcolor{posgreen!50}$+0.091$ & $0.491$ & $0.390$ & \cellcolor{posgreen!50}$+0.100$ \\
\bottomrule
\end{tabular}}
\end{table}

% NOTE: 표가 페이지 폭을 넘으면 paper3.md 프리앰블에 \usepackage{pdflscape}를 추가하고
%       \begin{landscape} ... \end{landscape}로 표 전체를 감싸 가로 회전하시기 바랍니다.
\begin{table}[!htbp]
\centering
\caption{전체 45 슬롯 Sortino 비율 결과 ($\Omega=$err 고정). $p_w \times \text{prior} \times q$ = $3 \times 3 \times 5$ = 45 슬롯 각각의 LSTM(L)·ANN(A) Sortino 비율 및 차이 $\Delta = \mathrm{L} - \mathrm{A}$를 5개 구간(All, R1 회복, R2 확장, R3 위기, R4 AI 랠리)에 대해 보고한다.}
\label{tab:app_grid_sortino_err}
\setlength{\tabcolsep}{2pt}
\renewcommand{\arraystretch}{1.05}
\scriptsize
\resizebox{\textwidth}{!}{%
\begin{tabular}{lll ccc ccc ccc ccc ccc}
\toprule
 & & & \multicolumn{3}{c}{All} & \multicolumn{3}{c}{R1} & \multicolumn{3}{c}{R2} & \multicolumn{3}{c}{R3} & \multicolumn{3}{c}{R4} \\
\cmidrule(lr){4-6} \cmidrule(lr){7-9} \cmidrule(lr){10-12} \cmidrule(lr){13-15} \cmidrule(lr){16-18}
$p_w$ & prior & $q$ & L & A & $\Delta$ & L & A & $\Delta$ & L & A & $\Delta$ & L & A & $\Delta$ & L & A & $\Delta$ \\
\midrule
mcap & mcap & lam & $1.464$ & $1.473$ & \cellcolor{negred!50}$-0.009$ & $1.295$ & $1.537$ & \cellcolor{negred!50}$-0.243$ & $1.311$ & $1.663$ & \cellcolor{negred!50}$-0.352$ & $2.261$ & $1.301$ & \cellcolor{posgreen!50}$+0.960$ & $1.198$ & $1.085$ & \cellcolor{posgreen!50}$+0.112$ \\
 &  & raw & $1.552$ & $1.530$ & \cellcolor{posgreen!50}$+0.021$ & $1.293$ & $1.544$ & \cellcolor{negred!50}$-0.251$ & $1.488$ & $1.923$ & \cellcolor{negred!50}$-0.435$ & $2.291$ & $1.315$ & \cellcolor{posgreen!50}$+0.976$ & $1.198$ & $1.085$ & \cellcolor{posgreen!50}$+0.112$ \\
 &  & inv & $1.395$ & $1.368$ & \cellcolor{posgreen!50}$+0.028$ & $1.447$ & $1.629$ & \cellcolor{negred!50}$-0.182$ & $1.200$ & $1.258$ & \cellcolor{negred!50}$-0.058$ & $1.980$ & $1.498$ & \cellcolor{posgreen!50}$+0.482$ & $1.425$ & $1.314$ & \cellcolor{posgreen!50}$+0.110$ \\
 &  & vsp & $1.403$ & $1.370$ & \cellcolor{posgreen!50}$+0.033$ & $1.345$ & $1.576$ & \cellcolor{negred!50}$-0.232$ & $1.209$ & $1.266$ & \cellcolor{negred!50}$-0.056$ & $2.139$ & $1.453$ & \cellcolor{posgreen!50}$+0.686$ & $1.314$ & $1.343$ & \cellcolor{negred!50}$-0.029$ \\
 &  & fpm & $1.320$ & $1.336$ & \cellcolor{negred!50}$-0.016$ & $1.190$ & $1.555$ & \cellcolor{negred!50}$-0.364$ & $1.481$ & $1.360$ & \cellcolor{posgreen!50}$+0.121$ & $1.174$ & $1.597$ & \cellcolor{negred!50}$-0.424$ & $1.513$ & $0.878$ & \cellcolor{posgreen!50}$+0.635$ \\
\cmidrule(lr){2-18}
 & eq & lam & $1.483$ & $1.526$ & \cellcolor{negred!50}$-0.043$ & $1.509$ & $1.731$ & \cellcolor{negred!50}$-0.222$ & $1.316$ & $1.699$ & \cellcolor{negred!50}$-0.382$ & $2.181$ & $1.532$ & \cellcolor{posgreen!50}$+0.649$ & $0.898$ & $0.888$ & \cellcolor{posgreen!50}$+0.011$ \\
 &  & raw & $1.594$ & $1.598$ & \cellcolor{negred!50}$-0.004$ & $1.528$ & $1.729$ & \cellcolor{negred!50}$-0.201$ & $1.547$ & $1.977$ & \cellcolor{negred!50}$-0.430$ & $2.221$ & $1.548$ & \cellcolor{posgreen!50}$+0.674$ & $0.898$ & $0.888$ & \cellcolor{posgreen!50}$+0.011$ \\
 &  & inv & $1.310$ & $1.356$ & \cellcolor{negred!50}$-0.046$ & $1.575$ & $1.796$ & \cellcolor{negred!50}$-0.221$ & $1.185$ & $1.238$ & \cellcolor{negred!50}$-0.053$ & $1.604$ & $1.475$ & \cellcolor{posgreen!50}$+0.129$ & $0.803$ & $0.929$ & \cellcolor{negred!50}$-0.125$ \\
 &  & vsp & $1.382$ & $1.410$ & \cellcolor{negred!50}$-0.028$ & $1.541$ & $1.752$ & \cellcolor{negred!50}$-0.210$ & $1.191$ & $1.290$ & \cellcolor{negred!50}$-0.099$ & $2.052$ & $1.637$ & \cellcolor{posgreen!50}$+0.415$ & $0.784$ & $0.926$ & \cellcolor{negred!50}$-0.141$ \\
 &  & fpm & $1.200$ & $1.271$ & \cellcolor{negred!50}$-0.071$ & $1.455$ & $1.755$ & \cellcolor{negred!50}$-0.300$ & $1.394$ & $1.300$ & \cellcolor{posgreen!50}$+0.094$ & $1.016$ & $1.548$ & \cellcolor{negred!50}$-0.532$ & $1.177$ & $0.522$ & \cellcolor{posgreen!50}$+0.655$ \\
\cmidrule(lr){2-18}
 & rp & lam & $1.499$ & $1.513$ & \cellcolor{negred!50}$-0.015$ & $1.679$ & $1.974$ & \cellcolor{negred!50}$-0.296$ & $1.325$ & $1.670$ & \cellcolor{negred!50}$-0.345$ & $2.108$ & $1.390$ & \cellcolor{posgreen!50}$+0.719$ & $0.877$ & $0.828$ & \cellcolor{posgreen!50}$+0.049$ \\
 &  & raw & $1.605$ & $1.585$ & \cellcolor{posgreen!50}$+0.020$ & $1.693$ & $1.972$ & \cellcolor{negred!50}$-0.279$ & $1.542$ & $1.907$ & \cellcolor{negred!50}$-0.365$ & $2.159$ & $1.406$ & \cellcolor{posgreen!50}$+0.753$ & $0.877$ & $0.828$ & \cellcolor{posgreen!50}$+0.049$ \\
 &  & inv & $1.318$ & $1.366$ & \cellcolor{negred!50}$-0.048$ & $1.677$ & $1.956$ & \cellcolor{negred!50}$-0.278$ & $1.245$ & $1.292$ & \cellcolor{negred!50}$-0.047$ & $1.533$ & $1.427$ & \cellcolor{posgreen!50}$+0.105$ & $0.740$ & $0.824$ & \cellcolor{negred!50}$-0.084$ \\
 &  & vsp & $1.418$ & $1.430$ & \cellcolor{negred!50}$-0.012$ & $1.688$ & $1.957$ & \cellcolor{negred!50}$-0.269$ & $1.265$ & $1.338$ & \cellcolor{negred!50}$-0.074$ & $2.007$ & $1.584$ & \cellcolor{posgreen!50}$+0.423$ & $0.737$ & $0.828$ & \cellcolor{negred!50}$-0.091$ \\
 &  & fpm & $1.233$ & $1.288$ & \cellcolor{negred!50}$-0.054$ & $1.595$ & $2.004$ & \cellcolor{negred!50}$-0.409$ & $1.455$ & $1.358$ & \cellcolor{posgreen!50}$+0.097$ & $0.970$ & $1.541$ & \cellcolor{negred!50}$-0.571$ & $1.251$ & $0.458$ & \cellcolor{posgreen!50}$+0.793$ \\
\midrule
eq & mcap & lam & $1.662$ & $1.523$ & \cellcolor{posgreen!50}$+0.139$ & $1.784$ & $1.765$ & \cellcolor{posgreen!50}$+0.019$ & $1.821$ & $1.974$ & \cellcolor{negred!50}$-0.153$ & $1.800$ & $1.127$ & \cellcolor{posgreen!50}$+0.673$ & $0.978$ & $0.872$ & \cellcolor{posgreen!50}$+0.106$ \\
 &  & raw & $1.696$ & $1.555$ & \cellcolor{posgreen!50}$+0.141$ & $1.800$ & $1.775$ & \cellcolor{posgreen!50}$+0.025$ & $1.947$ & $2.090$ & \cellcolor{negred!50}$-0.143$ & $1.795$ & $1.127$ & \cellcolor{posgreen!50}$+0.668$ & $0.978$ & $0.872$ & \cellcolor{posgreen!50}$+0.106$ \\
 &  & inv & $1.519$ & $1.367$ & \cellcolor{posgreen!50}$+0.152$ & $1.851$ & $1.774$ & \cellcolor{posgreen!50}$+0.077$ & $1.489$ & $1.431$ & \cellcolor{posgreen!50}$+0.057$ & $1.674$ & $1.183$ & \cellcolor{posgreen!50}$+0.491$ & $1.150$ & $1.034$ & \cellcolor{posgreen!50}$+0.116$ \\
 &  & vsp & $1.555$ & $1.378$ & \cellcolor{posgreen!50}$+0.177$ & $1.866$ & $1.716$ & \cellcolor{posgreen!50}$+0.150$ & $1.516$ & $1.534$ & \cellcolor{negred!50}$-0.019$ & $1.775$ & $1.110$ & \cellcolor{posgreen!50}$+0.665$ & $1.141$ & $1.022$ & \cellcolor{posgreen!50}$+0.119$ \\
 &  & fpm & $1.623$ & $1.548$ & \cellcolor{posgreen!50}$+0.075$ & $1.468$ & $1.653$ & \cellcolor{negred!50}$-0.185$ & $1.701$ & $1.721$ & \cellcolor{negred!50}$-0.020$ & $1.767$ & $1.307$ & \cellcolor{posgreen!50}$+0.460$ & $1.724$ & $1.642$ & \cellcolor{posgreen!50}$+0.082$ \\
\cmidrule(lr){2-18}
 & eq & lam & $1.715$ & $1.556$ & \cellcolor{posgreen!50}$+0.159$ & $2.088$ & $1.923$ & \cellcolor{posgreen!50}$+0.165$ & $1.844$ & $2.025$ & \cellcolor{negred!50}$-0.180$ & $1.961$ & $1.240$ & \cellcolor{posgreen!50}$+0.722$ & $0.529$ & $0.411$ & \cellcolor{posgreen!50}$+0.117$ \\
 &  & raw & $1.724$ & $1.567$ & \cellcolor{posgreen!50}$+0.157$ & $2.109$ & $1.939$ & \cellcolor{posgreen!50}$+0.170$ & $1.961$ & $2.082$ & \cellcolor{negred!50}$-0.121$ & $1.958$ & $1.241$ & \cellcolor{posgreen!50}$+0.717$ & $0.529$ & $0.411$ & \cellcolor{posgreen!50}$+0.117$ \\
 &  & inv & $1.539$ & $1.389$ & \cellcolor{posgreen!50}$+0.151$ & $2.035$ & $1.910$ & \cellcolor{posgreen!50}$+0.125$ & $1.508$ & $1.441$ & \cellcolor{posgreen!50}$+0.066$ & $1.737$ & $1.364$ & \cellcolor{posgreen!50}$+0.373$ & $0.730$ & $0.664$ & \cellcolor{posgreen!50}$+0.066$ \\
 &  & vsp & $1.623$ & $1.468$ & \cellcolor{posgreen!50}$+0.154$ & $2.041$ & $1.923$ & \cellcolor{posgreen!50}$+0.118$ & $1.575$ & $1.660$ & \cellcolor{negred!50}$-0.086$ & $2.041$ & $1.381$ & \cellcolor{posgreen!50}$+0.661$ & $0.665$ & $0.608$ & \cellcolor{posgreen!50}$+0.056$ \\
 &  & fpm & $1.680$ & $1.615$ & \cellcolor{posgreen!50}$+0.066$ & $1.749$ & $1.872$ & \cellcolor{negred!50}$-0.123$ & $1.738$ & $1.831$ & \cellcolor{negred!50}$-0.093$ & $2.009$ & $1.634$ & \cellcolor{posgreen!50}$+0.375$ & $0.952$ & $0.782$ & \cellcolor{posgreen!50}$+0.170$ \\
\cmidrule(lr){2-18}
 & rp & lam & $1.694$ & $1.572$ & \cellcolor{posgreen!50}$+0.122$ & $2.482$ & $2.213$ & \cellcolor{posgreen!50}$+0.268$ & $1.866$ & $2.029$ & \cellcolor{negred!50}$-0.164$ & $1.783$ & $1.164$ & \cellcolor{posgreen!50}$+0.618$ & $0.406$ & $0.292$ & \cellcolor{posgreen!50}$+0.114$ \\
 &  & raw & $1.710$ & $1.589$ & \cellcolor{posgreen!50}$+0.122$ & $2.500$ & $2.223$ & \cellcolor{posgreen!50}$+0.277$ & $1.973$ & $2.086$ & \cellcolor{negred!50}$-0.113$ & $1.779$ & $1.166$ & \cellcolor{posgreen!50}$+0.613$ & $0.406$ & $0.292$ & \cellcolor{posgreen!50}$+0.114$ \\
 &  & inv & $1.537$ & $1.393$ & \cellcolor{posgreen!50}$+0.145$ & $2.365$ & $2.067$ & \cellcolor{posgreen!50}$+0.299$ & $1.538$ & $1.457$ & \cellcolor{posgreen!50}$+0.081$ & $1.644$ & $1.349$ & \cellcolor{posgreen!50}$+0.295$ & $0.619$ & $0.536$ & \cellcolor{posgreen!50}$+0.083$ \\
 &  & vsp & $1.606$ & $1.485$ & \cellcolor{posgreen!50}$+0.121$ & $2.551$ & $2.151$ & \cellcolor{posgreen!50}$+0.399$ & $1.605$ & $1.668$ & \cellcolor{negred!50}$-0.063$ & $1.882$ & $1.348$ & \cellcolor{posgreen!50}$+0.535$ & $0.590$ & $0.476$ & \cellcolor{posgreen!50}$+0.114$ \\
 &  & fpm & $1.697$ & $1.671$ & \cellcolor{posgreen!50}$+0.026$ & $1.999$ & $2.228$ & \cellcolor{negred!50}$-0.229$ & $1.828$ & $1.856$ & \cellcolor{negred!50}$-0.028$ & $1.867$ & $1.615$ & \cellcolor{posgreen!50}$+0.252$ & $0.868$ & $0.685$ & \cellcolor{posgreen!50}$+0.183$ \\
\midrule
rp & mcap & lam & $1.644$ & $1.432$ & \cellcolor{posgreen!50}$+0.211$ & $1.672$ & $1.319$ & \cellcolor{posgreen!50}$+0.353$ & $1.746$ & $1.891$ & \cellcolor{negred!50}$-0.145$ & $1.794$ & $1.216$ & \cellcolor{posgreen!50}$+0.577$ & $1.099$ & $0.942$ & \cellcolor{posgreen!50}$+0.157$ \\
 &  & raw & $1.679$ & $1.481$ & \cellcolor{posgreen!50}$+0.198$ & $1.687$ & $1.321$ & \cellcolor{posgreen!50}$+0.366$ & $1.873$ & $2.070$ & \cellcolor{negred!50}$-0.197$ & $1.790$ & $1.218$ & \cellcolor{posgreen!50}$+0.573$ & $1.099$ & $0.942$ & \cellcolor{posgreen!50}$+0.157$ \\
 &  & inv & $1.476$ & $1.239$ & \cellcolor{posgreen!50}$+0.237$ & $1.906$ & $1.306$ & \cellcolor{posgreen!50}$+0.600$ & $1.377$ & $1.314$ & \cellcolor{posgreen!50}$+0.063$ & $1.629$ & $1.177$ & \cellcolor{posgreen!50}$+0.452$ & $1.312$ & $1.233$ & \cellcolor{posgreen!50}$+0.079$ \\
 &  & vsp & $1.540$ & $1.268$ & \cellcolor{posgreen!50}$+0.273$ & $1.818$ & $1.371$ & \cellcolor{posgreen!50}$+0.447$ & $1.441$ & $1.436$ & \cellcolor{posgreen!50}$+0.005$ & $1.771$ & $1.134$ & \cellcolor{posgreen!50}$+0.636$ & $1.320$ & $1.093$ & \cellcolor{posgreen!50}$+0.227$ \\
 &  & fpm & $1.586$ & $1.386$ & \cellcolor{posgreen!50}$+0.199$ & $1.438$ & $1.120$ & \cellcolor{posgreen!50}$+0.318$ & $1.643$ & $1.608$ & \cellcolor{posgreen!50}$+0.035$ & $1.749$ & $1.398$ & \cellcolor{posgreen!50}$+0.351$ & $1.779$ & $1.545$ & \cellcolor{posgreen!50}$+0.235$ \\
\cmidrule(lr){2-18}
 & eq & lam & $1.672$ & $1.465$ & \cellcolor{posgreen!50}$+0.207$ & $1.876$ & $1.538$ & \cellcolor{posgreen!50}$+0.338$ & $1.764$ & $1.935$ & \cellcolor{negred!50}$-0.171$ & $1.883$ & $1.324$ & \cellcolor{posgreen!50}$+0.560$ & $0.715$ & $0.536$ & \cellcolor{posgreen!50}$+0.179$ \\
 &  & raw & $1.692$ & $1.493$ & \cellcolor{posgreen!50}$+0.199$ & $1.895$ & $1.547$ & \cellcolor{posgreen!50}$+0.348$ & $1.894$ & $2.019$ & \cellcolor{negred!50}$-0.124$ & $1.881$ & $1.324$ & \cellcolor{posgreen!50}$+0.557$ & $0.715$ & $0.536$ & \cellcolor{posgreen!50}$+0.179$ \\
 &  & inv & $1.455$ & $1.272$ & \cellcolor{posgreen!50}$+0.183$ & $2.041$ & $1.494$ & \cellcolor{posgreen!50}$+0.547$ & $1.381$ & $1.330$ & \cellcolor{posgreen!50}$+0.051$ & $1.628$ & $1.359$ & \cellcolor{posgreen!50}$+0.269$ & $0.842$ & $0.805$ & \cellcolor{posgreen!50}$+0.036$ \\
 &  & vsp & $1.569$ & $1.348$ & \cellcolor{posgreen!50}$+0.220$ & $2.054$ & $1.546$ & \cellcolor{posgreen!50}$+0.509$ & $1.483$ & $1.472$ & \cellcolor{posgreen!50}$+0.012$ & $1.889$ & $1.414$ & \cellcolor{posgreen!50}$+0.475$ & $0.801$ & $0.750$ & \cellcolor{posgreen!50}$+0.052$ \\
 &  & fpm & $1.630$ & $1.445$ & \cellcolor{posgreen!50}$+0.185$ & $1.808$ & $1.317$ & \cellcolor{posgreen!50}$+0.491$ & $1.679$ & $1.679$ & $-0.000$ & $1.954$ & $1.635$ & \cellcolor{posgreen!50}$+0.319$ & $0.968$ & $0.800$ & \cellcolor{posgreen!50}$+0.167$ \\
\cmidrule(lr){2-18}
 & rp & lam & $1.696$ & $1.497$ & \cellcolor{posgreen!50}$+0.199$ & $2.457$ & $1.784$ & \cellcolor{posgreen!50}$+0.673$ & $1.806$ & $1.915$ & \cellcolor{negred!50}$-0.109$ & $1.747$ & $1.265$ & \cellcolor{posgreen!50}$+0.482$ & $0.583$ & $0.430$ & \cellcolor{posgreen!50}$+0.153$ \\
 &  & raw & $1.705$ & $1.529$ & \cellcolor{posgreen!50}$+0.176$ & $2.464$ & $1.799$ & \cellcolor{posgreen!50}$+0.665$ & $1.903$ & $2.019$ & \cellcolor{negred!50}$-0.115$ & $1.744$ & $1.269$ & \cellcolor{posgreen!50}$+0.475$ & $0.583$ & $0.430$ & \cellcolor{posgreen!50}$+0.153$ \\
 &  & inv & $1.458$ & $1.297$ & \cellcolor{posgreen!50}$+0.162$ & $2.476$ & $1.644$ & \cellcolor{posgreen!50}$+0.833$ & $1.414$ & $1.361$ & \cellcolor{posgreen!50}$+0.054$ & $1.534$ & $1.369$ & \cellcolor{posgreen!50}$+0.165$ & $0.739$ & $0.672$ & \cellcolor{posgreen!50}$+0.067$ \\
 &  & vsp & $1.561$ & $1.371$ & \cellcolor{posgreen!50}$+0.191$ & $2.592$ & $1.745$ & \cellcolor{posgreen!50}$+0.848$ & $1.514$ & $1.513$ & \cellcolor{posgreen!50}$+0.001$ & $1.756$ & $1.386$ & \cellcolor{posgreen!50}$+0.370$ & $0.691$ & $0.601$ & \cellcolor{posgreen!50}$+0.090$ \\
 &  & fpm & $1.676$ & $1.499$ & \cellcolor{posgreen!50}$+0.177$ & $1.983$ & $1.441$ & \cellcolor{posgreen!50}$+0.542$ & $1.711$ & $1.725$ & \cellcolor{negred!50}$-0.015$ & $1.845$ & $1.682$ & \cellcolor{posgreen!50}$+0.162$ & $0.882$ & $0.691$ & \cellcolor{posgreen!50}$+0.191$ \\
\bottomrule
\end{tabular}}
\end{table}

% NOTE: 표가 페이지 폭을 넘으면 paper3.md 프리앰블에 \usepackage{pdflscape}를 추가하고
%       \begin{landscape} ... \end{landscape}로 표 전체를 감싸 가로 회전하시기 바랍니다.
\begin{table}[!htbp]
\centering
\caption{전체 45 슬롯 최대낙폭 (MDD) 결과 ($\Omega=$err 고정). $p_w \times \text{prior} \times q$ = $3 \times 3 \times 5$ = 45 슬롯 각각의 LSTM(L)·ANN(A) 최대낙폭 (MDD) 및 차이 $\Delta = \mathrm{L} - \mathrm{A}$를 5개 구간(All, R1 회복, R2 확장, R3 위기, R4 AI 랠리)에 대해 보고한다. MDD는 음수가 손실이므로 $\Delta > 0$이 LSTM이 덜 빠진 경우이다.}
\label{tab:app_grid_mdd_err}
\setlength{\tabcolsep}{2pt}
\renewcommand{\arraystretch}{1.05}
\scriptsize
\resizebox{\textwidth}{!}{%
\begin{tabular}{lll ccc ccc ccc ccc ccc}
\toprule
 & & & \multicolumn{3}{c}{All} & \multicolumn{3}{c}{R1} & \multicolumn{3}{c}{R2} & \multicolumn{3}{c}{R3} & \multicolumn{3}{c}{R4} \\
\cmidrule(lr){4-6} \cmidrule(lr){7-9} \cmidrule(lr){10-12} \cmidrule(lr){13-15} \cmidrule(lr){16-18}
$p_w$ & prior & $q$ & L & A & $\Delta$ & L & A & $\Delta$ & L & A & $\Delta$ & L & A & $\Delta$ & L & A & $\Delta$ \\
\midrule
mcap & mcap & lam & $-16.48\%$ & $-16.03\%$ & \cellcolor{negred!50}$-0.45\%$ & $-16.22\%$ & $-13.51\%$ & \cellcolor{negred!50}$-2.72\%$ & $-16.48\%$ & $-13.08\%$ & \cellcolor{negred!50}$-3.40\%$ & $-9.95\%$ & $-16.03\%$ & \cellcolor{posgreen!50}$+6.08\%$ & $-6.87\%$ & $-7.07\%$ & \cellcolor{posgreen!50}$+0.20\%$ \\
 &  & raw & $-16.22\%$ & $-16.03\%$ & \cellcolor{negred!50}$-0.19\%$ & $-16.22\%$ & $-13.51\%$ & \cellcolor{negred!50}$-2.72\%$ & $-10.34\%$ & $-11.48\%$ & \cellcolor{posgreen!50}$+1.14\%$ & $-9.95\%$ & $-16.03\%$ & \cellcolor{posgreen!50}$+6.08\%$ & $-6.87\%$ & $-7.07\%$ & \cellcolor{posgreen!50}$+0.20\%$ \\
 &  & inv & $-15.77\%$ & $-16.06\%$ & \cellcolor{posgreen!50}$+0.29\%$ & $-13.25\%$ & $-11.41\%$ & \cellcolor{negred!50}$-1.83\%$ & $-15.77\%$ & $-16.06\%$ & \cellcolor{posgreen!50}$+0.29\%$ & $-12.88\%$ & $-15.89\%$ & \cellcolor{posgreen!50}$+3.02\%$ & $-8.11\%$ & $-7.47\%$ & \cellcolor{negred!50}$-0.64\%$ \\
 &  & vsp & $-16.78\%$ & $-15.70\%$ & \cellcolor{negred!50}$-1.09\%$ & $-15.22\%$ & $-12.80\%$ & \cellcolor{negred!50}$-2.43\%$ & $-16.78\%$ & $-15.70\%$ & \cellcolor{negred!50}$-1.09\%$ & $-11.82\%$ & $-15.21\%$ & \cellcolor{posgreen!50}$+3.39\%$ & $-7.82\%$ & $-7.45\%$ & \cellcolor{negred!50}$-0.38\%$ \\
 &  & fpm & $-20.48\%$ & $-18.39\%$ & \cellcolor{negred!50}$-2.10\%$ & $-17.12\%$ & $-12.54\%$ & \cellcolor{negred!50}$-4.58\%$ & $-17.57\%$ & $-16.10\%$ & \cellcolor{negred!50}$-1.47\%$ & $-16.53\%$ & $-16.14\%$ & \cellcolor{negred!50}$-0.39\%$ & $-13.95\%$ & $-18.39\%$ & \cellcolor{posgreen!50}$+4.43\%$ \\
\cmidrule(lr){2-18}
 & eq & lam & $-17.84\%$ & $-14.61\%$ & \cellcolor{negred!50}$-3.22\%$ & $-16.96\%$ & $-14.11\%$ & \cellcolor{negred!50}$-2.85\%$ & $-17.84\%$ & $-11.96\%$ & \cellcolor{negred!50}$-5.88\%$ & $-9.88\%$ & $-14.61\%$ & \cellcolor{posgreen!50}$+4.73\%$ & $-8.07\%$ & $-7.69\%$ & \cellcolor{negred!50}$-0.38\%$ \\
 &  & raw & $-16.96\%$ & $-14.61\%$ & \cellcolor{negred!50}$-2.35\%$ & $-16.96\%$ & $-14.11\%$ & \cellcolor{negred!50}$-2.85\%$ & $-10.73\%$ & $-9.93\%$ & \cellcolor{negred!50}$-0.80\%$ & $-9.88\%$ & $-14.61\%$ & \cellcolor{posgreen!50}$+4.73\%$ & $-8.07\%$ & $-7.69\%$ & \cellcolor{negred!50}$-0.38\%$ \\
 &  & inv & $-18.51\%$ & $-16.70\%$ & \cellcolor{negred!50}$-1.80\%$ & $-14.70\%$ & $-12.13\%$ & \cellcolor{negred!50}$-2.58\%$ & $-16.45\%$ & $-16.70\%$ & \cellcolor{posgreen!50}$+0.26\%$ & $-11.39\%$ & $-14.43\%$ & \cellcolor{posgreen!50}$+3.04\%$ & $-10.13\%$ & $-8.95\%$ & \cellcolor{negred!50}$-1.18\%$ \\
 &  & vsp & $-17.13\%$ & $-16.10\%$ & \cellcolor{negred!50}$-1.03\%$ & $-16.24\%$ & $-13.68\%$ & \cellcolor{negred!50}$-2.56\%$ & $-17.13\%$ & $-16.10\%$ & \cellcolor{negred!50}$-1.03\%$ & $-10.12\%$ & $-14.52\%$ & \cellcolor{posgreen!50}$+4.39\%$ & $-9.91\%$ & $-8.98\%$ & \cellcolor{negred!50}$-0.93\%$ \\
 &  & fpm & $-25.60\%$ & $-18.05\%$ & \cellcolor{negred!50}$-7.55\%$ & $-17.22\%$ & $-13.43\%$ & \cellcolor{negred!50}$-3.79\%$ & $-18.56\%$ & $-15.89\%$ & \cellcolor{negred!50}$-2.67\%$ & $-19.22\%$ & $-14.48\%$ & \cellcolor{negred!50}$-4.73\%$ & $-14.06\%$ & $-18.05\%$ & \cellcolor{posgreen!50}$+3.98\%$ \\
\cmidrule(lr){2-18}
 & rp & lam & $-16.41\%$ & $-13.97\%$ & \cellcolor{negred!50}$-2.44\%$ & $-14.40\%$ & $-11.99\%$ & \cellcolor{negred!50}$-2.41\%$ & $-16.41\%$ & $-11.06\%$ & \cellcolor{negred!50}$-5.35\%$ & $-10.15\%$ & $-13.97\%$ & \cellcolor{posgreen!50}$+3.82\%$ & $-6.99\%$ & $-6.85\%$ & \cellcolor{negred!50}$-0.15\%$ \\
 &  & raw & $-14.40\%$ & $-13.97\%$ & \cellcolor{negred!50}$-0.42\%$ & $-14.40\%$ & $-11.99\%$ & \cellcolor{negred!50}$-2.41\%$ & $-10.22\%$ & $-9.25\%$ & \cellcolor{negred!50}$-0.97\%$ & $-10.15\%$ & $-13.97\%$ & \cellcolor{posgreen!50}$+3.82\%$ & $-6.99\%$ & $-6.85\%$ & \cellcolor{negred!50}$-0.15\%$ \\
 &  & inv & $-17.22\%$ & $-15.53\%$ & \cellcolor{negred!50}$-1.69\%$ & $-12.38\%$ & $-10.70\%$ & \cellcolor{negred!50}$-1.68\%$ & $-15.44\%$ & $-15.53\%$ & \cellcolor{posgreen!50}$+0.09\%$ & $-10.93\%$ & $-13.79\%$ & \cellcolor{posgreen!50}$+2.85\%$ & $-8.80\%$ & $-7.96\%$ & \cellcolor{negred!50}$-0.84\%$ \\
 &  & vsp & $-15.68\%$ & $-14.82\%$ & \cellcolor{negred!50}$-0.86\%$ & $-13.87\%$ & $-11.74\%$ & \cellcolor{negred!50}$-2.13\%$ & $-15.68\%$ & $-14.82\%$ & \cellcolor{negred!50}$-0.86\%$ & $-10.26\%$ & $-13.76\%$ & \cellcolor{posgreen!50}$+3.51\%$ & $-8.69\%$ & $-8.01\%$ & \cellcolor{negred!50}$-0.68\%$ \\
 &  & fpm & $-23.79\%$ & $-17.35\%$ & \cellcolor{negred!50}$-6.44\%$ & $-14.74\%$ & $-11.57\%$ & \cellcolor{negred!50}$-3.17\%$ & $-17.50\%$ & $-14.96\%$ & \cellcolor{negred!50}$-2.53\%$ & $-17.73\%$ & $-13.84\%$ & \cellcolor{negred!50}$-3.89\%$ & $-12.93\%$ & $-17.35\%$ & \cellcolor{posgreen!50}$+4.42\%$ \\
\midrule
eq & mcap & lam & $-14.73\%$ & $-18.86\%$ & \cellcolor{posgreen!50}$+4.13\%$ & $-12.75\%$ & $-12.65\%$ & \cellcolor{negred!50}$-0.10\%$ & $-8.20\%$ & $-10.80\%$ & \cellcolor{posgreen!50}$+2.60\%$ & $-14.73\%$ & $-18.86\%$ & \cellcolor{posgreen!50}$+4.13\%$ & $-8.86\%$ & $-9.66\%$ & \cellcolor{posgreen!50}$+0.80\%$ \\
 &  & raw & $-14.73\%$ & $-18.86\%$ & \cellcolor{posgreen!50}$+4.13\%$ & $-12.75\%$ & $-12.65\%$ & \cellcolor{negred!50}$-0.10\%$ & $-8.35\%$ & $-11.00\%$ & \cellcolor{posgreen!50}$+2.65\%$ & $-14.73\%$ & $-18.86\%$ & \cellcolor{posgreen!50}$+4.13\%$ & $-8.86\%$ & $-9.66\%$ & \cellcolor{posgreen!50}$+0.80\%$ \\
 &  & inv & $-14.59\%$ & $-17.67\%$ & \cellcolor{posgreen!50}$+3.07\%$ & $-10.72\%$ & $-10.98\%$ & \cellcolor{posgreen!50}$+0.27\%$ & $-11.86\%$ & $-13.88\%$ & \cellcolor{posgreen!50}$+2.02\%$ & $-14.59\%$ & $-17.67\%$ & \cellcolor{posgreen!50}$+3.07\%$ & $-7.33\%$ & $-9.42\%$ & \cellcolor{posgreen!50}$+2.09\%$ \\
 &  & vsp & $-13.65\%$ & $-17.44\%$ & \cellcolor{posgreen!50}$+3.79\%$ & $-12.37\%$ & $-12.18\%$ & \cellcolor{negred!50}$-0.19\%$ & $-9.96\%$ & $-11.92\%$ & \cellcolor{posgreen!50}$+1.97\%$ & $-13.65\%$ & $-17.44\%$ & \cellcolor{posgreen!50}$+3.79\%$ & $-8.14\%$ & $-9.40\%$ & \cellcolor{posgreen!50}$+1.26\%$ \\
 &  & fpm & $-15.71\%$ & $-17.28\%$ & \cellcolor{posgreen!50}$+1.57\%$ & $-15.71\%$ & $-13.51\%$ & \cellcolor{negred!50}$-2.20\%$ & $-11.42\%$ & $-11.47\%$ & \cellcolor{posgreen!50}$+0.05\%$ & $-15.14\%$ & $-17.28\%$ & \cellcolor{posgreen!50}$+2.14\%$ & $-7.89\%$ & $-9.82\%$ & \cellcolor{posgreen!50}$+1.93\%$ \\
\cmidrule(lr){2-18}
 & eq & lam & $-13.65\%$ & $-18.55\%$ & \cellcolor{posgreen!50}$+4.90\%$ & $-13.36\%$ & $-12.73\%$ & \cellcolor{negred!50}$-0.63\%$ & $-8.17\%$ & $-10.27\%$ & \cellcolor{posgreen!50}$+2.10\%$ & $-13.65\%$ & $-18.55\%$ & \cellcolor{posgreen!50}$+4.90\%$ & $-8.95\%$ & $-9.87\%$ & \cellcolor{posgreen!50}$+0.92\%$ \\
 &  & raw & $-13.65\%$ & $-18.55\%$ & \cellcolor{posgreen!50}$+4.90\%$ & $-13.36\%$ & $-12.73\%$ & \cellcolor{negred!50}$-0.63\%$ & $-8.24\%$ & $-10.55\%$ & \cellcolor{posgreen!50}$+2.31\%$ & $-13.65\%$ & $-18.55\%$ & \cellcolor{posgreen!50}$+4.90\%$ & $-8.95\%$ & $-9.87\%$ & \cellcolor{posgreen!50}$+0.92\%$ \\
 &  & inv & $-17.57\%$ & $-16.55\%$ & \cellcolor{negred!50}$-1.02\%$ & $-11.18\%$ & $-11.09\%$ & \cellcolor{negred!50}$-0.09\%$ & $-12.02\%$ & $-14.23\%$ & \cellcolor{posgreen!50}$+2.21\%$ & $-11.38\%$ & $-15.26\%$ & \cellcolor{posgreen!50}$+3.88\%$ & $-8.30\%$ & $-10.02\%$ & \cellcolor{posgreen!50}$+1.72\%$ \\
 &  & vsp & $-13.74\%$ & $-15.65\%$ & \cellcolor{posgreen!50}$+1.91\%$ & $-12.50\%$ & $-12.25\%$ & \cellcolor{negred!50}$-0.25\%$ & $-10.13\%$ & $-11.52\%$ & \cellcolor{posgreen!50}$+1.39\%$ & $-11.48\%$ & $-15.65\%$ & \cellcolor{posgreen!50}$+4.18\%$ & $-8.64\%$ & $-10.08\%$ & \cellcolor{posgreen!50}$+1.44\%$ \\
 &  & fpm & $-15.10\%$ & $-14.93\%$ & \cellcolor{negred!50}$-0.16\%$ & $-15.10\%$ & $-14.06\%$ & \cellcolor{negred!50}$-1.04\%$ & $-11.49\%$ & $-11.00\%$ & \cellcolor{negred!50}$-0.49\%$ & $-11.81\%$ & $-14.93\%$ & \cellcolor{posgreen!50}$+3.12\%$ & $-9.58\%$ & $-11.01\%$ & \cellcolor{posgreen!50}$+1.43\%$ \\
\cmidrule(lr){2-18}
 & rp & lam & $-13.75\%$ & $-17.64\%$ & \cellcolor{posgreen!50}$+3.88\%$ & $-11.83\%$ & $-11.13\%$ & \cellcolor{negred!50}$-0.71\%$ & $-8.35\%$ & $-10.16\%$ & \cellcolor{posgreen!50}$+1.81\%$ & $-13.75\%$ & $-17.64\%$ & \cellcolor{posgreen!50}$+3.88\%$ & $-7.76\%$ & $-8.91\%$ & \cellcolor{posgreen!50}$+1.15\%$ \\
 &  & raw & $-13.75\%$ & $-17.64\%$ & \cellcolor{posgreen!50}$+3.88\%$ & $-11.83\%$ & $-11.13\%$ & \cellcolor{negred!50}$-0.71\%$ & $-8.48\%$ & $-10.27\%$ & \cellcolor{posgreen!50}$+1.79\%$ & $-13.75\%$ & $-17.64\%$ & \cellcolor{posgreen!50}$+3.88\%$ & $-7.76\%$ & $-8.91\%$ & \cellcolor{posgreen!50}$+1.15\%$ \\
 &  & inv & $-16.17\%$ & $-15.32\%$ & \cellcolor{negred!50}$-0.84\%$ & $-9.68\%$ & $-9.65\%$ & $-0.03\%$ & $-11.36\%$ & $-13.45\%$ & \cellcolor{posgreen!50}$+2.08\%$ & $-11.47\%$ & $-14.03\%$ & \cellcolor{posgreen!50}$+2.56\%$ & $-7.30\%$ & $-8.70\%$ & \cellcolor{posgreen!50}$+1.40\%$ \\
 &  & vsp & $-12.73\%$ & $-14.63\%$ & \cellcolor{posgreen!50}$+1.90\%$ & $-11.15\%$ & $-10.69\%$ & \cellcolor{negred!50}$-0.46\%$ & $-9.83\%$ & $-10.86\%$ & \cellcolor{posgreen!50}$+1.03\%$ & $-11.60\%$ & $-14.63\%$ & \cellcolor{posgreen!50}$+3.03\%$ & $-7.56\%$ & $-8.88\%$ & \cellcolor{posgreen!50}$+1.33\%$ \\
 &  & fpm & $-13.27\%$ & $-13.80\%$ & \cellcolor{posgreen!50}$+0.53\%$ & $-13.27\%$ & $-12.33\%$ & \cellcolor{negred!50}$-0.94\%$ & $-10.79\%$ & $-10.34\%$ & \cellcolor{negred!50}$-0.45\%$ & $-11.75\%$ & $-13.80\%$ & \cellcolor{posgreen!50}$+2.05\%$ & $-8.63\%$ & $-9.64\%$ & \cellcolor{posgreen!50}$+1.01\%$ \\
\midrule
rp & mcap & lam & $-14.01\%$ & $-18.62\%$ & \cellcolor{posgreen!50}$+4.61\%$ & $-14.01\%$ & $-15.13\%$ & \cellcolor{posgreen!50}$+1.12\%$ & $-8.45\%$ & $-10.60\%$ & \cellcolor{posgreen!50}$+2.15\%$ & $-13.43\%$ & $-18.62\%$ & \cellcolor{posgreen!50}$+5.19\%$ & $-8.74\%$ & $-9.34\%$ & \cellcolor{posgreen!50}$+0.60\%$ \\
 &  & raw & $-14.01\%$ & $-18.62\%$ & \cellcolor{posgreen!50}$+4.61\%$ & $-14.01\%$ & $-15.13\%$ & \cellcolor{posgreen!50}$+1.12\%$ & $-8.65\%$ & $-10.59\%$ & \cellcolor{posgreen!50}$+1.94\%$ & $-13.43\%$ & $-18.62\%$ & \cellcolor{posgreen!50}$+5.19\%$ & $-8.74\%$ & $-9.34\%$ & \cellcolor{posgreen!50}$+0.60\%$ \\
 &  & inv & $-14.72\%$ & $-17.59\%$ & \cellcolor{posgreen!50}$+2.87\%$ & $-10.96\%$ & $-15.30\%$ & \cellcolor{posgreen!50}$+4.34\%$ & $-13.57\%$ & $-14.73\%$ & \cellcolor{posgreen!50}$+1.16\%$ & $-14.72\%$ & $-17.59\%$ & \cellcolor{posgreen!50}$+2.87\%$ & $-6.94\%$ & $-8.38\%$ & \cellcolor{posgreen!50}$+1.44\%$ \\
 &  & vsp & $-13.98\%$ & $-17.51\%$ & \cellcolor{posgreen!50}$+3.53\%$ & $-12.67\%$ & $-15.17\%$ & \cellcolor{posgreen!50}$+2.50\%$ & $-11.30\%$ & $-13.23\%$ & \cellcolor{posgreen!50}$+1.92\%$ & $-13.98\%$ & $-17.51\%$ & \cellcolor{posgreen!50}$+3.53\%$ & $-7.78\%$ & $-8.57\%$ & \cellcolor{posgreen!50}$+0.79\%$ \\
 &  & fpm & $-19.24\%$ & $-17.17\%$ & \cellcolor{negred!50}$-2.07\%$ & $-19.24\%$ & $-16.10\%$ & \cellcolor{negred!50}$-3.14\%$ & $-12.58\%$ & $-12.43\%$ & \cellcolor{negred!50}$-0.14\%$ & $-15.02\%$ & $-17.17\%$ & \cellcolor{posgreen!50}$+2.15\%$ & $-7.54\%$ & $-9.50\%$ & \cellcolor{posgreen!50}$+1.96\%$ \\
\cmidrule(lr){2-18}
 & eq & lam & $-13.87\%$ & $-18.23\%$ & \cellcolor{posgreen!50}$+4.37\%$ & $-13.87\%$ & $-14.45\%$ & \cellcolor{posgreen!50}$+0.59\%$ & $-8.33\%$ & $-9.98\%$ & \cellcolor{posgreen!50}$+1.64\%$ & $-12.69\%$ & $-18.23\%$ & \cellcolor{posgreen!50}$+5.54\%$ & $-7.57\%$ & $-9.04\%$ & \cellcolor{posgreen!50}$+1.48\%$ \\
 &  & raw & $-13.87\%$ & $-18.23\%$ & \cellcolor{posgreen!50}$+4.37\%$ & $-13.87\%$ & $-14.44\%$ & \cellcolor{posgreen!50}$+0.58\%$ & $-8.38\%$ & $-10.26\%$ & \cellcolor{posgreen!50}$+1.88\%$ & $-12.69\%$ & $-18.23\%$ & \cellcolor{posgreen!50}$+5.54\%$ & $-7.57\%$ & $-9.04\%$ & \cellcolor{posgreen!50}$+1.48\%$ \\
 &  & inv & $-17.79\%$ & $-16.92\%$ & \cellcolor{negred!50}$-0.87\%$ & $-10.91\%$ & $-14.61\%$ & \cellcolor{posgreen!50}$+3.70\%$ & $-14.12\%$ & $-15.26\%$ & \cellcolor{posgreen!50}$+1.15\%$ & $-11.45\%$ & $-15.02\%$ & \cellcolor{posgreen!50}$+3.57\%$ & $-7.57\%$ & $-8.72\%$ & \cellcolor{posgreen!50}$+1.14\%$ \\
 &  & vsp & $-14.12\%$ & $-15.26\%$ & \cellcolor{posgreen!50}$+1.14\%$ & $-12.51\%$ & $-14.49\%$ & \cellcolor{posgreen!50}$+1.98\%$ & $-10.81\%$ & $-13.49\%$ & \cellcolor{posgreen!50}$+2.68\%$ & $-11.63\%$ & $-15.26\%$ & \cellcolor{posgreen!50}$+3.64\%$ & $-7.76\%$ & $-8.77\%$ & \cellcolor{posgreen!50}$+1.01\%$ \\
 &  & fpm & $-17.91\%$ & $-15.75\%$ & \cellcolor{negred!50}$-2.16\%$ & $-17.91\%$ & $-15.75\%$ & \cellcolor{negred!50}$-2.16\%$ & $-12.81\%$ & $-12.27\%$ & \cellcolor{negred!50}$-0.55\%$ & $-11.46\%$ & $-14.89\%$ & \cellcolor{posgreen!50}$+3.43\%$ & $-9.49\%$ & $-10.21\%$ & \cellcolor{posgreen!50}$+0.71\%$ \\
\cmidrule(lr){2-18}
 & rp & lam & $-12.84\%$ & $-17.33\%$ & \cellcolor{posgreen!50}$+4.49\%$ & $-11.88\%$ & $-12.91\%$ & \cellcolor{posgreen!50}$+1.04\%$ & $-8.59\%$ & $-9.90\%$ & \cellcolor{posgreen!50}$+1.31\%$ & $-12.84\%$ & $-17.33\%$ & \cellcolor{posgreen!50}$+4.49\%$ & $-6.89\%$ & $-8.15\%$ & \cellcolor{posgreen!50}$+1.26\%$ \\
 &  & raw & $-12.84\%$ & $-17.33\%$ & \cellcolor{posgreen!50}$+4.49\%$ & $-11.88\%$ & $-12.90\%$ & \cellcolor{posgreen!50}$+1.03\%$ & $-8.81\%$ & $-9.93\%$ & \cellcolor{posgreen!50}$+1.12\%$ & $-12.84\%$ & $-17.33\%$ & \cellcolor{posgreen!50}$+4.49\%$ & $-6.89\%$ & $-8.15\%$ & \cellcolor{posgreen!50}$+1.26\%$ \\
 &  & inv & $-16.42\%$ & $-15.59\%$ & \cellcolor{negred!50}$-0.83\%$ & $-9.59\%$ & $-13.06\%$ & \cellcolor{posgreen!50}$+3.47\%$ & $-13.26\%$ & $-14.32\%$ & \cellcolor{posgreen!50}$+1.05\%$ & $-11.55\%$ & $-13.86\%$ & \cellcolor{posgreen!50}$+2.31\%$ & $-7.13\%$ & $-7.66\%$ & \cellcolor{posgreen!50}$+0.53\%$ \\
 &  & vsp & $-12.95\%$ & $-14.40\%$ & \cellcolor{posgreen!50}$+1.45\%$ & $-10.84\%$ & $-12.94\%$ & \cellcolor{posgreen!50}$+2.11\%$ & $-10.22\%$ & $-12.53\%$ & \cellcolor{posgreen!50}$+2.31\%$ & $-11.72\%$ & $-14.40\%$ & \cellcolor{posgreen!50}$+2.68\%$ & $-7.27\%$ & $-7.85\%$ & \cellcolor{posgreen!50}$+0.58\%$ \\
 &  & fpm & $-15.69\%$ & $-13.96\%$ & \cellcolor{negred!50}$-1.73\%$ & $-15.69\%$ & $-13.96\%$ & \cellcolor{negred!50}$-1.73\%$ & $-11.87\%$ & $-11.38\%$ & \cellcolor{negred!50}$-0.48\%$ & $-11.64\%$ & $-13.54\%$ & \cellcolor{posgreen!50}$+1.91\%$ & $-8.60\%$ & $-9.19\%$ & \cellcolor{posgreen!50}$+0.59\%$ \\
\bottomrule
\end{tabular}}
\end{table}

% NOTE: 표가 페이지 폭을 넘으면 paper3.md 프리앰블에 \usepackage{pdflscape}를 추가하고
%       \begin{landscape} ... \end{landscape}로 표 전체를 감싸 가로 회전하시기 바랍니다.
\begin{table}[!htbp]
\centering
\caption{전체 45 슬롯 Sharpe 비율 결과 ($\Omega=$he 고정). $p_w \times \text{prior} \times q$ = $3 \times 3 \times 5$ = 45 슬롯 각각의 LSTM(L)·ANN(A) Sharpe 비율 및 차이 $\Delta = \mathrm{L} - \mathrm{A}$를 5개 구간(All, R1 회복, R2 확장, R3 위기, R4 AI 랠리)에 대해 보고한다.}
\label{tab:app_grid_sharpe_he}
\setlength{\tabcolsep}{2pt}
\renewcommand{\arraystretch}{1.05}
\scriptsize
\resizebox{\textwidth}{!}{%
\begin{tabular}{lll ccc ccc ccc ccc ccc}
\toprule
 & & & \multicolumn{3}{c}{All} & \multicolumn{3}{c}{R1} & \multicolumn{3}{c}{R2} & \multicolumn{3}{c}{R3} & \multicolumn{3}{c}{R4} \\
\cmidrule(lr){4-6} \cmidrule(lr){7-9} \cmidrule(lr){10-12} \cmidrule(lr){13-15} \cmidrule(lr){16-18}
$p_w$ & prior & $q$ & L & A & $\Delta$ & L & A & $\Delta$ & L & A & $\Delta$ & L & A & $\Delta$ & L & A & $\Delta$ \\
\midrule
mcap & mcap & lam & $0.968$ & $0.882$ & \cellcolor{posgreen!50}$+0.086$ & $1.041$ & $1.146$ & \cellcolor{negred!50}$-0.105$ & $1.201$ & $1.195$ & \cellcolor{posgreen!50}$+0.006$ & $0.709$ & $0.415$ & \cellcolor{posgreen!50}$+0.294$ & $0.677$ & $0.505$ & \cellcolor{posgreen!50}$+0.172$ \\
 &  & raw & $0.999$ & $0.910$ & \cellcolor{posgreen!50}$+0.089$ & $1.059$ & $1.156$ & \cellcolor{negred!50}$-0.097$ & $1.243$ & $1.243$ & $+0.000$ & $0.722$ & $0.425$ & \cellcolor{posgreen!50}$+0.298$ & $0.677$ & $0.505$ & \cellcolor{posgreen!50}$+0.172$ \\
 &  & inv & $0.983$ & $0.927$ & \cellcolor{posgreen!50}$+0.056$ & $1.390$ & $1.539$ & \cellcolor{negred!50}$-0.150$ & $1.237$ & $1.154$ & \cellcolor{posgreen!50}$+0.083$ & $0.567$ & $0.515$ & \cellcolor{posgreen!50}$+0.052$ & $0.906$ & $0.712$ & \cellcolor{posgreen!50}$+0.194$ \\
 &  & vsp & $0.965$ & $0.896$ & \cellcolor{posgreen!50}$+0.069$ & $1.149$ & $1.220$ & \cellcolor{negred!50}$-0.071$ & $1.250$ & $1.167$ & \cellcolor{posgreen!50}$+0.083$ & $0.567$ & $0.476$ & \cellcolor{posgreen!50}$+0.092$ & $0.880$ & $0.706$ & \cellcolor{posgreen!50}$+0.173$ \\
 &  & fpm & $0.830$ & $0.871$ & \cellcolor{negred!50}$-0.041$ & $0.882$ & $1.133$ & \cellcolor{negred!50}$-0.251$ & $1.178$ & $1.043$ & \cellcolor{posgreen!50}$+0.135$ & $0.580$ & $0.680$ & \cellcolor{negred!50}$-0.100$ & $0.599$ & $0.637$ & \cellcolor{negred!50}$-0.038$ \\
\cmidrule(lr){2-18}
 & eq & lam & $0.976$ & $0.913$ & \cellcolor{posgreen!50}$+0.063$ & $1.200$ & $1.283$ & \cellcolor{negred!50}$-0.083$ & $1.241$ & $1.221$ & \cellcolor{posgreen!50}$+0.019$ & $0.811$ & $0.530$ & \cellcolor{posgreen!50}$+0.282$ & $0.374$ & $0.369$ & \cellcolor{posgreen!50}$+0.006$ \\
 &  & raw & $1.001$ & $0.942$ & \cellcolor{posgreen!50}$+0.059$ & $1.213$ & $1.288$ & \cellcolor{negred!50}$-0.076$ & $1.266$ & $1.275$ & \cellcolor{negred!50}$-0.009$ & $0.823$ & $0.536$ & \cellcolor{posgreen!50}$+0.287$ & $0.374$ & $0.369$ & \cellcolor{posgreen!50}$+0.006$ \\
 &  & inv & $0.913$ & $0.918$ & \cellcolor{negred!50}$-0.004$ & $1.617$ & $1.828$ & \cellcolor{negred!50}$-0.211$ & $1.265$ & $1.217$ & \cellcolor{posgreen!50}$+0.048$ & $0.562$ & $0.534$ & \cellcolor{posgreen!50}$+0.028$ & $0.410$ & $0.385$ & \cellcolor{posgreen!50}$+0.025$ \\
 &  & vsp & $0.914$ & $0.906$ & \cellcolor{posgreen!50}$+0.008$ & $1.344$ & $1.419$ & \cellcolor{negred!50}$-0.075$ & $1.325$ & $1.273$ & \cellcolor{posgreen!50}$+0.052$ & $0.565$ & $0.521$ & \cellcolor{posgreen!50}$+0.044$ & $0.367$ & $0.374$ & \cellcolor{negred!50}$-0.007$ \\
 &  & fpm & $0.795$ & $0.813$ & \cellcolor{negred!50}$-0.019$ & $1.076$ & $1.317$ & \cellcolor{negred!50}$-0.241$ & $1.216$ & $1.014$ & \cellcolor{posgreen!50}$+0.202$ & $0.572$ & $0.628$ & \cellcolor{negred!50}$-0.056$ & $0.334$ & $0.387$ & \cellcolor{negred!50}$-0.053$ \\
\cmidrule(lr){2-18}
 & rp & lam & $0.959$ & $0.897$ & \cellcolor{posgreen!50}$+0.062$ & $1.361$ & $1.458$ & \cellcolor{negred!50}$-0.097$ & $1.189$ & $1.167$ & \cellcolor{posgreen!50}$+0.022$ & $0.742$ & $0.465$ & \cellcolor{posgreen!50}$+0.276$ & $0.339$ & $0.346$ & \cellcolor{negred!50}$-0.007$ \\
 &  & raw & $0.981$ & $0.935$ & \cellcolor{posgreen!50}$+0.046$ & $1.372$ & $1.465$ & \cellcolor{negred!50}$-0.092$ & $1.213$ & $1.241$ & \cellcolor{negred!50}$-0.028$ & $0.752$ & $0.474$ & \cellcolor{posgreen!50}$+0.278$ & $0.339$ & $0.346$ & \cellcolor{negred!50}$-0.007$ \\
 &  & inv & $0.906$ & $0.923$ & \cellcolor{negred!50}$-0.017$ & $1.706$ & $1.981$ & \cellcolor{negred!50}$-0.275$ & $1.254$ & $1.224$ & \cellcolor{posgreen!50}$+0.030$ & $0.537$ & $0.515$ & \cellcolor{posgreen!50}$+0.022$ & $0.357$ & $0.331$ & \cellcolor{posgreen!50}$+0.025$ \\
 &  & vsp & $0.918$ & $0.911$ & \cellcolor{posgreen!50}$+0.007$ & $1.524$ & $1.618$ & \cellcolor{negred!50}$-0.094$ & $1.281$ & $1.266$ & \cellcolor{posgreen!50}$+0.015$ & $0.566$ & $0.488$ & \cellcolor{posgreen!50}$+0.078$ & $0.318$ & $0.312$ & \cellcolor{posgreen!50}$+0.006$ \\
 &  & fpm & $0.805$ & $0.831$ & \cellcolor{negred!50}$-0.026$ & $1.194$ & $1.498$ & \cellcolor{negred!50}$-0.304$ & $1.218$ & $1.022$ & \cellcolor{posgreen!50}$+0.196$ & $0.557$ & $0.617$ & \cellcolor{negred!50}$-0.060$ & $0.289$ & $0.356$ & \cellcolor{negred!50}$-0.066$ \\
\midrule
eq & mcap & lam & $0.950$ & $0.907$ & \cellcolor{posgreen!50}$+0.043$ & $1.147$ & $1.159$ & \cellcolor{negred!50}$-0.011$ & $1.226$ & $1.207$ & \cellcolor{posgreen!50}$+0.020$ & $0.584$ & $0.425$ & \cellcolor{posgreen!50}$+0.160$ & $0.499$ & $0.505$ & \cellcolor{negred!50}$-0.007$ \\
 &  & raw & $0.958$ & $0.923$ & \cellcolor{posgreen!50}$+0.035$ & $1.164$ & $1.173$ & \cellcolor{negred!50}$-0.009$ & $1.220$ & $1.221$ & \cellcolor{negred!50}$-0.001$ & $0.590$ & $0.429$ & \cellcolor{posgreen!50}$+0.161$ & $0.499$ & $0.505$ & \cellcolor{negred!50}$-0.007$ \\
 &  & inv & $1.057$ & $1.038$ & \cellcolor{posgreen!50}$+0.019$ & $1.571$ & $1.738$ & \cellcolor{negred!50}$-0.167$ & $1.380$ & $1.326$ & \cellcolor{posgreen!50}$+0.054$ & $0.585$ & $0.534$ & \cellcolor{posgreen!50}$+0.051$ & $0.903$ & $0.865$ & \cellcolor{posgreen!50}$+0.037$ \\
 &  & vsp & $1.040$ & $1.011$ & \cellcolor{posgreen!50}$+0.028$ & $1.287$ & $1.256$ & \cellcolor{posgreen!50}$+0.031$ & $1.422$ & $1.389$ & \cellcolor{posgreen!50}$+0.033$ & $0.551$ & $0.522$ & \cellcolor{posgreen!50}$+0.029$ & $0.828$ & $0.820$ & \cellcolor{posgreen!50}$+0.008$ \\
 &  & fpm & $0.998$ & $0.988$ & \cellcolor{posgreen!50}$+0.010$ & $0.857$ & $0.987$ & \cellcolor{negred!50}$-0.130$ & $1.448$ & $1.312$ & \cellcolor{posgreen!50}$+0.136$ & $0.582$ & $0.588$ & \cellcolor{negred!50}$-0.006$ & $0.910$ & $0.961$ & \cellcolor{negred!50}$-0.050$ \\
\cmidrule(lr){2-18}
 & eq & lam & $0.908$ & $0.898$ & \cellcolor{posgreen!50}$+0.009$ & $1.329$ & $1.367$ & \cellcolor{negred!50}$-0.038$ & $1.199$ & $1.189$ & \cellcolor{posgreen!50}$+0.010$ & $0.564$ & $0.478$ & \cellcolor{posgreen!50}$+0.086$ & $0.237$ & $0.257$ & \cellcolor{negred!50}$-0.019$ \\
 &  & raw & $0.925$ & $0.905$ & \cellcolor{posgreen!50}$+0.020$ & $1.349$ & $1.373$ & \cellcolor{negred!50}$-0.024$ & $1.210$ & $1.180$ & \cellcolor{posgreen!50}$+0.030$ & $0.572$ & $0.486$ & \cellcolor{posgreen!50}$+0.087$ & $0.237$ & $0.257$ & \cellcolor{negred!50}$-0.019$ \\
 &  & inv & $0.954$ & $0.960$ & \cellcolor{negred!50}$-0.006$ & $1.654$ & $1.815$ & \cellcolor{negred!50}$-0.160$ & $1.394$ & $1.328$ & \cellcolor{posgreen!50}$+0.067$ & $0.526$ & $0.530$ & \cellcolor{negred!50}$-0.004$ & $0.396$ & $0.430$ & \cellcolor{negred!50}$-0.034$ \\
 &  & vsp & $0.935$ & $0.956$ & \cellcolor{negred!50}$-0.022$ & $1.513$ & $1.515$ & \cellcolor{negred!50}$-0.002$ & $1.429$ & $1.380$ & \cellcolor{posgreen!50}$+0.050$ & $0.422$ & $0.504$ & \cellcolor{negred!50}$-0.082$ & $0.343$ & $0.397$ & \cellcolor{negred!50}$-0.054$ \\
 &  & fpm & $0.924$ & $0.947$ & \cellcolor{negred!50}$-0.023$ & $1.093$ & $1.224$ & \cellcolor{negred!50}$-0.131$ & $1.474$ & $1.345$ & \cellcolor{posgreen!50}$+0.129$ & $0.511$ & $0.596$ & \cellcolor{negred!50}$-0.085$ & $0.315$ & $0.411$ & \cellcolor{negred!50}$-0.096$ \\
\cmidrule(lr){2-18}
 & rp & lam & $0.902$ & $0.886$ & \cellcolor{posgreen!50}$+0.016$ & $1.471$ & $1.529$ & \cellcolor{negred!50}$-0.058$ & $1.177$ & $1.159$ & \cellcolor{posgreen!50}$+0.018$ & $0.539$ & $0.424$ & \cellcolor{posgreen!50}$+0.116$ & $0.202$ & $0.246$ & \cellcolor{negred!50}$-0.044$ \\
 &  & raw & $0.915$ & $0.895$ & \cellcolor{posgreen!50}$+0.020$ & $1.480$ & $1.535$ & \cellcolor{negred!50}$-0.055$ & $1.186$ & $1.156$ & \cellcolor{posgreen!50}$+0.029$ & $0.546$ & $0.431$ & \cellcolor{posgreen!50}$+0.115$ & $0.202$ & $0.246$ & \cellcolor{negred!50}$-0.044$ \\
 &  & inv & $0.938$ & $0.947$ & \cellcolor{negred!50}$-0.009$ & $1.725$ & $1.865$ & \cellcolor{negred!50}$-0.140$ & $1.365$ & $1.310$ & \cellcolor{posgreen!50}$+0.056$ & $0.501$ & $0.504$ & \cellcolor{negred!50}$-0.002$ & $0.346$ & $0.394$ & \cellcolor{negred!50}$-0.047$ \\
 &  & vsp & $0.926$ & $0.951$ & \cellcolor{negred!50}$-0.024$ & $1.638$ & $1.693$ & \cellcolor{negred!50}$-0.055$ & $1.374$ & $1.339$ & \cellcolor{posgreen!50}$+0.035$ & $0.426$ & $0.477$ & \cellcolor{negred!50}$-0.051$ & $0.299$ & $0.362$ & \cellcolor{negred!50}$-0.063$ \\
 &  & fpm & $0.925$ & $0.935$ & \cellcolor{negred!50}$-0.010$ & $1.184$ & $1.330$ & \cellcolor{negred!50}$-0.146$ & $1.438$ & $1.312$ & \cellcolor{posgreen!50}$+0.126$ & $0.511$ & $0.562$ & \cellcolor{negred!50}$-0.051$ & $0.271$ & $0.354$ & \cellcolor{negred!50}$-0.083$ \\
\midrule
rp & mcap & lam & $0.935$ & $0.865$ & \cellcolor{posgreen!50}$+0.069$ & $1.088$ & $0.926$ & \cellcolor{posgreen!50}$+0.161$ & $1.186$ & $1.185$ & \cellcolor{posgreen!50}$+0.001$ & $0.619$ & $0.447$ & \cellcolor{posgreen!50}$+0.172$ & $0.534$ & $0.534$ & $+0.000$ \\
 &  & raw & $0.960$ & $0.893$ & \cellcolor{posgreen!50}$+0.067$ & $1.092$ & $0.931$ & \cellcolor{posgreen!50}$+0.161$ & $1.222$ & $1.216$ & \cellcolor{posgreen!50}$+0.006$ & $0.626$ & $0.471$ & \cellcolor{posgreen!50}$+0.156$ & $0.534$ & $0.534$ & $+0.000$ \\
 &  & inv & $1.061$ & $0.989$ & \cellcolor{posgreen!50}$+0.072$ & $1.590$ & $1.127$ & \cellcolor{posgreen!50}$+0.462$ & $1.353$ & $1.342$ & \cellcolor{posgreen!50}$+0.011$ & $0.593$ & $0.551$ & \cellcolor{posgreen!50}$+0.042$ & $0.929$ & $0.878$ & \cellcolor{posgreen!50}$+0.051$ \\
 &  & vsp & $1.030$ & $0.985$ & \cellcolor{posgreen!50}$+0.045$ & $1.233$ & $0.983$ & \cellcolor{posgreen!50}$+0.250$ & $1.394$ & $1.409$ & \cellcolor{negred!50}$-0.015$ & $0.560$ & $0.543$ & \cellcolor{posgreen!50}$+0.017$ & $0.851$ & $0.841$ & \cellcolor{posgreen!50}$+0.010$ \\
 &  & fpm & $0.969$ & $0.957$ & \cellcolor{posgreen!50}$+0.012$ & $0.810$ & $0.859$ & \cellcolor{negred!50}$-0.049$ & $1.421$ & $1.320$ & \cellcolor{posgreen!50}$+0.101$ & $0.577$ & $0.580$ & \cellcolor{negred!50}$-0.003$ & $0.883$ & $0.953$ & \cellcolor{negred!50}$-0.070$ \\
\cmidrule(lr){2-18}
 & eq & lam & $0.906$ & $0.875$ & \cellcolor{posgreen!50}$+0.031$ & $1.277$ & $1.112$ & \cellcolor{posgreen!50}$+0.166$ & $1.171$ & $1.181$ & \cellcolor{negred!50}$-0.010$ & $0.603$ & $0.515$ & \cellcolor{posgreen!50}$+0.088$ & $0.274$ & $0.303$ & \cellcolor{negred!50}$-0.029$ \\
 &  & raw & $0.933$ & $0.892$ & \cellcolor{posgreen!50}$+0.041$ & $1.289$ & $1.123$ & \cellcolor{posgreen!50}$+0.166$ & $1.207$ & $1.194$ & \cellcolor{posgreen!50}$+0.013$ & $0.613$ & $0.527$ & \cellcolor{posgreen!50}$+0.086$ & $0.274$ & $0.303$ & \cellcolor{negred!50}$-0.029$ \\
 &  & inv & $0.951$ & $0.913$ & \cellcolor{posgreen!50}$+0.038$ & $1.687$ & $1.227$ & \cellcolor{posgreen!50}$+0.459$ & $1.352$ & $1.321$ & \cellcolor{posgreen!50}$+0.031$ & $0.543$ & $0.537$ & \cellcolor{posgreen!50}$+0.006$ & $0.392$ & $0.438$ & \cellcolor{negred!50}$-0.045$ \\
 &  & vsp & $0.924$ & $0.916$ & \cellcolor{posgreen!50}$+0.008$ & $1.437$ & $1.140$ & \cellcolor{posgreen!50}$+0.297$ & $1.407$ & $1.387$ & \cellcolor{posgreen!50}$+0.020$ & $0.425$ & $0.513$ & \cellcolor{negred!50}$-0.088$ & $0.338$ & $0.401$ & \cellcolor{negred!50}$-0.062$ \\
 &  & fpm & $0.908$ & $0.925$ & \cellcolor{negred!50}$-0.017$ & $1.062$ & $1.079$ & \cellcolor{negred!50}$-0.016$ & $1.451$ & $1.356$ & \cellcolor{posgreen!50}$+0.095$ & $0.503$ & $0.595$ & \cellcolor{negred!50}$-0.091$ & $0.289$ & $0.399$ & \cellcolor{negred!50}$-0.110$ \\
\cmidrule(lr){2-18}
 & rp & lam & $0.890$ & $0.865$ & \cellcolor{posgreen!50}$+0.025$ & $1.425$ & $1.230$ & \cellcolor{posgreen!50}$+0.195$ & $1.140$ & $1.159$ & \cellcolor{negred!50}$-0.019$ & $0.560$ & $0.461$ & \cellcolor{posgreen!50}$+0.100$ & $0.210$ & $0.276$ & \cellcolor{negred!50}$-0.066$ \\
 &  & raw & $0.916$ & $0.888$ & \cellcolor{posgreen!50}$+0.029$ & $1.434$ & $1.240$ & \cellcolor{posgreen!50}$+0.194$ & $1.177$ & $1.186$ & \cellcolor{negred!50}$-0.009$ & $0.567$ & $0.470$ & \cellcolor{posgreen!50}$+0.098$ & $0.210$ & $0.276$ & \cellcolor{negred!50}$-0.066$ \\
 &  & inv & $0.933$ & $0.913$ & \cellcolor{posgreen!50}$+0.020$ & $1.739$ & $1.348$ & \cellcolor{posgreen!50}$+0.392$ & $1.333$ & $1.309$ & \cellcolor{posgreen!50}$+0.025$ & $0.514$ & $0.511$ & \cellcolor{posgreen!50}$+0.003$ & $0.354$ & $0.412$ & \cellcolor{negred!50}$-0.058$ \\
 &  & vsp & $0.916$ & $0.921$ & \cellcolor{negred!50}$-0.005$ & $1.577$ & $1.283$ & \cellcolor{posgreen!50}$+0.294$ & $1.357$ & $1.359$ & \cellcolor{negred!50}$-0.002$ & $0.417$ & $0.493$ & \cellcolor{negred!50}$-0.075$ & $0.299$ & $0.372$ & \cellcolor{negred!50}$-0.073$ \\
 &  & fpm & $0.914$ & $0.921$ & \cellcolor{negred!50}$-0.007$ & $1.170$ & $1.178$ & \cellcolor{negred!50}$-0.008$ & $1.423$ & $1.323$ & \cellcolor{posgreen!50}$+0.100$ & $0.499$ & $0.565$ & \cellcolor{negred!50}$-0.067$ & $0.242$ & $0.348$ & \cellcolor{negred!50}$-0.105$ \\
\bottomrule
\end{tabular}}
\end{table}

% NOTE: 표가 페이지 폭을 넘으면 paper3.md 프리앰블에 \usepackage{pdflscape}를 추가하고
%       \begin{landscape} ... \end{landscape}로 표 전체를 감싸 가로 회전하시기 바랍니다.
\begin{table}[!htbp]
\centering
\caption{전체 45 슬롯 Sortino 비율 결과 ($\Omega=$he 고정). $p_w \times \text{prior} \times q$ = $3 \times 3 \times 5$ = 45 슬롯 각각의 LSTM(L)·ANN(A) Sortino 비율 및 차이 $\Delta = \mathrm{L} - \mathrm{A}$를 5개 구간(All, R1 회복, R2 확장, R3 위기, R4 AI 랠리)에 대해 보고한다.}
\label{tab:app_grid_sortino_he}
\setlength{\tabcolsep}{2pt}
\renewcommand{\arraystretch}{1.05}
\scriptsize
\resizebox{\textwidth}{!}{%
\begin{tabular}{lll ccc ccc ccc ccc ccc}
\toprule
 & & & \multicolumn{3}{c}{All} & \multicolumn{3}{c}{R1} & \multicolumn{3}{c}{R2} & \multicolumn{3}{c}{R3} & \multicolumn{3}{c}{R4} \\
\cmidrule(lr){4-6} \cmidrule(lr){7-9} \cmidrule(lr){10-12} \cmidrule(lr){13-15} \cmidrule(lr){16-18}
$p_w$ & prior & $q$ & L & A & $\Delta$ & L & A & $\Delta$ & L & A & $\Delta$ & L & A & $\Delta$ & L & A & $\Delta$ \\
\midrule
mcap & mcap & lam & $1.690$ & $1.537$ & \cellcolor{posgreen!50}$+0.153$ & $1.678$ & $1.905$ & \cellcolor{negred!50}$-0.227$ & $1.854$ & $1.837$ & \cellcolor{posgreen!50}$+0.017$ & $1.601$ & $0.982$ & \cellcolor{posgreen!50}$+0.619$ & $1.190$ & $0.779$ & \cellcolor{posgreen!50}$+0.411$ \\
 &  & raw & $1.793$ & $1.625$ & \cellcolor{posgreen!50}$+0.168$ & $1.714$ & $1.912$ & \cellcolor{negred!50}$-0.198$ & $2.018$ & $2.014$ & \cellcolor{posgreen!50}$+0.004$ & $1.624$ & $1.013$ & \cellcolor{posgreen!50}$+0.611$ & $1.190$ & $0.779$ & \cellcolor{posgreen!50}$+0.411$ \\
 &  & inv & $1.392$ & $1.343$ & \cellcolor{posgreen!50}$+0.049$ & $1.907$ & $2.142$ & \cellcolor{negred!50}$-0.235$ & $1.520$ & $1.446$ & \cellcolor{posgreen!50}$+0.073$ & $1.026$ & $1.003$ & \cellcolor{posgreen!50}$+0.023$ & $1.577$ & $1.087$ & \cellcolor{posgreen!50}$+0.489$ \\
 &  & vsp & $1.453$ & $1.344$ & \cellcolor{posgreen!50}$+0.109$ & $1.775$ & $1.950$ & \cellcolor{negred!50}$-0.175$ & $1.563$ & $1.417$ & \cellcolor{posgreen!50}$+0.146$ & $1.123$ & $1.004$ & \cellcolor{posgreen!50}$+0.119$ & $1.460$ & $1.103$ & \cellcolor{posgreen!50}$+0.357$ \\
 &  & fpm & $1.180$ & $1.260$ & \cellcolor{negred!50}$-0.079$ & $1.462$ & $1.818$ & \cellcolor{negred!50}$-0.356$ & $1.585$ & $1.424$ & \cellcolor{posgreen!50}$+0.161$ & $0.966$ & $1.225$ & \cellcolor{negred!50}$-0.259$ & $0.779$ & $0.827$ & \cellcolor{negred!50}$-0.048$ \\
\cmidrule(lr){2-18}
 & eq & lam & $1.742$ & $1.595$ & \cellcolor{posgreen!50}$+0.147$ & $2.149$ & $2.418$ & \cellcolor{negred!50}$-0.269$ & $1.944$ & $1.914$ & \cellcolor{posgreen!50}$+0.030$ & $1.784$ & $1.051$ & \cellcolor{posgreen!50}$+0.733$ & $0.648$ & $0.640$ & \cellcolor{posgreen!50}$+0.007$ \\
 &  & raw & $1.796$ & $1.663$ & \cellcolor{posgreen!50}$+0.133$ & $2.174$ & $2.417$ & \cellcolor{negred!50}$-0.243$ & $1.998$ & $2.036$ & \cellcolor{negred!50}$-0.038$ & $1.805$ & $1.069$ & \cellcolor{posgreen!50}$+0.736$ & $0.648$ & $0.640$ & \cellcolor{posgreen!50}$+0.007$ \\
 &  & inv & $1.272$ & $1.311$ & \cellcolor{negred!50}$-0.039$ & $2.538$ & $2.951$ & \cellcolor{negred!50}$-0.414$ & $1.531$ & $1.538$ & \cellcolor{negred!50}$-0.007$ & $0.921$ & $0.881$ & \cellcolor{posgreen!50}$+0.040$ & $0.709$ & $0.675$ & \cellcolor{posgreen!50}$+0.034$ \\
 &  & vsp & $1.367$ & $1.378$ & \cellcolor{negred!50}$-0.011$ & $2.373$ & $2.572$ & \cellcolor{negred!50}$-0.199$ & $1.597$ & $1.597$ & $-0.000$ & $1.022$ & $0.945$ & \cellcolor{posgreen!50}$+0.077$ & $0.631$ & $0.664$ & \cellcolor{negred!50}$-0.033$ \\
 &  & fpm & $1.098$ & $1.183$ & \cellcolor{negred!50}$-0.085$ & $1.862$ & $2.395$ & \cellcolor{negred!50}$-0.533$ & $1.530$ & $1.383$ & \cellcolor{posgreen!50}$+0.147$ & $0.886$ & $1.010$ & \cellcolor{negred!50}$-0.124$ & $0.512$ & $0.584$ & \cellcolor{negred!50}$-0.073$ \\
\cmidrule(lr){2-18}
 & rp & lam & $1.727$ & $1.599$ & \cellcolor{posgreen!50}$+0.128$ & $2.532$ & $2.719$ & \cellcolor{negred!50}$-0.187$ & $1.927$ & $1.897$ & \cellcolor{posgreen!50}$+0.030$ & $1.616$ & $0.943$ & \cellcolor{posgreen!50}$+0.673$ & $0.553$ & $0.586$ & \cellcolor{negred!50}$-0.034$ \\
 &  & raw & $1.763$ & $1.675$ & \cellcolor{posgreen!50}$+0.088$ & $2.540$ & $2.717$ & \cellcolor{negred!50}$-0.177$ & $1.953$ & $2.027$ & \cellcolor{negred!50}$-0.075$ & $1.631$ & $0.966$ & \cellcolor{posgreen!50}$+0.665$ & $0.553$ & $0.586$ & \cellcolor{negred!50}$-0.034$ \\
 &  & inv & $1.263$ & $1.325$ & \cellcolor{negred!50}$-0.062$ & $2.577$ & $3.045$ & \cellcolor{negred!50}$-0.469$ & $1.527$ & $1.565$ & \cellcolor{negred!50}$-0.038$ & $0.906$ & $0.886$ & \cellcolor{posgreen!50}$+0.020$ & $0.590$ & $0.542$ & \cellcolor{posgreen!50}$+0.048$ \\
 &  & vsp & $1.377$ & $1.388$ & \cellcolor{negred!50}$-0.011$ & $2.585$ & $2.885$ & \cellcolor{negred!50}$-0.299$ & $1.603$ & $1.582$ & \cellcolor{posgreen!50}$+0.021$ & $1.045$ & $0.922$ & \cellcolor{posgreen!50}$+0.123$ & $0.525$ & $0.516$ & \cellcolor{posgreen!50}$+0.009$ \\
 &  & fpm & $1.122$ & $1.215$ & \cellcolor{negred!50}$-0.093$ & $2.204$ & $2.680$ & \cellcolor{negred!50}$-0.476$ & $1.535$ & $1.431$ & \cellcolor{posgreen!50}$+0.104$ & $0.892$ & $1.006$ & \cellcolor{negred!50}$-0.114$ & $0.428$ & $0.519$ & \cellcolor{negred!50}$-0.091$ \\
\midrule
eq & mcap & lam & $1.655$ & $1.572$ & \cellcolor{posgreen!50}$+0.083$ & $1.764$ & $1.867$ & \cellcolor{negred!50}$-0.103$ & $2.124$ & $2.092$ & \cellcolor{posgreen!50}$+0.032$ & $1.134$ & $0.863$ & \cellcolor{posgreen!50}$+0.270$ & $0.805$ & $0.768$ & \cellcolor{posgreen!50}$+0.037$ \\
 &  & raw & $1.670$ & $1.606$ & \cellcolor{posgreen!50}$+0.064$ & $1.796$ & $1.892$ & \cellcolor{negred!50}$-0.097$ & $2.097$ & $2.107$ & \cellcolor{negred!50}$-0.010$ & $1.146$ & $0.872$ & \cellcolor{posgreen!50}$+0.274$ & $0.805$ & $0.768$ & \cellcolor{posgreen!50}$+0.037$ \\
 &  & inv & $1.481$ & $1.444$ & \cellcolor{posgreen!50}$+0.037$ & $2.057$ & $2.049$ & \cellcolor{posgreen!50}$+0.007$ & $1.730$ & $1.738$ & \cellcolor{negred!50}$-0.008$ & $1.051$ & $0.963$ & \cellcolor{posgreen!50}$+0.088$ & $1.332$ & $1.214$ & \cellcolor{posgreen!50}$+0.118$ \\
 &  & vsp & $1.528$ & $1.487$ & \cellcolor{posgreen!50}$+0.041$ & $1.844$ & $1.849$ & \cellcolor{negred!50}$-0.005$ & $1.858$ & $1.765$ & \cellcolor{posgreen!50}$+0.092$ & $1.001$ & $1.003$ & \cellcolor{negred!50}$-0.002$ & $1.301$ & $1.210$ & \cellcolor{posgreen!50}$+0.091$ \\
 &  & fpm & $1.451$ & $1.548$ & \cellcolor{negred!50}$-0.097$ & $1.378$ & $1.618$ & \cellcolor{negred!50}$-0.240$ & $1.819$ & $1.889$ & \cellcolor{negred!50}$-0.070$ & $1.050$ & $1.088$ & \cellcolor{negred!50}$-0.039$ & $1.531$ & $1.653$ & \cellcolor{negred!50}$-0.123$ \\
\cmidrule(lr){2-18}
 & eq & lam & $1.637$ & $1.608$ & \cellcolor{posgreen!50}$+0.029$ & $2.268$ & $2.264$ & \cellcolor{posgreen!50}$+0.004$ & $2.048$ & $2.087$ & \cellcolor{negred!50}$-0.038$ & $1.175$ & $0.945$ & \cellcolor{posgreen!50}$+0.230$ & $0.384$ & $0.406$ & \cellcolor{negred!50}$-0.022$ \\
 &  & raw & $1.673$ & $1.627$ & \cellcolor{posgreen!50}$+0.046$ & $2.316$ & $2.282$ & \cellcolor{posgreen!50}$+0.033$ & $2.069$ & $2.088$ & \cellcolor{negred!50}$-0.019$ & $1.196$ & $0.964$ & \cellcolor{posgreen!50}$+0.232$ & $0.384$ & $0.406$ & \cellcolor{negred!50}$-0.022$ \\
 &  & inv & $1.319$ & $1.309$ & \cellcolor{posgreen!50}$+0.010$ & $2.278$ & $2.291$ & \cellcolor{negred!50}$-0.013$ & $1.776$ & $1.685$ & \cellcolor{posgreen!50}$+0.091$ & $0.902$ & $0.873$ & \cellcolor{posgreen!50}$+0.029$ & $0.605$ & $0.622$ & \cellcolor{negred!50}$-0.016$ \\
 &  & vsp & $1.390$ & $1.406$ & \cellcolor{negred!50}$-0.016$ & $2.336$ & $2.329$ & \cellcolor{posgreen!50}$+0.006$ & $1.931$ & $1.804$ & \cellcolor{posgreen!50}$+0.127$ & $0.765$ & $0.879$ & \cellcolor{negred!50}$-0.115$ & $0.513$ & $0.591$ & \cellcolor{negred!50}$-0.078$ \\
 &  & fpm & $1.401$ & $1.556$ & \cellcolor{negred!50}$-0.156$ & $1.961$ & $2.166$ & \cellcolor{negred!50}$-0.205$ & $1.833$ & $2.046$ & \cellcolor{negred!50}$-0.213$ & $0.948$ & $1.106$ & \cellcolor{negred!50}$-0.158$ & $0.515$ & $0.682$ & \cellcolor{negred!50}$-0.167$ \\
\cmidrule(lr){2-18}
 & rp & lam & $1.624$ & $1.575$ & \cellcolor{posgreen!50}$+0.048$ & $2.383$ & $2.519$ & \cellcolor{negred!50}$-0.136$ & $2.036$ & $2.002$ & \cellcolor{posgreen!50}$+0.034$ & $1.095$ & $0.835$ & \cellcolor{posgreen!50}$+0.260$ & $0.342$ & $0.394$ & \cellcolor{negred!50}$-0.052$ \\
 &  & raw & $1.661$ & $1.592$ & \cellcolor{posgreen!50}$+0.068$ & $2.395$ & $2.526$ & \cellcolor{negred!50}$-0.131$ & $2.084$ & $1.993$ & \cellcolor{posgreen!50}$+0.091$ & $1.109$ & $0.850$ & \cellcolor{posgreen!50}$+0.259$ & $0.342$ & $0.394$ & \cellcolor{negred!50}$-0.052$ \\
 &  & inv & $1.307$ & $1.304$ & \cellcolor{posgreen!50}$+0.002$ & $2.294$ & $2.263$ & \cellcolor{posgreen!50}$+0.031$ & $1.757$ & $1.682$ & \cellcolor{posgreen!50}$+0.075$ & $0.860$ & $0.835$ & \cellcolor{posgreen!50}$+0.025$ & $0.512$ & $0.585$ & \cellcolor{negred!50}$-0.072$ \\
 &  & vsp & $1.407$ & $1.415$ & \cellcolor{negred!50}$-0.008$ & $2.368$ & $2.532$ & \cellcolor{negred!50}$-0.164$ & $1.947$ & $1.816$ & \cellcolor{posgreen!50}$+0.131$ & $0.789$ & $0.838$ & \cellcolor{negred!50}$-0.049$ & $0.448$ & $0.549$ & \cellcolor{negred!50}$-0.101$ \\
 &  & fpm & $1.420$ & $1.557$ & \cellcolor{negred!50}$-0.136$ & $2.069$ & $2.320$ & \cellcolor{negred!50}$-0.251$ & $1.835$ & $2.033$ & \cellcolor{negred!50}$-0.198$ & $0.951$ & $1.054$ & \cellcolor{negred!50}$-0.103$ & $0.427$ & $0.590$ & \cellcolor{negred!50}$-0.163$ \\
\midrule
rp & mcap & lam & $1.625$ & $1.407$ & \cellcolor{posgreen!50}$+0.218$ & $1.814$ & $1.237$ & \cellcolor{posgreen!50}$+0.577$ & $2.010$ & $2.034$ & \cellcolor{negred!50}$-0.023$ & $1.196$ & $0.883$ & \cellcolor{posgreen!50}$+0.313$ & $0.855$ & $0.802$ & \cellcolor{posgreen!50}$+0.053$ \\
 &  & raw & $1.683$ & $1.473$ & \cellcolor{posgreen!50}$+0.210$ & $1.817$ & $1.250$ & \cellcolor{posgreen!50}$+0.567$ & $2.101$ & $2.129$ & \cellcolor{negred!50}$-0.027$ & $1.213$ & $0.935$ & \cellcolor{posgreen!50}$+0.278$ & $0.855$ & $0.802$ & \cellcolor{posgreen!50}$+0.053$ \\
 &  & inv & $1.489$ & $1.305$ & \cellcolor{posgreen!50}$+0.184$ & $2.194$ & $1.083$ & \cellcolor{posgreen!50}$+1.112$ & $1.692$ & $1.724$ & \cellcolor{negred!50}$-0.031$ & $1.087$ & $0.981$ & \cellcolor{posgreen!50}$+0.106$ & $1.402$ & $1.266$ & \cellcolor{posgreen!50}$+0.136$ \\
 &  & vsp & $1.520$ & $1.371$ & \cellcolor{posgreen!50}$+0.149$ & $1.943$ & $1.215$ & \cellcolor{posgreen!50}$+0.728$ & $1.770$ & $1.765$ & \cellcolor{posgreen!50}$+0.005$ & $1.021$ & $1.021$ & $+0.000$ & $1.342$ & $1.256$ & \cellcolor{posgreen!50}$+0.086$ \\
 &  & fpm & $1.421$ & $1.360$ & \cellcolor{posgreen!50}$+0.061$ & $1.400$ & $1.026$ & \cellcolor{posgreen!50}$+0.374$ & $1.763$ & $1.869$ & \cellcolor{negred!50}$-0.106$ & $1.043$ & $1.100$ & \cellcolor{negred!50}$-0.057$ & $1.568$ & $1.627$ & \cellcolor{negred!50}$-0.060$ \\
\cmidrule(lr){2-18}
 & eq & lam & $1.647$ & $1.488$ & \cellcolor{posgreen!50}$+0.159$ & $2.335$ & $1.578$ & \cellcolor{posgreen!50}$+0.756$ & $1.985$ & $2.018$ & \cellcolor{negred!50}$-0.033$ & $1.287$ & $1.015$ & \cellcolor{posgreen!50}$+0.272$ & $0.443$ & $0.484$ & \cellcolor{negred!50}$-0.041$ \\
 &  & raw & $1.708$ & $1.530$ & \cellcolor{posgreen!50}$+0.178$ & $2.350$ & $1.605$ & \cellcolor{posgreen!50}$+0.746$ & $2.065$ & $2.047$ & \cellcolor{posgreen!50}$+0.018$ & $1.312$ & $1.045$ & \cellcolor{posgreen!50}$+0.268$ & $0.443$ & $0.484$ & \cellcolor{negred!50}$-0.041$ \\
 &  & inv & $1.316$ & $1.210$ & \cellcolor{posgreen!50}$+0.106$ & $2.447$ & $1.224$ & \cellcolor{posgreen!50}$+1.222$ & $1.656$ & $1.645$ & \cellcolor{posgreen!50}$+0.012$ & $0.934$ & $0.875$ & \cellcolor{posgreen!50}$+0.059$ & $0.614$ & $0.636$ & \cellcolor{negred!50}$-0.022$ \\
 &  & vsp & $1.360$ & $1.298$ & \cellcolor{posgreen!50}$+0.062$ & $2.436$ & $1.477$ & \cellcolor{posgreen!50}$+0.959$ & $1.807$ & $1.765$ & \cellcolor{posgreen!50}$+0.042$ & $0.755$ & $0.891$ & \cellcolor{negred!50}$-0.136$ & $0.513$ & $0.598$ & \cellcolor{negred!50}$-0.085$ \\
 &  & fpm & $1.394$ & $1.402$ & \cellcolor{negred!50}$-0.008$ & $2.102$ & $1.323$ & \cellcolor{posgreen!50}$+0.778$ & $1.815$ & $2.007$ & \cellcolor{negred!50}$-0.192$ & $0.929$ & $1.109$ & \cellcolor{negred!50}$-0.180$ & $0.484$ & $0.689$ & \cellcolor{negred!50}$-0.205$ \\
\cmidrule(lr){2-18}
 & rp & lam & $1.612$ & $1.484$ & \cellcolor{posgreen!50}$+0.127$ & $2.476$ & $1.708$ & \cellcolor{posgreen!50}$+0.768$ & $1.955$ & $1.979$ & \cellcolor{negred!50}$-0.024$ & $1.140$ & $0.902$ & \cellcolor{posgreen!50}$+0.238$ & $0.351$ & $0.447$ & \cellcolor{negred!50}$-0.096$ \\
 &  & raw & $1.672$ & $1.532$ & \cellcolor{posgreen!50}$+0.140$ & $2.501$ & $1.742$ & \cellcolor{posgreen!50}$+0.758$ & $2.045$ & $2.028$ & \cellcolor{posgreen!50}$+0.017$ & $1.157$ & $0.924$ & \cellcolor{posgreen!50}$+0.233$ & $0.351$ & $0.447$ & \cellcolor{negred!50}$-0.096$ \\
 &  & inv & $1.295$ & $1.222$ & \cellcolor{posgreen!50}$+0.073$ & $2.231$ & $1.310$ & \cellcolor{posgreen!50}$+0.921$ & $1.673$ & $1.661$ & \cellcolor{posgreen!50}$+0.011$ & $0.883$ & $0.846$ & \cellcolor{posgreen!50}$+0.037$ & $0.533$ & $0.608$ & \cellcolor{negred!50}$-0.075$ \\
 &  & vsp & $1.363$ & $1.334$ & \cellcolor{posgreen!50}$+0.029$ & $2.458$ & $1.623$ & \cellcolor{posgreen!50}$+0.835$ & $1.820$ & $1.779$ & \cellcolor{posgreen!50}$+0.041$ & $0.746$ & $0.865$ & \cellcolor{negred!50}$-0.119$ & $0.452$ & $0.565$ & \cellcolor{negred!50}$-0.113$ \\
 &  & fpm & $1.420$ & $1.434$ & \cellcolor{negred!50}$-0.014$ & $2.040$ & $1.415$ & \cellcolor{posgreen!50}$+0.626$ & $1.840$ & $2.030$ & \cellcolor{negred!50}$-0.190$ & $0.930$ & $1.089$ & \cellcolor{negred!50}$-0.159$ & $0.393$ & $0.584$ & \cellcolor{negred!50}$-0.191$ \\
\bottomrule
\end{tabular}}
\end{table}

% NOTE: 표가 페이지 폭을 넘으면 paper3.md 프리앰블에 \usepackage{pdflscape}를 추가하고
%       \begin{landscape} ... \end{landscape}로 표 전체를 감싸 가로 회전하시기 바랍니다.
\begin{table}[!htbp]
\centering
\caption{전체 45 슬롯 최대낙폭 (MDD) 결과 ($\Omega=$he 고정). $p_w \times \text{prior} \times q$ = $3 \times 3 \times 5$ = 45 슬롯 각각의 LSTM(L)·ANN(A) 최대낙폭 (MDD) 및 차이 $\Delta = \mathrm{L} - \mathrm{A}$를 5개 구간(All, R1 회복, R2 확장, R3 위기, R4 AI 랠리)에 대해 보고한다. MDD는 음수가 손실이므로 $\Delta > 0$이 LSTM이 덜 빠진 경우이다.}
\label{tab:app_grid_mdd_he}
\setlength{\tabcolsep}{2pt}
\renewcommand{\arraystretch}{1.05}
\scriptsize
\resizebox{\textwidth}{!}{%
\begin{tabular}{lll ccc ccc ccc ccc ccc}
\toprule
 & & & \multicolumn{3}{c}{All} & \multicolumn{3}{c}{R1} & \multicolumn{3}{c}{R2} & \multicolumn{3}{c}{R3} & \multicolumn{3}{c}{R4} \\
\cmidrule(lr){4-6} \cmidrule(lr){7-9} \cmidrule(lr){10-12} \cmidrule(lr){13-15} \cmidrule(lr){16-18}
$p_w$ & prior & $q$ & L & A & $\Delta$ & L & A & $\Delta$ & L & A & $\Delta$ & L & A & $\Delta$ & L & A & $\Delta$ \\
\midrule
mcap & mcap & lam & $-11.65\%$ & $-13.81\%$ & \cellcolor{posgreen!50}$+2.16\%$ & $-10.74\%$ & $-10.37\%$ & \cellcolor{negred!50}$-0.38\%$ & $-11.65\%$ & $-11.33\%$ & \cellcolor{negred!50}$-0.32\%$ & $-10.05\%$ & $-13.81\%$ & \cellcolor{posgreen!50}$+3.77\%$ & $-4.91\%$ & $-5.91\%$ & \cellcolor{posgreen!50}$+1.00\%$ \\
 &  & raw & $-13.69\%$ & $-13.81\%$ & \cellcolor{posgreen!50}$+0.12\%$ & $-10.79\%$ & $-10.45\%$ & \cellcolor{negred!50}$-0.34\%$ & $-13.69\%$ & $-13.45\%$ & \cellcolor{negred!50}$-0.24\%$ & $-10.05\%$ & $-13.81\%$ & \cellcolor{posgreen!50}$+3.77\%$ & $-4.91\%$ & $-5.91\%$ & \cellcolor{posgreen!50}$+1.00\%$ \\
 &  & inv & $-14.76\%$ & $-15.37\%$ & \cellcolor{posgreen!50}$+0.61\%$ & $-7.43\%$ & $-7.13\%$ & \cellcolor{negred!50}$-0.30\%$ & $-11.76\%$ & $-12.87\%$ & \cellcolor{posgreen!50}$+1.11\%$ & $-13.39\%$ & $-15.37\%$ & \cellcolor{posgreen!50}$+1.98\%$ & $-5.31\%$ & $-7.17\%$ & \cellcolor{posgreen!50}$+1.86\%$ \\
 &  & vsp & $-12.73\%$ & $-14.35\%$ & \cellcolor{posgreen!50}$+1.63\%$ & $-9.54\%$ & $-9.26\%$ & \cellcolor{negred!50}$-0.28\%$ & $-9.46\%$ & $-11.42\%$ & \cellcolor{posgreen!50}$+1.96\%$ & $-12.48\%$ & $-14.35\%$ & \cellcolor{posgreen!50}$+1.87\%$ & $-5.19\%$ & $-7.19\%$ & \cellcolor{posgreen!50}$+2.01\%$ \\
 &  & fpm & $-19.84\%$ & $-18.91\%$ & \cellcolor{negred!50}$-0.92\%$ & $-11.90\%$ & $-10.47\%$ & \cellcolor{negred!50}$-1.43\%$ & $-16.60\%$ & $-12.84\%$ & \cellcolor{negred!50}$-3.76\%$ & $-19.84\%$ & $-15.17\%$ & \cellcolor{negred!50}$-4.66\%$ & $-18.74\%$ & $-18.91\%$ & \cellcolor{posgreen!50}$+0.17\%$ \\
\cmidrule(lr){2-18}
 & eq & lam & $-12.18\%$ & $-13.01\%$ & \cellcolor{posgreen!50}$+0.83\%$ & $-12.18\%$ & $-11.51\%$ & \cellcolor{negred!50}$-0.66\%$ & $-11.50\%$ & $-12.09\%$ & \cellcolor{posgreen!50}$+0.59\%$ & $-9.79\%$ & $-13.01\%$ & \cellcolor{posgreen!50}$+3.22\%$ & $-6.83\%$ & $-7.18\%$ & \cellcolor{posgreen!50}$+0.35\%$ \\
 &  & raw & $-14.19\%$ & $-14.05\%$ & \cellcolor{negred!50}$-0.14\%$ & $-12.18\%$ & $-11.51\%$ & \cellcolor{negred!50}$-0.66\%$ & $-14.19\%$ & $-14.05\%$ & \cellcolor{negred!50}$-0.14\%$ & $-9.79\%$ & $-13.01\%$ & \cellcolor{posgreen!50}$+3.22\%$ & $-6.83\%$ & $-7.18\%$ & \cellcolor{posgreen!50}$+0.35\%$ \\
 &  & inv & $-18.88\%$ & $-17.89\%$ & \cellcolor{negred!50}$-0.98\%$ & $-7.57\%$ & $-6.31\%$ & \cellcolor{negred!50}$-1.26\%$ & $-12.18\%$ & $-12.44\%$ & \cellcolor{posgreen!50}$+0.26\%$ & $-13.28\%$ & $-14.77\%$ & \cellcolor{posgreen!50}$+1.49\%$ & $-8.64\%$ & $-9.33\%$ & \cellcolor{posgreen!50}$+0.69\%$ \\
 &  & vsp & $-15.65\%$ & $-15.12\%$ & \cellcolor{negred!50}$-0.52\%$ & $-10.31\%$ & $-10.00\%$ & \cellcolor{negred!50}$-0.31\%$ & $-9.79\%$ & $-11.03\%$ & \cellcolor{posgreen!50}$+1.24\%$ & $-11.97\%$ & $-14.35\%$ & \cellcolor{posgreen!50}$+2.38\%$ & $-8.29\%$ & $-9.19\%$ & \cellcolor{posgreen!50}$+0.89\%$ \\
 &  & fpm & $-24.34\%$ & $-19.65\%$ & \cellcolor{negred!50}$-4.69\%$ & $-12.73\%$ & $-10.07\%$ & \cellcolor{negred!50}$-2.66\%$ & $-17.61\%$ & $-12.58\%$ & \cellcolor{negred!50}$-5.04\%$ & $-19.41\%$ & $-14.22\%$ & \cellcolor{negred!50}$-5.20\%$ & $-20.34\%$ & $-19.65\%$ & \cellcolor{negred!50}$-0.69\%$ \\
\cmidrule(lr){2-18}
 & rp & lam & $-12.56\%$ & $-12.65\%$ & \cellcolor{posgreen!50}$+0.09\%$ & $-10.44\%$ & $-9.77\%$ & \cellcolor{negred!50}$-0.67\%$ & $-11.70\%$ & $-12.65\%$ & \cellcolor{posgreen!50}$+0.95\%$ & $-10.15\%$ & $-12.24\%$ & \cellcolor{posgreen!50}$+2.09\%$ & $-6.81\%$ & $-6.84\%$ & $+0.03\%$ \\
 &  & raw & $-14.37\%$ & $-13.81\%$ & \cellcolor{negred!50}$-0.56\%$ & $-10.44\%$ & $-9.77\%$ & \cellcolor{negred!50}$-0.67\%$ & $-14.37\%$ & $-13.81\%$ & \cellcolor{negred!50}$-0.56\%$ & $-10.15\%$ & $-12.24\%$ & \cellcolor{posgreen!50}$+2.09\%$ & $-6.81\%$ & $-6.84\%$ & $+0.03\%$ \\
 &  & inv & $-17.57\%$ & $-16.73\%$ & \cellcolor{negred!50}$-0.84\%$ & $-6.56\%$ & $-5.56\%$ & \cellcolor{negred!50}$-1.00\%$ & $-11.78\%$ & $-11.81\%$ & $+0.04\%$ & $-12.56\%$ & $-14.13\%$ & \cellcolor{posgreen!50}$+1.57\%$ & $-7.58\%$ & $-8.31\%$ & \cellcolor{posgreen!50}$+0.73\%$ \\
 &  & vsp & $-14.51\%$ & $-14.19\%$ & \cellcolor{negred!50}$-0.33\%$ & $-8.69\%$ & $-8.24\%$ & \cellcolor{negred!50}$-0.45\%$ & $-9.46\%$ & $-10.57\%$ & \cellcolor{posgreen!50}$+1.11\%$ & $-11.40\%$ & $-13.84\%$ & \cellcolor{posgreen!50}$+2.44\%$ & $-7.40\%$ & $-8.24\%$ & \cellcolor{posgreen!50}$+0.84\%$ \\
 &  & fpm & $-22.39\%$ & $-19.01\%$ & \cellcolor{negred!50}$-3.38\%$ & $-10.85\%$ & $-8.38\%$ & \cellcolor{negred!50}$-2.47\%$ & $-16.89\%$ & $-11.93\%$ & \cellcolor{negred!50}$-4.97\%$ & $-17.57\%$ & $-13.43\%$ & \cellcolor{negred!50}$-4.14\%$ & $-19.62\%$ & $-19.01\%$ & \cellcolor{negred!50}$-0.60\%$ \\
\midrule
eq & mcap & lam & $-12.82\%$ & $-14.16\%$ & \cellcolor{posgreen!50}$+1.34\%$ & $-9.97\%$ & $-10.51\%$ & \cellcolor{posgreen!50}$+0.53\%$ & $-12.82\%$ & $-13.37\%$ & \cellcolor{posgreen!50}$+0.55\%$ & $-10.99\%$ & $-14.16\%$ & \cellcolor{posgreen!50}$+3.17\%$ & $-5.35\%$ & $-5.12\%$ & \cellcolor{negred!50}$-0.23\%$ \\
 &  & raw & $-15.16\%$ & $-15.46\%$ & \cellcolor{posgreen!50}$+0.30\%$ & $-9.97\%$ & $-10.51\%$ & \cellcolor{posgreen!50}$+0.53\%$ & $-15.16\%$ & $-15.46\%$ & \cellcolor{posgreen!50}$+0.30\%$ & $-10.99\%$ & $-14.16\%$ & \cellcolor{posgreen!50}$+3.17\%$ & $-5.35\%$ & $-5.12\%$ & \cellcolor{negred!50}$-0.23\%$ \\
 &  & inv & $-14.46\%$ & $-16.41\%$ & \cellcolor{posgreen!50}$+1.95\%$ & $-5.95\%$ & $-6.25\%$ & \cellcolor{posgreen!50}$+0.30\%$ & $-10.81\%$ & $-11.45\%$ & \cellcolor{posgreen!50}$+0.63\%$ & $-14.46\%$ & $-16.41\%$ & \cellcolor{posgreen!50}$+1.95\%$ & $-5.23\%$ & $-4.79\%$ & \cellcolor{negred!50}$-0.44\%$ \\
 &  & vsp & $-12.64\%$ & $-15.47\%$ & \cellcolor{posgreen!50}$+2.83\%$ & $-8.56\%$ & $-8.84\%$ & \cellcolor{posgreen!50}$+0.28\%$ & $-8.70\%$ & $-9.39\%$ & \cellcolor{posgreen!50}$+0.69\%$ & $-12.64\%$ & $-15.47\%$ & \cellcolor{posgreen!50}$+2.83\%$ & $-5.37\%$ & $-4.94\%$ & \cellcolor{negred!50}$-0.43\%$ \\
 &  & fpm & $-16.01\%$ & $-15.84\%$ & \cellcolor{negred!50}$-0.16\%$ & $-13.82\%$ & $-11.96\%$ & \cellcolor{negred!50}$-1.85\%$ & $-10.22\%$ & $-8.53\%$ & \cellcolor{negred!50}$-1.69\%$ & $-16.01\%$ & $-15.84\%$ & \cellcolor{negred!50}$-0.16\%$ & $-7.44\%$ & $-7.60\%$ & \cellcolor{posgreen!50}$+0.16\%$ \\
\cmidrule(lr){2-18}
 & eq & lam & $-13.31\%$ & $-13.52\%$ & \cellcolor{posgreen!50}$+0.20\%$ & $-10.36\%$ & $-10.46\%$ & \cellcolor{posgreen!50}$+0.10\%$ & $-13.31\%$ & $-13.52\%$ & \cellcolor{posgreen!50}$+0.20\%$ & $-10.46\%$ & $-12.41\%$ & \cellcolor{posgreen!50}$+1.95\%$ & $-7.06\%$ & $-6.78\%$ & \cellcolor{negred!50}$-0.28\%$ \\
 &  & raw & $-15.11\%$ & $-15.54\%$ & \cellcolor{posgreen!50}$+0.44\%$ & $-10.36\%$ & $-10.46\%$ & \cellcolor{posgreen!50}$+0.10\%$ & $-15.11\%$ & $-15.54\%$ & \cellcolor{posgreen!50}$+0.44\%$ & $-10.46\%$ & $-12.41\%$ & \cellcolor{posgreen!50}$+1.95\%$ & $-7.06\%$ & $-6.78\%$ & \cellcolor{negred!50}$-0.28\%$ \\
 &  & inv & $-16.87\%$ & $-16.79\%$ & \cellcolor{negred!50}$-0.08\%$ & $-5.80\%$ & $-5.79\%$ & $-0.01\%$ & $-10.57\%$ & $-11.07\%$ & \cellcolor{posgreen!50}$+0.51\%$ & $-13.23\%$ & $-14.15\%$ & \cellcolor{posgreen!50}$+0.92\%$ & $-7.37\%$ & $-6.81\%$ & \cellcolor{negred!50}$-0.55\%$ \\
 &  & vsp & $-14.20\%$ & $-14.57\%$ & \cellcolor{posgreen!50}$+0.36\%$ & $-8.24\%$ & $-8.81\%$ & \cellcolor{posgreen!50}$+0.57\%$ & $-8.59\%$ & $-9.19\%$ & \cellcolor{posgreen!50}$+0.60\%$ & $-12.71\%$ & $-13.75\%$ & \cellcolor{posgreen!50}$+1.04\%$ & $-7.59\%$ & $-7.07\%$ & \cellcolor{negred!50}$-0.52\%$ \\
 &  & fpm & $-14.26\%$ & $-13.59\%$ & \cellcolor{negred!50}$-0.68\%$ & $-12.43\%$ & $-11.48\%$ & \cellcolor{negred!50}$-0.95\%$ & $-9.84\%$ & $-8.23\%$ & \cellcolor{negred!50}$-1.61\%$ & $-13.84\%$ & $-13.59\%$ & \cellcolor{negred!50}$-0.25\%$ & $-9.03\%$ & $-9.18\%$ & \cellcolor{posgreen!50}$+0.15\%$ \\
\cmidrule(lr){2-18}
 & rp & lam & $-13.34\%$ & $-13.51\%$ & \cellcolor{posgreen!50}$+0.17\%$ & $-8.96\%$ & $-8.82\%$ & \cellcolor{negred!50}$-0.14\%$ & $-13.34\%$ & $-13.51\%$ & \cellcolor{posgreen!50}$+0.17\%$ & $-9.81\%$ & $-11.88\%$ & \cellcolor{posgreen!50}$+2.07\%$ & $-6.76\%$ & $-6.75\%$ & $-0.02\%$ \\
 &  & raw & $-15.07\%$ & $-15.53\%$ & \cellcolor{posgreen!50}$+0.46\%$ & $-8.96\%$ & $-8.82\%$ & \cellcolor{negred!50}$-0.14\%$ & $-15.07\%$ & $-15.53\%$ & \cellcolor{posgreen!50}$+0.46\%$ & $-9.81\%$ & $-11.88\%$ & \cellcolor{posgreen!50}$+2.07\%$ & $-6.76\%$ & $-6.75\%$ & $-0.02\%$ \\
 &  & inv & $-16.28\%$ & $-16.10\%$ & \cellcolor{negred!50}$-0.18\%$ & $-5.46\%$ & $-5.62\%$ & \cellcolor{posgreen!50}$+0.15\%$ & $-10.16\%$ & $-10.61\%$ & \cellcolor{posgreen!50}$+0.45\%$ & $-12.86\%$ & $-13.45\%$ & \cellcolor{posgreen!50}$+0.60\%$ & $-6.90\%$ & $-6.68\%$ & \cellcolor{negred!50}$-0.23\%$ \\
 &  & vsp & $-13.47\%$ & $-13.90\%$ & \cellcolor{posgreen!50}$+0.43\%$ & $-7.02\%$ & $-7.11\%$ & \cellcolor{posgreen!50}$+0.09\%$ & $-8.30\%$ & $-8.87\%$ & \cellcolor{posgreen!50}$+0.57\%$ & $-12.03\%$ & $-13.04\%$ & \cellcolor{posgreen!50}$+1.02\%$ & $-7.06\%$ & $-6.67\%$ & \cellcolor{negred!50}$-0.40\%$ \\
 &  & fpm & $-14.32\%$ & $-12.87\%$ & \cellcolor{negred!50}$-1.45\%$ & $-11.06\%$ & $-9.91\%$ & \cellcolor{negred!50}$-1.15\%$ & $-9.48\%$ & $-8.15\%$ & \cellcolor{negred!50}$-1.33\%$ & $-13.39\%$ & $-12.87\%$ & \cellcolor{negred!50}$-0.51\%$ & $-8.51\%$ & $-8.61\%$ & \cellcolor{posgreen!50}$+0.10\%$ \\
\midrule
rp & mcap & lam & $-11.87\%$ & $-14.19\%$ & \cellcolor{posgreen!50}$+2.32\%$ & $-11.87\%$ & $-14.19\%$ & \cellcolor{posgreen!50}$+2.32\%$ & $-11.64\%$ & $-11.84\%$ & \cellcolor{posgreen!50}$+0.20\%$ & $-11.06\%$ & $-13.94\%$ & \cellcolor{posgreen!50}$+2.87\%$ & $-5.25\%$ & $-4.96\%$ & \cellcolor{negred!50}$-0.29\%$ \\
 &  & raw & $-13.78\%$ & $-14.68\%$ & \cellcolor{posgreen!50}$+0.90\%$ & $-11.87\%$ & $-14.19\%$ & \cellcolor{posgreen!50}$+2.32\%$ & $-13.78\%$ & $-14.68\%$ & \cellcolor{posgreen!50}$+0.90\%$ & $-11.06\%$ & $-13.94\%$ & \cellcolor{posgreen!50}$+2.87\%$ & $-5.25\%$ & $-4.96\%$ & \cellcolor{negred!50}$-0.29\%$ \\
 &  & inv & $-14.61\%$ & $-16.25\%$ & \cellcolor{posgreen!50}$+1.64\%$ & $-6.36\%$ & $-13.99\%$ & \cellcolor{posgreen!50}$+7.63\%$ & $-11.85\%$ & $-12.03\%$ & \cellcolor{posgreen!50}$+0.18\%$ & $-14.61\%$ & $-16.25\%$ & \cellcolor{posgreen!50}$+1.64\%$ & $-5.12\%$ & $-4.84\%$ & \cellcolor{negred!50}$-0.28\%$ \\
 &  & vsp & $-12.88\%$ & $-15.39\%$ & \cellcolor{posgreen!50}$+2.51\%$ & $-9.51\%$ & $-14.09\%$ & \cellcolor{posgreen!50}$+4.58\%$ & $-9.18\%$ & $-9.63\%$ & \cellcolor{posgreen!50}$+0.45\%$ & $-12.88\%$ & $-15.39\%$ & \cellcolor{posgreen!50}$+2.51\%$ & $-5.19\%$ & $-4.93\%$ & \cellcolor{negred!50}$-0.26\%$ \\
 &  & fpm & $-16.24\%$ & $-15.66\%$ & \cellcolor{negred!50}$-0.58\%$ & $-16.24\%$ & $-13.81\%$ & \cellcolor{negred!50}$-2.43\%$ & $-10.49\%$ & $-8.54\%$ & \cellcolor{negred!50}$-1.95\%$ & $-15.93\%$ & $-15.66\%$ & \cellcolor{negred!50}$-0.27\%$ & $-7.60\%$ & $-7.51\%$ & \cellcolor{negred!50}$-0.09\%$ \\
\cmidrule(lr){2-18}
 & eq & lam & $-12.42\%$ & $-13.57\%$ & \cellcolor{posgreen!50}$+1.15\%$ & $-11.30\%$ & $-13.57\%$ & \cellcolor{posgreen!50}$+2.27\%$ & $-12.39\%$ & $-12.11\%$ & \cellcolor{negred!50}$-0.28\%$ & $-10.43\%$ & $-12.10\%$ & \cellcolor{posgreen!50}$+1.66\%$ & $-6.96\%$ & $-6.82\%$ & \cellcolor{negred!50}$-0.14\%$ \\
 &  & raw & $-14.16\%$ & $-14.63\%$ & \cellcolor{posgreen!50}$+0.47\%$ & $-11.30\%$ & $-13.57\%$ & \cellcolor{posgreen!50}$+2.26\%$ & $-14.16\%$ & $-14.63\%$ & \cellcolor{posgreen!50}$+0.47\%$ & $-10.43\%$ & $-12.10\%$ & \cellcolor{posgreen!50}$+1.66\%$ & $-6.96\%$ & $-6.82\%$ & \cellcolor{negred!50}$-0.14\%$ \\
 &  & inv & $-16.99\%$ & $-17.01\%$ & $+0.03\%$ & $-6.23\%$ & $-13.39\%$ & \cellcolor{posgreen!50}$+7.16\%$ & $-11.78\%$ & $-11.84\%$ & \cellcolor{posgreen!50}$+0.05\%$ & $-13.17\%$ & $-14.01\%$ & \cellcolor{posgreen!50}$+0.84\%$ & $-7.32\%$ & $-6.87\%$ & \cellcolor{negred!50}$-0.45\%$ \\
 &  & vsp & $-14.68\%$ & $-14.93\%$ & \cellcolor{posgreen!50}$+0.25\%$ & $-9.30\%$ & $-13.48\%$ & \cellcolor{posgreen!50}$+4.19\%$ & $-9.11\%$ & $-9.33\%$ & \cellcolor{posgreen!50}$+0.22\%$ & $-12.59\%$ & $-13.56\%$ & \cellcolor{posgreen!50}$+0.97\%$ & $-7.54\%$ & $-7.13\%$ & \cellcolor{negred!50}$-0.41\%$ \\
 &  & fpm & $-14.51\%$ & $-13.49\%$ & \cellcolor{negred!50}$-1.02\%$ & $-14.51\%$ & $-13.07\%$ & \cellcolor{negred!50}$-1.44\%$ & $-9.97\%$ & $-8.19\%$ & \cellcolor{negred!50}$-1.78\%$ & $-13.66\%$ & $-13.49\%$ & \cellcolor{negred!50}$-0.17\%$ & $-9.38\%$ & $-9.16\%$ & \cellcolor{negred!50}$-0.22\%$ \\
\cmidrule(lr){2-18}
 & rp & lam & $-13.11\%$ & $-12.18\%$ & \cellcolor{negred!50}$-0.94\%$ & $-9.58\%$ & $-12.13\%$ & \cellcolor{posgreen!50}$+2.55\%$ & $-12.57\%$ & $-12.18\%$ & \cellcolor{negred!50}$-0.39\%$ & $-9.90\%$ & $-11.90\%$ & \cellcolor{posgreen!50}$+1.99\%$ & $-6.77\%$ & $-6.75\%$ & $-0.03\%$ \\
 &  & raw & $-14.24\%$ & $-14.56\%$ & \cellcolor{posgreen!50}$+0.32\%$ & $-9.58\%$ & $-12.12\%$ & \cellcolor{posgreen!50}$+2.54\%$ & $-14.24\%$ & $-14.56\%$ & \cellcolor{posgreen!50}$+0.32\%$ & $-9.90\%$ & $-11.90\%$ & \cellcolor{posgreen!50}$+1.99\%$ & $-6.77\%$ & $-6.75\%$ & $-0.03\%$ \\
 &  & inv & $-16.41\%$ & $-16.26\%$ & \cellcolor{negred!50}$-0.14\%$ & $-5.57\%$ & $-11.94\%$ & \cellcolor{posgreen!50}$+6.36\%$ & $-11.10\%$ & $-11.21\%$ & \cellcolor{posgreen!50}$+0.11\%$ & $-12.81\%$ & $-13.36\%$ & \cellcolor{posgreen!50}$+0.55\%$ & $-6.83\%$ & $-6.73\%$ & \cellcolor{negred!50}$-0.10\%$ \\
 &  & vsp & $-13.91\%$ & $-14.03\%$ & \cellcolor{posgreen!50}$+0.13\%$ & $-7.82\%$ & $-12.03\%$ & \cellcolor{posgreen!50}$+4.21\%$ & $-8.82\%$ & $-9.03\%$ & \cellcolor{posgreen!50}$+0.21\%$ & $-12.05\%$ & $-12.98\%$ & \cellcolor{posgreen!50}$+0.94\%$ & $-7.00\%$ & $-6.72\%$ & \cellcolor{negred!50}$-0.28\%$ \\
 &  & fpm & $-13.90\%$ & $-12.70\%$ & \cellcolor{negred!50}$-1.21\%$ & $-12.72\%$ & $-11.34\%$ & \cellcolor{negred!50}$-1.38\%$ & $-9.55\%$ & $-7.99\%$ & \cellcolor{negred!50}$-1.56\%$ & $-13.19\%$ & $-12.70\%$ & \cellcolor{negred!50}$-0.50\%$ & $-8.83\%$ & $-8.60\%$ & \cellcolor{negred!50}$-0.23\%$ \\
\bottomrule
\end{tabular}}
\end{table}

\section{R1 회복 구간 손실의 시점 집중}

본 부록은 §5.2 (3)에서 인용한 R1 구간 손실의 시점 집중성(76\%, 40\%) 
및 §4.3 Table 5에서 보고한 $\mathbf{p}$ 의존적 비대칭의 정량적 근거를 
제공한다. 본 부록의 모든 수치는 결과론적 관찰에 머무르며, R1 LSTM 
열위의 원인에 대한 인과적 규명은 시도하지 않는다.

\subsection{R1 구간 $p_w$별 LSTM-ANN 누적 $ΔReturn$}

본문 §4.3 Table 5에서 Sharpe 비율 기준 $p_w$별 LSTM 우위 슬롯 수를 
보고하였다. 본 부록은 보조 지표로 누적 수익률 차이를 함께 제시한다
(각 $p_w$ 15 슬롯 평균, R1 30개월 누적).

\begin{table}[ht]
\centering
\caption{R1 회복 구간 P-matrix별 LSTM-vs-ANN 격차 — 평균 $\Delta$ Sharpe, 우위 슬롯 수,
누적 $\Delta$Return 세 지표. 각 $p_w$ 값에 대해 (Q $\times$ prior) 15개 슬롯 평균.}
\label{tab:r1_pmatrix_delta}
\setlength{\tabcolsep}{8pt}
\renewcommand{\arraystretch}{1.2}
\small
\begin{tabular}{lccc}
\toprule
$p_w$ & 평균 $\Delta$ Sharpe (LSTM$-$ANN) & LSTM 우위 슬롯 (Sharpe, /15) & R1 누적 $\Delta$Return \\
\midrule
mcap & $-0.214$ & 0 / 15  & $-7.75\%$ \\
eq   & $-0.046$ & 1 / 15  & $-1.74\%$ \\
rp   & $+0.081$ & 14 / 15 & $+0.19\%$ \\
\bottomrule
\end{tabular}
\par\smallskip
\footnotesize
\raggedright
$\Delta$Return = (LSTM 누적 수익) $-$ (ANN 누적 수익), 각 $p_w$ 15 슬롯 평균.
\end{table}

R1 LSTM 열위는 $p_w=mcap$ 조건에서 가장 두드러지며 ($-7.75\%$), $p_w=eq$에서 격차가 큰 폭 완화되고 ($-1.74\%$), $p_w=rp$에서는 사실상 사라진다 ($+0.19\%$).

\subsection{mcap 누적 손실의 시점 집중성}

$p_w=mcap$ 누적 손실 −7.75\%의 월별 분포를 분석하면, 약 76\%가 2011년 
3·4·8월의 세 시점에 집중되어 있다.

\begin{table}[ht]
\centering
\caption{$p_w=mcap$ 누적 손실의 시점 집중성 — R1 30개월 중 2011년 3·4·8월 세 시점이 누적 손실의 약 76\%를 차지한다.}
\label{tab:r1_loss_concentration}
\setlength{\tabcolsep}{8pt}
\renewcommand{\arraystretch}{1.2}
\small
\begin{tabular}{llrr}
\toprule
시점 & 시장 이벤트 & $\Delta$Return (mcap) & 기여도 \\
\midrule
2011-03 & Tohoku 지진 + 일본 시장 충격     & $-1.19$\%p & 15.4\% \\
2011-04 & 유럽 채무 재발 + 신용 경색 우려   & $-1.65$\%p & 21.3\% \\
2011-08 & 유로존 위기 + 미국 신용등급 강등  & $-3.08$\%p & 39.7\% \\
\midrule
위 3개 합계   & --                              & $-5.92$\%p & 76.4\% \\
나머지 27개월 & --                              & $-1.83$\%p & 23.6\% \\
\midrule
R1 누적 합계  & --                              & $-7.75$\%p & 100.0\% \\
\bottomrule
\end{tabular}
\end{table}

특히 2011-08 단일 월이 R1 누적 손실의 약 40\%를 견인한다. R1 30개월
이라는 표본 크기와 누적 손실이 소수 시점에 집중되는 분포 특성을 
고려할 때, 본 결과를 "LSTM 기반 BL은 회복 구간에서 일관되게 약하다" 
와 같은 일반화로 확장하는 것은 신중해야 한다.

\section{Q=fpm 부호 뒤집힘과 $\mathbf{p}$ 벡터의 완화 효과}
% 많이 수정 필요

본 부록은 §\ref{result}에서 언급한 "\citet{pyo2018}가 사용한 Fama-French 회귀 기반 $q^{\mathrm{fpm}}$ 이 시장 상황에 따라 부호가 뒤집히면서 저변동 전략의 본질이 약화될 수 있다"는 관찰의 정량적 근거를 제공한다. 본 부록의 모든 분석은 사후적 관찰에 머무르며, 인과적 메커니즘의 규명은 본 연구의 범위를 벗어난다.

\subsection{앵커 슬롯의 레짐별 비중 추이}
\citet{pyo2018}가 제안한 앵커 구성의 저변동·고변동 종목 비중을 4개 레짐에 걸쳐 집계하면 다음과 같다.

\begin{table}[ht]
\centering
\caption{Anchor 슬롯(mcap-mcap-fpm-err)의 레짐별 저변동·고변동 비중 추이.
저변동/고변동은 변동성 예측값 하위 30\% / 상위 30\%로 정의하며, 각 셀은 월별 평균.}
\label{tab:r4_anchor_weights}
\setlength{\tabcolsep}{8pt}
\renewcommand{\arraystretch}{1.2}
\small
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{3}{c}{LSTM} & \multicolumn{3}{c}{ANN} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7}
Regime & 저변동 & 고변동 & 스프레드 & 저변동 & 고변동 & 스프레드 \\
\midrule
R1 회복   & 0.556 & 0.087 & $+0.469$          & 0.598 & 0.082 & $+0.516$          \\
R2 확장   & 0.431 & 0.205 & $+0.226$          & 0.453 & 0.181 & $+0.272$          \\
R3 위기   & 0.403 & 0.212 & $+0.190$          & 0.420 & 0.198 & $+0.222$          \\
R4 정상화 & 0.250 & 0.395 & $\mathbf{-0.145}$ & 0.263 & 0.366 & $\mathbf{-0.103}$ \\
\bottomrule
\end{tabular}
\end{table}

R1에서 R3까지는 저변동 비중이 고변동 비중보다 큰 저변동 전략이 유지되었으나, R4에서 LSTM·ANN 모두 변동성 스프레드가 음수로 전환되어 저변동 전략의 본질이 사실상 뒤집힌 상태로 운용되었다.

\subsection{$q^{\mathrm{fpm}}$의 레짐별 부호 분포}

앵커 슬롯에서 $q_{\mathrm{fpm}}$의 월별 평균값과 음수 비율을 레짐별로 집계한 결과는 다음과 같다.

\begin{table}[ht]
\centering
\caption{Anchor 슬롯(mcap-mcap-fpm-err)의 $q^{\mathrm{fpm}}$ 레짐별 부호 분포 — 
R4에서 평균값과 음수 빈도가 모두 극단적으로 증가.}
\label{tab:r4_q_sign}
\setlength{\tabcolsep}{8pt}
\renewcommand{\arraystretch}{1.2}
\small
\begin{tabular}{lcccc}
\toprule
& \multicolumn{2}{c}{LSTM} & \multicolumn{2}{c}{ANN} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
Regime & mean$(q)$ & \% $q<0$ & mean$(q)$ & \% $q<0$ \\
\midrule
R1 회복   & $+0.0001$          & 43.3\%            & $+0.0022$          & 3.3\%             \\
R2 확장   & $-0.0010$          & 61.1\%            & $-0.0001$          & 44.4\%            \\
R3 위기   & $-0.0009$          & 69.0\%            & $-0.0000$          & 40.5\%            \\
R4 정상화 & $\mathbf{-0.0100}$ & $\mathbf{80.0}$\% & $\mathbf{-0.0099}$ & $\mathbf{90.0}$\% \\
\bottomrule
\end{tabular}
\end{table}

R4에서 $q^{\mathrm{fpm}}$의 평균값은 다른 국면 대비 높은 음수 값을 보이며, 음수 발생 빈도도 80–90\% 수준에 도달한다. Black-Litterman 사후 추정에서 $P \times q$항의 부호는 $q$의 부호에 의해 결정되므로, R4에서는 저변동(P의 양의 가중)·고변동(P의 음의 가중)에 대한 뷰의 방향이 반전되어 고변동 long·저변동 short로 작동한다.

\begin{figure}[ht]
\centering
\includegraphics[width=0.95\textwidth]{Figure/r4_fpm_lam.png}
\caption{$q^{\mathrm{fpm}}$(파란 원, FF3 OLS 기반)과 $q^{\mathrm{lam}}$(초록 사각, $\sigma$-direct 기반)의 월별 시계열 — 앵커 슬롯 mcap-mcap-fpm-err 기준. 빨간 점선은 부호 반전 임계선(0), 음영은 4개 레짐. $q^{\mathrm{lam}}$은 전 기간 양수를 유지하나, $q^{\mathrm{fpm}}$은 빈번한 부호 반전 후 R4에서 강한 음수로 수렴한다.}
\label{fig:q_timeline}
\end{figure}

그림 \ref{fig:q_timeline}은 \citet{pyo2018}가 제안한 앵커 슬롯의 $q^{\mathrm{fpm}}$ 과 동일 앵커에서 $q$만 $q^{\mathrm{lam}}$ 으로 교체한 경우의 값을 시각화한 것이다. FF3 기반의 $q^{\mathrm{fpm}}$ 은 R1\textendash R3 구간에서도 부호 반전을 자주 보이며 R4에 들어서면 0 아래에 머물러 뷰 방향이 반전된 상태가 유지되는 반면, $\sigma$ 직접 변환 기반의 $q^{\mathrm{lam}}$ 은 전 기간 양수를 유지하여 저변동 뷰의 일관성을 보존한다.

\subsection{$\mathbf{p}$ 벡터 변경의 완화 효과}

R4 정상화 구간에서 prior=mcap·omega=err을 고정한 채 $\mathbf{p}$와 Q-mode(q)를 변경한 모든 조합의 스프레드는 다음과 같다(LSTM 기준).

\begin{table}[ht]
\centering
\caption{R4 정상화 구간에서 prior=mcap, $\omega$=err 고정 후 $\mathbf{p}$와 
Q-mode($q$)를 변경한 모든 조합의 스프레드(저변동 $-$ 고변동, LSTM 기준). 
$p_w$=mcap $\cdot$ $q$=$q^{\mathrm{fpm}}$ 한 조합에서만 음수로 반전된다.}
\label{tab:r4_pq_spread}
\setlength{\tabcolsep}{10pt}
\renewcommand{\arraystretch}{1.2}
\small
\begin{tabular}{lccccc}
\toprule
spread (저$-$고) & $q^{\mathrm{lam}}$ & $q^{\mathrm{raw}}$ & $q^{\mathrm{inv}}$ & $q^{\mathrm{vsp}}$ & $q^{\mathrm{fpm}}$ \\
\midrule
$p_w$ = mcap & $+0.290$ & $+0.290$ & $+0.229$ & $+0.236$ & $\mathbf{-0.145}$ \\
$p_w$ = eq   & $+0.335$ & $+0.335$ & $+0.267$ & $+0.278$ & $+0.146$ \\
$p_w$ = rp   & $+0.332$ & $+0.332$ & $+0.265$ & $+0.277$ & $+0.153$ \\
\bottomrule
\end{tabular}
\end{table}

R4에서 스프레드가 음수로 전환되는 조합은 $(p_w=\text{mcap}, q=q^{\mathrm{fpm}})$ 1개에 불과하며, 동일한 $q^{\mathrm{fpm}}$ 을 사용하더라도 $p_w$를 eq 또는 rp로 변경하면 스프레드가 양수($+0.146, +0.153$)로 회복된다. ANN 기준에서도 같은 패턴이 관찰되며(mcap$\cdot$fpm: $-0.103$, eq$\cdot$fpm: $+0.165$, rp$\cdot$fpm: $+0.179$), 양수 제약 $Q$와 결합된 $\mathbf{p}$는 모든 조합에서 스프레드 $0.20$\textendash $0.34$ 수준의 저변동 우위 뷰를 유지한다.

즉 R4에서 전략 본질의 뒤집힘은 mcap 가중과 $q^{\mathrm{fpm}}$ 의 결합 조건에서만 관찰되며, 두 요소 중 어느 한쪽이라도 변경되면 저변동 본질이 부분적 또는 완전 회복된다.

\subsection{결론 요약}

본 부록의 관찰은 다음과 같이 요약된다.

\begin{enumerate}
\item R4 정상화 구간에서 $q^{\mathrm{fpm}}$ 은 다른 레짐 대비 음수 빈도(80\textendash 90\%)와 평균값 크기(LSTM 약 11배 증가, ANN 사실상 $0 \to$ 강음수 전환)가 극단적으로 증가한다.
\item 이로 인해 앵커 구성(mcap-mcap-fpm-err)은 R4에서 저변동 전략 본질이 뒤집혀 고변동 over-weight 상태로 운용된다.
\item P-matrix를 eq 또는 rp로 변경하면 동일 $q^{\mathrm{fpm}}$ 조건에서도 저변동 over-weight가 회복되며, 양수 제약 $Q$를 사용하면 P-matrix 선택과 무관하게 저변동 본질이 강하게 유지된다.
\end{enumerate}

이러한 관찰은 적응형 뷰 산출(Fama-French 회귀)이 시장 국면 변화에 따라 전략 본질을 의도치 않게 변경할 수 있음을 시사한다. 다만 본 연구는 이러한 관찰의 사후적 기술에 머무르며, 뷰 산출 방식과 전략 본질 간 인과적 메커니즘의 규명은 향후 연구 과제로 남긴다.

\section{벤치마크 대비 비교}\label{app:benchmark}\label{app:r4_mechanism}

\S\ref{res:benchmark}절(벤치마크 성능 비교)에서 보고한 Sharpe 비율 결과(표~\ref{tab:benchmark_sharpe})에 대응하여, 동일한 슬롯과 5개 기간(All, R1 회복, R2 확장, R3 위기, R4 AI 랠리)에 대한 보조 지표 3종을 첨부한다. 표~\ref{tab:benchmark_sortino}은 Sortino 비율, 표~\ref{tab:benchmark_mdd}은 최대낙폭(MDD), 표~\ref{tab:benchmark_annret}은 연환산 수익률을 보고하며, 각 표에서 4개 LSTM 슬롯 중 컬럼별 최우수 값을 굵게 표시한다. Sortino 비율과 연환산 수익률은 클수록, MDD는 0에 가까울수록 우위이다.

% ── Sortino ratio ───────────────────────────────────────────
\begin{table}[ht]
\centering
\caption{벤치마크 성능 비교 — Sortino 비율 $\times$ 5 기간. 4개 LSTM 슬롯 중 컬럼별 최대값 굵게 표시.}
\label{tab:benchmark_sortino}
\setlength{\tabcolsep}{6pt}
\renewcommand{\arraystretch}{1.15}
\small
\begin{tabular}{lccccc}
\toprule
Strategy & All & R1 & R2 & R3 & R4 \\
\midrule
SPY (market)                       & 1.38 & 1.41 & 1.66 & 1.06 & 2.36 \\
1/N (equal-weight)                 & 1.21 & 1.32 & 1.47 & 1.08 & 0.88 \\
Risk Parity ($1/\sigma$)           & 1.25 & 1.72 & 1.50 & 1.05 & 0.81 \\
ANN-anchor \citep{pyo2018}         & 1.34 & 1.56 & 1.36 & 1.60 & 0.88 \\
LSTM-anchor ($\sigma$ swap)        & 1.32 & 1.19 & 1.48 & 1.17 & 1.51 \\
\midrule
\textbf{LSTM-defensive-EQ} & \textbf{1.66} & \textbf{1.78} & \textbf{1.82} & \textbf{1.80} & 0.98 \\
\textbf{LSTM-defensive-RP} & 1.64 & 1.67 & 1.75 & 1.79 & 1.10 \\
\textbf{LSTM-adaptive-EQ}  & 1.62 & 1.47 & 1.70 & 1.77 & 1.72 \\
\textbf{LSTM-adaptive-RP}  & 1.59 & 1.44 & 1.64 & 1.75 & \textbf{1.78} \\
\bottomrule
\end{tabular}
\end{table}

% ── MDD (max drawdown) ──────────────────────────────────────
\begin{table}[ht]
\centering
\caption{벤치마크 성능 비교 — Maximum Drawdown $\times$ 5 기간 (0에 가까울수록 우수). 
4개 LSTM 슬롯 중 컬럼별 최우수(최소 절대값) 굵게 표시.}
\label{tab:benchmark_mdd}
\setlength{\tabcolsep}{6pt}
\renewcommand{\arraystretch}{1.15}
\small
\begin{tabular}{lccccc}
\toprule
Strategy & All & R1 & R2 & R3 & R4 \\
\midrule
SPY (market)                       & $-23.9$\% & $-16.2$\% & $-13.5$\% & $-23.9$\% & $-7.6$\%  \\
1/N (equal-weight)                 & $-24.3$\% & $-21.4$\% & $-17.3$\% & $-16.8$\% & $-12.0$\% \\
Risk Parity ($1/\sigma$)           & $-22.0$\% & $-18.2$\% & $-15.8$\% & $-16.1$\% & $-10.4$\% \\
ANN-anchor \citep{pyo2018}         & $-18.4$\% & $-12.5$\% & $-16.1$\% & $-16.1$\% & $-18.4$\% \\
LSTM-anchor ($\sigma$ swap)        & $-20.5$\% & $-17.1$\% & $-17.6$\% & $-16.5$\% & $-14.0$\% \\
\midrule
\textbf{LSTM-defensive-EQ} & $-14.7$\% & $\mathbf{-12.8}$\% & $\mathbf{-8.2}$\%  & $-14.7$\%          & $-8.9$\%  \\
\textbf{LSTM-defensive-RP} & $\mathbf{-14.0}$\% & $-14.0$\% & $-8.5$\%  & $\mathbf{-13.4}$\% & $-8.7$\%  \\
\textbf{LSTM-adaptive-EQ}  & $-15.7$\% & $-15.7$\% & $-11.4$\% & $-15.1$\%          & $-7.9$\%  \\
\textbf{LSTM-adaptive-RP}  & $-19.2$\% & $-19.2$\% & $-12.6$\% & $-15.0$\%          & $\mathbf{-7.5}$\%  \\
\bottomrule
\end{tabular}
\end{table}

% ── Annualized return ───────────────────────────────────────
\begin{table}[ht]
\centering
\caption{벤치마크 성능 비교 — 연환산 수익률 $\times$ 5 기간. 4개 LSTM 슬롯 중 컬럼별 최대값 굵게 표시.}
\label{tab:benchmark_annret}
\setlength{\tabcolsep}{6pt}
\renewcommand{\arraystretch}{1.15}
\small
\begin{tabular}{lccccc}
\toprule
Strategy & All & R1 & R2 & R3 & R4 \\
\midrule
SPY (market)                       & $+14.4$\% & $+12.7$\% & $+14.2$\% & $+12.4$\% & $+19.7$\% \\
1/N (equal-weight)                 & $+14.8$\% & $+14.3$\% & $+16.1$\% & $+15.5$\% & $+10.5$\% \\
Risk Parity ($1/\sigma$)           & $+14.2$\% & $+14.9$\% & $+15.6$\% & $+13.9$\% & $+9.8$\%  \\
ANN-anchor \citep{pyo2018}         & $+14.3$\% & $+16.6$\% & $+12.9$\% & $+15.0$\% & $+15.2$\% \\
LSTM-anchor ($\sigma$ swap)        & $+15.1$\% & $+11.4$\% & $+15.9$\% & $+14.2$\% & $+17.9$\% \\
\midrule
\textbf{LSTM-defensive-EQ} & $+15.1$\% & $\mathbf{+17.2}$\% & $+15.1$\% & $\mathbf{+16.7}$\% & $+11.1$\% \\
\textbf{LSTM-defensive-RP} & $+14.7$\% & $+15.2$\% & $+14.7$\% & $+16.5$\% & $+11.9$\% \\
\textbf{LSTM-adaptive-EQ}  & $\mathbf{+15.9}$\% & $+14.0$\% & $\mathbf{+16.4}$\% & $+15.7$\% & $+16.4$\% \\
\textbf{LSTM-adaptive-RP}  & $+15.3$\% & $+13.1$\% & $+15.6$\% & $+15.3$\% & $\mathbf{+16.8}$\% \\
\bottomrule
\end{tabular}
\end{table}

\end{document}