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

\paragraph{ANN (비교 baseline).} 본 연구는 변동성 예측 모델의 비교 baseline으로 \cite{pyo2018}이 블랙-리터만 view 산출에 사용한 인공신경망(Artificial Neural Network, ANN) 구조를 채택한다. 해당 ANN은 종목 $i$의 직전 10개월 변동성을 입력으로 받아 다음 1개월 변동성을 출력한다.
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

\begin{figure}[ht]
\centering
\renewcommand{\arraystretch}{1.3}
\setlength{\tabcolsep}{4pt}
\footnotesize
\begin{tabular}{l|c|c|c|c}
\hline
\textbf{폴드} & \textbf{In-Sample (IS, 1{,}250)} & \textbf{Purge (21)} & \textbf{Embargo (63)} & \textbf{OOS (21)} \\
\hline
$k$ & day $1 \sim 1{,}250$ & day $1{,}251 \sim 1{,}271$ & day $1{,}272 \sim 1{,}334$ & day $1{,}335 \sim 1{,}355$ \\
$k+1$ & day $22 \sim 1{,}271$ & day $1{,}272 \sim 1{,}292$ & day $1{,}293 \sim 1{,}355$ & day $1{,}356 \sim 1{,}376$ \\
$k+2$ & day $43 \sim 1{,}292$ & day $1{,}293 \sim 1{,}313$ & day $1{,}314 \sim 1{,}376$ & day $1{,}377 \sim 1{,}397$ \\
\hline
\multicolumn{5}{c}{각 폴드는 직전 폴드 대비 step $=21$ 영업일 우측으로 이동} \\
\end{tabular}
\caption{Walk-Forward 폴드 구조 (step $=21$ 영업일). 폴드 $k$, $k+1$, $k+2$의 day 인덱스 예시.}
\label{fig:walk_forward}
\end{figure}

\paragraph{파라미터 근거.} IS 1{,}250영업일(약 5영업년)은 LSTM 학습 표본 크기와 시장 체제 다양성을 동시에 확보하기 위한 범위이며, Purge 21영업일과 OOS 21영업일은 각각 종속변수의 forward horizon(식~\ref{eq:lstm_target}) 및 포트폴리오 월별 리밸런싱 주기와 일치한다. Embargo 63영업일은 SPY와 QQQ 일별 실현변동성의 자기상관함수(autocorrelation function, ACF)가 Bartlett 95\% 신뢰구간 내로 진입하여 안정화되는 시점으로 확인된 값이며, 종목별 ACF 재추정에 따른 임의성을 배제하기 위해 평가 대상 전 종목에 일괄 적용한다. 종목별 ACF 분포에 따른 민감도 분석은 본 연구 범위 밖이며, 향후 연구 과제로 남긴다.

\paragraph{비교 baseline ANN의 학습 구조.} 식~\ref{eq:ann_pred}에서 정의한 ANN은 \cite{pyo2018}의 원 사양에 따라 \textit{월별} 데이터를 사용하므로, 일별 데이터를 사용하는 LSTM 및 HAR-RV와 학습·평가 단위가 다르다. 시점 $t$ 기준 직전 60개월을 학습 윈도우로 종목별 모델을 적합하고, 동일 시점에서 직전 10개월의 변동성을 입력으로 다음 한 달의 변동성을 1회 예측한다. 학습 윈도우는 매월 1개월씩 이동하여, ANN의 OOS 1개월 단위는 LSTM·HAR-RV·앙상블의 OOS 21영업일 단위와 정확히 일치한다. 본 연구는 ANN의 출력을 log-daily 변동성 공간으로 통일하여, 모든 모델이 동일 시점·동일 단위에서 직접 비교 가능하도록 한다.


\subsection{블랙-리터만 구조}

블랙-리터만(Black–Litterman, BL, \cite{black1992}) 모델은 시장 균형 수익률과 투자자의 뷰를 베이지안 방식으로 결합하는 포트폴리오 최적화 프레임워크로, Markowitz 평균-분산 최적화가 갖는 추정 오차 민감성을 완화하는 데 효과적이다. 모델은 시장 균형 초과수익률 벡터 $\boldsymbol{\pi}$에서 출발하며, 이는 이차 효용 최대화 문제의 해로 도출된다.

\begin{align}\label{1st}
    & \boldsymbol{\pi} = \lambda\Sigma\mathbf{w_{mkt}}
\end{align}

여기서 $\boldsymbol{\pi} \in R^{n + 1}$은 균형 초과수익률 벡터, $\Sigma_{n \times n}$는 초과수익률 공분산 행렬, $\mathbf{w_{mkt}} \in R^{n+1}$은 시장 비중 벡터다. 위험회피계수 $\lambda$는 본 연구에서 $\lambda = 2.5$로 고정하였으며, 이는 BL 문헌의 일반적 관행을 따른 것이다(\cite{helitterman1999}). 공분산 행렬 $\Sigma$는 직전 60개월(약 1,250 거래일) 일별 수익률에 Ledoit-Wolf 수축 추정량을 적용하여 산출하며, 이는 고차원 추정 시 발생하는 표본 오차를 완화하기 위함이다(\cite{ledoitwolf2004}).

사전 분포(prior)는 다음과 같이 설정된다.

\begin{align}\label{2st}
    \boldsymbol{\mu} \sim N(\boldsymbol{\pi}, \tau\Sigma)
\end{align}

$\tau$는 사전 분포의 불확실성을 조절하며, 본 연구에서는 $\tau = 0.1$로 설정하였다. 투자자의 뷰는 다음과 같이 표현된다.

\begin{align}\label{3rd}
    q|\boldsymbol{\mu} \sim N(\mathbf{p}^T\boldsymbol{\mu}, \omega)
\end{align}

여기서 $\mathbf{p} \in R^{n \times 1}$은 뷰 포트폴리오 가중치 벡터, $q \in R$은 단일 상대 뷰의 기대 수익률 스칼라, $\omega$는 뷰의 불확실성을 나타내는 스칼라다. 사전 분포와 뷰 우도를 결합하면 사후 기대수익률과 공분산은 다음과 같이 도출된다.

\begin{align}\label{4th}
    & \Sigma_{BL} = [\tau\Sigma^{-1} + \mathbf{p}^T\omega^{-1}\mathbf{p}]^{-1}\\
    \label{5th}
    & \boldsymbol{\mu}_{BL} = [\tau\Sigma^{-1} + \mathbf{p}^T\omega^{-1}\mathbf{p}]^{-1}[(\tau\Sigma)^{-1}+\mathbf{p}^T\omega^{-1}q]
\end{align}

최적 포트폴리오 비중은 다음을 풀어 산출한다.

\begin{align}
    min\frac{1}{2}\lambda\mathbf{w}^T\Sigma_{BL}\mathbf{w}-\mathbf{w}^T\boldsymbol{\mu}_{BL} \\
    \text{s.t.} \quad \sum_{i=1}^{n}{\mathbf{w_i}} = 1, \mathbf{w_i} \leq 0.1
\end{align}

단일 종목 비중 상한 10\%는 극단적 집중 투자를 방지하기 위한 제약이다.

\subsubsection{제안 슬롯}\label{bl:proposed_slot}

본 연구의 핵심은 \ref{subsec:vol_pred}절의 변동성 예측 결과를 BL 프레임워크의 각 슬롯에 체계적으로 연결하는 것이다. 예측 변동성이 $\mathbf{p}$ 벡터를 통해 자산을 분류하고, 이를 기반으로 q와 $\omega$가 결합되어 사후 기대수익률 $\boldsymbol{\mu}_{BL}$이 산출된다. 본 절에서 각 슬롯의 제안 설정을 기술한다.

\paragraph{사전 분포(Prior)}

균형 수익률 계산에 필요한 시장 비중 $\mathbf{w}_{mkt}$를 세 가지 방식으로 구성한다.

\begin{itemize}
    \item $\mathbf{w}^{\mathrm{mcap}}$ : 시가총액 비중 가중. \cite{pyo2018}의 원안이며, CAPM의 표준적 해석이다.
    \item $\mathbf{w}^{\mathrm{eq}}$ : 동일가중. 투자 가능 종목군 전체에 균등한 사전 정보를 부여한다.
    \item $\mathbf{w}^{\mathrm{rp}}$ : 21일 실현 변동성의 역수($1/\sigma_{\mathrm{realized}}$) 비중 가중치. 저변동 종목에 더 큰 가중치를 배분하는 리스크 패리티(\cite{maillard2010}) 철학에 기반하였다. 단, 사용하는 변동성은 LSTM 예측값이 아닌 전월의 실현 변동성이다.
\end{itemize} 

\paragraph{뷰 포트폴리오($\textbf{p}$)}

$\mathbf{p}$ 벡터는 저변동 자산이 고변동 자산을 초과 성과할 것이라는 뷰를 인코딩한다. 
각 리밸런싱 시점 $t$에서 투자 가능 종목군 내 종목을 LSTM 예측 변동성 $\hat{\sigma}_{i,t}$ 기준으로 
오름차순 정렬 후, 하위 30\%(저변동 그룹 $L$)에 양($+$)의 가중치, 상위 30\%(고변동 그룹 $H$)에 
음($-$)의 가중치를 부여하며, 나머지 40\%는 0으로 처리한다.

$\mathbf{p}$ 벡터의 가중 방식으로 세 가지를 제안한다.

\begin{itemize}
  \item $\mathbf{p}^{\mathrm{mcap}}$: 시가총액 비례 가중. \citet{pyo2018}의 원안으로, 
        각 그룹 내 시가총액 합계로 정규화한다.
        \begin{equation}
          p_i^{\mathrm{mcap}} =
          \begin{cases}
            +\dfrac{\mathrm{cap}_i}{\sum_{j \in L} \mathrm{cap}_j}, & i \in L, \\[8pt]
            -\dfrac{\mathrm{cap}_i}{\sum_{j \in H} \mathrm{cap}_j}, & i \in H, \\[6pt]
            \phantom{+}0, & \text{otherwise},
          \end{cases}
          \label{eq:p_mcap}
        \end{equation}
        여기서 $\mathrm{cap}_i$는 종목 $i$의 시가총액이다.

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
월 수익이 0.4--0.7\% 범위임을 확인한다. 본 연구는 예측 모델의 오판 가능성과 분석 기간이 저위험 이상현상과는 거리가 먼 것을 고려하여 실증치의 약 절반(0.3\%)을 기준값으로 채택한다.

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

위 수식에서 동적 조정에 사용하는 시장 위험회피도 $\lambda_q$는 $\boldsymbol{\pi}$ 계산용 고정 $\lambda=2.5$와 
별개로, 매 시점 다음과 같이 추정한다.
\begin{equation}
  \lambda_q = \mathrm{clip}\!\left(\bar{r}_{\mathrm{SPY}}^{\mathrm{ex}}/\sigma_{\mathrm{mkt}}^2,\ 0.5,\ 10.0\right).
\end{equation}
$\bar{\lambda}$는 훈련 기간 내 $\lambda_q$의 평균이다.
% clip 그대로 써도 될지? r_spy같은 것들도 수정해야 함

\vspace{1em}

\textbf{예측 변동성 기반 슬롯.}
$q^{\mathrm{vsp}}$는 LSTM 예측 변동성 격차를 훈련 기간 중앙값으로 정규화하여 $q$의 
크기를 조정한다. $\lambda_q$ 분류가 매크로 시장 신호를 사용하는 반면, $q^{\mathrm{vsp}}$는 
LSTM 예측 변동성을 $q$에 직접 반영한다.
\begin{equation}
  q^{\mathrm{vsp}} = q^{\mathrm{base}} \times 
  \mathrm{clip}\!\left(\Delta\hat{\sigma}_t/\mathrm{median}(\Delta\hat{\sigma}_{\mathrm{train}}),\ 
  0.1,\ 3.0\right),
\end{equation}
여기서 $\Delta\hat{\sigma}_t = \mathrm{mean}(\hat{\sigma}_{i\in H}) - 
\mathrm{mean}(\hat{\sigma}_{i\in L})$이며, $H$, $L$의 정의에 의해 $\Delta\hat{\sigma}_t>0$이 
보장된다.

\vspace{1em}

\textbf{팩터 모형 기반 슬롯.}
$q^{\mathrm{fpm}}$은 FF3 훈련 윈도우(60개월) 내 팩터 평균을 다음 달 기대 팩터 프리미엄으로 
사용하는 방식이다 \citep{fama1973risk, cochrane2005}. 단기 팩터 프리미엄 추정의 
불안정성을 감안하여 장기 평균을 기대치로 대입함으로써 추정 안정성을 높인다. 종목별 FF3 
롤링 회귀로 추정한 기대수익률 $\hat{r}_i$와 $\mathbf{p}$ 벡터의 내적으로 $q$를 산출하며, 
훈련 기간 내 데이터만 사용하므로 look-ahead bias가 없다. \citet{pyo2018}의 FF3 기반 $q$ 
산출 방식을 장기 평균 팩터 프리미엄 관점으로 재해석한 것으로, 본 연구의 앵커(anchor) 
슬롯에 해당한다.
\begin{equation}
  q^{\mathrm{fpm}} = \mathbf{p}^{\top}\hat{\mathbf{r}}, \qquad
  \hat{r}_i = \hat{\alpha}_i + \hat{\beta}_i^{\mathrm{mkt}}\bar{f}_{\mathrm{mkt}} 
  + \hat{\beta}_i^{\mathrm{SMB}}\bar{f}_{\mathrm{SMB}} 
  + \hat{\beta}_i^{\mathrm{HML}}\bar{f}_{\mathrm{HML}},
\end{equation}
여기서 $\bar{f}_k$는 훈련 윈도우 내 팩터 $k$의 평균값이다.

\paragraph{뷰 신뢰도($\omega$)}

$\omega$는 사전 균형 수익률 대비 투자자 뷰의 신뢰도를 결정하며, 본 연구는 두 가지 
설정을 비교한다.

\begin{itemize}
  \item \textbf{$\omega^{\mathrm{he}}$ (He-Litterman 표준)}: 뷰 포트폴리오의 사전 
        분산에 비례하여 불확실성을 설정하는 BL 문헌의 표준 방식이다 
        \citep{helitterman1999}.
        \begin{equation}
          \omega^{\mathrm{he}} = \tau \cdot \mathbf{p}\,\Sigma\,\mathbf{p}^{\top}.
        \end{equation}

  \item \textbf{$\omega^{\mathrm{err}}$ (전월 예측 오차 기반)}: 각 시점 $t$에서 
        전월($t-1$)의 뷰 포트폴리오 예측 수익률과 실현 수익률 간 오차 제곱으로 
        불확실성을 추정하며, \citet{pyo2018}의 방식을 따른다.
        \begin{equation}
          \omega^{\mathrm{err}} = \left(q_{t-1} - \mathbf{p}_{t-1}^{\top}\mathbf{r}_{t-1}\right)^{2},
        \end{equation}
        여기서 $\mathbf{r}_{t-1}$은 $t-1$ 시점의 실현 수익률 벡터다. 전월 뷰 예측이 
        크게 빗나간 시점에서 불확실성이 자동으로 증가하는 자기교정적 구조이며, 첫 달은 
        $\omega^{\mathrm{he}}$로 초기화한다.
\end{itemize}

\subsubsection{Walk-Forward 구조}
\label{subsec:bl_walkforward}

포트폴리오 비중은 \ref{subsec:vol_wf}절의 변동성 예측과 동일한 원칙을 
따르는 Walk-Forward 구조로 산출한다. 각 리밸런싱 시점 $t$(월 1회)에서 직전 60개월 
일별 수익률로 공분산 행렬 $\Sigma_t$를 추정하고, 해당 시점의 시가총액 비중 
$w_{\mathrm{mkt},t}$와 위험회피도 $\lambda = 2.5$로 균형 수익률 
$\pi_t = \lambda \Sigma_t w_{\mathrm{mkt},t}$를 계산한다. 이후 LSTM(또는 ANN 베이스라인)의 
예측 변동성 $\hat{\sigma}_{i,t}$로 $\mathbf{p}_t$를 구성하고, 설정에 따라 $q_t$와 
$\omega_t$를 산출한 뒤 BL 사후 분포의 평균 $\mu_{\mathrm{BL},t}$와 공분산 
$\Sigma_{\mathrm{BL},t}$를 계산하여 포트폴리오 비중 최적화를 수행한다.

각 리밸런싱 시점에서 훈련 윈도우(60개월) 내 일별 수익률 데이터가 90\% 이상 존재하는 
종목만 유효 투자 가능 종목군로 선별한다. 거래비용은 편측 20bp를 적용하며, 월별 실제 비중 변화량 $\sum_i |\Delta w_{i,t}|$에 
비례하여 차감한다. 이는 S\&P 500 구성 종목의 실제 매매 스프레드 대비 보수적인 설정으로, $\omega^{\mathrm{err}}$ 슬롯의 뷰 교정이 고변동 구간에서 회전율을 높일 수 
있다는 점을 고려한 것이다. 무위험 수익률은 미국 3개월 국채 수익률을 사용하며, Sharpe 비율 및 
Sortino 비율 산출 시 초과수익률 계산에 적용한다.
% 평균 N종목 들어가면 좋을 듯?
% \Delta w_{i,t}에 비례 차감은 확인해봐야 함

\subsubsection{평가 방법론}
\label{subsec:bl_eval}

전체 분석 기간은 2010년 1월부터 2025년 12월까지(192개월)이며, 시장 국면에 따라 성과가 크게 달라질 수 있으므로 전체 기간 평가와 레짐 별 성과 분해를 병행한다.

레짐 분류는 Appendix A1에 상세히 기술하며, Hidden Markov Model(HMM, \cite{hamilton1989new}) 기반 통계적 구조전환점 탐지와 금융 이벤트 도메인 지식을 결합하여 네 개의 비중첩 구간을 식별하였다(표 \ref{tab:regimes}).

각 레짐 및 전체 기간에 대해 Sharpe 비율, Sortino 비율, MDD(Maximum Drawdown)를 보고한다. 레짐별 분해를 통해 LSTM 기반 BL 포트폴리오의 ANN 대비 우위가 특정 시장 국면에 집중된 것인지, 전반적으로 일관된 것인지를 검증한다. 이를 위해 LSTM과 ANN 베이스라인은 동일한 108개 슬롯 조합에 대해 1:1 대응 비교하며, 슬롯 구성은 두 모델 간 완전히 동일하게 유지한다.

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

\section{실증 분석}

본 연구에서 활용한 야후 파이낸스의 주가 데이터는 2010년부터 2025년까지 S\&P500에 한 번이라도 포함된 총 617 종목을 대상으로 한다. 포트폴리오는 매월 말 리밸런싱하며(총 168개월), 거래비용은 20bp를 적용한다. 시장 벤치마크는 SPY, 무위험 수익률은 1개월 미국 국채 수익률을 사용한다.

\subsection{Volatility Forecast Performance}

표 \ref{tab:sigma_pred}는 LSTM과 ANN의 변동성 예측 성능을 전체 기간과 4개 레짐 국면 별로 비교한 결과이다. LSTM은 RMSE, Spearman 순위상관, 상\/하위 30\% Hit rate 지표 모두에서 ANN을 일관되게 상회한다. 이는 시장 국면과 무관하게 LSTM의 변동성 예측이 ANN 대비 우위에 있음을 시사한다.

특히, 이 차이는 위기 구간(R3)에서 가장 두드러진다. R3에서 다른 구간 대비 격차가 가장 크게 나타나며, 특히 Spearman 순위 상관 격차가 +0.326으로 가장 두드러지는데, 이는 LSTM이 종목 간 변동성의 상대적 순위를 금융 위기 구간에서 보다 잘 보존함을 알 수 있다.

Hit Rate 또한 동일한 패턴을 보여 R3에서 하위 30\% 식별 정확도가 +0.047, 상위 30\% 식별 정확도가 +0.037 향상되는데, 이는 LSTM이 저변동성 종목을 정확히 선별하는 능력뿐 아니라 고변동성 종목을 정확히 배제하는 양방향 선별력에서도 ANN을 능가함을 보여준다. 본 연구의 포트폴리오는 저변동성 종목의 비중 확대와 고변동성 종목의 비중 축소가 모두 작동해야 하므로, Hit Rate 양 끝단에서의 우위는 포트폴리오 단계 성과로 직접 전이될 가능성이 높다. 이러한 예측 성능 차이가 포트폴리오 단계에서 어떻게 전이되는지는 5.2장에서 살펴본다.

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

\subsection{Baseline Slot}

본 절에서는 \citet{pyo2018}이 제안한 슬롯을 기준 슬롯(mcap\_mcap\_fpm\_err, 표 \ref{tab:1step_variation}에서 $\star$로 표기)으로 두고, \textbf{q}, \textbf{p}, prior, $\omega$ 네 차원을 하나씩 변경하며 LSTM과 ANN의 Sharpe 비율 차이를 비교한다.

전체 기간의 결과를 참고했을 때, LSTM은 대부분의 슬롯에서 ANN 대비 우위를 보이나 기준 슬롯 자체에서의 격차는 $\Delta = +0.000$으로 미미하다. 이는 변동성 예측 모델의 우위가 포트폴리오 성과로 전이되기 위해서는 슬롯 구성 차원의 설계가 함께 동반되어야 함을 시사한다. 아래에서는 4개 차원별로 모델 간 차이의 패턴을 분리해 분석한다.

표 \ref{tab:1step_variation}의 \textbf{q} 변경에서 기존 $\textbf{q}$는 파마-프랜치 3 팩터 회귀 기반으로 산출되어 부호가 유동적인 반면, 본 연구에서 제안한 방식은 항상 $\textbf{q}>0$이 보장되는 구조이다. 이 방식으로 변경하는 경우, R3 위기 구간에서 LSTM 우위가 $\Delta = +0.156 \sim +0.357$로 강하게 나타난다. 반면 기준 슬롯의 fpm은 R3에서 오히려 ANN이 우위를 보인다($\Delta = -0.148$). 이는 위기 구간에서의 모델 차이가 $\textbf{q}$ 부호의 안정성에 의존하며, 부호가 유동적일 경우 위기 구간에서 LSTM의 변동성 순위 식별력이 포트폴리오 단계에서 희석됨을 시사한다.

\textbf{p} 행렬을 기준 슬롯의 시가총액 가중(mcap)에서 동일 가중(eq) 또는 역변동성 가중(rp)으로 변경하는 경우, 전체 기간 Sharpe 비율이 LSTM은 0.946에서 1.099, 1.054로, ANN은 0.945에서 1.049, 0.970으로 양 모델 모두 상승한다. \textbf{P} 벡터는 개별 종목의 $\sigma$ 예측을 포트폴리오 뷰로 종합하는 가중치를 결정하는데, 동일 가중과 역변동성 가중치는 대형주 비중 집중을 완화하여 예측 우위가 포트폴리오 성과로 보다 효과적으로 전이된다. 모델 자체보다 슬롯 구조의 영향이 더 결정적임을 보여주는 결과이며, 이후 \ref{subsec:slot_config}에서 전체 슬롯에 걸친 분포로 강건성을 확인한다.

사전 분포(prior)를 시가총액 비중에서 동일 가중, 역변동성 가중으로 변경하더라도 $\Delta$의 부호 패턴은 기준 슬롯과 동일하게 유지된다. 즉, R1, R3에서 ANN 우위, R2, R4에서 LSTM 우위 구조가 prior 변경과 무관하게 보존되며, 격차의 크기만 조정된다. 사전 분포는 모델 간 차이의 방향이 아닌 크기에만 영향을 미치는 차원으로 해석된다.

$\Omega$ 변경에서 전월 대비 오차(err)의 경우 예측이 크게 빗나간 직후에는 포트폴리오 뷰에 대한 신뢰도를 자동으로 낮춰 사전 분포에 더 의존하므로, 예측력이 시점에 따라 변동하는 환경에서 자동으로 손실을 통제하는 효과를 갖는다. 기준 슬롯이 전월 대비 오차를 채택한 효과가 양 모델 모두에서 공통적으로 나타난다.

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


\subsection{Slot Configuration}

본 절에서는 \S 4.2의 한 차원 변경 결과가 전체 슬롯 조합에 대해서도 일관되게 나타나는지 확인하고, 각 슬롯 차원의 한계 효과(marginal effect)를 분포로 시각화한다. 전체 45 슬롯의 상세 성과는 Appendix \ref{app:full_grid}에 첨부하였다.

표 \ref{tab:p_matrix_avg}는 \textbf{$\Omega$}를 err로 고정한 상태에서 $\textbf{P}$ 행렬 가중치를 mcap/eq/rp 세 가지로 변경하고, 각 조합에 대해 $q$ 5개와 prior 3개의 15개 슬롯을 평균낸 결과이다. 전체 표본 기준으로 mcap 가중 시 LSTM의 변동성 예측 우위에도 불구하고 포트폴리오 Sharpe는 ANN이 $\Delta = -0.019$로 소폭 우위를 보인다. 반면, eq, rp로 가중치를 변경하면 LSTM의 Sharpe 비율은 0.929에서 각각 1.038과 1.008로 상승하며, ANN도 mcap의 0.948에서 eq의 0.970으로 상승한다 (rp는 0.920으로 소폭 감소). 이는 변동성 예측의 우위가 포트폴리오 성과로 전이되기 위해서는 P 행렬을 대형주 비중 집중이 적은 가중치로 구성해야 함을 시사한다. 그림 \ref{fig:marginal_boxplot}에서도 eq, rp 구성 시 LSTM과 ANN의 분포 차이가 가장 두드러진다.

위기 구간(R3)에서는 $\textbf{P}$ 행렬 구성과 무관하게 LSTM이 일관된 우위를 보이며 ($\Delta = +0.134, +0.253, +0.212$), 표 \ref{tab:p_matrix_win}에서도 거의 모든 슬롯에서 LSTM이 ANN을 상회한다 (mcap 12/15, eq 15/15, rp 15/15). 변동성 분포가 가장 분산되는 위기 시점에 종목 순위 식별의 가치가 극대화되며, LSTM의 변동성 예측 우위가 포트폴리오 성과로 가장 효과적으로 전이된다.
% MDD도 추가하긴 해야할 것 같은데...흠 어떻게 추가해보지

반면 회복기(R1)와 확장기(R2)에서는 mcap과 eq 가중 모두 LSTM이 ANN에 패배하는 슬롯이 더 많지만 (R1: mcap 0/15, eq 1/15; R2: mcap·eq 모두 5/15), 손실 규모 측면에서는 두 가중치가 크게 다르다. R1에서 mcap의 평균 손실 $\Delta = -0.214$는 2011년 8월 미국 국가신용등급 강등을 포함한 소수 이벤트 월에 손실이 집중되고, 이것이 대형주 비중 집중이 강한 mcap에서 증폭되어 나타난 결과이다. 반면 eq에서는 균등 배분으로 충격의 영향이 작아져 손실이 $-0.046$으로 통제된다. 즉 eq 가중은 회복기에 LSTM 우위를 잃는 빈도는 mcap과 유사하나, 손실 규모는 mcap의 대형주 비중 집중이 야기하는 대규모 손실 위험을 효과적으로 완화한다. 자세한 분석은 Appendix \ref{app:r1}에서 확인할 수 있다.
% eq, rp, mcap 일단 막 적고 용어 통일은 한 번 해야할 것 같음. 너무 무지성이네

그림 \ref{fig:marginal_boxplot}의 q 차원을 보면, R3 위기 구간에서 q=lam과 q=raw처럼 항상 $q>0$이 보장되어 저변동성 종목의 비중을 일관되게 확대하는 방식이 LSTM과 ANN의 성과 차이를 가장 크게 만든다. 부호가 유동적인 fpm은 R3에서 오히려 ANN 우위로 역전되는데, 이는 LSTM이 식별한 저변동성 종목의 정확한 순위가 q 부호의 안정성을 통해서만 포트폴리오 가중치로 보존됨을 시사한다.

본 절에서 확인한 슬롯 구성이 단순 벤치마크 대비 실제로 의미 있는 성과를 갖는지는 \S 4.4에서 확인한다.


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

\begin{figure}[hbt!]
    \centering
    \includegraphics[width=15cm]{Figure/marginal_mean.png}
    \caption{Marginal Effects of Black-Litterman Slots}
    \label{fig:marginal_boxplot}
\end{figure}

\subsection{벤치마크 성능 비교}

SPY, 1/N, risk parity, ANN(논문 앵커)랑 결과 비교한 거 그림 표 보여주기

\section{Conclusion \& Limitation}

\bibliographystyle{apalike}
\bibliography{references}

% --------------- appendix 시작 ---------------
\appendix
\renewcommand{\thesection}{부록 \Alph{section}}
\renewcommand{\thefigure}{A.\arabic{figure}}
\renewcommand{\thetable}{A.\arabic{table}}
\setcounter{figure}{0}
\setcounter{table}{0}

\section{HMM 기반 시장 레짐 분류}


\end{document}