# 논문 내 수식 통일
> 논문 내 사용되는 LaTeX 수식은 다음 수식들로 통일한다.
### LSTM
입력


\mathbf{x}[t] = \bigl[\log RV_d[t],\ \log RV_w[t],\ \log RV_m[t],\ \log \text{VIX}[t]\bigr]^\top





예측


y[t] = \log(\text{std}(r[t+1], \ldots, r[t+21]))


\hat{y}^{\text{LSTM}}[t] = f_{\text{LSTM}}(\mathbf{X}_{t-L+1}, \ldots, \mathbf{X}_t)

### HAR-RV
전체 수식


\log RV_h[t+h] = \beta_0 + \beta_d \cdot \log RV_d[t] + \beta_w \cdot \log RV_w[t] + \beta_m \cdot \log RV_m[t]


입력


RV_d[t] = r[t]^2


RV_w[t] = \frac{1}{5}\sum_{s=0}^{4} r[t-s]^2


RV_m[t] = \frac{1}{22}\sum_{s=0}^{21} r[t-s]^2


예측
\hat{y}^{\text{HAR}}[t] = \beta_0 + \beta_d \log RV_d[t] + \beta_w \log RV_w[t] + \beta_m \log RV_m[t]

### 앙상블
앙상블 가중치



w_k^{\text{LSTM}} = \frac{1/\text{RMSE}_{k-1}^{\text{LSTM}}}{1/\text{RMSE}_{k-1}^{\text{LSTM}} + 1/\text{RMSE}_{k-1}^{\text{HAR}}}



w_k^{\text{HAR}} = 1 - w_k^{\text{LSTM}}




최종 예측


\hat{y}^{\text{ens}}[t] = w_k^{\text{LSTM}} \cdot \hat{y}^{\text{LSTM}}[t] + w_k^{\text{HAR}} \cdot \hat{y}^{\text{HAR}}[t]




연환산 변환(BL 입력용)


\hat{\sigma}[t] = \exp(\hat{y}[t]) \times \sqrt{252}














\hat{y}^{\text{ens}}[t] \xrightarrow{\exp(\cdot)\times\sqrt{252}} \hat{\sigma}[t] \longrightarrow \text{BL}
### Black-Litterman
공분산 행렬    ( R[t] : 수익률행렬 Txn )


\boldsymbol{\Sigma} = \text{LedoitWolf}(\mathbf{R}[t]) \times 21


균형 기대수익률


\boldsymbol{\pi} = \lambda\,\boldsymbol{\Sigma}\,\mathbf{w}_{\text{mkt}}
 


w_i^{\text{mcap}} = \frac{m_i}{\sum_j m_j}         w_i^{\text{eq}} = \frac{1}{n}         w_i^{\text{rp}} = \frac{1/\hat{\sigma}_i}{\sum_j 1/\hat{\sigma}_j}

사전확률 
\boldsymbol{\mu}_{\text{prior}} \sim \mathcal{N}(\boldsymbol{\pi},\ \tau\boldsymbol{\Sigma})

기대수익률
q \sim \mathcal{N}(\mathbf{p}^\top\boldsymbol{\mu}_{\text{prior}},\ \omega)

사후확률
\boldsymbol{\mu}_{\text{post}} \mid q \sim \mathcal{N}(\boldsymbol{\mu}_{BL},\ \boldsymbol{\Sigma}_{BL})

P : 뷰 포트폴리오(L=하위 30%, H=상위 30%)



p_i = \begin{cases} +1/n_g & i \in \mathcal{L} \\ -1/n_g & i \in \mathcal{H} \\ 0 & \text{otherwise} \end{cases}



p_i^{\text{mcap}} = \begin{cases} +m_i / \sum_{j \in \mathcal{L}} m_j & i \in \mathcal{L} \\ -m_i / \sum_{j \in \mathcal{H}} m_j & i \in \mathcal{H} \\ 0 & \text{otherwise} \end{cases}



p_i^{\text{eq}} = \begin{cases} +1/n_g & i \in \mathcal{L} \\ -1/n_g & i \in \mathcal{H} \\ 0 & \text{otherwise} \end{cases}



p_i^{\text{rp}} = \begin{cases} +(1/\hat{\sigma}_i) / \sum_{j \in \mathcal{L}} (1/\hat{\sigma}_j) & i \in \mathcal{L} \\ -\hat{\sigma}_i / \sum_{j \in \mathcal{H}} \hat{\sigma}_j & i \in \mathcal{H} \\ 0 & \text{otherwise} \end{cases}

q : 뷰 수익률(q0 = 0.003, lambda_bar = 2.5)


q^{\text{fixed}} = q_0


q^{\text{lambda}} = q_0 \cdot \text{clip}\!\left(\lambda_t / \bar{\lambda},\ 0.1,\ 3.0\right)


q^{\text{inv\_lambda}} = q_0 \cdot \text{clip}\!\left(\bar{\lambda} / \lambda_t,\ 0.1,\ 3.0\right)


q^{\text{raw\_lam}} = \max\!\left(0,\ q_0 \cdot \lambda_t^{\text{raw}} / \bar{\lambda}\right)


q^{\text{vol\_spread}} = q_0 \cdot \text{clip}\!\left(\Delta\hat{\sigma}_t / \overline{\Delta\hat{\sigma}},\ 0.1,\ 3.0\right)


\lambda_t = \text{clip}\!\left(\frac{r_{\text{SPY},t} - r_{f,t}}{\text{Var}(r_{\text{SPY},t})},\ 0.5,\ 10.0\right)

\Delta\hat{\sigma}_t = \bar{\hat{\sigma}}_{\mathcal{H},t} - \bar{\hat{\sigma}}_{\mathcal{L},t}

ω : 뷰 불확실성


\omega^{\text{He-Litterman}} = \tau \cdot \mathbf{p}^\top \Sigma\, \mathbf{p}


\omega^{\text{ff3}} = \max\!\left((q_{t-1} - \mathbf{p}_{t-1}^\top \mathbf{r}_t)^2,\ 10^{-8}\right)

사후분포


\mu_{\text{BL}} = \pi + \tau\Sigma \mathbf{p} \cdot \frac{Q - \mathbf{p}^\top \pi}{\mathbf{p}^\top(\tau\Sigma)\mathbf{p} + \omega}


\Sigma_{BL} = \tau\Sigma - \frac{(\tau\Sigma \mathbf{p}^T)(\mathbf{p}\tau\Sigma)}{\mathbf{p}\tau\Sigma \mathbf{p}^T + \Omega}



MVO


\mathbf{w}^* = \arg\min_{\mathbf{w}}\ \frac{\lambda}{2}\mathbf{w}^\top \Sigma_{\text{BL}}\mathbf{w} - \mathbf{w}^\top \mu_{\text{BL}}



\text{s.t.}\ \sum_i w_i = 1,\quad 0 \leq w_i \leq 0.1

> 행렬 : 대문자+볼드     벡터 : 소문자+볼드    스칼라 : 소문자+이탤릭

기호	타입	코드 타입
π
\pi	벡터 n×1	pd.Series
μBL
\mu_{BL}	벡터 n×1	pd.Series
p
\mathbf{p}	벡터 1×n	pd.Series
Σ
\Sigma	행렬 n×n	pd.DataFrame
ΣBL
\Sigma_{BL}	행렬 n×n	np.ndarray
q
q	스칼라	float
Ω / ω
\Omega	스칼라	float
τ
\tau	스칼라	float
λ
\lambda	스칼라	float
w
w	벡터 n×1	pd.Series

### ANN
Net input (가중합)

\text{net}_j = \sum_{i} w_{ij} x_i

Activation function (ReLU)

f(S) = \max(0, S)

Output

o_j = f(\text{net}_j)

Weight update (Backpropagation)


w_{ij}^{(t+1)} = w_{ij}^{(t)} - \eta \frac{\partial \mathcal{L}}{\partial w_{ij}}

논문 ANN 변동성

\hat{\sigma}_{i,t} = f(\sigma_{i,t-1}, \sigma_{i,t-2}, \dots, \sigma_{i,t-10}), \quad i \in \{1, \dots, n\}

FF3 수익률 예측


\hat{r}_i = \alpha + \beta_{mkt}(r_{mkt} - r_f) + \beta_{SMB} \cdot SMB + \beta_{HML} \cdot HML, \quad \forall i \in \{1, \dots, n\}

View return Q


q = P\hat{\mathbf{r}}, \quad \hat{\mathbf{r}} = [\hat{r}_1, \hat{r}_2, \dots, \hat{r}_n]^T

Ω 추정


\Omega_t = \text{Var}(P\hat{\mu}_{t-1} - P\mu_{t-1}) = (q_{t-1} - Pr_{t-1})^2

r_{mkt} - r_f                      SMB = r_{small} - r_{big}                   HML = r_{high} - r_{low}

### HMM
관측 피처 벡터

\mathbf{x}_t = [\text{mkt\_rf},\ \text{vol\_21d\_daily},\ \text{vol\_ratio},\ \text{vix\_basis},\ \text{t10y2y\_chg},\ \text{mom}]^\top

초기 분포

\boldsymbol{\pi} = [\pi_1, \ldots, \pi_K], \quad \pi_i = P(z_1 = i)

전이 행렬


A_{ij} = P(z_t = j \mid z_{t-1} = i), \quad \sum_j A_{ij} = 1

방출 분포 (Gaussian, full covariance)

P(\mathbf{x}_t \mid z_t = i) = \mathcal{N}(\mathbf{x}_t;\,\boldsymbol{\mu}_i,\,\boldsymbol{\Sigma}_i)