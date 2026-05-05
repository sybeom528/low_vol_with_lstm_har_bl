"""
bl_config.py — Black-Litterman 실험 정의

새 실험 추가 방법:
  - 기존 방식의 파라미터만 바꿀 때 → EXPERIMENTS에 dict 한 줄 추가
  - 새 계산 방식 도입 시 → bl_functions.py에 함수 추가 + 여기에 dict + 99_run dispatcher에 분기

슬롯 키 정리:
  p_mode    : 'trailing_vol21' | 'lstm_predicted'
  p_weight  : 'mcap' | 'eq' | 'rp' | 'vol_mcap'
  q_mode    : 'fixed' | 'ff3_regression' | 'realized_spread' | 'regime' | 'lambda' | 'raw_lam' | 'vol_spread' | 'none'
  q_value   : float  (q_mode='fixed'|'lambda'|'raw_lam' 일 때 q_base로 사용, 기본 0.003)
  lam_mean  : float  (q_mode='lambda'|'raw_lam' 일 때 기준 λ, 기본 2.5)
  q_regime_table : dict (q_mode='regime' 일 때 사용)
  omega_mode: 'he_litterman' | 'scaled'
  omega_scale: float (omega_mode='scaled' 일 때 사용, 기본 1.0)
  prior     : 'capm_mcap' | 'capm_eq' | 'capm_rp'   # capm_rp = 1/σ 정규화 Risk Parity
  tc        : float  (거래비용, 편도 turnover 기준, 기본 0.001 = 10bp)
  max_weight: float  (단일 종목 상한, 기본 0.10)
  lstm_pred_path: str | None  (p_mode='lstm_predicted' 또는 omega_mode='rmse' 시 경로)
"""
from pathlib import Path

# ── LSTM 예측 파일 경로 ────────────────────────────────────────────────────────
# 위치에 따라 자동 탐색. 없으면 LSTM 실험은 자동 스킵됨.
_PHASE3_DIR = Path(__file__).parent / 'phase3(data_outputs)'
_LSTM_PRED_DEFAULT = _PHASE3_DIR / 'data' / 'ensemble_predictions_stockwise.csv'

# ── 기준선 실험 (모든 실험의 default 값) ────────────────────────────────────
BASELINE = {
    'name'        : 'baseline',
    'p_mode'      : 'trailing_vol21',   # vol_21d 기준 분류
    'p_weight'    : 'mcap',             # 시총가중 P
    'q_mode'      : 'fixed',
    'q_value'     : 0.003,              # 월 0.3% (연 3.6%)
    'omega_mode'  : 'he_litterman',     # τ·P·Σ·P^T
    'omega_scale' : 1.0,
    'prior'       : 'capm_mcap',        # 시총가중 균형수익률
    'tc'          : 0.001,              # 10bp
    'max_weight'  : 0.10,               # 여기는 팀 합의가 필요한 부분. 몇 %까지 가져갈지에 대한 내용
    'lstm_pred_path': str(_LSTM_PRED_DEFAULT),
}

# ── 실험 목록 ────────────────────────────────────────────────────────────────
EXPERIMENTS = [

    # ── 기준선 (CAPM 시총가중 Prior, P 시총가중, vol21) ───────────────────────
    BASELINE,

    # ── [Prior] CAPM 시총가중 vs 1/N 균등가중 ────────────────────────────────
    {**BASELINE, 'name': 'prior_eq',
     'prior': 'capm_eq'},               # 1/N 균등가중 prior

    # ── [P 슬롯] P 행렬 가중 방식 ───────────────────────────────────────────────
    {**BASELINE, 'name': 'p_rp',
     'p_weight': 'rp'},                 # 1/σ 역변동성 가중

    {**BASELINE, 'name': 'p_eq',
     'p_weight': 'eq'},                 # 동일가중

    {**BASELINE, 'name': 'p_vol_mcap',
     'p_weight': 'vol_mcap'},           # 롱 (1/σ)×mcap, 숏 σ×mcap

    # ── [비교군] BL 없음 ──────────────────────────────────────────────────────
    {**BASELINE, 'name': 'capm_no_bl',
     'q_mode': 'capm'},                 # CAPM prior π 직접 최적화, 전체 유니버스, BL 없음

    {**BASELINE, 'name': 'naive_lowvol',
     'q_mode': 'none'},                 # 저변동 시총가중 직접 보유 (BL 생략)

    # ── [LSTM] CAPM 시총가중 Prior × LSTM 예측 vol ───────────────────────────
    {**BASELINE, 'name': 'p_lstm_mcap',
     'p_mode': 'lstm_predicted'},

    {**BASELINE, 'name': 'p_lstm_eq',
     'p_mode': 'lstm_predicted', 'p_weight': 'eq'},

    {**BASELINE, 'name': 'p_lstm_rp',
     'p_mode': 'lstm_predicted', 'p_weight': 'rp'},

    {**BASELINE, 'name': 'p_lstm_vol_mcap',
     'p_mode': 'lstm_predicted', 'p_weight': 'vol_mcap'},

    # ── [LSTM] 1/N Prior × LSTM 예측 vol ─────────────────────────────────────
    {**BASELINE, 'name': 'prior_eq_p_lstm_mcap',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted'},

    {**BASELINE, 'name': 'prior_eq_p_lstm_eq',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted', 'p_weight': 'eq'},

    {**BASELINE, 'name': 'prior_eq_p_lstm_rp',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted', 'p_weight': 'rp'},

    {**BASELINE, 'name': 'prior_eq_p_lstm_vol_mcap',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted', 'p_weight': 'vol_mcap'},

    # ── [Q_lambda] 시장 위험회피계수 λ 기반 Q 조절 ───────────────────────────
    # Q = q_base × clip(λ / lam_mean, 0.1, 3.0)
    # 시장 안정(λ↑) → Q 강화 / 시장 불안(λ↓) → Q 약화
    {**BASELINE, 'name': 'q_lambda',
     'q_mode': 'lambda', 'q_value': 0.003, 'lam_mean': 2.5},

    # Q_lambda × LSTM vol P 조합
    {**BASELINE, 'name': 'q_lambda_p_lstm',
     'q_mode': 'lambda', 'q_value': 0.003, 'lam_mean': 2.5,
     'p_mode': 'lstm_predicted'},

    # ── [inv_lambda_Q] 역방향 λ: 위기일수록 Q 강화 ───────────────────────────
    # λ 낮음(위기) → Q 높음, λ 높음(강세) → Q 낮음
    {**BASELINE, 'name': 'q_inv_lambda',
     'q_mode': 'inv_lambda', 'q_value': 0.003, 'lam_mean': 2.5},

    # inv_lambda × LSTM vol P 조합
    {**BASELINE, 'name': 'q_inv_lambda_p_lstm',
     'q_mode': 'inv_lambda', 'q_value': 0.003, 'lam_mean': 2.5,
     'p_mode': 'lstm_predicted'},

    # ── [raw_lam_Q] raw λ 부호 기반 자연 게이팅 ─────────────────────────────
    # SPY 하락 → lam_raw 음수 → Q=0 자연 도달 (하드스탑 없이)
    {**BASELINE, 'name': 'q_raw_lam',
     'q_mode': 'raw_lam', 'q_value': 0.003, 'lam_mean': 2.5},

    # ── [최고성과 조합] prior_eq × lstm_rp × q_lambda ────────────────────────
    {**BASELINE, 'name': 'prior_eq_p_lstm_rp_q_lambda',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted', 'p_weight': 'rp',
     'q_mode': 'lambda', 'q_value': 0.003, 'lam_mean': 2.5},

    # inv_lambda × LSTM P × prior_eq
    {**BASELINE, 'name': 'prior_eq_q_inv_lambda_p_lstm',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted', 'p_weight': 'mcap',
     'q_mode': 'inv_lambda', 'q_value': 0.003, 'lam_mean': 2.5},

    # lambda × LSTM P × prior_eq
    {**BASELINE, 'name': 'prior_eq_q_lambda_p_lstm',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted', 'p_weight': 'mcap',
     'q_mode': 'lambda', 'q_value': 0.003, 'lam_mean': 2.5},

    # ── [논문 FF3] Q = 직전월 실현팩터 (Ω = he_litterman) ────────────────────
    {**BASELINE, 'name': 'q_ff3_paper',
     'q_mode': 'ff3_paper'},

    # ── [논문 완전 구현] Q = 직전월 실현팩터, Ω = 전월 예측오차² ─────────────
    {**BASELINE, 'name': 'q_ff3_paper_omega_paper',
     'q_mode': 'ff3_paper', 'omega_mode': 'ff3_paper'},

    # ── [논문 Ω만] fixed Q + 논문 방식 Ω ──────────────────────────────────────
    {**BASELINE, 'name': 'omega_paper',
     'omega_mode': 'ff3_paper'},

    # ── [논문 Ω + LSTM P] ──────────────────────────────────────────────────────
    {**BASELINE, 'name': 'omega_paper_p_lstm',
     'omega_mode': 'ff3_paper', 'p_mode': 'lstm_predicted'},

    # ════════════════════════════════════════════════════════════════════════
    # [윤서 추가 실험] prior=1/N × Q 동적 (LSTM 전용 정책: trailing 버전은 제외)
    # ════════════════════════════════════════════════════════════════════════
    # 아래 2개만 trailing 유지 (이미 .pkl 있음, 비교용)
    # 5개 trailing은 LSTM 전용 정책에 따라 제거: p_vol_mcap_q_lambda,
    # p_vol_mcap_q_raw_lam, prior_eq_p_vol_mcap, prior_eq_p_vol_mcap_q_lambda,
    # prior_eq_p_vol_mcap_q_raw_lam (.pkl은 보존됨)
    # ════════════════════════════════════════════════════════════════════════

    {**BASELINE, 'name': 'prior_eq_q_lambda',
     'prior': 'capm_eq', 'q_mode': 'lambda', 'q_value': 0.003, 'lam_mean': 2.5},

    {**BASELINE, 'name': 'prior_eq_q_raw_lam',
     'prior': 'capm_eq', 'q_mode': 'raw_lam', 'q_value': 0.003, 'lam_mean': 2.5},

    # ════════════════════════════════════════════════════════════════════════
    # [윤서 추가 LSTM 버전] 위 7개 실험에 LSTM 예측 vol 결합 (5개 신규)
    # ════════════════════════════════════════════════════════════════════════
    # 참고: 다음 2개는 팀이 이미 추가함
    #   - prior_eq_q_lambda + LSTM      = prior_eq_q_lambda_p_lstm (기존)
    #   - prior_eq_p_vol_mcap + LSTM    = prior_eq_p_lstm_vol_mcap (기존)
    # ════════════════════════════════════════════════════════════════════════

    # ── prior=1/N × Q raw_lam × LSTM ──────────────────────────────────────
    {**BASELINE, 'name': 'prior_eq_q_raw_lam_p_lstm',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted',
     'q_mode': 'raw_lam', 'q_value': 0.003, 'lam_mean': 2.5},

    # ── p=vol_mcap × Q 동적 × LSTM ────────────────────────────────────────
    {**BASELINE, 'name': 'p_vol_mcap_q_lambda_p_lstm',
     'p_mode': 'lstm_predicted', 'p_weight': 'vol_mcap',
     'q_mode': 'lambda', 'q_value': 0.003, 'lam_mean': 2.5},

    {**BASELINE, 'name': 'p_vol_mcap_q_raw_lam_p_lstm',
     'p_mode': 'lstm_predicted', 'p_weight': 'vol_mcap',
     'q_mode': 'raw_lam', 'q_value': 0.003, 'lam_mean': 2.5},

    # ── 모두 변경 + LSTM: 1/N + vol_mcap + Q 동적 + LSTM ──────────────────
    {**BASELINE, 'name': 'prior_eq_p_vol_mcap_q_lambda_p_lstm',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted', 'p_weight': 'vol_mcap',
     'q_mode': 'lambda', 'q_value': 0.003, 'lam_mean': 2.5},

    {**BASELINE, 'name': 'prior_eq_p_vol_mcap_q_raw_lam_p_lstm',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted', 'p_weight': 'vol_mcap',
     'q_mode': 'raw_lam', 'q_value': 0.003, 'lam_mean': 2.5},

    # ════════════════════════════════════════════════════════════════════════
    # [윤서 LSTM 4×4 매트릭스] prior=1/N + LSTM 고정, p_weight × q_mode 조합
    # ════════════════════════════════════════════════════════════════════════
    # p_weight ∈ {mcap, eq, rp, vol_mcap}
    # q_mode ∈ {fixed, raw_lam, lambda, vol_spread}
    # 16개 중 9개는 기존 실험으로 커버, 7개를 신규 추가
    # ════════════════════════════════════════════════════════════════════════

    # ── mcap × vol_spread ─────────────────────────────────────────────────
    {**BASELINE, 'name': 'prior_eq_p_lstm_mcap_q_vol_spread',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted', 'p_weight': 'mcap',
     'q_mode': 'vol_spread', 'q_value': 0.003},

    # ── eq × {raw_lam, lambda, vol_spread} ────────────────────────────────
    {**BASELINE, 'name': 'prior_eq_p_lstm_eq_q_raw_lam',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted', 'p_weight': 'eq',
     'q_mode': 'raw_lam', 'q_value': 0.003, 'lam_mean': 2.5},

    {**BASELINE, 'name': 'prior_eq_p_lstm_eq_q_lambda',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted', 'p_weight': 'eq',
     'q_mode': 'lambda', 'q_value': 0.003, 'lam_mean': 2.5},

    {**BASELINE, 'name': 'prior_eq_p_lstm_eq_q_vol_spread',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted', 'p_weight': 'eq',
     'q_mode': 'vol_spread', 'q_value': 0.003},

    # ── rp × {raw_lam, vol_spread} ────────────────────────────────────────
    {**BASELINE, 'name': 'prior_eq_p_lstm_rp_q_raw_lam',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted', 'p_weight': 'rp',
     'q_mode': 'raw_lam', 'q_value': 0.003, 'lam_mean': 2.5},

    {**BASELINE, 'name': 'prior_eq_p_lstm_rp_q_vol_spread',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted', 'p_weight': 'rp',
     'q_mode': 'vol_spread', 'q_value': 0.003},

    # ── vol_mcap × vol_spread ─────────────────────────────────────────────
    {**BASELINE, 'name': 'prior_eq_p_lstm_vol_mcap_q_vol_spread',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted', 'p_weight': 'vol_mcap',
     'q_mode': 'vol_spread', 'q_value': 0.003},

    # ════════════════════════════════════════════════════════════════════════
    # [윤서 RP Prior] capm_rp (Risk Parity, 1/σ 정규화)
    # ════════════════════════════════════════════════════════════════════════
    # 04 Part 3 미니 BL: 1/N과 RP의 w* 상관 0.98 → 거의 동등 예상
    # final 백테스트로 portfolio Sharpe 직접 검증
    # ════════════════════════════════════════════════════════════════════════

    # ── 기본 비교: prior_rp (baseline + RP만 변경) ─────────────────────
    {**BASELINE, 'name': 'prior_rp',
     'prior': 'capm_rp'},

    # ── prior=RP × LSTM × P 가중치 4개 ───────────────────────────────
    {**BASELINE, 'name': 'prior_rp_p_lstm_mcap',
     'prior': 'capm_rp', 'p_mode': 'lstm_predicted'},

    {**BASELINE, 'name': 'prior_rp_p_lstm_eq',
     'prior': 'capm_rp', 'p_mode': 'lstm_predicted', 'p_weight': 'eq'},

    {**BASELINE, 'name': 'prior_rp_p_lstm_rp',
     'prior': 'capm_rp', 'p_mode': 'lstm_predicted', 'p_weight': 'rp'},

    {**BASELINE, 'name': 'prior_rp_p_lstm_vol_mcap',
     'prior': 'capm_rp', 'p_mode': 'lstm_predicted', 'p_weight': 'vol_mcap'},

    # ── prior=RP × LSTM × Q 동적 (lambda, raw_lam, vol_spread) ─────────
    {**BASELINE, 'name': 'prior_rp_p_lstm_mcap_q_lambda',
     'prior': 'capm_rp', 'p_mode': 'lstm_predicted',
     'q_mode': 'lambda', 'q_value': 0.003, 'lam_mean': 2.5},

    {**BASELINE, 'name': 'prior_rp_p_lstm_mcap_q_raw_lam',
     'prior': 'capm_rp', 'p_mode': 'lstm_predicted',
     'q_mode': 'raw_lam', 'q_value': 0.003, 'lam_mean': 2.5},

    {**BASELINE, 'name': 'prior_rp_p_lstm_mcap_q_vol_spread',
     'prior': 'capm_rp', 'p_mode': 'lstm_predicted',
     'q_mode': 'vol_spread', 'q_value': 0.003},

    # ════════════════════════════════════════════════════════════════════════
    # [윤서 Ω 슬롯 비교군] omega_paper(=ff3_paper)와 같은 레벨에 scaled/rmse 추가
    # ════════════════════════════════════════════════════════════════════════
    # 이미 존재: omega_paper (fixed Q + ff3_paper), omega_paper_p_lstm
    # 신규: scaled(0.5/2.0), rmse — 각각 trailing/LSTM 두 버전
    # ════════════════════════════════════════════════════════════════════════

    {**BASELINE, 'name': 'omega_scaled_half',
     'omega_mode': 'scaled', 'omega_scale': 0.5},

    {**BASELINE, 'name': 'omega_scaled_half_p_lstm',
     'omega_mode': 'scaled', 'omega_scale': 0.5, 'p_mode': 'lstm_predicted'},

    {**BASELINE, 'name': 'omega_scaled_double',
     'omega_mode': 'scaled', 'omega_scale': 2.0},

    {**BASELINE, 'name': 'omega_scaled_double_p_lstm',
     'omega_mode': 'scaled', 'omega_scale': 2.0, 'p_mode': 'lstm_predicted'},

    {**BASELINE, 'name': 'omega_rmse',
     'omega_mode': 'rmse'},

    {**BASELINE, 'name': 'omega_rmse_p_lstm',
     'omega_mode': 'rmse', 'p_mode': 'lstm_predicted'},

    # ════════════════════════════════════════════════════════════════════════
    # [윤서 Ω 비교 매트릭스] prior_eq + LSTM + p=mcap × {q_lambda, q_raw_lam} × Ω 4종
    # ════════════════════════════════════════════════════════════════════════
    # 이미 존재: q × omega=he_litterman 2개
    #   prior_eq_q_lambda_p_lstm, prior_eq_q_raw_lam_p_lstm
    # 신규: 2 q × {scaled_half, scaled_double, rmse, ff3_paper} = 8개
    # ════════════════════════════════════════════════════════════════════════

    # ── q_lambda × Ω 4종 ────────────────────────────────────────────────
    {**BASELINE, 'name': 'prior_eq_q_lambda_p_lstm_omega_scaled_half',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted',
     'q_mode': 'lambda', 'q_value': 0.003, 'lam_mean': 2.5,
     'omega_mode': 'scaled', 'omega_scale': 0.5},

    {**BASELINE, 'name': 'prior_eq_q_lambda_p_lstm_omega_scaled_double',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted',
     'q_mode': 'lambda', 'q_value': 0.003, 'lam_mean': 2.5,
     'omega_mode': 'scaled', 'omega_scale': 2.0},

    {**BASELINE, 'name': 'prior_eq_q_lambda_p_lstm_omega_rmse',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted',
     'q_mode': 'lambda', 'q_value': 0.003, 'lam_mean': 2.5,
     'omega_mode': 'rmse'},

    {**BASELINE, 'name': 'prior_eq_q_lambda_p_lstm_omega_paper',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted',
     'q_mode': 'lambda', 'q_value': 0.003, 'lam_mean': 2.5,
     'omega_mode': 'ff3_paper'},

    # ── q_raw_lam × Ω 4종 ───────────────────────────────────────────────
    {**BASELINE, 'name': 'prior_eq_q_raw_lam_p_lstm_omega_scaled_half',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted',
     'q_mode': 'raw_lam', 'q_value': 0.003, 'lam_mean': 2.5,
     'omega_mode': 'scaled', 'omega_scale': 0.5},

    {**BASELINE, 'name': 'prior_eq_q_raw_lam_p_lstm_omega_scaled_double',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted',
     'q_mode': 'raw_lam', 'q_value': 0.003, 'lam_mean': 2.5,
     'omega_mode': 'scaled', 'omega_scale': 2.0},

    {**BASELINE, 'name': 'prior_eq_q_raw_lam_p_lstm_omega_rmse',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted',
     'q_mode': 'raw_lam', 'q_value': 0.003, 'lam_mean': 2.5,
     'omega_mode': 'rmse'},

    {**BASELINE, 'name': 'prior_eq_q_raw_lam_p_lstm_omega_paper',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted',
     'q_mode': 'raw_lam', 'q_value': 0.003, 'lam_mean': 2.5,
     'omega_mode': 'ff3_paper'},

    # ════════════════════════════════════════════════════════════════════════
    # [윤서 prior_rp 매트릭스] prior=rp + LSTM × q ∈ {lambda, raw_lam} × p ∈ {mcap, eq, rp} × Ω ∈ {he, paper, rmse}
    # ════════════════════════════════════════════════════════════════════════
    # 기존 2개: prior_rp_p_lstm_mcap_q_lambda, prior_rp_p_lstm_mcap_q_raw_lam (둘 다 ω=he)
    # 신규 16개로 18-cell 매트릭스 완성
    # ════════════════════════════════════════════════════════════════════════

    # ── q=lambda × p ∈ {eq, rp} × ω=he (2개) ─────────────────────
    {**BASELINE, 'name': 'prior_rp_p_lstm_eq_q_lambda',
     'prior': 'capm_rp', 'p_mode': 'lstm_predicted', 'p_weight': 'eq',
     'q_mode': 'lambda', 'q_value': 0.003, 'lam_mean': 2.5},

    {**BASELINE, 'name': 'prior_rp_p_lstm_rp_q_lambda',
     'prior': 'capm_rp', 'p_mode': 'lstm_predicted', 'p_weight': 'rp',
     'q_mode': 'lambda', 'q_value': 0.003, 'lam_mean': 2.5},

    # ── q=lambda × p × ω ∈ {paper, rmse} (6개) ───────────────────
    {**BASELINE, 'name': 'prior_rp_p_lstm_mcap_q_lambda_omega_paper',
     'prior': 'capm_rp', 'p_mode': 'lstm_predicted',
     'q_mode': 'lambda', 'q_value': 0.003, 'lam_mean': 2.5,
     'omega_mode': 'ff3_paper'},

    {**BASELINE, 'name': 'prior_rp_p_lstm_mcap_q_lambda_omega_rmse',
     'prior': 'capm_rp', 'p_mode': 'lstm_predicted',
     'q_mode': 'lambda', 'q_value': 0.003, 'lam_mean': 2.5,
     'omega_mode': 'rmse'},

    {**BASELINE, 'name': 'prior_rp_p_lstm_eq_q_lambda_omega_paper',
     'prior': 'capm_rp', 'p_mode': 'lstm_predicted', 'p_weight': 'eq',
     'q_mode': 'lambda', 'q_value': 0.003, 'lam_mean': 2.5,
     'omega_mode': 'ff3_paper'},

    {**BASELINE, 'name': 'prior_rp_p_lstm_eq_q_lambda_omega_rmse',
     'prior': 'capm_rp', 'p_mode': 'lstm_predicted', 'p_weight': 'eq',
     'q_mode': 'lambda', 'q_value': 0.003, 'lam_mean': 2.5,
     'omega_mode': 'rmse'},

    {**BASELINE, 'name': 'prior_rp_p_lstm_rp_q_lambda_omega_paper',
     'prior': 'capm_rp', 'p_mode': 'lstm_predicted', 'p_weight': 'rp',
     'q_mode': 'lambda', 'q_value': 0.003, 'lam_mean': 2.5,
     'omega_mode': 'ff3_paper'},

    {**BASELINE, 'name': 'prior_rp_p_lstm_rp_q_lambda_omega_rmse',
     'prior': 'capm_rp', 'p_mode': 'lstm_predicted', 'p_weight': 'rp',
     'q_mode': 'lambda', 'q_value': 0.003, 'lam_mean': 2.5,
     'omega_mode': 'rmse'},

    # ── q=raw_lam × p ∈ {eq, rp} × ω=he (2개) ────────────────────
    {**BASELINE, 'name': 'prior_rp_p_lstm_eq_q_raw_lam',
     'prior': 'capm_rp', 'p_mode': 'lstm_predicted', 'p_weight': 'eq',
     'q_mode': 'raw_lam', 'q_value': 0.003, 'lam_mean': 2.5},

    {**BASELINE, 'name': 'prior_rp_p_lstm_rp_q_raw_lam',
     'prior': 'capm_rp', 'p_mode': 'lstm_predicted', 'p_weight': 'rp',
     'q_mode': 'raw_lam', 'q_value': 0.003, 'lam_mean': 2.5},

    # ── q=raw_lam × p × ω ∈ {paper, rmse} (6개) ──────────────────
    {**BASELINE, 'name': 'prior_rp_p_lstm_mcap_q_raw_lam_omega_paper',
     'prior': 'capm_rp', 'p_mode': 'lstm_predicted',
     'q_mode': 'raw_lam', 'q_value': 0.003, 'lam_mean': 2.5,
     'omega_mode': 'ff3_paper'},

    {**BASELINE, 'name': 'prior_rp_p_lstm_mcap_q_raw_lam_omega_rmse',
     'prior': 'capm_rp', 'p_mode': 'lstm_predicted',
     'q_mode': 'raw_lam', 'q_value': 0.003, 'lam_mean': 2.5,
     'omega_mode': 'rmse'},

    {**BASELINE, 'name': 'prior_rp_p_lstm_eq_q_raw_lam_omega_paper',
     'prior': 'capm_rp', 'p_mode': 'lstm_predicted', 'p_weight': 'eq',
     'q_mode': 'raw_lam', 'q_value': 0.003, 'lam_mean': 2.5,
     'omega_mode': 'ff3_paper'},

    {**BASELINE, 'name': 'prior_rp_p_lstm_eq_q_raw_lam_omega_rmse',
     'prior': 'capm_rp', 'p_mode': 'lstm_predicted', 'p_weight': 'eq',
     'q_mode': 'raw_lam', 'q_value': 0.003, 'lam_mean': 2.5,
     'omega_mode': 'rmse'},

    {**BASELINE, 'name': 'prior_rp_p_lstm_rp_q_raw_lam_omega_paper',
     'prior': 'capm_rp', 'p_mode': 'lstm_predicted', 'p_weight': 'rp',
     'q_mode': 'raw_lam', 'q_value': 0.003, 'lam_mean': 2.5,
     'omega_mode': 'ff3_paper'},

    {**BASELINE, 'name': 'prior_rp_p_lstm_rp_q_raw_lam_omega_rmse',
     'prior': 'capm_rp', 'p_mode': 'lstm_predicted', 'p_weight': 'rp',
     'q_mode': 'raw_lam', 'q_value': 0.003, 'lam_mean': 2.5,
     'omega_mode': 'rmse'},

    # ── q_inv_lambda × Ω 4종 (매트릭스 완성) ─────────────────────────────
    {**BASELINE, 'name': 'prior_eq_q_inv_lambda_p_lstm_omega_scaled_half',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted',
     'q_mode': 'inv_lambda', 'q_value': 0.003, 'lam_mean': 2.5,
     'omega_mode': 'scaled', 'omega_scale': 0.5},

    {**BASELINE, 'name': 'prior_eq_q_inv_lambda_p_lstm_omega_scaled_double',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted',
     'q_mode': 'inv_lambda', 'q_value': 0.003, 'lam_mean': 2.5,
     'omega_mode': 'scaled', 'omega_scale': 2.0},

    {**BASELINE, 'name': 'prior_eq_q_inv_lambda_p_lstm_omega_rmse',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted',
     'q_mode': 'inv_lambda', 'q_value': 0.003, 'lam_mean': 2.5,
     'omega_mode': 'rmse'},

    {**BASELINE, 'name': 'prior_eq_q_inv_lambda_p_lstm_omega_paper',
     'prior': 'capm_eq', 'p_mode': 'lstm_predicted',
     'q_mode': 'inv_lambda', 'q_value': 0.003, 'lam_mean': 2.5,
     'omega_mode': 'ff3_paper'},

]


def get_changed_slots(cfg: dict, baseline: dict = None) -> set:
    """
    baseline 대비 바뀐 슬롯 이름 반환.
    99_analyze.ipynb에서 조건부 차트 선택에 사용.
    """
    if baseline is None:
        baseline = BASELINE
    ignore = {'name', 'lstm_pred_path'}
    return {k for k in set(cfg) | set(baseline)
            if k not in ignore and cfg.get(k) != baseline.get(k)}
