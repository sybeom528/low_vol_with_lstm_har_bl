"""
bl_config_ann.py — ANN baseline (Pyo & Lee 2018) 실험 정의

bl_config.EXPERIMENTS 의 LSTM 매트릭스 108 슬롯을 1:1 미러링하여
p_mode='ann_predicted', name 끝에 '_ann' 접미사 부여.

용도
----
ANN 학습 결과 (data/paper_ann_predictions.csv) 와 LSTM 결과를 동일한
BL 프레임워크 하에서 head-to-head 비교. 학술지 제출용 sensitivity table 의
ANN 베이스라인.

데이터 의존성
-------------
- data/paper_ann_predictions.csv  (_dev/_train_paper_ann.py 산출물)
  스키마: date, ticker, y_true, y_pred_ensemble
  y_pred_ensemble 단위: log(daily_std), LSTM 과 동일.
- ANN spec (Pyo & Lee 2018): MLP(10→4→1), Adam, lookback 10mo, train 60mo rolling.

ANN 학습 자체는 _dev/_train_paper_ann.py 로 walk-forward 수행 (각 pred_date 에서
직전 60개월 데이터만 사용 → leakage-free). 본 config 는 그 산출 CSV 를 단순 소비.

사용
----
    from bl_config import EXPERIMENTS as EXPERIMENTS_LSTM
    from bl_config_ann import EXPERIMENTS as EXPERIMENTS_ANN

    ALL = EXPERIMENTS_LSTM + EXPERIMENTS_ANN   # 216개

    # walk_forward 호출 시 ann_state 인자 전달:
    #   ann_state = load_pred_csv('data/paper_ann_predictions.csv', pred_dates)
"""
from pathlib import Path
from bl_config import EXPERIMENTS as _LSTM_EXPERIMENTS, BASELINE


_ANN_PRED_DEFAULT = Path(__file__).parent / 'data' / 'paper_ann_predictions.csv'


def _mirror_to_ann(cfg: dict) -> dict:
    """LSTM 슬롯 → ANN 슬롯 변환 (p_mode 교체 + name 접미사)."""
    new = {**cfg}
    new['p_mode'] = 'ann_predicted'
    new['name']   = cfg['name'] + '_ann'
    new['ann_pred_path'] = str(_ANN_PRED_DEFAULT)
    return new


# ── ANN 미러 매트릭스 (LSTM 108 슬롯 ↔ ANN 108 슬롯) ────────────
EXPERIMENTS = [_mirror_to_ann(cfg) for cfg in _LSTM_EXPERIMENTS]


# ── sanity check ───────────────────────────────────────────────
assert len(EXPERIMENTS) == len(_LSTM_EXPERIMENTS), \
    'ANN/LSTM 슬롯 수 불일치'
for ann_cfg, lstm_cfg in zip(EXPERIMENTS, _LSTM_EXPERIMENTS):
    assert ann_cfg['name'] == lstm_cfg['name'] + '_ann'
    assert ann_cfg['p_mode'] == 'ann_predicted'
    # p_mode 와 name 외 모든 키 동일
    for k in lstm_cfg:
        if k in ('name', 'p_mode'):
            continue
        assert ann_cfg.get(k) == lstm_cfg.get(k), \
            f'키 {k!r} 가 {ann_cfg["name"]} 에서 불일치'
