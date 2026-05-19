"""
_run_all_parallel.py — 04 노트북의 walk_forward 실행을 N 워커 병렬로.

i5-1135G7 (4코어/8thread) 기준 권장 N=3. 25h → ~7-9h 단축.

사용 (final_pt 루트에서):
    python _dev/_run_all_parallel.py            # 기본 n_workers=3
    python _dev/_run_all_parallel.py 4          # 4 워커 (CPU 좀 더 빠듯)

설계
----
1. main 프로세스: 데이터 + monthly_cache 1회 빌드 → _cache_temp.pkl 저장
2. ProcessPoolExecutor 로 N 워커 spawn
3. 각 워커: cache pkl 1회 로드 (initializer) → 슬롯들 처리
4. 결과 → results/{name}.pkl (04 노트북과 동일 형식)

안전장치
--------
- 기존 results/{name}.pkl 있으면 skip (SKIP_IF_EXISTS=True 기본 동작)
- p_mode 가용성 게이팅 (ANN CSV 없으면 ANN 슬롯 skip)
- 각 슬롯 에러는 isolated — 다른 슬롯 영향 X
"""
import sys, os, time, pickle, io
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Windows cp949 -> UTF-8 (✓/✗ 같은 unicode 출력용)
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))

from bl_runner import load_pred_csv, build_monthly_cache, walk_forward
from bl_config import EXPERIMENTS as EXPERIMENTS_LSTM, BASELINE
from bl_config_ann import EXPERIMENTS as EXPERIMENTS_ANN

EXPERIMENTS = EXPERIMENTS_LSTM + EXPERIMENTS_ANN

DATA_DIR    = PROJECT / 'data'
RESULTS_DIR = PROJECT / 'results'
CACHE_PKL   = HERE / '_state_temp.pkl'

TRAIN_WINDOW = 60
THRESH_DAILY = 0.9
TAU          = 0.1
PCT_GROUP    = 0.30
START_PRED   = '2010-01-01'


def _build_state():
    """data + cache + 예측 state 빌드 (main 프로세스에서 1회)."""
    print(f'[main] 데이터 로드')
    panel = pd.read_csv(DATA_DIR / 'monthly_panel.csv', parse_dates=['date']).set_index(['date','ticker'])
    daily_ret = pd.read_pickle(DATA_DIR / 'daily_returns.pkl')

    all_dates  = panel.index.get_level_values('date').unique().sort_values()
    pred_dates = all_dates[all_dates >= START_PRED]
    spy_series = panel['spy_ret'].groupby(level='date').first()
    rf_series  = panel['rf_1m'].groupby(level='date').first()
    ret_pivot  = panel['ret_1m'].unstack('ticker')
    ff3        = pd.read_csv(DATA_DIR / 'ff3_monthly.csv', index_col=0, parse_dates=True)

    print(f'[main] 예측 CSV 로드 (LSTM + ANN)')
    lstm_state = load_pred_csv(BASELINE['lstm_pred_path'], pred_dates)
    ann_state  = load_pred_csv(DATA_DIR / 'paper_ann_predictions.csv', pred_dates)
    print(f'  LSTM available: {lstm_state["available"]}')
    print(f'  ANN  available: {ann_state["available"]}')

    print(f'[main] monthly_cache 빌드 ({len(pred_dates)}개월)')
    t0 = time.time()
    cache = build_monthly_cache(
        panel=panel, daily_ret=daily_ret,
        pred_dates=pred_dates, all_dates=all_dates,
        spy_series=spy_series, rf_series=rf_series,
        train_window=TRAIN_WINDOW, thresh_daily=THRESH_DAILY, verbose=False,
    )
    print(f'  cache 완료: {(time.time()-t0)/60:.1f}분, {len(cache)} entries')

    return dict(
        monthly_cache = cache,
        pred_dates    = pred_dates,
        spy_series    = spy_series,
        rf_series     = rf_series,
        ret_pivot     = ret_pivot,
        ff3           = ff3,
        lstm_state    = lstm_state,
        ann_state     = ann_state,
    )


# ── 워커 전역 (각 워커 프로세스 안) ────────────────────────────
_STATE = None


def _worker_init():
    """워커 시작 시 1회 — state pkl 로드."""
    global _STATE
    with open(CACHE_PKL, 'rb') as f:
        _STATE = pickle.load(f)


def _worker_run(cfg):
    """워커 — 단일 슬롯 walk_forward 실행 + pkl 저장."""
    global _STATE
    name = cfg['name']
    out_path = RESULTS_DIR / f'{name}.pkl'

    if out_path.exists():
        return ('skip-exists', name, 0.0)

    p_mode = cfg.get('p_mode', 'lstm_predicted')
    if p_mode == 'lstm_predicted' and not _STATE['lstm_state']['available']:
        return ('skip-noavail', name, 0.0)
    if p_mode == 'ann_predicted'  and not _STATE['ann_state']['available']:
        return ('skip-noavail', name, 0.0)

    t0 = time.time()
    try:
        result = walk_forward(
            cfg, _STATE['monthly_cache'], _STATE['pred_dates'], _STATE['lstm_state'],
            spy_series=_STATE['spy_series'], tau=TAU, pct_group=PCT_GROUP, verbose=False,
            ann_state=_STATE['ann_state'],
            ret_pivot=_STATE['ret_pivot'], ff3=_STATE['ff3'], rf_series=_STATE['rf_series'],
        )
        with open(out_path, 'wb') as f:
            pickle.dump(result, f)
        return ('done', name, time.time() - t0)
    except Exception as e:
        import traceback
        return ('error', name, f'{type(e).__name__}: {e}\n{traceback.format_exc()[:500]}')


def main():
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(f'=== _run_all_parallel.py | N_WORKERS={n_workers} ===\n')

    RESULTS_DIR.mkdir(exist_ok=True)

    # 1. state 빌드 (main 프로세스, 1회)
    t_setup = time.time()
    state = _build_state()
    print(f'\n[main] state 임시 저장: {CACHE_PKL}')
    with open(CACHE_PKL, 'wb') as f:
        pickle.dump(state, f, protocol=4)
    print(f'  완료 ({CACHE_PKL.stat().st_size / 1024**2:.1f} MB)')
    setup_min = (time.time() - t_setup) / 60
    print(f'[main] 셋업 합계: {setup_min:.1f}분\n')

    # 2. 실행 대상 필터
    run_list = []
    skip_exists = 0
    for cfg in EXPERIMENTS:
        if (RESULTS_DIR / f"{cfg['name']}.pkl").exists():
            skip_exists += 1
            continue
        run_list.append(cfg)
    print(f'[main] 실행 대상: {len(run_list)} / {len(EXPERIMENTS)} (skip-exists {skip_exists})')
    print(f'  - LSTM 슬롯: {sum(1 for c in run_list if c.get("p_mode")=="lstm_predicted")}')
    print(f'  - ANN  슬롯: {sum(1 for c in run_list if c.get("p_mode")=="ann_predicted")}')

    if not run_list:
        print('실행할 슬롯 없음 — 종료')
        CACHE_PKL.unlink(missing_ok=True)
        return

    # 3. 병렬 실행
    print(f'\n[main] {n_workers} 워커 spawn — 슬롯 처리 시작')
    t_all = time.time()
    completed, errors = [], []
    elapsed_per_slot = []

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init) as ex:
        futures = {ex.submit(_worker_run, cfg): cfg['name'] for cfg in run_list}

        for fut in as_completed(futures):
            status, name, info = fut.result()
            elapsed_all = (time.time() - t_all) / 60
            n_done = len(completed) + len(errors) + 1

            if status == 'done':
                completed.append(name)
                elapsed_per_slot.append(info)
                avg = np.mean(elapsed_per_slot[-20:]) / 60   # 최근 20 평균 (분)
                remaining = len(run_list) - n_done
                eta = avg * remaining / n_workers   # 병렬 보정
                print(f'  [{n_done}/{len(run_list)}] ✓ {name} ({info:.1f}s) | '
                      f'전체 {elapsed_all:.1f}분 | ETA {eta:.1f}분',
                      flush=True)
            elif status == 'error':
                errors.append((name, info))
                print(f'  [{n_done}/{len(run_list)}] ✗ {name}: {info[:200]}', flush=True)
            elif status.startswith('skip'):
                print(f'  [{n_done}/{len(run_list)}] - {name}: {status}', flush=True)

    # 4. 정리
    CACHE_PKL.unlink(missing_ok=True)
    total = (time.time() - t_all) / 60
    print(f'\n{"="*60}')
    print(f'완료: {len(completed)}  / 에러: {len(errors)} / 총 {total:.1f}분 ({total/60:.1f}시간)')
    print(f'결과: {RESULTS_DIR}')
    if errors:
        print(f'\n=== 에러 슬롯 ({len(errors)}개) ===')
        for name, info in errors[:10]:
            print(f'  {name}: {info[:200]}')


if __name__ == '__main__':
    main()
