"""
02a_v2.ipynb + 02b_phase15_cross_sectional.ipynb 에 panel 확장 + incremental 학습 셀 추가/수정.

작업:
1. Panel 확장 셀 신규 추가 (학습 셀 직전, 1회 실행 후 자동 skip)
2. 학습 셀 수정: INCREMENTAL_TRAIN flag + run_ensemble_*_parallel(incremental=True)

사용:
    python scripts/_apply_panel_extension_changes.py [--end-date 2026-04-30]
"""
import json
import sys
import io
import uuid
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

NB_DIR = Path(__file__).resolve().parent.parent

# --- 02a_v2.ipynb 학습 셀 (Cell id = f96904f7) 수정 후 source ---
SEC02A_NEW_CELL_8_SOURCE = '''# §3. 8-way 병렬 학습 (V4_BEST_CONFIG, 전체 S&P 500)
# ⭐ 캐시 + Incremental 학습 (2026-05-01 v2)
import pandas as pd

out_path = DATA_DIR / 'ensemble_predictions_stockwise.csv'

# ⭐ Incremental 학습 (panel 확장 후 마지막 fold + 신규 fold 만 재학습)
INCREMENTAL_TRAIN = True   # ⭐ panel forward 확장 후 True. 사용 후 False 권장.
FORCE_RETRAIN = False      # True 시 캐시 무시 + 전체 재학습 (수 시간)

if INCREMENTAL_TRAIN:
    print('⚡ Incremental 학습 모드 (Panel 확장 후, 마지막 + 신규 fold 만)')
    print('  종목당 ~5 fold 학습 → ~10-30분 (8-way 병렬)')
    t0 = time.time()
    ensemble_sw = run_ensemble_for_universe_parallel(
        panel_csv=panel_path,
        universe_csv=universe_path,
        out_dir=DATA_DIR,
        config=V4_BEST_CONFIG,
        n_workers=8,
        out_name='ensemble_predictions_stockwise.csv',
        verbose=True,
        incremental=True,   # ⭐ 신규 인자 (volatility_ensemble.py)
    )
    elapsed = time.time() - t0
    print(f'\\n⏱️ Incremental 학습 시간: {elapsed/60:.1f} 분')
elif out_path.exists() and not FORCE_RETRAIN:
    print(f'⚡ 학습 결과 캐시 사용: {out_path.name} (재학습 생략)')
    print(f'  강제 재학습: FORCE_RETRAIN = True 변경 후 재실행 (GPU 비용 큼)')
    print(f'  Incremental 학습: INCREMENTAL_TRAIN = True 변경 후 재실행 (Panel 확장 후)')
    ensemble_sw = pd.read_csv(out_path, parse_dates=['date'])
    print(f'  ensemble_sw: {ensemble_sw.shape}')
    print(f'  unique 종목: {ensemble_sw["ticker"].nunique()}')
else:
    if FORCE_RETRAIN:
        print('⚠️ FORCE_RETRAIN=True → 8-way GPU 전체 재학습 (~수 시간)')
    t0 = time.time()
    ensemble_sw = run_ensemble_for_universe_parallel(
        panel_csv=panel_path,
        universe_csv=universe_path,
        out_dir=DATA_DIR,
        config=V4_BEST_CONFIG,
        n_workers=8,
        out_name='ensemble_predictions_stockwise.csv',
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f'\\n⏱️ 총 소요 시간: {elapsed/60:.1f} 분')

print(f'결과: {ensemble_sw.shape}')
print(f'  date 범위: {ensemble_sw["date"].min().date()} ~ {ensemble_sw["date"].max().date()}')
ensemble_sw.head()'''

# --- 02a/02b 공통: Panel 확장 셀 source (학습 셀 직전 추가) ---
PANEL_EXTENSION_CELL_SOURCE = '''# §1.5 Panel Forward 확장 (1회 실행, 재실행 시 자동 skip)
# ⭐ 2026-05-01 추가: Panel 끝점을 forward 확장 → 마지막 fold y_true 채움
#    LSTM 의 y_true = 다음 21일 변동성 → panel 끝점 부족 시 마지막 1개월 학습 누락
#    이 셀 1회 실행 → daily_panel.csv, market_data.csv, vix_daily.csv 갱신
from scripts.universe_extended import extend_panel_forward

PANEL_END_DATE = '2026-04-30'   # ⭐ 원하는 panel 끝점

stats = extend_panel_forward(
    end_date=PANEL_END_DATE,
    backup=True,        # 기존 파일 .bak 백업
    verbose=True,
)
print(f'\\n📊 Panel: {stats["panel_old_max"]} → {stats["panel_new_max"]}')
print(f'📊 추가 행: {stats["n_added_rows"]:+,d}')
print('\\n다음 단계: Cell 8 (학습 셀) 의 INCREMENTAL_TRAIN=True 확인 후 실행')'''

# --- 02b 학습 셀 수정 후 source ---
SEC02B_NEW_CELL_13_SOURCE = '''t0 = time.time()

# ⭐ Incremental 학습 (2026-05-01 추가, panel 확장 후 마지막 + 신규 fold 만)
INCREMENTAL_TRAIN = True   # ⭐ panel forward 확장 후 True. 사용 후 False 권장.

if INCREMENTAL_TRAIN:
    print('⚡ Cross-Sectional Incremental 학습 (panel 확장 후 마지막 + 신규 fold 만)')
    print('  ~5 fold 학습 → ~30-60분 (single GPU)')
    ensemble_cs = run_ensemble_cross_sectional(
        panel_csv=panel_path,
        universe_csv=universe_path,
        out_dir=DATA_DIR,
        config=CS_V4_BEST_CONFIG,
        device=device,
        use_har=True,
        out_name='ensemble_predictions_crosssec.csv',
        verbose=True,
        incremental=True,   # ⭐ 신규 인자
    )
else:
    ensemble_cs = run_ensemble_cross_sectional(
        panel_csv=panel_path,
        universe_csv=universe_path,           # ⭐ universe_full_history.csv (624 종목)
        out_dir=DATA_DIR,
        config=CS_V4_BEST_CONFIG,
        device=device,
        use_har=True,
        out_name='ensemble_predictions_crosssec.csv',
        verbose=True,
    )

elapsed = time.time() - t0
print(f'\\n⏱️ 총 소요 시간: {elapsed/60:.1f} 분')
print(f'결과: {ensemble_cs.shape}')
print(f'  date 범위: {ensemble_cs["date"].min().date()} ~ {ensemble_cs["date"].max().date()}')
ensemble_cs.head()'''


def split_src(s):
    lines = s.rstrip('\n').split('\n')
    return [ln + '\n' for ln in lines[:-1]] + [lines[-1]]


def make_cell(src_str, cell_type='code'):
    cell = {
        "cell_type": cell_type,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": split_src(src_str),
    }
    if cell_type == 'code':
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def modify_notebook(nb_path: Path, target_cell_id: str, new_source: str,
                    insert_before: bool = True, insert_source: str = None):
    """노트북 수정:
    - target_cell_id 셀의 source 를 new_source 로 교체
    - insert_before=True 시 그 셀 직전에 insert_source 의 신규 셀 추가
    """
    if not nb_path.exists():
        print(f'  ❌ 파일 부재: {nb_path.name}')
        return False

    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 1. target cell source 교체
    target_idx = None
    for i, cell in enumerate(nb['cells']):
        if cell.get('id') == target_cell_id:
            cell['source'] = split_src(new_source)
            cell['outputs'] = []
            cell['execution_count'] = None
            target_idx = i
            break

    if target_idx is None:
        print(f'  ⚠️ target cell id "{target_cell_id}" 부재 → 셀 위치 직접 확인 필요')
        return False

    # 2. 신규 셀 insert (이미 있으면 skip)
    if insert_before and insert_source is not None:
        # 중복 검사: 이미 'extend_panel_forward' 호출 셀 있으면 skip
        already_inserted = any(
            'extend_panel_forward' in ''.join(c.get('source', []))
            and c.get('id') != target_cell_id
            for c in nb['cells']
        )
        if already_inserted:
            print(f'  ⚡ Panel 확장 셀 이미 존재 → insert skip')
        else:
            new_cell = make_cell(insert_source)
            nb['cells'].insert(target_idx, new_cell)
            print(f'  ✅ Panel 확장 셀 insert (idx {target_idx}, 학습 셀 직전)')

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f'  💾 저장: {nb_path.name}')
    return True


def main():
    print('=' * 70)
    print('  Panel 확장 + Incremental 학습 셀 적용')
    print('=' * 70)
    print()

    # 02a_v2.ipynb 수정 (Cell id = f96904f7)
    nb_02a = NB_DIR / '02a_v2.ipynb'
    print(f'[1/2] 02a_v2.ipynb 수정 (학습 셀 incremental + panel 확장 셀 insert)')
    modify_notebook(
        nb_02a,
        target_cell_id='f96904f7',
        new_source=SEC02A_NEW_CELL_8_SOURCE,
        insert_before=True,
        insert_source=PANEL_EXTENSION_CELL_SOURCE,
    )
    print()

    # 02b_phase15_cross_sectional.ipynb 수정 (Cell 13)
    nb_02b = NB_DIR / '02b_phase15_cross_sectional.ipynb'
    print(f'[2/2] 02b_phase15_cross_sectional.ipynb 수정')

    if not nb_02b.exists():
        print(f'  ⚠️ {nb_02b.name} 부재')
        return

    with open(nb_02b, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 02b Cell 13 의 cell id 확인
    cell_13 = nb['cells'][13]
    cell_13_id = cell_13.get('id')
    print(f'  Cell 13 id: {cell_13_id}')

    modify_notebook(
        nb_02b,
        target_cell_id=cell_13_id,
        new_source=SEC02B_NEW_CELL_13_SOURCE,
        insert_before=True,
        insert_source=PANEL_EXTENSION_CELL_SOURCE,
    )
    print()
    print('=' * 70)
    print('  ✅ 완료. 두 노트북에 panel 확장 + incremental flag 추가.')
    print('  사용자 실행 가이드: 재실행_가이드_panel_extension.md 참조')
    print('=' * 70)


if __name__ == '__main__':
    main()
