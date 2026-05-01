# Panel 확장 + Incremental 재학습 사용자 실행 가이드

**작성일**: 2026-05-01
**목적**: panel forward 확장 (2026-04-30) + 마지막 fold y_true 채움 → 02a/02b ensemble 정상화 + 모든 시점 fresh ML 예측
**사용자 작업 시간**: 약 **2-3시간** (병렬 진행 시)

---

## 사전 확인

### 적용된 코드 변경

| 파일 | 변경 내용 |
|---|---|
| `scripts/volatility_ensemble.py` | `compute_performance_weights` finite_mask y_true 제거 (1 line) |
| `scripts/volatility_ensemble.py` | `run_walkforward_for_ticker` 에 `existing_folds` 인자 추가 |
| `scripts/volatility_ensemble.py` | `_train_ticker_worker` args 6-tuple backward 호환 |
| `scripts/volatility_ensemble.py` | `run_ensemble_for_universe_parallel` 에 `incremental` 인자 추가 |
| `scripts/volatility_ensemble.py` | `run_ensemble_cross_sectional` (02b) 에 `incremental` 인자 추가 |
| `scripts/universe_extended.py` | `extend_panel_forward(end_date)` 함수 신규 추가 |
| `02a_v2.ipynb` | Panel 확장 셀 (Cell 8) + 학습 셀 (Cell 9) incremental flag 추가 |
| `02b_phase15_cross_sectional.ipynb` | Panel 확장 셀 (Cell 13) + 학습 셀 (Cell 14) incremental flag 추가 |

### 검증 (시작 전 확인)
```bash
# 1. 모듈 syntax OK
python -c "import ast; ast.parse(open('scripts/volatility_ensemble.py', encoding='utf-8').read()); print('OK')"
python -c "import ast; ast.parse(open('scripts/universe_extended.py', encoding='utf-8').read()); print('OK')"

# 2. 노트북 열어서 셀 추가 확인
# 02a_v2.ipynb: Cell 8 = Panel 확장, Cell 9 = 학습 (INCREMENTAL_TRAIN=True)
# 02b: Cell 13 = Panel 확장, Cell 14 = 학습
```

---

## Step 1 — 02a_v2.ipynb Panel 확장 + Incremental 학습 (~25분)

### Step 1.1 — Panel 확장 (1회만, ~15분)

**위치**: 02a_v2.ipynb 의 Cell 8 (`§1.5 Panel Forward 확장`)

**실행:**
1. VS Code 에서 02a_v2.ipynb 열기
2. Cell 8 실행 (Shift+Enter)
3. 로그에서 다음 확인:
```
Panel Forward 확장: end_date = 2026-04-30
기존 panel: 3,344,502 rows, max date = 2025-12-31
[download] yfinance: 615 종목, 2026-01-01 ~ 2026-04-30
...
✅ panel 저장: 3,344,502 → ~3,500,000 rows, max 2025-12-31 → 2026-04-30
```

**예상 시간**: ~10-20분 (yfinance API 속도 영향)

**검증:**
```python
import pandas as pd
panel = pd.read_csv('data/daily_panel.csv', parse_dates=['date'], usecols=['date'])
print(f'panel max date: {panel["date"].max()}')
# 기대: 2026-04-30 부근
```

### Step 1.2 — 02a Incremental 학습 (~25분, 8-way 병렬)

**위치**: 02a_v2.ipynb 의 Cell 9 (`§3. 8-way 병렬 학습`)

**확인:**
- `INCREMENTAL_TRAIN = True` 로 설정되어 있음 (자동 적용됨)

**실행:**
1. Cell 9 실행
2. 로그에서:
```
⚡ Incremental 학습 모드 (Panel 확장 후, 마지막 + 신규 fold 만)
  종목당 ~5 fold 학습 → ~10-30분 (8-way 병렬)
  [parallel] ⚡ Incremental 모드: fold_predictions_stockwise.csv 로드
  [parallel] 마지막 fold (223) + 신규 fold (panel 확장 후) 만 학습
  [parallel] 615 종목 × 8-way 병렬 학습 시작
  [1/615] AAPL: ... fold ...
  ...
⏱️ Incremental 학습 시간: ~25분
```

**예상 시간**: ~20-40분 (615 종목 × 5 fold ÷ 8 worker)

**검증:**
```python
ens = pd.read_csv('data/ensemble_predictions_stockwise.csv', parse_dates=['date'], usecols=['date'])
print(f'ensemble max date: {ens["date"].max()}')
# 기대: 2026-04 부근 (정확한 마지막 일자)
print(f'2025-12 unique dates: {ens[ens["date"] >= "2025-12-01"]["date"].nunique()}')
# 기대: 22 (2025-12 영업일 모두)
```

---

## Step 2 — 02b 노트북 Incremental 학습 (~30-60분, 별도 터미널/창)

### Step 2.1 — Panel 확장 셀 (자동 skip, ~5초)

**위치**: 02b_phase15_cross_sectional.ipynb 의 Cell 13 (`§1.5 Panel Forward 확장`)

**실행:**
- 이미 Step 1.1 에서 panel 확장됐으므로 → **자동 skip 메시지** 출력:
```
⚡ Panel 이 이미 2026-04-30 까지 → 확장 불필요
```

### Step 2.2 — 02b Incremental 학습 (~30-60분, single GPU)

**위치**: 02b_phase15_cross_sectional.ipynb 의 Cell 14 (학습 셀)

**확인:**
- `INCREMENTAL_TRAIN = True`

**실행:**
- Cell 14 실행
- 로그에서:
```
⚡ Cross-Sectional Incremental 학습 (panel 확장 후 마지막 + 신규 fold 만)
  ~5 fold 학습 → ~30-60분 (single GPU)
  [cross-sec] ⚡ Incremental 모드: ensemble_predictions_crosssec.csv 로드
  [cross-sec] 마지막 fold (223) + 신규 fold (panel 확장 후) 만 학습
  [fold 224/228] epoch~13, val=0.2282, ...
  ...
⏱️ 총 소요 시간: ~30-60분
```

**병렬 가능**: Step 1.2 와 Step 2.2 는 **동시 실행 가능** (GPU 별도, CPU 서로 안 겹침).

---

## Step 3 — BL 백테스트 재실행 (~76분, terminal)

ensemble 갱신 후 BL 가중치 재산출 필요 (모든 192 시점 + 신규 시점에 fresh ML 예측 사용).

**실행 (terminal):**
```bash
cd "시계열_Test/Phase3_Robust_Extensions"
python scripts/_run_02a_v2_sec6.py
```

**예상 시간**: ~76분 (단순 BL 백테스트 복제)

**검증:**
```python
import pickle
with open('data/bl_weights_v2_sanity_check.pkl', 'rb') as f:
    cache = pickle.load(f)
for s, w in cache['weights'].items():
    sorted_dates = sorted(w.keys())
    print(f'{s}: 마지막 시점 = {sorted_dates[-1]}, n = {len(sorted_dates)}')
# 기대: 마지막 시점 = 2026-04-30 부근 (panel 끝점 갱신됨)
# 기대: n = ~196 시점 (180 OOS + 12 hold-out + ~4 신규)
```

---

## Step 4 — 노트북 nbconvert 재실행 (~10분, terminal)

02a_v2.ipynb 의 §6/§7 셀 출력 모두 갱신 (캐시 hit + fresh 데이터).

**실행:**
```powershell
# PowerShell (한글 경로 안전)
$env:PYTHONIOENCODING = "utf-8"
jupyter nbconvert --to notebook --execute --inplace "02a_v2.ipynb" `
    --ExecutePreprocessor.timeout=1800 `
    --ExecutePreprocessor.kernel_name=python3
```

또는 Bash:
```bash
cd "시계열_Test/Phase3_Robust_Extensions" && \
jupyter nbconvert --to notebook --execute --inplace 02a_v2.ipynb \
    --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=python3
```

---

## Step 5 — 결과 검증 (5분)

### 5.1 ensemble 끝점 검증
```python
import pandas as pd
for fname in ['ensemble_predictions_stockwise.csv', 'ensemble_predictions_crosssec.csv']:
    ens = pd.read_csv(f'data/{fname}', parse_dates=['date'], usecols=['date'])
    print(f'{fname}: max = {ens["date"].max()}, 2025-12 dates = {ens[ens["date"] >= "2025-12-01"]["date"].nunique()}')
# 기대 (양쪽 동일):
#   max = 2026-04-30 부근
#   2025-12 dates = 22 (모든 영업일)
```

### 5.2 BL 가중치 끝점 검증
```python
import pickle
with open('data/bl_weights_v2_sanity_check.pkl', 'rb') as f:
    cache = pickle.load(f)
for s, w in cache['weights'].items():
    print(f'{s}: 마지막 시점 = {sorted(w.keys())[-1]}')
# 기대: 모든 6 시나리오 = 2026-04 부근
```

### 5.3 §6 메트릭 갱신 확인
- 02a_v2.ipynb 열기 → §6-4 메트릭 표 확인
- Hold-out 11m → ~14-15m 으로 확장 가능 (2026-01,02,03 추가)

---

## 시간 추정 종합

| 단계 | 작업 | 예상 시간 |
|---|---|---|
| Step 1.1 | 02a Panel 확장 (yfinance) | ~15분 |
| Step 1.2 | 02a Incremental 학습 | ~25분 |
| Step 2.1 | 02b Panel 확장 (auto skip) | ~5초 |
| Step 2.2 | 02b Incremental 학습 | ~30-60분 |
| Step 3 | BL 백테스트 재실행 | ~76분 |
| Step 4 | nbconvert 재실행 | ~10분 |
| Step 5 | 검증 | ~5분 |
| **총 합계 (순차)** | | **~3시간** |
| **총 합계 (Step 1.2 + Step 2.2 병렬)** | | **~2시간** |

---

## 트러블슈팅

### Q1: Panel 확장 시 yfinance 다운로드 실패
- 네트워크 확인
- `extend_panel_forward(end_date='2026-04-30', verbose=True)` 다시 실행
- 일부 종목 실패해도 chunk 단위 진행 (전체 실패 X)

### Q2: Incremental 학습 시 "fold_predictions_stockwise.csv 부재" 메시지
- → 기존 학습 결과 부재 시 **전체 학습으로 자동 fallback** (~5-7시간)
- 02b 도 동일 (ensemble_predictions_crosssec.csv 부재 시 fallback)

### Q3: BL 재실행 시 ml_pred_pivot 의 시점 부족
- `data/bl_weights_v2_sanity_check.pkl` 삭제 후 재실행
- 또는 `_run_02a_v2_sec6.py` 의 FORCE_RECOMPUTE=True 변경

### Q4: nbconvert 의 timeout
- `--ExecutePreprocessor.timeout=3600` 으로 증가
- 또는 standalone 우회: 이미 BL 결과 캐시 활용

### Q5: 02a/02b 동시 실행 시 GPU 메모리 부족
- 02a 가 8-way 병렬 → GPU 메모리 큰 사용
- 02b 와 별도 터미널이지만 **동일 GPU 공유** → 메모리 충돌 가능
- 해결: 순차 실행 (02a → 02b) 권장

---

## 백업 파일 (롤백)

`extend_panel_forward(backup=True)` 가 자동 생성:
- `data/daily_panel.csv.bak`
- `data/market_data.csv.bak`
- `data/vix_daily.csv.bak`

문제 발생 시 `.bak` 파일 → 원본 복원.

---

## 사용 후 정리

이 작업이 완료되면:

1. **02a_v2.ipynb 의 INCREMENTAL_TRAIN flag**:
   - 향후 다시 panel 확장 후 incremental 학습 시 활용
   - 일반 사용 시 `INCREMENTAL_TRAIN = False` 권장 (캐시 hit)

2. **02b 동일** (INCREMENTAL_TRAIN = False 권장)

3. **재사용 가능 모듈** (영구 보존):
   - `scripts/universe_extended.py:extend_panel_forward()`
   - `scripts/volatility_ensemble.py` 의 `incremental` 인자 (02a + 02b)
   - 향후 panel 확장 시 동일 절차 활용

---

## 최종 점검

위 Step 1-5 모두 완료 시:
- ✅ panel 끝점 = 2026-04-30
- ✅ ensemble 끝점 = 2026-04 부근 (2025-12 22 영업일 모두)
- ✅ BL 가중치 모든 시점 fresh ML 예측 사용
- ✅ 노트북 §6/§7 결과 갱신
- ✅ Hold-out 평가 11m → 14-15m 확장 (옵션)

이상 없으면 **분석 완료**. WORKLOG_v2.md 에 본 작업 결과 기재 권장.
