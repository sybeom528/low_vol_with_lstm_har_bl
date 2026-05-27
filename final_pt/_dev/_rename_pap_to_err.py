"""Rename display labels: 'pap' → 'err' (Ω naming). Code identifiers (pkl keys) untouched."""
import json, os, re
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

nb_path = '99_main_analysis.ipynb'
nb = json.load(open(nb_path, encoding='utf-8'))

# 텍스트 치환 규칙 — 라벨/문장 한정 (slot key 안 건드림)
# 정규식 사용: 'pap' 앞뒤에 underscore 가 있으면 (mat_..._pap, _pap_ann) 건드리지 않음.
LABEL_SUBS = [
    # 의미 라벨: omega=pap / Ω=pap / ω=pap → err
    (r'omega\s*=\s*pap', 'omega=err'),
    (r'Ω\s*=\s*pap', 'Ω=err'),
    (r'ω\s*=\s*pap', 'ω=err'),
    (r'\$\\omega=\$pap', r'$\\omega=$err'),
    # 카테고리 표현: "pap★" / "pap ★" / "pap > he" 등 (단어 경계)
    (r'(?<![_\w])pap(?![_\w])', 'err'),
]

def transform(text: str) -> str:
    out = text
    for pat, repl in LABEL_SUBS:
        out = re.sub(pat, repl, out)
    return out

# 단, "Idzorek-style, Pyo-Lee 채택" 의 'Idzorek-style' 은 사실 부정확 (실제 = realized-error).
# 추가로 그 문장 정정.
EXTRA_SUBS = [
    ('(Idzorek-style, Pyo-Lee 채택)', '(realized-error: 직전월 view 예측오차 제곱)'),
]
def extra(text: str) -> str:
    out = text
    for old, new in EXTRA_SUBS:
        out = out.replace(old, new)
    return out

n_cells_changed = 0
for c in nb['cells']:
    if c['cell_type'] not in ('markdown', 'code'): continue
    src = c['source']
    new_lines = []
    changed = False
    for line in src:
        new_line = transform(line)
        new_line = extra(new_line)
        if new_line != line:
            changed = True
        new_lines.append(new_line)
    if changed:
        c['source'] = new_lines
        n_cells_changed += 1

    # 캐시된 output (text/html, text/plain, stream) 도 정리 — 표 라벨에 pap 남으면 곤란
    if c['cell_type'] == 'code':
        for out in c.get('outputs', []):
            if 'text' in out:
                txt = out['text']
                if isinstance(txt, list):
                    out['text'] = [extra(transform(t)) for t in txt]
                else:
                    out['text'] = extra(transform(txt))
            if 'data' in out:
                for mime, val in out['data'].items():
                    if mime in ('text/plain', 'text/html'):
                        if isinstance(val, list):
                            out['data'][mime] = [extra(transform(t)) for t in val]
                        else:
                            out['data'][mime] = extra(transform(val))

json.dump(nb, open(nb_path, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(f'Cells changed: {n_cells_changed}')
