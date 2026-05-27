"""Update [3e] — output complete LaTeX table block (copy-paste ready) instead of per-row output."""
import json, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

nb_path = '99_main_analysis.ipynb'
nb = json.load(open(nb_path, encoding='utf-8'))

target_idx = None
for i, c in enumerate(nb['cells']):
    if c['cell_type']=='code' and '[3e]' in ''.join(c['source']):
        target_idx = i; break
assert target_idx is not None

new_code = r'''# ── [3e] P 행렬 평균 Sharpe 표 — §4.3 메인 표 (LaTeX 출력) ──
# 각 p_w 값에 대해 (Q × prior) 15 슬롯 평균 LSTM/ANN Sharpe + Δ
# 출력: (1) 콘솔 친화 (2) overleaf 바로 복붙 가능한 완성 LaTeX 표

_periods = list(dict.fromkeys(df_p_sh.columns.get_level_values("period")))

# ── 콘솔 친화 표 ──
print("="*100)
print("P 행렬 평균 Sharpe — Q 5개 × prior 3개 = 15 슬롯 평균")
print("="*100)
hdr = f"{'p_w':<6s}" + "".join(f"{p:^24s}" for p in _periods)
print(hdr); print("-"*len(hdr))
sub_hdr = "      " + "".join(f"{'L':>7s}{'A':>8s}{'Δ':>9s}" for _ in _periods)
print(sub_hdr)

results = {}
for pw in P_WEIGHTS:
    cells_text = []
    cells_data = {}
    for p in _periods:
        L = df_p_sh[(p, "L")].xs(pw, level="p_w").dropna()
        A = df_p_sh[(p, "A")].xs(pw, level="p_w").dropna()
        Lm, Am = L.mean(), A.mean()
        delta = Lm - Am
        cells_text.append(f"{Lm:>7.3f}{Am:>8.3f}{delta:>+9.3f}")
        cells_data[p] = (Lm, Am, delta)
    results[pw] = cells_data
    print(f"{pw:<6s}" + "".join(cells_text))

# ── LSTM win count (콘솔 보조) ──
print("\n" + "="*60)
print("LSTM win count (Δ>0) — 15 슬롯 중")
print("="*60)
hdr2 = f"{'p_w':<6s}" + "".join(f"{p:>10s}" for p in _periods)
print(hdr2); print("-"*len(hdr2))
for pw in P_WEIGHTS:
    cells = []
    for p in _periods:
        d = (df_p_sh[(p, "L")] - df_p_sh[(p, "A")]).xs(pw, level="p_w").dropna()
        cells.append(f"{int((d>0).sum())}/{len(d)}")
    print(f"{pw:<6s}" + "".join(f"{c:>10s}" for c in cells))

# ── 완성 LaTeX 표 (Overleaf 복붙용) ──
n_periods = len(_periods)
col_spec = "l " + " ".join(["ccc"] * n_periods)
period_hdr = " & ".join([f"\\multicolumn{{3}}{{c}}{{{p}}}" for p in _periods])
cmidrules = " ".join([f"\\cmidrule(lr){{{2 + 3*i}-{4 + 3*i}}}" for i in range(n_periods)])
sub_hdr_l = " & ".join(["L & A & $\\Delta$"] * n_periods)

latex_lines = []
latex_lines.append(r"\begin{table}[ht]")
latex_lines.append(r"\centering")
latex_lines.append(r"\caption{P 행렬 변경에 따른 평균 Sharpe 비율 ($Q$ 5개 $\times$ prior 3개 = 15 슬롯 평균, $\omega=$err 고정). LSTM(L)과 ANN(A)의 평균 Sharpe 비율 및 차이 $\Delta = \mathrm{L} - \mathrm{A}$를 전체 표본과 4개 구간(R1 회복, R2 확장, R3 위기, R4 AI랠리)에 대해 비교한다.}")
latex_lines.append(r"\label{tab:p_matrix_avg}")
latex_lines.append(r"\setlength{\tabcolsep}{3pt}")
latex_lines.append(r"\renewcommand{\arraystretch}{1.15}")
latex_lines.append(r"\footnotesize")
latex_lines.append(r"\resizebox{\textwidth}{!}{%")
latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
latex_lines.append(r"\toprule")
latex_lines.append(f"  & {period_hdr} \\\\")
latex_lines.append(cmidrules)
latex_lines.append(f"$p_w$ & {sub_hdr_l} \\\\")
latex_lines.append(r"\midrule")

for pw in P_WEIGHTS:
    parts = [pw]
    for p in _periods:
        Lm, Am, d = results[pw][p]
        if d > 0.001:
            d_str = f"\\cellcolor{{posgreen!50}}$+{d:.3f}$"
        elif d < -0.001:
            d_str = f"\\cellcolor{{negred!50}}${d:.3f}$"
        else:
            d_str = f"${d:+.3f}$"
        parts.extend([f"{Lm:.3f}", f"{Am:.3f}", d_str])
    latex_lines.append(" & ".join(parts) + r" \\")

latex_lines.append(r"\bottomrule")
latex_lines.append(r"\end{tabular}%")
latex_lines.append(r"}")
latex_lines.append(r"\end{table}")

print("\n" + "="*100)
print("Overleaf 복붙용 LaTeX 표")
print("="*100)
print("\n".join(latex_lines))
'''

nb['cells'][target_idx]['source'] = new_code.splitlines(keepends=True)
nb['cells'][target_idx]['outputs'] = []
nb['cells'][target_idx]['execution_count'] = None
json.dump(nb, open(nb_path, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(f'Updated [3e] at idx {target_idx} — full LaTeX table output')
