"""Tier-1 pass rule (decided 2026-11-08): every banked cell within 1e-12 relative of
the bank (text cells equal), and every published number reproduces exactly.
Applied to the pinned-image run's regenerated tables (rerun/step1/kaggle_ridge/out/).
Run from the repo root."""
import os, shutil, subprocess, sys, tempfile
import numpy as np, pandas as pd
REG = "rerun/step1/kaggle_ridge/out"
cmp = pd.read_csv(f"{REG}/cell_comparison.csv", dtype=str, keep_default_na=False)   # all 2,707 cells
def within(s, n):
    try: s, n = float(s), float(n)
    except ValueError: return s == n
    return abs(n - s) <= 1e-12 * abs(s) if s != 0 else abs(n) <= 1e-12
cmp["rule_1e12"] = [within(s, n) for s, n in zip(cmp.stored, cmp.regenerated)]
cmp["rel_diff"] = [abs(float(n) - float(s)) / abs(float(s)) if s not in ("", "0", "0.0") and n != "" and s.replace('.','',1).replace('-','',1).replace('e','',1).isdigit() else np.nan
                   for s, n in zip(cmp.stored, cmp.regenerated)]
cmp.to_csv("rerun/step1/rule_check_cells.csv", index=False)
print(f"banked cells within 1e-12 relative: {int(cmp.rule_1e12.sum())}/{len(cmp)}  (strict: {int(cmp.status.str.startswith('MATCH').sum())}/{len(cmp)})")
print(f"max relative difference: {np.nanmax(cmp.rel_diff):.3g}")
# published numbers: paper/README cross-check evaluated on the pinned-image tables
tmp = tempfile.mkdtemp()
shutil.copytree("data", f"{tmp}/data"); os.makedirs(f"{tmp}/rerun/step1")
for f in os.listdir(REG):
    if f.startswith("table__"): shutil.copy(f"{REG}/{f}", f"{tmp}/data/{f[7:]}")
subprocess.run([sys.executable, os.path.abspath("rerun/step1/paper_check.py")], cwd=tmp, capture_output=True, check=True)
pc = pd.read_csv(f"{tmp}/rerun/step1/paper_crosscheck.csv")
pc.to_csv("rerun/step1/rule_check_published.csv", index=False)
print(pc.status.str.split(" \\(").str[0].value_counts().to_dict())
same = pc.equals(pd.read_csv("rerun/step1/paper_crosscheck.csv"))
print("identical to the bank-based cross-check:", same)
