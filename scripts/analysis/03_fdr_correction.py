# -*- coding: utf-8 -*-
"""
03_fdr_correction.py  --  Hieu chinh da kiem dinh Benjamini-Hochberg (FDR) (§6).

Khi kiem dinh huong tren NHIEU bo dac trung x NHIEU chan troi, xac suat co it
nhat 1 ket qua "co y nghia" gia tang. BH-FDR kiem soat ti le phat hien sai.

Doc p-value tu results/direction_test.csv (cot PT_p) va ap BH o q=0.05.
Cung ho tro nhap tay 1 danh sach p-value qua --pvals.

Xuat: results/fdr_direction.csv
Chay:  python analysis/03_fdr_correction.py
"""
import os, sys, argparse
import numpy as np
import pandas as pd

os.makedirs("results/statistical", exist_ok=True)

def benjamini_hochberg(pvals, q=0.05):
    """Tra ve (reject[bool], p_adj[BH-adjusted])."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    # nguong BH: p(k) <= (k/n)*q
    thresh = (np.arange(1, n + 1) / n) * q
    below = ranked <= thresh
    k_max = np.max(np.where(below)[0]) + 1 if below.any() else 0
    reject = np.zeros(n, dtype=bool)
    if k_max > 0:
        reject[order[:k_max]] = True
    # p-adjusted (BH step-up)
    p_adj_sorted = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    p_adj = np.empty(n); p_adj[order] = np.clip(p_adj_sorted, 0, 1)
    return reject, p_adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--src", default="E:\\FPT\\AI\\SEM8_AI\\DAP391m\\project\\CofPred\\results\\statistical\\direction_test.csv")
    ap.add_argument("--pvals", type=float, nargs="*", default=None,
                    help="Nhap tay danh sach p-value (thay cho doc file).")
    a = ap.parse_args()

    if a.pvals:
        df = pd.DataFrame({"label": [f"test{i+1}" for i in range(len(a.pvals))],
                           "PT_p": a.pvals})
    else:
        if not os.path.exists(a.src):
            sys.exit(f"[loi] khong thay {a.src}. Chay 02_direction_test.py truoc.")
        df = pd.read_csv(a.src)
        df["label"] = df["feature_set"].astype(str) + "_h" + df["h"].astype(str)

    reject, p_adj = benjamini_hochberg(df["PT_p"].values, q=a.q)
    df["p_adj_BH"] = np.round(p_adj, 5)
    df["survive_q%.2f" % a.q] = reject
    df.to_csv("results/statistical/fdr_direction.csv", index=False)

    n_sig_raw = int((df["PT_p"] < a.q).sum())
    n_sig_bh = int(reject.sum())
    print(f"Tong kiem dinh: {len(df)}")
    print(f"Co y nghia THO (p<{a.q}): {n_sig_raw}")
    print(f"Sau BH-FDR (q={a.q}): {n_sig_bh} song sot")
    print("\n" + df.to_string(index=False))
    print("\n[done] -> results/statistical/fdr_direction.csv")


if __name__ == "__main__":
    main()
