# -*- coding: utf-8 -*-
"""
make_figures_en.py — Generate the 7 figures (ENGLISH) for the Journal of Forecasting paper
"Complex Models Do Not Beat the Random Walk: MCS Evidence from
 Vietnamese Domestic Robusta Coffee Prices"

Run:     python make_figures_en.py
Output:  <OUT_DIR>/Fig1..Fig7  (each figure as .png 300dpi + .pdf vector)

Requires: pandas, numpy, matplotlib
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
DATA_MASTER  = "E:\\FPT\AI\\SEM8_AI\\DAP391m\\project\\CofPred\\data\\processed\\gia_cafe_master_full.csv"
PREDS_DIR    = "E:\\FPT\AI\\SEM8_AI\\DAP391m\\project\\CofPred\\results\\preds"
BACKTEST_DIR = "E:\\FPT\AI\\SEM8_AI\\DAP391m\\project\\CofPred\\results\\backtest"
OUT_DIR      = "E:\\FPT\AI\\SEM8_AI\\DAP391m\\project\\CofPred\\results\\figures"

DATE_COL   = "Ngay"
TARGET_COL = "Gia_target"
INITIAL_FRAC = 0.60

# Fig 6: label -> file in PREDS_DIR (horizon h = 5). Columns: Ngay / y_true / pred
# Cleaner view: Actual + Chronos (best) + LSTM (DL representative)
PREDS_H5 = {
    "Chronos": "Chronos__h5.csv",
    "LSTM":    "LSTM__h5.csv",
}
PRED_DATE_COL, PRED_TRUE_COL, PRED_PRED_COL = "Ngay", "y_true", "pred"

# Fig 7: equity curve + permutation distribution per horizon
BACKTEST_H = {
    "h = 5":  {"equity": "backtest_h5_equity.csv",  "perm": "backtest_h5_perm.csv"},
    "h = 10": {"equity": "backtest_h10_equity.csv", "perm": "backtest_h10_perm.csv"},
}

# ----------------------------------------------------------------------
# Shared style
# ----------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.unicode_minus": False,
    "figure.dpi": 120,
    "savefig.bbox": "tight",
})

C_NAIVE   = "#111111"
C_CHRONOS = "#1b7837"
C_CLASSIC = "#4575b4"
C_ML      = "#f1a340"
C_DL      = "#d73027"
C_IN      = "#2c7fb8"
C_OUT     = "#e34a33"
C_NA      = "#d9d9d9"
C_TRAIN   = "#e8eef5"
C_TEST    = "#fdece1"


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  ✓ {name}.png / {name}.pdf")


# ======================================================================
# FIGURE 1 — Price series & evaluation period
# ======================================================================
def fig1_price_series():
    df = pd.read_csv(DATA_MASTER)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.dropna(subset=[TARGET_COL]).sort_values(DATE_COL).reset_index(drop=True)

    split = int(len(df) * INITIAL_FRAC)
    split_date = df[DATE_COL].iloc[split]
    peak_i = df[TARGET_COL].idxmax()

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.axvspan(df[DATE_COL].iloc[0], split_date, color=C_TRAIN,
               label="Initial training (60%)")
    ax.axvspan(split_date, df[DATE_COL].iloc[-1], color=C_TEST,
               label="Walk-forward region (out-of-sample)")
    ax.plot(df[DATE_COL], df[TARGET_COL] / 1000, color=C_NAIVE, lw=1.1)

    ax.axvline(split_date, color="black", ls="--", lw=0.8)
    ax.set_title("Figure 1. Vietnam domestic Robusta price and evaluation period (2020–2025)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Price (thousand VND/kg)")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    _save(fig, "Fig1_price_series")


# ======================================================================
# FIGURE 2 — Last-anchored walk-forward scheme (illustration)
# ======================================================================
def fig2_walkforward_schema():
    n, start, STEP = 40, 20, 5
    origins = sorted(range(n - 1 - 1, start - 1, -STEP))
    fig, ax = plt.subplots(figsize=(9, 3.6))

    ax.hlines(0, 0, n - 1, color="black", lw=1)
    for t in range(0, n, 5):
        ax.vlines(t, -0.1, 0.1, color="black", lw=0.6)
        ax.text(t, -0.35, f"t{t}", ha="center", fontsize=7, color="dimgray")

    ax.axvspan(0, start, color=C_TRAIN)
    ax.text(start / 2, 0.9, "Initial training (60%)",
            ha="center", fontsize=9, color="#33506e")

    for k, o in enumerate(origins):
        ax.plot(o, 0, "o", color=C_IN, ms=7, zorder=3)
        ax.vlines(o, 0, 0.55, color=C_IN, lw=0.8, ls=":")
        ax.hlines(0.55, o, min(o + 5, n - 1), color=C_DL, lw=2, alpha=0.7)
    ax.set_ylim(-0.7, 1.15)
    ax.set_xlim(-1, n)
    ax.axis("off")
    ax.set_title("Figure 2. Last-anchored walk-forward scheme (unified date grid)")
    legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor=C_IN,
                     markersize=8, label="Forecast origin"),
              Line2D([0], [0], color=C_DL, lw=2, label="Forecast window (horizon h)")]
    ax.legend(handles=legend, loc="lower right", frameon=False, fontsize=8.5)
    _save(fig, "Fig2_walkforward_schema")


# ======================================================================
# FIGURE 3 — MAE by forecast horizon (representative models)
# ======================================================================
def fig3_mae_by_horizon():
    horizons = [1, 5, 21, 63]
    mae = {
        "Naive":   ([1567.3, 3508.1, 8646.5, 15878.7], C_NAIVE),
        "Chronos": ([1589.7, 3543.0, 8907.7, 15737.5], C_CHRONOS),
        "ARIMA":   ([1576.8, 3555.0, 9587.4, 18601.3], C_CLASSIC),
        "SVR":     ([1692.2, 4089.0, 10161.2, 17193.1], C_ML),
        "LSTM":    ([2097.5, 5312.2, 12175.5, 20236.5], C_DL),
        "NHITS":   ([np.nan, 4147.2, 9882.8, 25366.1], "#7b3294"),
    }
    fig, ax = plt.subplots(figsize=(9, 4.4))
    labels = list(mae.keys())
    nH, nM = len(horizons), len(labels)
    w = 0.8 / nM
    x = np.arange(nH)
    for j, lab in enumerate(labels):
        vals = np.array(mae[lab][0]) / 1000.0
        ax.bar(x + j * w - 0.4 + w / 2, vals, w, label=lab, color=mae[lab][1])
    ax.set_xticks(x)
    ax.set_xticklabels([f"h = {h}" for h in horizons])
    ax.set_ylabel("MAE (thousand VND/kg)")
    ax.set_ylim(0, 27)
    ax.set_title("Figure 3. MAE by forecast horizon for representative models")
    ax.legend(ncol=3, frameon=False, fontsize=9, loc="upper left")
    _save(fig, "Fig3_mae_by_horizon")


# ======================================================================
# MCS data (IN / eliminated) — for Figures 4 & 5
# ======================================================================
MODELS = ["Naive", "Chronos", "ARIMA", "ARIMAX", "AutoARIMA", "SARIMAX", "VECM",
          "ETS", "Theta", "Combination",
          "SVR", "Ridge", "Lasso", "ElasticNet", "RandomForest", "XGBoost",
          "LightGBM", "kNN", "GRU", "LSTM", "NHITS", "NBEATSx"]

MCS_MAE_IN = {
    1:  {"Naive","ETS","Combination","ARIMA","Theta","Chronos",
         "AutoARIMA","ARIMAX","SARIMAX","VECM","SVR"},
    5:  {"Theta","Naive","Chronos","ARIMA","Combination","ARIMAX","SARIMAX",
         "ETS","AutoARIMA","VECM","NBEATSx","SVR","GRU","NHITS",
         "Ridge","ElasticNet","LSTM"},
    21: set(MODELS) - {"VECM","GRU","Lasso","LSTM"},
    63: set(MODELS),
}
MCS_MSE_IN = {
    1:  {"SVR","Naive"},
    5:  {"Chronos","Naive","Theta","Combination","ARIMAX","ARIMA",
         "SARIMAX","VECM","AutoARIMA","ETS"},
    21: set(MODELS),
    63: set(MODELS),
}
NA_AT_H1 = {"NHITS", "NBEATSx"}


def _mcs_grid(in_map, title, fname, pvals):
    horizons = [1, 5, 21, 63]
    grid = np.full((len(MODELS), len(horizons)), 0)  # 0=NA,1=IN,2=OUT
    for c, h in enumerate(horizons):
        for r, m in enumerate(MODELS):
            if h == 1 and m in NA_AT_H1:
                grid[r, c] = 0
            elif m in in_map[h]:
                grid[r, c] = 1
            else:
                grid[r, c] = 2

    cmap = mpl.colors.ListedColormap([C_NA, C_IN, C_OUT])
    fig, ax = plt.subplots(figsize=(6.4, 8.2))
    ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0, vmax=2)

    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels([f"h = {h}\n(p={p})" for h, p in zip(horizons, pvals)],
                       fontsize=9)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS, fontsize=9)
    ax.set_xticks(np.arange(-.5, len(horizons)), minor=True)
    ax.set_yticks(np.arange(-.5, len(MODELS)), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="minor", length=0)

    for r in range(len(MODELS)):
        for c in range(len(horizons)):
            txt = {1: "IN", 2: "out", 0: "—"}[grid[r, c]]
            col = "white" if grid[r, c] in (1, 2) else "grey"
            ax.text(c, r, txt, ha="center", va="center", fontsize=7.5, color=col)

    ax.set_title(title, fontsize=11)
    legend = [Patch(facecolor=C_IN, label="IN (superior set)"),
              Patch(facecolor=C_OUT, label="Eliminated"),
              Patch(facecolor=C_NA, label="Not evaluated")]
    ax.legend(handles=legend, bbox_to_anchor=(0.5, -0.06), loc="upper center",
              ncol=3, frameon=False, fontsize=8.5)
    _save(fig, fname)


def fig4_mcs_mae():
    _mcs_grid(MCS_MAE_IN,
              "Figure 4. MCS superior set by MAE (α = 0.10)",
              "Fig4_mcs_mae", pvals=["0.286", "0.125", "0.216", "0.116"])


def fig5_mcs_mse():
    _mcs_grid(MCS_MSE_IN,
              "Figure 5. MCS superior set by MSE (α = 0.10)\n"
              "— sharpest at h = 1 (2/20)",
              "Fig5_mcs_mse", pvals=["0.608", "0.467", "0.200", "0.331"])


# ======================================================================
# FIGURE 6 — Forecast vs. actual (h = 5)
# ======================================================================
def fig6_forecast_vs_actual():
    colors = {"Naive": C_NAIVE, "Chronos": C_CHRONOS, "LSTM": C_DL}
    fig, ax = plt.subplots(figsize=(9, 4.2))
    plotted_actual = False
    for lab, fname in PREDS_H5.items():
        path = os.path.join(PREDS_DIR, fname)
        if not os.path.exists(path):
            print(f"  [Fig6] Skip (file not found): {path}")
            continue
        d = pd.read_csv(path)
        d[PRED_DATE_COL] = pd.to_datetime(d[PRED_DATE_COL])
        d = d.sort_values(PRED_DATE_COL)
        if not plotted_actual and PRED_TRUE_COL in d.columns:
            ax.plot(d[PRED_DATE_COL], d[PRED_TRUE_COL] / 1000, color="grey",
                    lw=2.4, label="Actual", zorder=1)
            plotted_actual = True
        ax.plot(d[PRED_DATE_COL], d[PRED_PRED_COL] / 1000, lw=1.3,
                color=colors.get(lab, None), label=lab)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title("Figure 6. Forecast vs. actual at horizon h = 5 (one trading week)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price (thousand VND/kg)")
    ax.legend(frameon=False, fontsize=9, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.22))
    _save(fig, "Fig6_forecast_vs_actual")


# ======================================================================
# FIGURE 7 — Backtest equity curve + permutation test (h = 5, h = 10)
# ======================================================================
def fig7_backtest_equity():
    palette = {"h = 5": C_CHRONOS, "h = 10": C_ML}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4),
                                   gridspec_kw={"width_ratios": [1.6, 1.0]})

    # --- Left panel: equity curve over time ---
    for lab, files in BACKTEST_H.items():
        epath = os.path.join(BACKTEST_DIR, files["equity"])
        if not os.path.exists(epath):
            print(f"  [Fig7] Skip equity (not found): {epath}")
            continue
        e = pd.read_csv(epath)
        e["date"] = pd.to_datetime(e["date"])
        e = e.sort_values("date")
        axL.plot(e["date"], e["equity"], color=palette[lab], lw=1.8, label=lab)
    axL.axhline(1.0, color="black", lw=0.8, ls="--")
    # Sparser time axis: one tick per year
    axL.xaxis.set_major_locator(mdates.YearLocator())
    axL.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axL.set_title("(a) Strategy equity curve from forecast signals")
    axL.set_xlabel("Time")
    axL.set_ylabel("Account value (start = 1)")
    axL.legend(frameon=False, fontsize=9, loc="upper left")

    # --- Right panel: permutation distribution + actual total return ---
    for lab, files in BACKTEST_H.items():
        ppath = os.path.join(BACKTEST_DIR, files["perm"])
        epath = os.path.join(BACKTEST_DIR, files["equity"])
        if not (os.path.exists(ppath) and os.path.exists(epath)):
            print(f"  [Fig7] Skip perm: {ppath}")
            continue
        perm = pd.read_csv(ppath)["perm_sum"].to_numpy(dtype=float)
        actual = pd.read_csv(epath)["strat_logret"].sum()
        pval = float(np.mean(perm >= actual))
        axR.hist(perm, bins=40, color=palette[lab], alpha=0.35, density=True)
        axR.axvline(actual, color=palette[lab], lw=2,
                    label=f"{lab}: actual (p = {pval:.3f})")
    axR.axvline(0.0, color="black", lw=0.8, ls="--")
    axR.set_title("(b) Permutation distribution of total log-return")
    axR.set_xlabel("Total log-return (signal-order permutation)")
    axR.set_ylabel("Density")
    axR.legend(frameon=False, fontsize=8.5, loc="upper left")

    fig.suptitle("Figure 7. Backtest equity curve and permutation test (h = 5, h = 10)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "Fig7_backtest_equity")


# ======================================================================
if __name__ == "__main__":
    print("Generating figures (English)...")
    fig1_price_series()
    fig2_walkforward_schema()
    fig3_mae_by_horizon()
    fig4_mcs_mae()
    fig5_mcs_mse()
    fig6_forecast_vs_actual()
    fig7_backtest_equity()
    print(f"Done. See folder: {OUT_DIR}")
