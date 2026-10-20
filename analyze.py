#!/usr/bin/env python3
"""
analyze.py - Compare rPPG (Green/CHROM/POS) against ECG over all recordings.
Prepared for Gourab - Build Club AOT.

Expects: data/<P>/<C>_R<k>/{ecg.csv, rgb.csv}
Writes into results/:
  windows.csv, summary.csv, stats.txt
  bland_altman.png, mae_by_condition.png, example_trace.png

Run:
  python analyze.py --data-dir data --out results
"""

import argparse
import glob
import os
import re
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import rppg_lib as L

FS_RPPG = 30.0
WIN, STRIDE = 30.0, 1.0


def process_recording(folder):
    ecg_path = os.path.join(folder, "ecg.csv")
    rgb_path = os.path.join(folder, "rgb.csv")
    if not os.path.exists(ecg_path) or not os.path.exists(rgb_path):
        return None

    ecg = pd.read_csv(ecg_path)
    rgb = pd.read_csv(rgb_path)

    if len(ecg) < 1000 or len(rgb) < 300:
        return None

    # ECG time axis: ESP32 microsecond clock (t_us)
    t_ecg = ecg["t_us"].values / 1e6
    peaks, _, _ = L.ecg_r_peaks(t_ecg, ecg["adc"].values)

    t_u, rgb_u = L.resample_uniform(rgb["t"].values, rgb[["R", "G", "B"]].values, FS_RPPG)
    t_end = min(t_u[-1], t_ecg[-1])

    starts = np.arange(max(t_u[0], 2.0), t_end - WIN, STRIDE)
    if len(starts) == 0:
        return None

    ref = L.ecg_bpm_windows(peaks, starts, WIN)
    lo = L.leadoff_fraction(t_ecg, ecg["leadoff"].values, starts, WIN)
    ref[lo > 0.05] = np.nan  # Discard if electrode detached > 5% of window

    out = {"starts": starts, "ecg": ref}
    traces = {}
    for name, fn in L.METHODS.items():
        s = fn(rgb_u, FS_RPPG)
        traces[name] = s
        out[name] = L.windowed_bpm(t_u, s, FS_RPPG, starts, WIN)

    return out, traces, t_u


def metrics(est, ref):
    m = np.isfinite(est) & np.isfinite(ref)
    if m.sum() < 5:
        return dict(n=int(m.sum()), MAE=np.nan, RMSE=np.nan, r=np.nan, bias=np.nan)
    e = est[m] - ref[m]
    return dict(
        n=int(m.sum()),
        MAE=float(np.abs(e).mean()),
        RMSE=float(np.sqrt((e**2).mean())),
        r=float(np.corrcoef(est[m], ref[m])[0, 1]),
        bias=float(e.mean()),
    )


def bland_altman(ax, est, ref, title):
    m = np.isfinite(est) & np.isfinite(ref)
    if m.sum() < 3:
        ax.set_title(f"{title}\nInsufficient data")
        return
    mean = (est[m] + ref[m]) / 2.0
    diff = est[m] - ref[m]
    md = float(diff.mean())
    sd = float(diff.std(ddof=1))
    ax.scatter(mean, diff, s=6, alpha=0.35, color="teal")
    ax.axhline(md, color="k", linewidth=1.2)
    ax.axhline(md + 1.96 * sd, color="r", ls="--", linewidth=1)
    ax.axhline(md - 1.96 * sd, color="r", ls="--", linewidth=1)
    ax.set_title(f"{title}\nbias {md:.2f}, LoA [{md - 1.96 * sd:.1f}, {md + 1.96 * sd:.1f}] BPM", fontsize=10)
    ax.set_xlabel("Mean of rPPG and ECG (BPM)", fontsize=9)
    ax.set_ylabel("rPPG - ECG (BPM)", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows, example = [], None
    folders = sorted(glob.glob(os.path.join(a.data_dir, "*", "*_R*")))

    for folder in folders:
        clean_folder = folder.replace("\\", "/")
        mt = re.search(r"(P\d+)/(C\d+)_R(\d+)$", clean_folder)
        if not mt or not os.path.exists(os.path.join(folder, "rgb.csv")):
            continue

        res = process_recording(folder)
        if res is None:
            print(f"Skipping (too short / missing data): {folder}")
            continue

        out, traces, t_u = res
        if example is None:
            example = (out, traces, t_u)

        for i, s in enumerate(out["starts"]):
            row = dict(
                participant=mt.group(1),
                condition=mt.group(2),
                repeat=int(mt.group(3)),
                start=s,
                ECG=out["ecg"][i],
            )
            for k in L.METHODS:
                row[k] = out[k][i]
            rows.append(row)
        print(f"[+] Processed: {folder}")

    if not rows:
        raise SystemExit(f"No usable recordings found in '{a.data_dir}'. Ensure recordings have rgb.csv and ecg.csv.")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "windows.csv"), index=False)

    # 1. Summary Table: Method x Condition
    summ = []
    for cond, g in df.groupby("condition"):
        for k in L.METHODS:
            summ.append(dict(condition=cond, method=k, **metrics(g[k].values, g["ECG"].values)))
    for k in L.METHODS:
        summ.append(dict(condition="ALL", method=k, **metrics(df[k].values, df["ECG"].values)))

    summ = pd.DataFrame(summ).round(3)
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    print("\n--- RESULTS SUMMARY ---")
    print(summ.to_string(index=False))

    # 2. Statistics: Per-participant MAE, Wilcoxon test between methods
    lines = []
    per = {}
    for k in L.METHODS:
        per[k] = (
            df.assign(ae=(df[k] - df["ECG"]).abs())
            .groupby(["participant", "condition"])["ae"]
            .mean()
        )

    names = list(L.METHODS)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            x, y = per[names[i]].align(per[names[j]], join="inner")
            ok = x.notna() & y.notna()
            if ok.sum() >= 6:
                w, p = stats.wilcoxon(x[ok], y[ok])
                lines.append(
                    f"Wilcoxon MAE {names[i]} vs {names[j]}: n={int(ok.sum())} W={w:.1f} p={p:.4f}"
                )
            else:
                lines.append(f"{names[i]} vs {names[j]}: not enough paired samples (n={ok.sum()})")

    with open(os.path.join(a.out, "stats.txt"), "w") as f:
        f.write("\n".join(lines))
    print("\n--- STATISTICAL TESTS ---")
    print("\n".join(lines))

    # 3. Plots: Bland-Altman Plots
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, k in zip(axs, L.METHODS):
        bland_altman(ax, df[k].values, df["ECG"].values, k)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "bland_altman.png"), dpi=150)
    plt.close(fig)

    # 4. Plots: MAE by Condition Bar Chart
    pv = summ[summ.condition != "ALL"].pivot(index="condition", columns="method", values="MAE")
    ax = pv.plot.bar(figsize=(7, 4), colormap="viridis")
    ax.set_ylabel("MAE (BPM)")
    ax.set_title("rPPG Heart Rate Error by Experimental Condition")
    plt.grid(axis="y", linestyle=":", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, "mae_by_condition.png"), dpi=150)
    plt.close()

    # 5. Plots: Example Trace Overlay
    if example is not None:
        out, traces, t_u = example
        fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=False)
        seg = (t_u >= 10) & (t_u <= 25)
        if seg.any():
            ax[0].plot(t_u[seg], traces["POS"][seg], color="forestgreen", lw=1.2)
            ax[0].set_title("POS rPPG Waveform (15 s snippet)", fontsize=10)
            ax[0].set_xlabel("Time (s)", fontsize=9)
            ax[0].grid(True, linestyle=":", alpha=0.5)

        ax[1].plot(out["starts"], out["ecg"], label="Ground Truth ECG", color="black", lw=1.8)
        for k in L.METHODS:
            ax[1].plot(out["starts"], out[k], label=f"rPPG {k}", alpha=0.8, linestyle="--")
        ax[1].set_ylabel("Heart Rate (BPM)", fontsize=9)
        ax[1].set_xlabel("Window Start Time (s)", fontsize=9)
        ax[1].set_title("Time-Aligned Heart Rate Tracking Comparison", fontsize=10)
        ax[1].legend(loc="upper right", fontsize=8)
        ax[1].grid(True, linestyle=":", alpha=0.5)

        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "example_trace.png"), dpi=150)
        plt.close(fig)

    print(f"\n[+] All results, tables, and figures successfully saved to: {a.out}")


if __name__ == "__main__":
    main()
