#!/usr/bin/env python3
"""Generate aggregate validation figures from TractoInferno summary.tsv.

Usage:
    python scripts/plot_validation_summary.py [SUMMARY_TSV] [--outdir DIR]

Defaults:
    SUMMARY_TSV = /mnt/neurodata/csttool_runs/2026-02-01_git-1433ac4/outputs/summary.tsv
    --outdir     = scripts/figures/
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DEFAULT_TSV = "/mnt/neurodata/csttool_runs/2026-02-01_git-1433ac4/outputs/summary.tsv"
DPI = 300


def load_data(tsv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load summary.tsv and return (full_df, valid_df)."""
    df = pd.read_csv(tsv_path, sep="\t")
    valid = df[df["status"] == "OK"].copy()
    return df, valid


def plot_dice_distribution(valid: pd.DataFrame, outdir: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(valid["dice"], bins=30, kde=True, color="#4C72B0", ax=ax)
    mean_d = valid["dice"].mean()
    med_d = valid["dice"].median()
    std_d = valid["dice"].std()
    ax.axvline(mean_d, color="#C44E52", ls="--", lw=1.5, label=f"Mean = {mean_d:.3f}")
    ax.axvline(med_d, color="#DD8452", ls=":", lw=1.5, label=f"Median = {med_d:.3f}")
    ax.set_xlabel("Dice Coefficient (Voxel Overlap)")
    ax.set_ylabel("Count")
    ax.set_title("Dice Coefficient Distribution — TractoInferno Validation")
    ax.annotate(
        f"n = {len(valid)}\nmean ± SD = {mean_d:.3f} ± {std_d:.3f}",
        xy=(0.97, 0.95), xycoords="axes fraction", ha="right", va="top",
        fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(outdir / "dice_distribution.png", dpi=DPI)
    plt.close(fig)


def plot_precision_recall(valid: pd.DataFrame, outdir: Path):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    colors = {"L": "#4C72B0", "R": "#C44E52"}
    for hemi, grp in valid.groupby("hemi"):
        ax.scatter(
            grp["o_ref_in_cand"], grp["o_cand_in_ref"],
            c=colors[hemi], label=f"{hemi} hemisphere",
            alpha=0.6, s=20, edgecolors="none",
        )
    ax.set_xlabel("Recall (reference voxels captured by candidate)")
    ax.set_ylabel("Precision (candidate voxels within reference)")
    ax.set_title("Precision vs Recall — Per-Hemisphere")
    ax.set_xlim(0, 0.55)
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color="grey", ls=":", lw=0.8, alpha=0.5)
    ax.axvline(0.5, color="grey", ls=":", lw=0.8, alpha=0.5)
    ax.annotate(
        "High precision,\nlow recall\n(conservative)",
        xy=(0.15, 0.90), xycoords="axes fraction", ha="center", va="top",
        fontsize=9, fontstyle="italic", color="#555555",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "precision_recall.png", dpi=DPI)
    plt.close(fig)


def plot_dice_by_hemisphere(valid: pd.DataFrame, outdir: Path):
    fig, ax = plt.subplots(figsize=(4.5, 5))
    sns.violinplot(
        data=valid, x="hemi", y="dice", hue="hemi", order=["L", "R"],
        palette={"L": "#4C72B0", "R": "#C44E52"}, inner="box", legend=False, ax=ax,
    )
    ax.set_xlabel("Hemisphere")
    ax.set_ylabel("Dice Coefficient")
    ax.set_title("Dice by Hemisphere")
    for hemi in ["L", "R"]:
        sub = valid[valid["hemi"] == hemi]["dice"]
        ax.text(
            0 if hemi == "L" else 1, sub.max() + 0.01,
            f"{sub.mean():.3f} ± {sub.std():.3f}",
            ha="center", va="bottom", fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(outdir / "dice_by_hemisphere.png", dpi=DPI)
    plt.close(fig)


def plot_streamline_counts(valid: pd.DataFrame, outdir: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        valid["n_ref"], valid["n_cand"],
        c="#4C72B0", alpha=0.5, s=18, edgecolors="none",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Reference Streamline Count")
    ax.set_ylabel("Candidate Streamline Count (csttool)")
    ax.set_title("Streamline Counts — Candidate vs Reference")
    # identity line
    lims = [
        min(valid["n_ref"].min(), valid["n_cand"].min()) * 0.5,
        max(valid["n_ref"].max(), valid["n_cand"].max()) * 2,
    ]
    ax.plot(lims, lims, "k--", lw=0.8, alpha=0.4, label="1:1 line")
    ax.annotate(
        f"Mean candidate: {valid['n_cand'].mean():,.0f}\n"
        f"Mean reference: {valid['n_ref'].mean():,.0f}",
        xy=(0.03, 0.97), xycoords="axes fraction", ha="left", va="top",
        fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(outdir / "streamline_counts.png", dpi=DPI)
    plt.close(fig)


def plot_success_rate(df: pd.DataFrame, outdir: Path):
    counts = df["status"].value_counts()
    n_ok = counts.get("OK", 0)
    n_invalid = counts.get("INVALID_SPACE", 0)
    n_total = len(df)

    fig, ax = plt.subplots(figsize=(4, 4))
    wedges, texts, autotexts = ax.pie(
        [n_ok, n_invalid],
        labels=["OK", "INVALID_SPACE"],
        colors=["#4C72B0", "#C44E52"],
        autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct / 100 * n_total))})",
        startangle=90,
        textprops={"fontsize": 10},
    )
    ax.set_title(f"Validation Status (n = {n_total})")
    fig.tight_layout()
    fig.savefig(outdir / "success_rate.png", dpi=DPI)
    plt.close(fig)


def plot_subject_paired(valid: pd.DataFrame, outdir: Path):
    # Pivot to get L and R side-by-side per subject
    paired = valid.pivot(index="subject", columns="hemi", values=["dice", "o_cand_in_ref", "o_ref_in_cand"])
    # Keep only subjects with both hemispheres
    paired = paired.dropna()
    paired = paired.sort_values(("dice", "L"))

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    metrics = [
        ("dice", "Dice Coefficient"),
        ("o_cand_in_ref", "Precision"),
        ("o_ref_in_cand", "Recall"),
    ]

    for ax, (col, title) in zip(axes, metrics):
        left = paired[(col, "L")].values
        right = paired[(col, "R")].values
        n = len(left)
        x = np.arange(n)

        # Connecting lines — color by which hemisphere is higher
        for i in range(n):
            color = "#4C72B0" if left[i] >= right[i] else "#C44E52"
            ax.plot([x[i], x[i]], [left[i], right[i]], color=color, alpha=0.25, lw=0.8)

        ax.scatter(x, left, c="#4C72B0", s=12, alpha=0.7, label="Left", zorder=3)
        ax.scatter(x, right, c="#C44E52", s=12, alpha=0.7, label="Right", zorder=3)

        ax.set_xlabel("Subjects (sorted by left Dice)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_xticks([])

        # Annotate mean difference
        diff = right - left
        ax.annotate(
            f"Mean R−L = {diff.mean():+.3f}\n|R−L| mean = {np.abs(diff).mean():.3f}",
            xy=(0.03, 0.97), xycoords="axes fraction", ha="left", va="top",
            fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )

    fig.suptitle(
        f"Subject-Level Paired L vs R Comparison (n = {len(paired)} subjects)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(outdir / "subject_paired.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def print_summary(df: pd.DataFrame, valid: pd.DataFrame):
    n_total = len(df)
    n_ok = len(valid)
    n_invalid = n_total - n_ok
    print("=" * 60)
    print("  TractoInferno Validation Summary")
    print("=" * 60)
    print(f"  Subjects:         {df['subject'].nunique()}")
    print(f"  Hemispheres:      {n_total} ({n_ok} OK, {n_invalid} INVALID_SPACE)")
    print(f"  Success rate:     {n_ok / n_total * 100:.1f}%")
    print()
    print(f"  Dice coefficient:")
    print(f"    Mean ± SD:      {valid['dice'].mean():.3f} ± {valid['dice'].std():.3f}")
    print(f"    Median:         {valid['dice'].median():.3f}")
    print(f"    Range:          [{valid['dice'].min():.3f}, {valid['dice'].max():.3f}]")
    print()
    print(f"  Precision (candidate voxels within reference):")
    print(f"    Mean ± SD:      {valid['o_cand_in_ref'].mean():.3f} ± {valid['o_cand_in_ref'].std():.3f}")
    print()
    print(f"  Recall (reference voxels captured):")
    print(f"    Mean ± SD:      {valid['o_ref_in_cand'].mean():.3f} ± {valid['o_ref_in_cand'].std():.3f}")
    print()
    print(f"  Streamline counts:")
    print(f"    Candidate mean: {valid['n_cand'].mean():,.0f} (range: {valid['n_cand'].min():,}–{valid['n_cand'].max():,})")
    print(f"    Reference mean: {valid['n_ref'].mean():,.0f} (range: {valid['n_ref'].min():,}–{valid['n_ref'].max():,})")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tsv", nargs="?", default=DEFAULT_TSV, help="Path to summary.tsv")
    parser.add_argument("--outdir", default="scripts/figures", help="Output directory for figures")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df, valid = load_data(args.tsv)

    plot_dice_distribution(valid, outdir)
    plot_precision_recall(valid, outdir)
    plot_dice_by_hemisphere(valid, outdir)
    plot_streamline_counts(valid, outdir)
    plot_success_rate(df, outdir)
    plot_subject_paired(valid, outdir)

    print_summary(df, valid)
    print(f"\nFigures saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
