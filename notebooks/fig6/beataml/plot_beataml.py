"""Plot BeatAML personalized drug-combination synergy results.

One figure per split ("Split by patients" / "Split by drugs"), each showing Patient-centric and
Drug-centric AUROC for the five model variants. Bars are the mean over all fits (pretraining runs x
training seeds); error bars are the std across those fits.

    python plot_beataml.py                # uses the cached results table if it is up to date
    python plot_beataml.py --refresh      # force re-reading the metrics_*.pkl files

"""
import os
import re
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))

# Paper font: notebooks/arial.ttf
FONT_PATH = os.path.normpath(os.path.join(HERE, "..", "..", "arial.ttf"))
if os.path.exists(FONT_PATH):
    fm.fontManager.addfont(FONT_PATH)
    FONT_NAME = fm.FontProperties(fname=FONT_PATH).get_name()
else:
    FONT_NAME = plt.rcParams["font.sans-serif"][0]
plt.rcParams["font.family"] = FONT_NAME
FONT_SIZE = 16  # single size shared by tick labels, axis label and legend
Y_MAJOR = 0.1   # labelled y-tick spacing (minor ticks fall halfway between)


def _base_dir():
    """BASE_DIR from the environment, else from Madrigal/.env."""
    if os.getenv("BASE_DIR"):
        return os.getenv("BASE_DIR")
    env = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".env"))
    if os.path.exists(env):
        for line in open(env):
            if line.strip().startswith("BASE_DIR"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("BASE_DIR is not set (see README.md).")


OUTPUT_DIR = _base_dir() + "model_output/BeatAML/"
SUBDIR = ("BeatAML_ft_freeze_madrigal_combo_128_dims_256_128_dropout_0.2"
          "_mut_True_lr_0.001")
CACHE = os.path.join(HERE, "figs", "beataml_results.csv")

# model_type -> (legend label, colour). Order = plotting order.
MODELS = [
    ("madrigal",            "Madrigal and patient information (genomics, clinical)", "#8B4BA8"),
    ("no_patient_clin",     "Madrigal and patient information (genomics)",           "#B47CC7"),
    ("no_patient_genomics", "Madrigal and patient information (clinical)",           "#D3A9DE"),
    ("no_patient",          "Madrigal",                                              "#F2DDF0"),
    ("no_drug",             "Patient information (genomics, clinical)",              "#BFBFBF"),
]
METRICS = [("test_patient_macro_auroc", "Patient-centric"),
           ("test_drug_pair_macro_auroc", "Drug-centric")]
SPLIT_TITLE = {"patient": "Split by patients", "drug": "Split by drugs"}


def _model_type_of(basename):
    """Longest model type the file name starts with."""
    for mt in sorted((mt for mt, _, _ in MODELS), key=len, reverse=True):
        if basename.startswith(f"metrics_{mt}_"):
            return mt
    return None


def _metrics_files(split):
    return glob.glob(os.path.join(OUTPUT_DIR, split, SUBDIR, "metrics_*.pkl"))


def _cache_is_fresh():
    if not os.path.exists(CACHE):
        return False
    files = _metrics_files("patient") + _metrics_files("drug")
    if not files:
        return False
    return os.path.getmtime(CACHE) >= max(os.path.getmtime(f) for f in files)


def read_all():
    """Tidy DataFrame: one row per (split, model, metric, fit). Cached to CSV."""
    if _cache_is_fresh():
        return pd.read_csv(CACHE)

    import torch  # lazy: only needed when actually re-reading the pickles
    rows = []
    for split in ["patient", "drug"]:
        d = os.path.join(OUTPUT_DIR, split, SUBDIR)
        for mt, label, _ in MODELS:
            for f in sorted(glob.glob(os.path.join(d, f"metrics_{mt}_*.pkl"))):
                if _model_type_of(os.path.basename(f)) != mt:
                    continue
                m = torch.load(f, map_location="cpu")
                for key, metric in METRICS:
                    rows.append({"Split": split, "Model": label, "Metric": metric,
                                 "AUROC": float(m[key])})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    df.to_csv(CACHE, index=False)
    print(f"  cached aggregated results -> {CACHE}")
    return df


def plot(split, df, out_dir):
    labels = [lab for _, lab, _ in MODELS if lab in set(df["Model"])]
    palette = {lab: col for _, lab, col in MODELS}

    sns.set(context="paper", style="ticks", font=FONT_NAME)  # sns.set resets rcParams
    plt.rcParams.update({
        "svg.fonttype": "none",  # keep text as text -> editable in Illustrator
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # one size for every piece of text: tick labels, axis label, legend
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
    })
    fig, ax = plt.subplots(figsize=(4.2, 2.6), dpi=300)
    sns.barplot(df, x="Metric", y="AUROC", hue="Model",
                order=[m for _, m in METRICS], hue_order=labels,
                palette=palette, errorbar="sd", err_kws={"color": "0.25", "linewidth": 1.2},
                capsize=0.0, ax=ax)
    ax.set_ylim(0.4, None)
    ax.set_xlabel("")
    ax.set_ylabel("AUROC")
    # y ticks: labelled every Y_MAJOR, with one unlabelled minor tick halfway between
    ax.yaxis.set_major_locator(MultipleLocator(Y_MAJOR))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="major", length=4)
    ax.tick_params(axis="y", which="minor", length=2.5)
    ax.tick_params(axis="x", which="minor", bottom=False)  # no minor ticks on the categorical axis
    leg = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    sns.despine()

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"beataml_split_by_{split}")
    for ext in ("svg", "png"):
        fig.savefig(f"{stem}.{ext}", format=ext, dpi=300,
                    bbox_inches="tight", bbox_extra_artists=(leg,))
    plt.close(fig)
    print(f"  saved {stem}.svg / .png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-read metrics_*.pkl, ignoring the cache")
    args = ap.parse_args()
    if args.refresh and os.path.exists(CACHE):
        os.remove(CACHE)

    out_dir = os.path.join(HERE, "figs")
    all_df = read_all()
    for split in ["patient", "drug"]:
        df = all_df[all_df["Split"] == split]
        if df.empty:
            print(f"[{split}] no metrics found under {os.path.join(OUTPUT_DIR, split, SUBDIR)}")
            continue
        n_fits = len(df[df["Model"] == MODELS[0][1]]) // len(METRICS)
        print(f"\n===== {SPLIT_TITLE[split]} — mean±std across {n_fits} fits "
              f"(pretraining runs x training seeds) =====")
        for _, label, _ in MODELS:
            sub = df[df["Model"] == label]
            if sub.empty:
                continue
            cells = []
            for _, metric in METRICS:
                v = sub[sub["Metric"] == metric]["AUROC"].values
                cells.append(f"{metric}={v.mean():.3f}±{v.std():.3f}")
            print(f"  {label:54s} " + "   ".join(cells) + f"   (n={len(sub)//len(METRICS)})")
        plot(split, df, out_dir)


if __name__ == "__main__":
    main()
