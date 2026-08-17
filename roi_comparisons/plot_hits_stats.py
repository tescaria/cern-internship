import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("darkgrid")

def count_hits(filename):
    """Count hits (rows) in one event."""
    with open(filename, "r") as f:
        return sum(1 for _ in f) - 1   # subtract header


def collect_hit_counts(directory, skip=0):
    """Collect total hit counts for each event"""

    files = sorted(glob.glob(os.path.join(directory, "event?????????-cells.csv")))[skip:]
    hit_counts = []
    for f in files:
        hit_counts.append(count_hits(f))

    return hit_counts

def plot_dup_stats(df):
    """Plot distribution of duplicate fraction across SuperRoI events"""
    
    plt.figure(figsize=(8, 5))
    duplicate_fraction = df["duplicate_fraction"] * 100

    bins = range(0, int(duplicate_fraction.max()) + 5, 5)
    plt.hist(duplicate_fraction, bins=bins)
    plt.xlabel("Duplicate Fraction [%]", fontsize=13)
    plt.ylabel("Number of Events", fontsize=13)
    plt.title("Duplicate fraction across jet superRoI events")
    plt.xticks(bins)
    plt.tight_layout()
    plt.savefig("/eos/user/t/tcostaes/traccc_outputs/plots/roi_comparisons/dup_stats.png")


def plot_hit_distributions(superroi_hits, roi_hits):

    plt.figure(figsize=(8, 5))

    all_hits = superroi_hits + roi_hits

    bins = np.logspace(np.log10(min(all_hits)), np.log10(max(all_hits)), 25)

    # Fraction of events in each bin
    superroi_weights = np.ones(len(superroi_hits)) / len(superroi_hits)
    roi_weights = np.ones(len(roi_hits)) / len(roi_hits)

    plt.hist(superroi_hits, bins=bins, weights=superroi_weights, alpha=0.6, label="Jet superRoIs")
    plt.hist(roi_hits, bins=bins, weights=roi_weights, alpha=0.6, label="Muon RoIs")
    plt.xscale("log")

    plt.xlabel("Number of Hits", fontsize=13)
    plt.ylabel("Fraction of Events", fontsize=13)
    plt.legend()
    plt.tight_layout()

    plt.savefig("/eos/user/t/tcostaes/traccc_outputs/plots/roi_comparisons/hit_distributions.png")


def plot_hit_distributions_3datasets(superroi_hits, roi_hits, full_hits):

    plt.figure(figsize=(8, 5))

    all_hits = superroi_hits + roi_hits + full_hits

    bins = np.logspace(np.log10(min(all_hits)), np.log10(max(all_hits)), 25)

    # normalise each dataset
    superroi_weights = np.ones(len(superroi_hits)) / len(superroi_hits)
    roi_weights = np.ones(len(roi_hits)) / len(roi_hits)
    full_weights = np.ones(len(full_hits)) / len(full_hits)

    plt.hist(superroi_hits, bins=bins, weights=superroi_weights, alpha=0.6, label="Jet superRoIs")
    plt.hist(roi_hits, bins=bins, weights=roi_weights, alpha=0.6, label="Muon RoIs")
    plt.hist(full_hits, bins=bins, weights=full_weights, alpha=0.6, label="Full data")

    plt.xscale("log")

    plt.xlabel("Number of Hits", fontsize=13)
    plt.ylabel("Fraction of Events", fontsize=13)
    plt.legend()
    plt.tight_layout()

    plt.savefig("/eos/user/t/tcostaes/traccc_outputs/plots/roi_comparisons/hit_distributions_3datasets.png")
    
def main():

    dup_file = "/eos/user/t/tcostaes/traccc_outputs/duplicate_stats.csv"
    df_dup = pd.read_csv(dup_file)
    plot_dup_stats(df_dup)

    superroi_nodup_dir = "/eos/user/t/tcostaes/traccc_outputs/roiInputJets_nodup"
    roi_dir = "/eos/user/e/exochell/traccc/traccc_athena_plots/g200/traccc-athena/data/roiInputMuon"
    superroi_hits = collect_hit_counts(superroi_nodup_dir)
    roi_hits = collect_hit_counts(roi_dir, skip=5)

    plot_hit_distributions(superroi_hits, roi_hits)

    #full_dir = "/eos/project/a/atlas-eftracking/GPU/ITk_data/traccc_standalone_data/ttbar_mu200"
    #full_hits = collect_hit_counts(full_dir)
    #plot_hit_distributions_3datasets(superroi_hits, roi_hits, full_hits)


if __name__ == "__main__":
    main()