import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

def plot_dup_stats(df):
    """Plot duplicate statistics per event of SuperRoI"""
    
    plt.figure(figsize=(12, 5))
    plt.bar(df["event"], df["duplicate_fraction"] * 100)
    plt.xlabel("Event", fontsize=13)
    plt.ylabel("Duplicate fraction [%]", fontsize=13)
    plt.title("Duplicate fraction per jet superRoI event")
    plt.xticks(df["event"])
    plt.tight_layout()
    plt.savefig("/eos/user/t/tcostaes/traccc_outputs/plots/roi_comparisons/dup_stats.png")


def superroi_roi_hits(df):
    """Plot bar comparison of SuperRoI and RoI hits per event"""

    x = range(len(df))
    width = 0.4

    plt.figure(figsize=(12, 5))
    plt.bar([i - width/2 for i in x], df["superRoI_hits"], width=width, label="Jet superRoIs")
    plt.bar([i + width/2 for i in x], df["RoI_hits"], width=width, label="Muon RoIs")

    plt.xlabel("Event", fontsize=13)
    plt.ylabel("Number of hits", fontsize=13)
    plt.xticks(x, df["event"])
    plt.legend()
    plt.tight_layout()
    plt.savefig("/eos/user/t/tcostaes/traccc_outputs/plots/roi_comparisons/superroi_roi_hits.png")


def ratio(df):
    """Plot SuperRoI / RoI hit ratio"""
    
    plt.figure(figsize=(12, 5))
    plt.bar(df["event"], df["ratio"])

    plt.xlabel("Event", fontsize=13)
    plt.ylabel("SuperRoI / RoI hit ratio", fontsize=13)
    plt.title("SuperRoI / RoI hit ratio per event")
    plt.xticks(df["event"])
    plt.tight_layout()
    plt.savefig("/eos/user/t/tcostaes/traccc_outputs/plots/roi_comparisons/superroi_roi_hits_ratio.png")
    
def main():
    dup_file = "/eos/user/t/tcostaes/traccc_outputs/duplicate_stats.csv"
    hits_file = "/eos/user/t/tcostaes/traccc_outputs/superroi_roi_hits.csv"

    df_dup = pd.read_csv(dup_file)
    df_hits = pd.read_csv(hits_file)

    plot_dup_stats(df_dup)
    superroi_roi_hits(df_hits)
    ratio(df_hits)

if __name__ == "__main__":
    main()