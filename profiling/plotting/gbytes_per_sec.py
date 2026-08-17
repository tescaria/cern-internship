import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")   

def relative_performance(df_full, df_roi):
    """Calculates the achieved DRAM bandwidth as a fraction of each GPU's theoretical performance"""
    theoretical_gbytes = {
        "Tesla T4": 320,
        "RTX 5000 Ada": 576,   
        "H100 NVL": 3940        
    }

    for df in [df_full, df_roi]:
        for gpu in ["t4", "ada", "h100"]:
            df[f"{gpu}_relative"] = (
                df[f"{gpu}_gbytes_s"] / 
                theoretical_gbytes[{"t4": "Tesla T4", "ada": "RTX 5000 Ada", "h100": "H100 NVL"}[gpu]] * 100)

            df[f"{gpu}_relative_std"] = (
                df[f"{gpu}_gbytes_s_std"] /
                theoretical_gbytes[{"t4": "Tesla T4", "ada": "RTX 5000 Ada", "h100": "H100 NVL"}[gpu]] * 100)

    return df_full, df_roi, theoretical_gbytes

def plot_gbytes_s(df_full, df_roi, theoretical_gbytes, output_path=None, nthreads=1, nevent=10):

    fig, axes = plt.subplots(1, 2, figsize=(18, 9), sharey=True)

    height = 0.25
    y = range(len(df_full))

    colours = {
        "Tesla T4": "tab:blue",
        "RTX 5000 Ada": "tab:orange",
        "H100 NVL": "tab:green"
    }

    # Full data
    axes[0].barh([i - height for i in y], df_full["t4_relative"], height=height, xerr=df_full["t4_relative_std"],
        capsize=4, label=f"Tesla T4 ({theoretical_gbytes['Tesla T4']:.2f} GB/s)", color=colours["Tesla T4"])
    axes[0].barh(y, df_full["ada_relative"], height=height, xerr=df_full["ada_relative_std"],
        capsize=4, label=f"RTX 5000 Ada ({theoretical_gbytes['RTX 5000 Ada']:.2f} GB/s)", color=colours["RTX 5000 Ada"])
    axes[0].barh([i + height for i in y], df_full["h100_relative"], height=height, xerr=df_full["h100_relative_std"],
        capsize=4, label=f"H100 NVL ({theoretical_gbytes['H100 NVL']/1000:.2f} TB/s)", color=colours["H100 NVL"])

    # RoI
    axes[1].barh([i - height for i in y], df_roi["t4_relative"], height=height, xerr=df_roi["t4_relative_std"],
        capsize=4, color=colours["Tesla T4"])
    axes[1].barh(y, df_roi["ada_relative"], height=height, xerr=df_roi["ada_relative_std"],
        capsize=4, color=colours["RTX 5000 Ada"])
    axes[1].barh([i + height for i in y], df_roi["h100_relative"], height=height, xerr=df_roi["h100_relative_std"],
        capsize=4, color=colours["H100 NVL"])

    axes[0].set_title("Full Data", fontsize=18)
    axes[1].set_title("RoI", fontsize=18)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df_full["Kernel_Name"], fontsize=15)

    axes[0].set_ylabel("Kernel", fontsize=17)
    axes[0].set_xlabel("Achieved DRAM Bandwidth / Theoretical Peak [%]", fontsize=17)
    axes[1].set_xlabel("Achieved DRAM Bandwidth / Theoretical Peak [%]", fontsize=17)
    
    # Increase x-axis tick labels
    axes[0].tick_params(axis="x", labelsize=15)
    axes[1].tick_params(axis="x", labelsize=15)

    axes[0].set_xlim(0, 90)
    axes[1].set_xlim(0, 90)

    axes[0].legend(title="GPU", fontsize=14, title_fontsize=15, loc="upper right")

    fig.suptitle(
        f"Kernel DRAM Bandwidth Relative Performance  "
        f"({nthreads} thread, {nevent} events)",
        fontsize=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():

    nthreads = 1
    nevent = 10

    input_path_full = f"/eos/user/t/tcostaes/traccc_outputs/profiling/{nthreads}t_{nevent}ev_1rep/kernel_stats/gbytes_plots_full.csv"
    input_path_roi = f"/eos/user/t/tcostaes/traccc_outputs/profiling/{nthreads}t_{nevent}ev_1rep/kernel_stats/gbytes_plots_roi.csv"
    output_path = f"/eos/user/t/tcostaes/traccc_outputs/plots/profiling/gbytes_s_{nthreads}t_{nevent}ev.png"
    df_full = pd.read_csv(input_path_full)
    df_roi = pd.read_csv(input_path_roi)

    df_full, df_roi, theoretical_gbytes = relative_performance(df_full, df_roi)
    plot_gbytes_s(df_full, df_roi, theoretical_gbytes, output_path, nthreads, nevent)


if __name__ == "__main__":
    main()