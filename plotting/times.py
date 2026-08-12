import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")

def plot_times_std(df, output_path=None, nthreads=1, nevent=10):
    """Plots mean kernel time with standard deviation."""

    plt.figure(figsize=(12, 9))

    # Position of the three GPU bars
    y = range(len(df))
    height = 0.25

    plt.barh([i - height for i in y], df["t4_time"], height=height, xerr=df["t4_time_std"],
        capsize=4, label="Tesla T4")
    plt.barh(y, df["ada_time"], height=height, xerr=df["ada_time_std"],
        capsize=4, label="RTX 5000 Ada")
    plt.barh([i + height for i in y], df["h100_time"], height=height, xerr=df["h100_time_std"],
        capsize=4, label="H100 NVL")

    plt.yticks(y, df["Kernel_Name"])
    plt.xlabel("Time [ms]", fontsize=13)
    plt.xlim(0, 80) # 18 for roi, 80 for full
    plt.ylabel("Kernel", fontsize=13)
    plt.legend(title="GPU", fontsize=12)
    plt.title(f"Time Per Event - Full Data ({nthreads} thread, {nevent} events)", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_percentage(df_full, df_roi, output_path=None, nthreads=1, nevent=10):

    fig, axes = plt.subplots(1, 2, figsize=(18, 9), sharey=True)

    height = 0.25
    y = range(len(df_full))

    colours = {
        "Tesla T4": "tab:blue",
        "RTX 5000 Ada": "tab:orange",
        "H100 NVL": "tab:green"
    }

    # Full data
    axes[0].barh([i - height for i in y], df_full["t4_time_per"], height=height, xerr=df_full["t4_time_per_std"],
        capsize=4, label="Tesla T4", color=colours["Tesla T4"])
    axes[0].barh(y, df_full["ada_time_per"], height=height, xerr=df_full["ada_time_per_std"],
        capsize=4, label="RTX 5000 Ada", color=colours["RTX 5000 Ada"])
    axes[0].barh([i + height for i in y], df_full["h100_time_per"], height=height, xerr=df_full["h100_time_per_std"],
        capsize=4, label="H100 NVL", color=colours["H100 NVL"])

    # RoI
    axes[1].barh([i - height for i in y], df_roi["t4_time_per"], height=height, xerr=df_roi["t4_time_per_std"],
        capsize=4, color=colours["Tesla T4"])
    axes[1].barh(y, df_roi["ada_time_per"], height=height, xerr=df_roi["ada_time_per_std"],
        capsize=4, color=colours["RTX 5000 Ada"])
    axes[1].barh([i + height for i in y], df_roi["h100_time_per"], height=height, xerr=df_roi["h100_time_per_std"],
        capsize=4, color=colours["H100 NVL"])

    axes[0].set_title("Full Data")
    axes[1].set_title("RoI")

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df_full["Kernel_Name"])

    axes[0].set_ylabel("Kernel", fontsize=13)
    axes[0].set_xlabel("Percentage of Total Event Time [%]", fontsize=13)
    axes[1].set_xlabel("Percentage of Total Event Time [%]", fontsize=13)

    axes[0].set_xlim(0, 80)
    axes[1].set_xlim(0, 80)

    axes[0].legend(title="GPU", fontsize=11)

    fig.suptitle(
        f"Kernel Contribution to Event Processing Time "
        f"({nthreads} thread, {nevent} events)",
        fontsize=14
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():

    nthreads = 1
    nevent = 10
    data = "full"

    #input_path = f"/eos/user/t/tcostaes/traccc_outputs/profiling/{nthreads}t_{nevent}ev_1rep/kernel_stats/time_plots_{data}.csv"
    #output_path = f"/eos/user/t/tcostaes/traccc_outputs/plots/profiling/time_{data}_{nthreads}t_{nevent}ev.png"

    #df = pd.read_csv(input_path)
    #plot_times_std(df, output_path, nthreads, nevent)
    input_path_full = f"/eos/user/t/tcostaes/traccc_outputs/profiling/{nthreads}t_{nevent}ev_1rep/kernel_stats/time_plots_full.csv"
    input_path_roi = f"/eos/user/t/tcostaes/traccc_outputs/profiling/{nthreads}t_{nevent}ev_1rep/kernel_stats/time_plots_roi.csv"
    output_path = f"/eos/user/t/tcostaes/traccc_outputs/plots/profiling/time_percentage_{nthreads}t_{nevent}ev.png"
    df_full = pd.read_csv(input_path_full)
    df_roi = pd.read_csv(input_path_roi)
    plot_percentage(df_full, df_roi, output_path, nthreads, nevent)


if __name__ == "__main__":
    main()