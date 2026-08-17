import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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

def getTime_reconstructed(df):
    """Get the average total event processing time"""

    time_std = []
    for gpu in ["t4", "ada", "h100"]:
        total_times = df[f"{gpu}_time"] / (df[f"{gpu}_time_per"] / 100)
        total_time = np.mean(total_times)
        std = np.std(total_times)
        time_std.append(total_time)
        time_std.append(std)

    return time_std

def getTime(nthreads, nevent, df_full, df_roi):
    # actual times (average of all events)
    if nthreads==1 and nevent==10:
        times_full = [155.94, 19.04, 74.09, 10.53, 43.33, 4.93]
        times_roi = [15.11, 3.41, 19.28, 3.42, 13.62, 1.61]
    elif nthreads==6 and nevent==20:
        times_full = [117.72, 35.21, 73.72, 9.31, 44.48, 4.18]
        times_roi = [17.09, 2.13, 20.06, 2.54, 13.71, 1.66]
    else:
        times_full = getTime_reconstructed(df_full)
        times_roi = getTime_reconstructed(df_roi)

    return times_full, times_roi


def plot_percentage(df_full, df_roi, output_path=None, nthreads=1, nevent=10):

    fig, axes = plt.subplots(1, 2, figsize=(18, 9), sharey=True)

    height = 0.25
    y = range(len(df_full))

    colours = {
        "Tesla T4": "tab:blue",
        "RTX 5000 Ada": "tab:orange",
        "H100 NVL": "tab:green"
    }

    # get total times for labels 
    times_full, times_roi = getTime(nthreads, nevent, df_full, df_roi)
    t4_full, t4_full_std, ada_full, ada_full_std, h100_full, h100_full_std = times_full
    t4_roi, t4_roi_std, ada_roi, ada_roi_std, h100_roi, h100_roi_std = times_roi

    # Full data
   # axes[0].barh([i - height for i in y], df_full["t4_time_per"], height=height, xerr=df_full["t4_time_per_std"],
   #     capsize=4, label=rf"Tesla T4 ({t4_full:.2f} $\pm$ {t4_full_std:.2f} ms)", color=colours["Tesla T4"])
    axes[0].barh(y, df_full["ada_time_per"], height=height, xerr=df_full["ada_time_per_std"],
        capsize=4, label=rf"RTX 5000 Ada ({ada_full:.2f} $\pm$ {ada_full_std:.2f} ms)", color=colours["RTX 5000 Ada"])
    axes[0].barh([i + height for i in y], df_full["h100_time_per"], height=height, xerr=df_full["h100_time_per_std"],
        capsize=4, label=rf"H100 NVL ({h100_full:.2f} $\pm$ {h100_full_std:.2f} ms)", color=colours["H100 NVL"])

    # RoI
    axes[1].barh([i - height for i in y], df_roi["t4_time_per"], height=height, xerr=df_roi["t4_time_per_std"],
        capsize=4, label=rf"Tesla T4 ({t4_roi:.2f} $\pm$ {t4_roi_std:.2f} ms)", color=colours["Tesla T4"])
    axes[1].barh(y, df_roi["ada_time_per"], height=height, xerr=df_roi["ada_time_per_std"],
        capsize=4, label=rf"RTX 5000 Ada ({ada_roi:.2f} $\pm$ {ada_roi_std:.2f} ms)", color=colours["RTX 5000 Ada"])
    axes[1].barh([i + height for i in y], df_roi["h100_time_per"], height=height, xerr=df_roi["h100_time_per_std"],
        capsize=4, label=rf"H100 NVL ({h100_roi:.2f} $\pm$ {h100_roi_std:.2f} ms)", color=colours["H100 NVL"])

    axes[0].set_title("Full Data", fontsize=18)
    axes[1].set_title("RoI", fontsize=18)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df_full["Kernel_Name"], fontsize=15)

    axes[0].set_ylabel("Kernel", fontsize=17)
    axes[0].set_xlabel("Percentage of Total Event Time [%]", fontsize=17)
    axes[1].set_xlabel("Percentage of Total Event Time [%]", fontsize=17)

    # Increase x-axis tick labels
    axes[0].tick_params(axis="x", labelsize=15)
    axes[1].tick_params(axis="x", labelsize=15)

    axes[0].set_xlim(0, 80)
    axes[1].set_xlim(0, 80)

    axes[0].legend(title="GPU — Total Event Time", fontsize=14, title_fontsize=15, loc="upper right")
    axes[1].legend(title="GPU — Total Event Time", fontsize=14, title_fontsize=15, loc="upper right")

    fig.suptitle(
        f"Kernel Contribution to Event Processing Time "
        f"({nthreads} thread, {nevent} events)",
        fontsize=20
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():

    nthreads = 6
    nevent = 20

    #data = "full"
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