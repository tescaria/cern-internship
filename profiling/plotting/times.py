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

def getTime(nthreads, nevent, df_full, df_roi, df_superroi=None):
    # actual times and std (average of all events) in ms
    if nthreads==1 and nevent==10:
        times_full = [155.94, 19.04, 74.09, 10.53, 43.33, 4.93]
        times_roi = [15.11, 3.41, 19.28, 3.42, 13.62, 1.61]
        times_superroi = [32.82, 6.50, 30.53, 3.70, 20.45, 2.33]
    elif nthreads==6 and nevent==20:
        times_full = [117.72, 35.21, 73.72, 9.31, 44.48, 4.18]
        times_roi = [17.09, 2.13, 20.06, 2.54, 13.71, 1.66]
        times_superroi = getTime_reconstructed(df_superroi)
    else:
        times_full = getTime_reconstructed(df_full)
        times_roi = getTime_reconstructed(df_roi)
        times_superroi = getTime_reconstructed(df_superroi)
    
    if df_superroi is not None:
        return times_full, times_roi, times_superroi
    else: 
        return times_full, times_roi


def plot_gpu_bars(ax, df, times, y, height, title, colours):

    t4, t4_std, ada, ada_std, h100, h100_std = times
    ax.barh([i - height for i in y], df["t4_time_per"], height=height, xerr=df["t4_time_per_std"],
        capsize=4, label=rf"Tesla T4 ({t4:.2f} $\pm$ {t4_std:.2f} ms)", color=colours["Tesla T4"])
    ax.barh(y, df["ada_time_per"], height=height, xerr=df["ada_time_per_std"],
        capsize=4, label=rf"RTX 5000 Ada ({ada:.2f} $\pm$ {ada_std:.2f} ms)", color=colours["RTX 5000 Ada"])
    ax.barh([i + height for i in y], df["h100_time_per"], height=height, xerr=df["h100_time_per_std"],
        capsize=4, label=rf"H100 NVL ({h100:.2f} $\pm$ {h100_std:.2f} ms)", color=colours["H100 NVL"])

    ax.set_title(title, fontsize=18)


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

    plot_gpu_bars(axes[0], df_full, times_full, y, height, "Full Data", colours)
    plot_gpu_bars(axes[1], df_roi, times_roi, y, height, "Muon RoI", colours)
    
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


def plot_percentage_3datasets(df_full, df_roi, df_superroi, output_path=None, nthreads=1, nevent=10):

    fig, axes = plt.subplots(1, 3, figsize=(24, 9), sharey=True)

    height = 0.25
    y = range(len(df_full))

    colours = {
        "Tesla T4": "tab:blue",
        "RTX 5000 Ada": "tab:orange",
        "H100 NVL": "tab:green"
    }

    # get total times for labels
    times_full, times_roi, times_superroi = getTime(nthreads, nevent, df_full, df_roi, df_superroi)
    
    plot_gpu_bars(axes[0], df_full, times_full, y, height, "Full Data", colours)
    plot_gpu_bars(axes[1], df_roi, times_roi, y, height, "Muon RoI", colours)
    plot_gpu_bars(axes[2], df_superroi, times_superroi, y, height,"Jet SuperRoI", colours)

    # formatting
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df_full["Kernel_Name"], fontsize=15)
    axes[0].set_ylabel("Kernel", fontsize=17)

    for ax in axes:
        ax.set_xlabel("Percentage of Total Event Time [%]", fontsize=17)
        ax.tick_params(axis="x", labelsize=15)
        ax.set_xlim(0, 80)

    axes[0].legend(title="GPU — Total Event Time", fontsize=14, title_fontsize=15, loc="upper right")
    axes[1].legend(title="GPU — Total Event Time", fontsize=14, title_fontsize=15, loc="upper right")
    axes[2].legend(title="GPU — Total Event Time", fontsize=14, title_fontsize=15, loc="upper right")

    fig.suptitle(f"Kernel Contribution to Event Processing Time ({nthreads} thread, {nevent} events)", fontsize=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_kernel_times_3datasets(df_full, df_roi, df_superroi, gpu="t4", output_path=None, nthreads=1, nevent=10):
    """Compare kernel execution times for Full, Muon RoI and Jet SuperRoI on a single GPU."""

    fig, ax = plt.subplots(figsize=(14, 10))

    height = 0.25
    y = np.arange(len(df_full))

    # Dataset colours
    colours = {
        "Muon RoI": "#3f90da",
        "Full Data": "#744A84",
        "Jet SuperRoI": "#cc79a7"
    }

    # GPU names
    gpu_names = {
        "t4": "Tesla T4",
        "ada": "RTX 5000 Ada",
        "h100": "H100 NVL"
    }

    gpu_name = gpu_names[gpu]

    # Muon RoI
    ax.barh(y + height, df_roi[f"{gpu}_time"], height=height, xerr=df_roi[f"{gpu}_time_std"],
        capsize=4, label="Muon RoI", color=colours["Muon RoI"])

    # Full Data
    ax.barh(y - height, df_full[f"{gpu}_time"], height=height, xerr=df_full[f"{gpu}_time_std"],
        capsize=4, label="Full Data", color=colours["Full Data"])

    # Jet SuperRoI
    ax.barh(y, df_superroi[f"{gpu}_time"], height=height, xerr=df_superroi[f"{gpu}_time_std"],
        capsize=4, label="Jet SuperRoI", color=colours["Jet SuperRoI"])

    # Formatting
    ax.set_yticks(y)
    ax.set_yticklabels(df_full["Kernel_Name"], fontsize=15)
    ax.set_ylabel("Kernel", fontsize=17)
    ax.set_xlabel("Kernel Time [ms]", fontsize=17)
    ax.tick_params(axis="x", labelsize=15)

    ax.legend(title="Input Dataset", fontsize=14, title_fontsize=15, loc="upper right")
    ax.set_title(f"Kernel Execution Time — {gpu_name}\n"
        f"({nthreads} thread, {nevent} events)",
        fontsize=20
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():

    nthreads = 1
    nevent = 10

    #data = "full"
    #input_path = f"/eos/user/t/tcostaes/traccc_outputs/profiling/{nthreads}t_{nevent}ev_1rep/kernel_stats/time_plots_{data}.csv"
    #output_path = f"/eos/user/t/tcostaes/traccc_outputs/plots/profiling/time_{data}_{nthreads}t_{nevent}ev.png"
    #df = pd.read_csv(input_path)
    #plot_times_std(df, output_path, nthreads, nevent)

    input_path_full = f"/eos/user/t/tcostaes/traccc_outputs/profiling/{nthreads}t_{nevent}ev_1rep/kernel_stats/time_plots_full.csv"
    input_path_roi = f"/eos/user/t/tcostaes/traccc_outputs/profiling/{nthreads}t_{nevent}ev_1rep/kernel_stats/time_plots_roi.csv"
    input_path_superroi = f"/eos/user/t/tcostaes/traccc_outputs/profiling/{nthreads}t_{nevent}ev_1rep/kernel_stats/time_plots_superroi.csv"
    output_path = f"/eos/user/t/tcostaes/traccc_outputs/plots/profiling/time_percentage_{nthreads}t_{nevent}ev_3datasets.png"

    df_full = pd.read_csv(input_path_full)
    df_roi = pd.read_csv(input_path_roi)
    df_superroi = pd.read_csv(input_path_superroi)

    #plot_percentage(df_full, df_roi, output_path, nthreads, nevent)
    plot_percentage_3datasets(df_full, df_roi, df_superroi, output_path, nthreads, nevent)
    plot_kernel_times_3datasets(df_full, df_roi, df_superroi, gpu="h100", 
                output_path=f"/eos/user/t/tcostaes/traccc_outputs/plots/profiling/kernel_times_3datasets_h100_{nthreads}t_{nevent}ev.png", 
                nthreads=nthreads,nevent=nevent)
    

if __name__ == "__main__":
    main()