import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")

def plot_times(df, output_path=None):
    """Plots time by kernel and GPU"""
    # create data frame for plotting (keep only time columns)
    df_plot = df[["Kernel_Name", "t4_time", "ada_time", "h100_time"]].melt(
        id_vars=["Kernel_Name"], var_name="GPU", value_name="Time [ms]")

    # make GPU names nicer 
    df_plot["GPU"] = df_plot["GPU"].replace({
        "t4_time": "Tesla T4",
        "ada_time": "RTX 5000 Ada",
        "h100_time": "H100 NVL"})

    # plot
    plt.figure(figsize=(12, 9))
    sns.barplot(data=df_plot, x="Time [ms]", y="Kernel_Name", hue="GPU")
    plt.xlabel("Time [ms]", fontsize=13)
    plt.ylabel("Kernel", fontsize=13)
    plt.legend(title="GPU", fontsize=12)
    plt.title("Time by Kernel - Full Data (1 thread)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

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




def main():

    nthreads = 1
    nevent = 100
    data = "full"

    input_path = f"/eos/user/t/tcostaes/traccc_outputs/profiling/{nthreads}t_{nevent}ev_1rep/kernel_stats/time_plots_{data}.csv"
    output_path = f"/eos/user/t/tcostaes/traccc_outputs/plots/profiling/time_{data}_{nthreads}t_{nevent}ev.png"

    df = pd.read_csv(input_path)
    plot_times_std(df, output_path, nthreads, nevent)

if __name__ == "__main__":
    main()