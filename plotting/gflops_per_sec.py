import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")   

def relative_performance(df_full, df_roi):
    """Calculates the achieved FP32 performance as a fraction of each GPU's theoretical performance"""
    theoretical_gflops = {
        "Tesla T4": 8141,
        "RTX 5000 Ada": 65280,   
        "H100 NVL": 60320        
    }

    for df in [df_full, df_roi]:
        for gpu in ["t4", "ada", "h100"]:
            df[f"{gpu}_relative"] = (
                df[f"{gpu}_gflops_s"] / 
                theoretical_gflops[{"t4": "Tesla T4", "ada": "RTX 5000 Ada", "h100": "H100 NVL"}[gpu]] * 100)

            df[f"{gpu}_relative_std"] = (
                df[f"{gpu}_gflops_s_std"] /
                theoretical_gflops[{"t4": "Tesla T4", "ada": "RTX 5000 Ada", "h100": "H100 NVL"}[gpu]] * 100)

    return df_full, df_roi, theoretical_gflops

def plot_glops_s(df_full, df_roi, theoretical_gflops, output_path=None, nthreads=1, nevent=10):

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
        capsize=4, label=f"Tesla T4 ({theoretical_gflops['Tesla T4']/1000:.2f} TFLOP/s)", color=colours["Tesla T4"])
    axes[0].barh(y, df_full["ada_relative"], height=height, xerr=df_full["ada_relative_std"],
        capsize=4, label=f"RTX 5000 Ada ({theoretical_gflops['RTX 5000 Ada']/1000:.2f} TFLOP/s)", color=colours["RTX 5000 Ada"])
    axes[0].barh([i + height for i in y], df_full["h100_relative"], height=height, xerr=df_full["h100_relative_std"],
        capsize=4, label=f"H100 NVL ({theoretical_gflops['H100 NVL']/1000:.2f} TFLOP/s)", color=colours["H100 NVL"])

    # RoI
    axes[1].barh([i - height for i in y], df_roi["t4_relative"], height=height, xerr=df_roi["t4_relative_std"],
        capsize=4, color=colours["Tesla T4"])
    axes[1].barh(y, df_roi["ada_relative"], height=height, xerr=df_roi["ada_relative_std"],
        capsize=4, color=colours["RTX 5000 Ada"])
    axes[1].barh([i + height for i in y], df_roi["h100_relative"], height=height, xerr=df_roi["h100_relative_std"],
        capsize=4, color=colours["H100 NVL"])

    axes[0].set_title("Full Data")
    axes[1].set_title("RoI")

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df_full["Kernel_Name"])

    axes[0].set_ylabel("Kernel", fontsize=13)
    axes[0].set_xlabel("Achieved FP32 Performance / Theoretical Peak [%]", fontsize=13)
    axes[1].set_xlabel("Achieved FP32 Performance / Theoretical Peak [%]", fontsize=13)

    axes[0].set_xlim(0, 30)
    axes[1].set_xlim(0, 30)

    axes[0].legend(title="GPU", fontsize=11)

    fig.suptitle(
        f"Kernel GFLOPS/s Relative Performance  "
        f"({nthreads} thread, {nevent} events)",
        fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():

    nthreads = 1
    nevent = 10

    input_path_full = f"/eos/user/t/tcostaes/traccc_outputs/profiling/{nthreads}t_{nevent}ev_1rep/kernel_stats/gflops_plots_full.csv"
    input_path_roi = f"/eos/user/t/tcostaes/traccc_outputs/profiling/{nthreads}t_{nevent}ev_1rep/kernel_stats/gflops_plots_roi.csv"
    output_path = f"/eos/user/t/tcostaes/traccc_outputs/plots/profiling/gflops_s_{nthreads}t_{nevent}ev.png"
    df_full = pd.read_csv(input_path_full)
    df_roi = pd.read_csv(input_path_roi)

    df_full, df_roi, theoretical_gflops = relative_performance(df_full, df_roi)
    plot_glops_s(df_full, df_roi, theoretical_gflops, output_path, nthreads, nevent)


if __name__ == "__main__":
    main()