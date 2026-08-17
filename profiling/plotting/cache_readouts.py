import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")

def plot_dram(df):
    """Plots DRAM accessed by kernel and GPU"""
    # create data frame for plotting (keep only dram columns)
    df_plot = df[["Kernel_Name", "T4_DRAM", "Ada_DRAM", "H100_DRAM"]].melt(
        id_vars=["Kernel_Name"], var_name="GPU", value_name="DRAM Accessed [MB]")

    # make GPU names nicer 
    df_plot["GPU"] = df_plot["GPU"].replace({
        "T4_DRAM": "Tesla T4",
        "Ada_DRAM": "RTX 5000 Ada",
        "H100_DRAM": "H100 NVL"})

    # plot
    plt.figure(figsize=(12, 9))
    sns.barplot(data=df_plot, x="DRAM Accessed [MB]", y="Kernel_Name", hue="GPU")
    plt.xlabel("DRAM Accessed [MB]", fontsize=17)
    plt.ylabel("Kernel", fontsize=17)
    plt.yticks(fontsize=15)
    plt.legend(title="GPU", fontsize=14)
    plt.title("DRAM Traffic by Kernel - Full Data (1 thread)", fontsize=14)
    plt.tight_layout()
    plt.savefig("/eos/user/t/tcostaes/traccc_outputs/plots/profiling/cache_readouts_dram.png", dpi=300)


def plot_l1_dram(df):
    """Plots L1 requested vs DRAM accesed by kernel and GPU"""
    df_plot = pd.concat([
        df[["Kernel_Name", "T4_L1", "T4_DRAM"]].assign(GPU="Tesla T4")
        .rename(columns={"T4_L1": "L1", "T4_DRAM": "DRAM"}),

    df[["Kernel_Name", "Ada_L1", "Ada_DRAM"]].assign(GPU="RTX 5000 Ada")
        .rename(columns={"Ada_L1": "L1", "Ada_DRAM": "DRAM"}),

    df[["Kernel_Name", "H100_L1", "H100_DRAM"]].assign(GPU="H100 NVL")
        .rename(columns={"H100_L1": "L1", "H100_DRAM": "DRAM"})], ignore_index=True)

    plt.figure(figsize=(12, 7))
    # colour = kernel, marker = GPU
    sns.scatterplot(data=df_plot,   x="L1", y="DRAM", hue="Kernel_Name", style="GPU", s=150)

    plt.xlabel("L1 requested [MB]", fontsize=17)
    plt.ylabel("DRAM accessed [MB]", fontsize=17)
    plt.title("DRAM Traffic vs L1 Requested — Full Data (1 thread)", fontsize=18)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig("/eos/user/t/tcostaes/traccc_outputs/plots/profiling/cache_readouts_l1_dram.png", dpi=300)



def main():

    df = pd.read_csv("/eos/user/t/tcostaes/traccc_outputs/profiling/1t_10ev_1rep/kernel_stats/cache_readouts_plots_full.csv")
    plot_dram(df)
    plot_l1_dram(df)

if __name__ == "__main__":
    main()