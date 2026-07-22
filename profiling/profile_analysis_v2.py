import pandas as pd
import numpy as np

# process the data frame 
def processDF(df):

    # extract units row (as different runs may have different units)
    units = df.iloc[0]
    dram_unit = units.get("dram__bytes.sum", None)
    time_unit = units.get("gpu__time_duration.sum", None)
    dram_rate_unit = units.get("dram__bytes.sum.per_second", None)

    # remove empty/unit rows
    df = df[df["Kernel Name"].notna()].reset_index(drop=True)
    # remove unessessary columns
    cols_to_drop = [
        "Process ID", "Process Name", "Host Name",
        "Context", "Stream", "Device", "CC", "Section Name"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # convert metric columns to numbers
    metric_cols = [
        "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
        "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
        "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
        "dram__bytes.sum",
        "dram__bytes.sum.per_second",
        "gpu__time_duration.sum",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"]

    for col in metric_cols:
        if col in df.columns:
            df[col] = (df[col].astype(str).str.replace(",", ""))
            df[col] = pd.to_numeric(df[col], errors="coerce")


    # assign event id based on ccl_kernel
    df["event_id"] = df["Kernel Name"].str.contains("ccl_kernel").cumsum() - 1

    return df, dram_unit, time_unit, dram_rate_unit

def _sum_or_nan(df, col):
    """Return column sum or NaN if column is missing."""
    return df[col].sum() if col in df.columns else np.nan


def convert_units(df, dram_unit, time_unit, dram_rate_unit):
    """
    Convert units to bytes and seconds.
    Use after processing data frame and before metric retrieving functions.
    """
    dram_scale = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9}
    time_scale = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1}
    dram_rate_scale = {"byte/s": 1, "Kbyte/s": 1e3, "Mbyte/s": 1e6, "Gbyte/s": 1e9,}

    if "dram__bytes.sum" in df.columns:
        df["dram__bytes.sum"] *= dram_scale[dram_unit]

    if "gpu__time_duration.sum" in df.columns:
        df["gpu__time_duration.sum"] *= time_scale[time_unit]

    if "dram__bytes.sum.per_second" in df.columns:
        df["dram__bytes.sum.per_second"] *= dram_rate_scale[dram_rate_unit]

    return df


def getFLOPS(df, event_id=None):
    """Get amount of GFLOPS performed"""
    if event_id is not None:
        df = df[df["event_id"] == event_id]

    fma = _sum_or_nan(df, "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum") / 1e9
    add = _sum_or_nan(df, "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum") / 1e9
    mul = _sum_or_nan(df, "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum") / 1e9

    total = 2*fma + add + mul

    return total


def getBytes(df, event_id=None):
    """Get amount of GB transferred"""
    if event_id is not None:
        df = df[df["event_id"] == event_id]

    bytes_transferred = _sum_or_nan(df, "dram__bytes.sum") 

    return bytes_transferred / 1e9


def getTime(df, event_id=None):
    """Get run execution time"""
    if event_id is not None:
        df = df[df["event_id"] == event_id]

    return _sum_or_nan(df, "gpu__time_duration.sum")


def getKernelStats(df, sorting_method="time_mean", df_mem=None, n=10):
    """
    Get average kernel contribution per processed event.
    Sorted top ones by chosen method.
    """

    time_col = "gpu__time_duration.sum"
    sm_col = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
    mem_col = "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"

    # time-weighted throughput per kernel per event
    def weighted_avg(group, metric):
        if metric not in group.columns or time_col not in group.columns:
            return np.nan
        total_time = group[time_col].sum()
        if total_time == 0:
            return np.nan
        return (group[metric] * group[time_col]).sum() / total_time

    # collapse multiple launches of the same kernel within each processed
    # event into a single set of per-event metrics.
    per_event_metrics = (df.groupby(["event_id", "Kernel Name"]).apply(lambda x: pd.Series({
            "time": getTime(x),
            "sm_throughput": weighted_avg(x, sm_col),
            "memory_throughput": weighted_avg(x, mem_col),
            "bytes": (
                getBytes(df_mem[(df_mem["event_id"] == x.name[0]) &
                                (df_mem["Kernel Name"] == x.name[1])])
                if df_mem is not None else getBytes(x)),
            "time_bytes": (
                getTime(df_mem[(df_mem["event_id"] == x.name[0]) &
                                (df_mem["Kernel Name"] == x.name[1])])
                if df_mem is not None else getTime(x)),
            "gbytes_s":  weighted_avg(x, "dram__bytes.sum.per_second") / 1e9,
            "manual_gbytes_s": (
                getBytes(df_mem[(df_mem["event_id"] == x.name[0]) &
                                (df_mem["Kernel Name"] == x.name[1])]) /
                getTime(df_mem[(df_mem["event_id"] == x.name[0]) &
                            (df_mem["Kernel Name"] == x.name[1])])
                if df_mem is not None else getBytes(x) / getTime(x)),
            "gflop": getFLOPS(x),
            "gflop_s": getFLOPS(x) / getTime(x)
            }),  include_groups=False).reset_index())

    # average each kernel's per-event metrics across all processed events
    kernel_stats = (per_event_metrics.groupby("Kernel Name").agg({
            "time": ["mean", "std"],
            "sm_throughput": ["mean", "std"],
            "memory_throughput": ["mean", "std"],
            "bytes": ["mean", "std"],
            "time_bytes": ["mean", "std"],
            "gbytes_s": ["mean", "std"],
            "manual_gbytes_s": ["mean", "std"],
            "gflop": ["mean", "std"],
            "gflop_s": ["mean", "std"]
            }))

    # flatten column names
    kernel_stats.columns = ["_".join(col) for col in kernel_stats.columns]

    # sort by average time and keep top n
    top = (kernel_stats.sort_values(sorting_method, ascending=False).head(n))

    return top


def getStats(fname, sorting_method, path_mem=None):
    """
    Get overall statistics averaged per event.
    Get average statistics per kernel and per event by calling getKernelStats.
    """
    df = pd.read_csv(fname, low_memory=False)
    df, dram_unit, time_unit, dram_rate_unit = processDF(df)
    df = convert_units(df, dram_unit, time_unit, dram_rate_unit)

    # Remove cold runs
    df = df[df["event_id"] >= 5].copy()
    df["event_id"] -= 5

    df_mem = None
    if path_mem is not None:
        df_mem = pd.read_csv(path_mem, low_memory=False)
        df_mem, dram_unit_mem, time_unit_mem, dram_rate_unit_mem = processDF(df_mem)
        # convert units to bytes and seconds
        df_mem = convert_units(df_mem, dram_unit_mem, time_unit_mem, dram_rate_unit_mem)
        # remove cold runs
        df_mem = df_mem[df_mem["event_id"] >= 5].copy()
        df_mem["event_id"] -= 5

    result=[]
    for i in range(df.event_id.max() + 1):
        tmp = df[df.event_id == i]
        flops = getFLOPS(tmp)
        time = getTime(tmp)
        if path_mem is not None:
            tmp_mem = df_mem[df_mem["event_id"] == i]
            bytes_transferred = getBytes(tmp_mem)
            time_bytes = getTime(tmp_mem)
        else:
            bytes_transferred = getBytes(tmp)
            time_bytes = getTime(tmp)
        result.append([i, bytes_transferred, time_bytes, flops, time])

    df_overall_stats = pd.DataFrame(result, columns=["event_id", "gb", "time_bytes", "gflop", "time"])

    print("Overall event statistics:")
    print(f"GB transferred: {df_overall_stats['gb'].mean():.2f} "f"+/- {df_overall_stats['gb'].std():.2f} GB")
    print(f"Time (for bytes): {df_overall_stats['time_bytes'].mean():.5f} "f"+/- {df_overall_stats['time_bytes'].std():.5f} s")
    print(f"FLOP: {df_overall_stats['gflop'].mean():.2f} "f"+/- {df_overall_stats['gflop'].std():.2f} GFLOP")
    print(f"Time: {df_overall_stats['time'].mean():.5f} "f"+/- {df_overall_stats['time'].std():.5f} s")

    # kernel statistics
    df_kernel_stats = getKernelStats(df, sorting_method, df_mem)
    
    return df_overall_stats, df_kernel_stats,  df_overall_stats["time"].mean()

def saveKernelStats(kernel_stats, total_time, filename):

    df = kernel_stats.reset_index()
    df["Kernel Name"] = df["Kernel Name"]
    # percentages
    df["time_percentage"] = (df["time_mean"] / total_time * 100)
    # save
    df.to_csv(filename, index=False)

def main():
    fname = "/eos/user/t/tcostaes/traccc_outputs/profiling/1t_10ev_1rep/raw_for_analysis/mldev02_full_1t_10ev_memory_check_v2.csv"
    #path_mem = "/eos/user/t/tcostaes/traccc_outputs/profiling/1t_10ev_1rep/raw_for_analysis/lxplus_full_1t_10ev_mem.csv"
    path_mem = None
    # OPTIONS: "time_mean", "sm_throughput_mean", "memory_throughput_mean", 
    # "bytes_mean", "time_bytes_mean", "gflop_mean", "gbytes_s_mean", "manual_gbytes_s_mean", "gflop_s_mean"
    sorting_method="memory_throughput_mean"
    overall_stats, kernel_stats, total_time = getStats(fname, sorting_method, path_mem)
    print("\nKernel statistics:")
    print(kernel_stats)
    
    csv_filename = "/eos/user/t/tcostaes/traccc_outputs/profiling/1t_10ev_1rep/kernel_stats/mldev02_full_memory_check_v2_mem.csv"
    saveKernelStats(kernel_stats, total_time, filename=csv_filename)

if __name__ == "__main__":
    main()
