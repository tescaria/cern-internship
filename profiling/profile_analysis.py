import pandas as pd
import numpy as np
import re

# process the data frame 
def processDF(df):

    # extract units row (as different runs may have different units)
    units = df.iloc[0]
    dram_unit = units.get("dram__bytes.sum", None)
    time_unit = units.get("gpu__time_duration.sum", None)

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
        "gpu__time_duration.sum",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"]

    for col in metric_cols:
        if col in df.columns:
            df[col] = (df[col].astype(str).str.replace(",", ""))
            df[col] = pd.to_numeric(df[col], errors="coerce")


    # assign event id based on ccl_kernel
    df["event_id"] = df["Kernel Name"].str.contains("ccl_kernel").cumsum() - 1

    return df, dram_unit, time_unit

def convert_units(df, dram_unit, time_unit):
    # convert units to bytes and seconds
    dram_scale = {
        "byte": 1,
        "Kbyte": 1e3,
        "Mbyte": 1e6,
        "Gbyte": 1e9
    }

    time_scale = {
        "ns": 1e-9,
        "us": 1e-6,
        "ms": 1e-3,
        "s": 1
    }

    if "dram__bytes.sum" in df.columns:
        df["dram__bytes.sum"] *= dram_scale[dram_unit]

    if "gpu__time_duration.sum" in df.columns:
        df["gpu__time_duration.sum"] *= time_scale[time_unit]

    return df

# get metrics functions
# giga flops
def getFLOPS(df, event_id=None):
    if event_id is not None:
        df = df[df["event_id"] == event_id]

    fma = df["smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"].sum() / 1e9
    add = df["smsp__sass_thread_inst_executed_op_fadd_pred_on.sum"].sum() / 1e9
    mul = df["smsp__sass_thread_inst_executed_op_fmul_pred_on.sum"].sum() / 1e9

    total = 2*fma + add + mul

    return total

# get memory bandwidth in GB/s
def getBytes(df, event_id=None):
    if event_id is not None:
        df = df[df["event_id"] == event_id]

    bytes_transferred = df["dram__bytes.sum"].sum() 
    time = df["gpu__time_duration.sum"].sum() 

    bandwidth = bytes_transferred / time / 1e9  # convert to GB/s

    return bandwidth, bytes_transferred / 1e9, time  # return bandwidth in GB/s, bytes in GB, time in seconds

def getTime(df, event_id=None):
    if event_id is not None:
        df = df[df["event_id"] == event_id]

    return df["gpu__time_duration.sum"].sum() 

def getKernelStats(df, df_mem=None, n=10):
    # df input already without cold runs and event_id adjusted

    time_col = "gpu__time_duration.sum"
    sm_col = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
    mem_col = "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"

    # time-weighted throughput per kernel per event
    def weighted_avg(group, metric):
        return ((group[metric] * group[time_col]).sum() / group[time_col].sum())

    per_event_metrics = (df.groupby(["event_id", "Kernel Name"]).apply(lambda x: pd.Series({
            "time": x[time_col].sum(),
            "sm_throughput": weighted_avg(x, sm_col),
            "memory_throughput": weighted_avg(x, mem_col),
            "bandwidth": (
                getBytes(df_mem[(df_mem["event_id"] == x.name[0]) &
                                (df_mem["Kernel Name"] == x.name[1])])[0]
                if df_mem is not None else getBytes(x)[0]),
            "bytes": (
                getBytes(df_mem[(df_mem["event_id"] == x.name[0]) &
                                (df_mem["Kernel Name"] == x.name[1])])[1]
                if df_mem is not None else getBytes(x)[1]),
            "time_bw": (
                getBytes(df_mem[(df_mem["event_id"] == x.name[0]) &
                                (df_mem["Kernel Name"] == x.name[1])])[2]
                if df_mem is not None else getBytes(x)[2]),
            "gflop": getFLOPS(x),
            "gflop_s": getFLOPS(x) / x[time_col].sum()
            }),  include_groups=False).reset_index())

    # average over processed events
    kernel_stats = (per_event_metrics.groupby("Kernel Name").agg({
            "time": ["mean", "std"],
            "sm_throughput": ["mean", "std"],
            "memory_throughput": ["mean", "std"],
            "bandwidth": ["mean", "std"],
            "bytes": ["mean", "std"],
            "time_bw": ["mean", "std"],
            "gflop": ["mean", "std"],
            "gflop_s": ["mean", "std"]}))

    # flatten column names
    kernel_stats.columns = ["_".join(col) for col in kernel_stats.columns]

    # sort by average time and keep top 5
    top5 = (kernel_stats.sort_values("bandwidth_mean", ascending=False).head(n))

    return top5

# get stats: overall and per kernel 
def getStats(fname, path_mem=None):
    df = pd.read_csv(fname, low_memory=False)
    df, dram_unit, time_unit = processDF(df)
    df = convert_units(df, dram_unit, time_unit)

    # Remove cold runs
    df = df[df["event_id"] >= 5].copy()
    df["event_id"] -= 5

    df_mem = None
    if path_mem is not None:
        df_mem = pd.read_csv(path_mem, low_memory=False)
        df_mem, dram_unit_mem, time_unit_mem = processDF(df_mem)
        # convert units to bytes and seconds
        df_mem = convert_units(df_mem, dram_unit_mem, time_unit_mem)
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
            bandwidth, bytes_transferred, time_bw  = getBytes(tmp_mem)
        else:
            bandwidth, bytes_transferred, time_bw = getBytes(tmp)
        result.append([i, bandwidth, bytes_transferred, time_bw, flops, flops/time, time])

    df_overall_stats = pd.DataFrame(result, columns=["event_id", "gb/s", "gb", "time_bw", "gflop", "gflop/s", "time"])

    print("Overall event statistics:")
    print(f"Bandwidth: {df_overall_stats['gb/s'].mean():.2f} "f"+/- {df_overall_stats['gb/s'].std():.2f} GB/s")
    print(f"GB transferred: {df_overall_stats['gb'].mean():.2f} "f"+/- {df_overall_stats['gb'].std():.2f} GB")
    print(f"Time (to calc bw): {df_overall_stats['time_bw'].mean():.5f} "f"+/- {df_overall_stats['time_bw'].std():.5f} s")
    print(f"FLOP: {df_overall_stats['gflop'].mean():.2f} "f"+/- {df_overall_stats['gflop'].std():.2f} GFLOP")
    print(f"Performance: {df_overall_stats['gflop/s'].mean():.2f} " f"+/- {df_overall_stats['gflop/s'].std():.2f} GFLOP/s")
    print(f"Time: {df_overall_stats['time'].mean():.5f} "f"+/- {df_overall_stats['time'].std():.5f} s")

    # kernel statistics
    df_kernel_stats = getKernelStats(df, df_mem)

    return df_overall_stats, df_kernel_stats,  df_overall_stats["time"].mean()

def saveKernelStats(kernel_stats, total_time, theo_bandwidth, theo_gflop, filename):

    df = kernel_stats.reset_index()

    df["Kernel Name"] = df["Kernel Name"]
    # percentages
    df["time_percentage"] = (df["time_mean"] / total_time * 100)
    df["bandwidth_percentage"] = (df["bandwidth_mean"] / theo_bandwidth * 100)
    df["gflop_percentage"] = (df["gflop_s_mean"] / theo_gflop * 100)
    # save
    df.to_csv(filename, index=False)


def main():
    fname = "/eos/user/t/tcostaes/traccc_outputs/profiling/1t_10ev_1rep/raw_for_analysis/mldev02_full_1t_10ev.csv"
    #path_mem = "/eos/user/t/tcostaes/traccc_outputs/profiling/1t_10ev_1rep/raw_for_analysis/h100_full_1t_10ev_mem.csv"
    path_mem = None
    overall_stats, kernel_stats, total_time = getStats(fname, path_mem)
    print("\nKernel statistics:")
    print(kernel_stats)

    #csv_filename = "/eos/user/t/tcostaes/traccc_outputs/profiling/1t_10ev_1rep/kernel_stats/h100_full_gflop.csv"
    #saveKernelStats(kernel_stats, total_time, theo_bandwidth=3940, theo_gflop=60320 ,filename=csv_filename)

if __name__ == "__main__":
    main()
