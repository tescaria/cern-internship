import pandas as pd
import numpy as np

# process the data frame 
def processDF(df):

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

    bytes_transferred = df["dram__bytes.sum"].sum() # MB
    time_ms = df["gpu__time_duration.sum"].sum() # ms
    # MB/ms = GB/s
    bandwidth = bytes_transferred / time_ms

    return bandwidth

def getTime(df, event_id=None):
    if event_id is not None:
        df = df[df["event_id"] == event_id]

    return df["gpu__time_duration.sum"].sum() 

def getKernelStats(df, n=5):
    # df input already without cold runs and event_id adjusted

    time_col = "gpu__time_duration.sum"
    sm_col = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
    mem_col = "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"

    # time-weighted throughput per kernel per event
    def weighted_avg(group, metric):
        return ((group[metric] * group[time_col]).sum() / group[time_col].sum())

    per_event_metrics = (df.groupby(["event_id", "Kernel Name"]).apply(lambda x: pd.Series({
            "time_ms": x[time_col].sum(),
            "sm_throughput": weighted_avg(x, sm_col),
            "memory_throughput": weighted_avg(x, mem_col)}),  include_groups=False
        ).reset_index())

    # average over processed events
    kernel_stats = (per_event_metrics.groupby("Kernel Name").agg({
            "time_ms": ["mean", "std"],
            "sm_throughput": ["mean", "std"],
            "memory_throughput": ["mean", "std"]}))

    # flatten column names
    kernel_stats.columns = ["_".join(col) for col in kernel_stats.columns]

    # sort by average time and keep top 5
    top5 = (kernel_stats.sort_values("time_ms_mean", ascending=False).head(n))

    return top5

# get stats: overall and per kernel 
def getStats(fname):
    df = pd.read_csv(fname, low_memory=False)
    df = processDF(df)

    # Remove cold runs
    df = df[df["event_id"] >= 5].copy()
    df["event_id"] -= 5

    result=[]
    for i in range(df.event_id.max() + 1):
        tmp = df[df.event_id == i]
        flops = getFLOPS(tmp)
        time_ms = getTime(tmp)
        result.append([i, getBytes(tmp), flops, flops / (time_ms * 1e-3), time_ms])

    
    df_overall_stats = pd.DataFrame(result, columns=["event_id", "gb/s", "gflop", "gflop/s", "time_ms"])

    print("Overall event statistics:")
    print(f"Bandwidth: {df_overall_stats['gb/s'].mean():.2f} "f"+/- {df_overall_stats['gb/s'].std():.2f} GB/s")
    print(f"FLOP: {df_overall_stats['gflop'].mean():.2f} "f"+/- {df_overall_stats['gflop'].std():.2f} GFLOP")
    print(f"Performance: {df_overall_stats['gflop/s'].mean():.2f} " f"+/- {df_overall_stats['gflop/s'].std():.2f} GFLOP/s")
    print(f"Time: {df_overall_stats['time_ms'].mean():.2f} "f"+/- {df_overall_stats['time_ms'].std():.2f} ms")

    # kernel statistics
    df_kernel_stats = getKernelStats(df)

    return df_overall_stats, df_kernel_stats

def main():
    fname = "/eos/user/t/tcostaes/traccc_outputs/profiling/1t_10ev_1rep/mldev02_full_1t_10ev.csv"
    overall_stats, kernel_stats = getStats(fname)
    print("\nKernel statistics:")
    print(kernel_stats)


if __name__ == "__main__":
    main()
