import pandas as pd
import numpy as np

# process the data frame 
def processDF(df):

    # extract units row (as different runs may have different units)
    units = df.iloc[0]
    dram_unit = units.get("dram__bytes.sum", None)
    time_unit = units.get("gpu__time_duration.sum", None)
    dram_rate_unit = units.get("dram__bytes.sum.per_second", None)
    l1tex_unit = units.get("l1tex__t_bytes.sum", None)
    lts_unit = units.get("lts__t_bytes.sum", None)

    # remove empty/unit rows
    df = df[df["Kernel Name"].notna()].reset_index(drop=True)

    # assign event id based on ccl_kernel, separately for each CUDA stream
    df["is_event_start"] = df["Kernel Name"].str.contains("ccl_kernel")
    df["event_id"] = (df.groupby("Stream")["is_event_start"].cumsum() - 1)

    # remove unessessary columns (do not drop Stream)
    cols_to_drop = [
        "Process ID", "Process Name", "Host Name",
        "Context", "Device", "CC", "Section Name"]
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
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        "lts__t_bytes.sum",
        "l1tex__t_bytes.sum"]

    for col in metric_cols:
        if col in df.columns:
            df[col] = (df[col].astype(str).str.replace(",", ""))
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, dram_unit, time_unit, dram_rate_unit, l1tex_unit, lts_unit

def _sum_or_nan(df, col):
    """Return column sum or NaN if column is missing."""
    return df[col].sum() if col in df.columns else np.nan

def remove_cold_events(df, n_cold=5):
    """Removes cold run events"""
    ccl = df[df["is_event_start"]]
    cold_events = list(zip(ccl.iloc[:n_cold]["Stream"], ccl.iloc[:n_cold]["event_id"]))
    df = df[~df[["Stream", "event_id"]].apply(tuple, axis=1).isin(cold_events)].copy()

    return df


def convert_units(df, dram_unit, time_unit, dram_rate_unit, l1tex_unit, lts_unit):
    """
    Convert units to bytes and seconds.
    Use after processing data frame and before metric retrieving functions.
    """
    dram_scale = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9}
    time_scale = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1}
    dram_rate_scale = {"byte/s": 1, "Kbyte/s": 1e3, "Mbyte/s": 1e6, "Gbyte/s": 1e9,}
    l1tex_scale = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9}
    lts_scale = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9}

    if "dram__bytes.sum" in df.columns:
        df["dram__bytes.sum"] *= dram_scale[dram_unit]

    if "gpu__time_duration.sum" in df.columns:
        df["gpu__time_duration.sum"] *= time_scale[time_unit]

    if "dram__bytes.sum.per_second" in df.columns:
        df["dram__bytes.sum.per_second"] *= dram_rate_scale[dram_rate_unit]

    if "l1tex__t_bytes.sum" in df.columns:
        df["l1tex__t_bytes.sum"] *= l1tex_scale[l1tex_unit]

    if "lts__t_bytes.sum" in df.columns:
        df["lts__t_bytes.sum"] *= lts_scale[lts_unit]

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

def getBytesL1LTS(df, event_id=None):
    """Get amount of GB requested from L1 and L2"""
    if event_id is not None:
        df = df[df["event_id"] == event_id]

    l1tex_bytes = _sum_or_nan(df, "l1tex__t_bytes.sum") 
    lts_bytes = _sum_or_nan(df, "lts__t_bytes.sum") 

    return l1tex_bytes/1e9, lts_bytes/1e9

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

    # total event time for each processed event
    event_times = (df.groupby(["Stream", "event_id"])[time_col].sum().rename("total_event_time"))

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
    per_event_metrics = (df.groupby(["Stream", "event_id", "Kernel Name"]).apply(lambda x: pd.Series({
            "time": getTime(x),
            "sm_throughput": weighted_avg(x, sm_col),
            "memory_throughput": weighted_avg(x, mem_col),
            "bytes": (
                getBytes(df_mem[(df_mem["Stream"] == x.name[0]) &
                                (df_mem["event_id"] == x.name[1]) &
                                (df_mem["Kernel Name"] == x.name[2])])
                if df_mem is not None else getBytes(x)),
            "l1_bytes": (
                getBytesL1LTS(df_mem[
                    (df_mem["Stream"] == x.name[0]) &
                    (df_mem["event_id"] == x.name[1]) &
                    (df_mem["Kernel Name"] == x.name[2])])[0]
                if df_mem is not None else getBytesL1LTS(x)[0]),
            "lts_bytes": (
                getBytesL1LTS(df_mem[(df_mem["Stream"] == x.name[0]) &
                                    (df_mem["event_id"] == x.name[1]) &
                                    (df_mem["Kernel Name"] == x.name[2])])[1]
                if df_mem is not None else getBytesL1LTS(x)[1]),
            "time_bytes": (
                getTime(df_mem[(df_mem["Stream"] == x.name[0]) &
                                (df_mem["event_id"] == x.name[1]) &
                                (df_mem["Kernel Name"] == x.name[2])])
                if df_mem is not None else getTime(x)),
            "gbytes_s":  weighted_avg(x, "dram__bytes.sum.per_second") / 1e9,
            "manual_gbytes_s": (
                getBytes(df_mem[(df_mem["Stream"] == x.name[0]) &
                                (df_mem["event_id"] == x.name[1]) &
                                (df_mem["Kernel Name"] == x.name[2])]) /
                getTime(df_mem[(df_mem["Stream"] == x.name[0]) &
                                (df_mem["event_id"] == x.name[1]) &
                                (df_mem["Kernel Name"] == x.name[2])])
                if df_mem is not None else getBytes(x) / getTime(x)),
            "gflop": getFLOPS(x),
            "gflop_s": getFLOPS(x) / getTime(x)
            }),  include_groups=False).reset_index())

    # add total event time to per-event metrics
    per_event_metrics = per_event_metrics.merge(event_times, on=["Stream", "event_id"])

    # kernel contribution to total event time, calculated per event
    per_event_metrics["time_percentage"] = (per_event_metrics["time"] / per_event_metrics["total_event_time"] * 100)

    # average each kernel's per-event metrics across all processed events
    kernel_stats = (per_event_metrics.groupby("Kernel Name").agg({
            "time": ["mean", "std"],
            "time_percentage": ["mean", "std"],
            "sm_throughput": ["mean", "std"],
            "memory_throughput": ["mean", "std"],
            "bytes": ["mean", "std"],
            "l1_bytes": ["mean", "std"],
            "lts_bytes": ["mean", "std"],
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


def getKernelStatsPerCall(df, sorting_method="time_mean", df_mem=None, n=10):
    """
    Get average kernel metrics per kernel launch (call).
    """

    time_col = "gpu__time_duration.sum"
    sm_col = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
    mem_col = "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"

    per_call_metrics = (df.apply(lambda x: pd.Series({
        "Kernel Name": x["Kernel Name"],
        "time": getTime(x.to_frame().T),
        "sm_throughput": x[sm_col],
        "memory_throughput": x[mem_col],
        "bytes": (
            getBytes(df_mem.iloc[[x.name]]) 
            if df_mem is not None and x.name < len(df_mem)
            else np.nan if df_mem is not None 
            else getBytes(x.to_frame().T)
        ),
        "time_bytes": (
            getTime(df_mem.iloc[[x.name]]) 
            if df_mem is not None and x.name < len(df_mem)
            else np.nan if df_mem is not None 
            else getTime(x.to_frame().T)
        ),
        "gbytes_s": (
            x.get("dram__bytes.sum.per_second", np.nan) / 1e9
        ),
        "manual_gbytes_s": (
            getBytes(df_mem.iloc[[x.name]]) / getTime(df_mem.iloc[[x.name]])
            if df_mem is not None and x.name < len(df_mem)
            else np.nan if df_mem is not None
            else getBytes(x.to_frame().T) / getTime(x.to_frame().T)
        ),
        "gflop": getFLOPS(x.to_frame().T),
        "gflop_s": getFLOPS(x.to_frame().T) / getTime(x.to_frame().T)
    }), axis=1)
    )
    
    kernel_stats = (per_call_metrics.groupby("Kernel Name").agg({
            "time": ["mean", "std"],
            "sm_throughput": ["mean", "std"],
            "memory_throughput": ["mean", "std"],
            "bytes": ["mean", "std"],
            "time_bytes": ["mean", "std"],
            "gbytes_s": ["mean", "std"],
            "manual_gbytes_s": ["mean", "std"],
            "gflop": ["mean", "std"],
            "gflop_s": ["mean", "std"]}))

    # flatten column names
    kernel_stats.columns = ["_".join(col) for col in kernel_stats.columns]
    # sort by average time and keep top n
    top = (kernel_stats.sort_values(sorting_method, ascending=False).head(n))

    return top

def getStats(fname, sorting_method="time_mean", path_mem=None, kernel_function=getKernelStats):
    """
    Get overall statistics averaged per event.
    Get average statistics per kernel and per event by calling getKernelStats.
    """
    df = pd.read_csv(fname, low_memory=False)
    df, dram_unit, time_unit, dram_rate_unit, l1tex_unit, lts_unit = processDF(df)
    df = convert_units(df, dram_unit, time_unit, dram_rate_unit, l1tex_unit, lts_unit)

    # remove cold runs
    df = remove_cold_events(df, 5)
    
    df_mem = None
    if path_mem is not None:
        df_mem = pd.read_csv(path_mem, low_memory=False)
        df_mem, dram_unit_mem, time_unit_mem, dram_rate_unit_mem, l1tex_unit_mem, lts_unit_mem = processDF(df_mem)
        # convert units to bytes and seconds
        df_mem = convert_units(df_mem, dram_unit_mem, time_unit_mem, dram_rate_unit_mem, l1tex_unit_mem, lts_unit_mem)
        # remove cold runs
        df_mem = remove_cold_events(df_mem, 5)

    events = df[["Stream", "event_id"]].drop_duplicates()
    result=[]
    for stream, event_id in events.itertuples(index=False):
        tmp = df[(df["Stream"] == stream) & (df["event_id"] == event_id)]
        flops = getFLOPS(tmp)
        time = getTime(tmp)
        if path_mem is not None:
            tmp_mem = df_mem[(df_mem["Stream"] == stream) & (df_mem["event_id"] == event_id)]
            bytes_transferred = getBytes(tmp_mem)
            time_bytes = getTime(tmp_mem)
        else:
            bytes_transferred = getBytes(tmp)
            time_bytes = getTime(tmp)
        result.append([stream, event_id, bytes_transferred, time_bytes, flops, time])

    df_overall_stats = pd.DataFrame(result, columns=["Stream", "event_id", "gb", "time_bytes", "gflop", "time"])

    print("Overall event statistics:")
    print(f"GB transferred: {df_overall_stats['gb'].mean():.2f} "f"+/- {df_overall_stats['gb'].std():.2f} GB")
    print(f"Time (for bytes): {df_overall_stats['time_bytes'].mean():.5f} "f"+/- {df_overall_stats['time_bytes'].std():.5f} s")
    print(f"FLOP: {df_overall_stats['gflop'].mean():.2f} "f"+/- {df_overall_stats['gflop'].std():.2f} GFLOP")
    print(f"Time: {df_overall_stats['time'].mean():.5f} "f"+/- {df_overall_stats['time'].std():.5f} s")

    # kernel statistics
    df_kernel_stats = kernel_function(df, sorting_method, df_mem)
    
    return df_overall_stats, df_kernel_stats

def saveKernelStats(kernel_stats, filename):

    df = kernel_stats.reset_index()
    df.to_csv(filename, index=False)



def main():
    # setup parameters
    nthreads = 4
    nevent = 10
    gpu = "lxplus"
    data = "full"
    # paths
    fname = f"/eos/user/t/tcostaes/traccc_outputs/profiling/{nthreads}t_{nevent}ev_1rep/raw_for_analysis/{gpu}_{data}_{nthreads}t_{nevent}ev.csv"
    path_mem = f"/eos/user/t/tcostaes/traccc_outputs/profiling/{nthreads}t_{nevent}ev_1rep/raw_for_analysis/{gpu}_{data}_{nthreads}t_{nevent}ev_mem.csv"
    #path_mem = None 

    # OPTIONS: "time_mean", "sm_throughput_mean", "memory_throughput_mean", 
    # "bytes_mean", "time_bytes_mean", "gflop_mean", "gbytes_s_mean", "manual_gbytes_s_mean", "gflop_s_mean"
    sorting_method="time_mean"
    overall_stats, kernel_stats = getStats(fname, sorting_method, path_mem, kernel_function=getKernelStats)
    print("\nKernel statistics:")
    print(kernel_stats)
    
    csv_filename = f"/eos/user/t/tcostaes/traccc_outputs/profiling/{nthreads}t_{nevent}ev_1rep/kernel_stats/time_ordered/{gpu}_{data}.csv"
    saveKernelStats(kernel_stats, filename=csv_filename)

if __name__ == "__main__":
    main()
