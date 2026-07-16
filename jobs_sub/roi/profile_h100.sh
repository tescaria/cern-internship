#!/bin/bash
export ALRB_localConfigDir="/etc/hepix/sh/GROUP/zp/alrb"
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase

# Save this script's command-line arguments
args=("$@")

# Hide them from atlasLocalSetup.sh
set --

source $ATLAS_LOCAL_ROOT_BASE/user/atlasLocalSetup.sh

asetup Athena,25.0.58,cuda

# Restore the original arguments
set -- "${args[@]}"

METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__time_duration.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum, \
dram__bytes.sum.per_second, \
dram__bytes.sum"

ncu \
        --target-processes all \
        --metrics "$METRICS" \
        -f \
        -o h100_roi_1t_10ev \
        /afs/cern.ch/user/t/tcostaes/project/traccc/extras/traccc_itk_throughput_mt_profiler_roi.sh \
        -m 1 \
        -t 1 \
        -r 1 \
        -c profile_h100_roi_1t_10ev.csv
