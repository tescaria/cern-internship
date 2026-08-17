import os
import glob
import csv
import pandas as pd

def count_hits(filename):
    """Count hits (rows) in one event."""
    with open(filename, "r") as f:
        return sum(1 for _ in f) - 1   # subtract header


def event_hit_stats(filename):
    """Returns the total hits, unique hits, and duplicate hits for one event"""

    with open(filename, "r") as f:
        reader = csv.reader(f)
        # skip header
        next(reader)
        rows = list(reader)

    total_hits = len(rows)
    unique_hits = len(set(tuple(row) for row in rows))
    duplicate_hits = total_hits - unique_hits
    duplicate_fraction = duplicate_hits / total_hits if total_hits > 0 else 0

    return total_hits, unique_hits, duplicate_hits, duplicate_fraction


def duplicate_stats(dir=None, output_file=None):
    """Collect the duplicate statistics for each RoI jet event"""

    files = sorted(glob.glob(os.path.join(dir, "event?????????-cells.csv")))
    stats = []

    for f in files:
        total, unique, duplicates, fraction = event_hit_stats(f)

        event_number = int(os.path.basename(f).replace("event", "").replace("-cells.csv", ""))

        stats.append({
            "event": event_number,
            "total_hits": total,
            "unique_hits": unique,
            "duplicates": duplicates,
            "duplicate_fraction": fraction
        })
        
        #print(
        #    os.path.basename(f),
        #    f"total = {total}, "
        #    f"unique = {unique}, "
        #    f"duplicates = {duplicates} "
        #    f"({100*fraction:.2f}%)")

    # save stats
    df = pd.DataFrame(stats)
    if output_file:
        df.to_csv(output_file, index=False)
        
    return df

def compare_hits(superroi_dir=None, roi_dir=None, output_file=None, n_events=35):
    """Compare hits in events for muon RoIs and jet RoIs"""

    # use superroi directory with no duplicates
    # compare events 5-34 since muon RoI dir only has 35 events and first 5 overwritten
    superroi_files = sorted(glob.glob(os.path.join(superroi_dir, "event?????????-cells.csv")))[5:n_events]
    roi_files = sorted(glob.glob(os.path.join(roi_dir, "event?????????-cells.csv")))[5:n_events]

    comparisons = []
    for superroi, roi in zip(superroi_files, roi_files):

        superroi_hits = count_hits(superroi)
        roi_hits = count_hits(roi)

        event_number = int(os.path.basename(superroi).replace("event", "").replace("-cells.csv", ""))

        comparisons.append({
            "event": event_number,
            "superRoI_hits": superroi_hits,
            "RoI_hits": roi_hits,
            "ratio": superroi_hits / roi_hits if roi_hits > 0 else 0
        })

    df = pd.DataFrame(comparisons)
    if output_file:
        df.to_csv(output_file, index=False)

    return df


def main():
    superroi_dir = "/eos/user/t/tcostaes/traccc_outputs/roiInputJets"
    superroi_nodup_dir = "/eos/user/t/tcostaes/traccc_outputs/roiInputJets_nodup"
    roi_dir = "/eos/user/e/exochell/traccc/traccc_athena_plots/g200/traccc-athena/data/roiInputMuon"
    
    dup_stats = duplicate_stats(superroi_dir, "/eos/user/t/tcostaes/traccc_outputs/duplicate_stats.csv")
    comparisons = compare_hits(superroi_nodup_dir, roi_dir, "/eos/user/t/tcostaes/traccc_outputs/superroi_roi_hits.csv")
    
    print(dup_stats)
    print("\n")
    print(comparisons)



if __name__ == "__main__":
    main()