import os
import glob
import numpy as np
import ROOT as root
import atlasplots as aplt

def count_hits(filename):
    """Count hits (rows) in one event."""
    with open(filename, "r") as f:
        return sum(1 for _ in f) - 1   # subtract header


def collect_hit_counts(directory, skip=0):
    """Collect total hit counts for each event"""

    files = sorted(glob.glob(os.path.join(directory, "event?????????-cells.csv")))[skip:]
    hit_counts = []
    for f in files:
        hit_counts.append(count_hits(f))

    return hit_counts

def make_histogram(data, bins, name):
    """Create a normalised ROOT histogram."""
    hist = root.TH1D(name, "", len(bins) - 1, bins)

    # Normalise to fraction of events
    weight = 1.0 / len(data)

    for value in data:
        hist.Fill(value, weight)

    return hist

def plot_hit_distributions(superroi_hits, roi_hits, full_hits=None):

    # ATLAS style 
    aplt.set_atlas_style()
    root.gStyle.SetOptStat(0)
    root.gStyle.SetGridColor(root.kGray)
    root.gStyle.SetGridStyle(3)
    root.gStyle.SetGridWidth(1)

    # binning 
    all_hits = superroi_hits + roi_hits
    if full_hits is not None:
        all_hits += full_hits
    bins = np.logspace(np.log10(min(all_hits)), np.log10(max(all_hits)), 25)
    # ROOT wants double precision arrays
    bins = np.asarray(bins, dtype=np.float64)

    # histograms 
    h_superroi = make_histogram(superroi_hits, bins, "h_superroi")
    h_roi = make_histogram(roi_hits, bins, "h_roi")
    if full_hits is not None:
        h_full = make_histogram(full_hits, bins, "h_full")

    # colours
    blue = root.TColor.GetColor("#1f77b4")
    orange = root.TColor.GetColor("#ff7f0e")
    green = root.TColor.GetColor("#2ca02c")

    h_superroi.SetLineColor(blue)
    h_superroi.SetFillColorAlpha(blue, 0.25)
    h_superroi.SetLineWidth(2)

    h_roi.SetLineColor(orange)
    h_roi.SetFillColorAlpha(orange, 0.25)
    h_roi.SetLineWidth(2)

    if full_hits is not None:
        h_full.SetLineColor(green)
        h_full.SetFillColorAlpha(green, 0.25)
        h_full.SetLineWidth(2)

    # canvas 
    c = root.TCanvas("c", "Hit distributions", 800, 600)
    c.SetLogx()
    c.SetLeftMargin(0.13)
    c.SetRightMargin(0.05)
    c.SetTopMargin(0.06)
    c.SetBottomMargin(0.15)

    # frame
    ymax = max(
        h_superroi.GetMaximum(),
        h_roi.GetMaximum(),
        h_full.GetMaximum() if full_hits is not None else 0
    )

    h_frame = c.DrawFrame(
        min(all_hits),
        0,
        max(all_hits),
        ymax * 1.25
    )

    h_frame.GetXaxis().SetTitle("Number of Hits")
    h_frame.GetYaxis().SetTitle("Fraction of Events")
    h_frame.GetYaxis().SetTitleOffset(1.7)

    # draw histograms
    h_superroi.Draw("HIST SAME")
    h_roi.Draw("HIST SAME")

    if full_hits is not None:
        h_full.Draw("HIST SAME")

    # legend
    leg = root.TLegend(0.58, 0.68, 0.88, 0.87)
    leg.SetTextSize(18)
    leg.SetBorderSize(0)

    leg.AddEntry(h_superroi, "Jet superRoIs", "L")
    leg.AddEntry(h_roi, "Muon RoIs", "L")

    if full_hits is not None:
        leg.AddEntry(h_full, "Full data", "L")

    leg.Draw()

    c.SaveAs("/eos/user/t/tcostaes/traccc_outputs/plots/roi_comparisons/hit_distributions_root.png")


def main():

    superroi_nodup_dir = ("/eos/user/t/tcostaes/traccc_outputs/roiInputJets_nodup")
    roi_dir = ("/eos/user/e/exochell/traccc/traccc_athena_plots/g200/traccc-athena/data/roiInputMuon")

    superroi_hits = collect_hit_counts(superroi_nodup_dir)
    roi_hits = collect_hit_counts(roi_dir, skip=5)

    # Full data if wanted:
    #full_dir = ("/eos/project/a/atlas-eftracking/GPU/ITk_data/traccc_standalone_data/ttbar_mu200")
    #full_hits = collect_hit_counts(full_dir)
    #plot_hit_distributions(superroi_hits, roi_hits, full_hits)

    plot_hit_distributions(superroi_hits, roi_hits)


if __name__ == "__main__":
    root.gROOT.SetBatch()
    main()
