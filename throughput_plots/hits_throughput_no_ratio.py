import csv
import glob
import os
import numpy as np
import ROOT as root
import atlasplots as aplt

def count_hits(filename):
    """Count hits (rows) in one event."""
    with open(filename, "r") as f:
        return sum(1 for _ in f) - 1   # subtract header


def dataset_hits(directory, n_events=35):
    """Get average hits/event from the first n_events."""
    
    files = sorted(glob.glob(os.path.join(directory, "event?????????-cells.csv")))

    # Use only the events actually loaded by traccc
    # temporary fix since first events were overwritten
    files = files[5:n_events]

    hits = []

    for f in files:
        n_hits = count_hits(f)
        hits.append(n_hits)
        print(os.path.basename(f), n_hits)

    avg_hits = sum(hits) / len(hits) # per event

    return avg_hits, hits


def preprocess_data(path):
    data = np.genfromtxt(path, delimiter=',', names=True, dtype=None, encoding='utf-8')
    return data

def hits_per_second(processed_events, processing_time_ns, avg_hits):
    total_hits = avg_hits * processed_events
    return total_hits / (processing_time_ns * 1e-9)

def aggregate_by_threads(data, avg_hits):
    threads = data['threads'].astype(int)
    throughput = hits_per_second(
        data['processed_events'].astype(float),
        data['processing_time'].astype(float),
        avg_hits)

    unique_threads = np.unique(threads)
    mean_throughput = np.array([np.mean(throughput[threads == t]) for t in unique_threads])
    yerr = np.array([np.std(throughput[threads == t], ddof=1) if np.sum(threads == t) > 1 else 0.0 for t in unique_threads])
    min_visible_yerr = 0.02 * mean_throughput
    yerr = np.maximum(yerr, min_visible_yerr)

    return unique_threads, mean_throughput, yerr

def make_graph(x, y, yerr, name):
    x_vals = np.asarray(x, dtype=np.float64)
    y_vals = np.asarray(y, dtype=np.float64)
    xerr_vals = np.zeros_like(x_vals)
    yerr_vals = np.asarray(yerr, dtype=np.float64)
    graph = root.TGraphErrors(len(x_vals), x_vals, y_vals, xerr_vals, yerr_vals)
    graph.SetName(name)
    return graph

def main():

    aplt.set_atlas_style()
    root.gStyle.SetEndErrorSize(0)
    root.gStyle.SetGridColor(root.kGray)
    root.gStyle.SetGridStyle(3)
    root.gStyle.SetGridWidth(1)

    # GPUs and their csv files
    gpu_files = {
        "NVIDIA Tesla T4": "/eos/user/t/tcostaes/traccc_outputs/full/lxplus.csv",
        "NVIDIA RTX 5000 Ada": "/eos/user/t/tcostaes/traccc_outputs/full/v2_mldev02.csv",
        #"NVIDIA RTX A5000": "/eos/user/t/tcostaes/traccc_outputs/roi/v2_mldev01.csv",
        #"NVIDIA A100-PCIE-40GB": "/eos/user/t/tcostaes/traccc_outputs/roi/a100.csv",
        #"NVIDIA H100L-2-24C MIG 2g.24gb": "/eos/user/t/tcostaes/traccc_outputs/roi/h100_24c.csv",
        "NVIDIA H100 NVL": "/eos/user/t/tcostaes/traccc_outputs/full/h100_nvl.csv",
        #"NVIDIA H100L-1-12C MIG 1g.12gb": "/eos/user/t/tcostaes/traccc_outputs/roi/h100_12c.csv"
        
    }

    c = root.TCanvas("c", "Throughput", 800, 600)
    c.SetGridx()
    c.SetGridy()
    c.SetLeftMargin(0.13)
    c.SetRightMargin(0.05)
    c.SetTopMargin(0.06)
    c.SetBottomMargin(0.15)

    #colors = [root.kBlue, root.kRed, root.kGreen + 2, root.kMagenta + 1, root.kOrange + 7, root.kCyan + 2, root.kCyan - 7]
    # custom colours 
    blue = root.TColor.GetColor("#1f77b4")    # tab:blue
    orange = root.TColor.GetColor("#ff7f0e")  # tab:orange
    green = root.TColor.GetColor("#2ca02c")   # tab:green
    colors = [blue, orange, green]

    all_x = []
    all_y = []

    #data_dir = "/eos/user/e/exochell/traccc/traccc_athena_plots/g200/traccc-athena/data/roiInputMuon"
    data_dir = "/eos/project/a/atlas-eftracking/GPU/ITk_data/traccc_standalone_data/ttbar_mu200"

    for path in gpu_files.values():
        avg_hits, hits = dataset_hits(data_dir)
        data = preprocess_data(path)
        x, y, _ = aggregate_by_threads(data, avg_hits)
        mask = x <= 25
        x = x[mask]
        y = y[mask]
        all_x.extend(x)
        all_y.extend(y)

    ax_xmin = min(all_x) - 0.5
    ax_xmax = max(all_x) + 0.5
    ax_ymin = 0
    ax_ymax = max(all_y) * 1.1

    h_frame = c.DrawFrame(ax_xmin, ax_ymin, ax_xmax, ax_ymax)
    h_frame.GetXaxis().SetTitle("Number of threads")
    h_frame.GetYaxis().SetTitle("Throughput [hits/s]")
    h_frame.GetYaxis().SetTitleOffset(1.8)

    # legend for full case
    leg = root.TLegend(0.57, 0.35, 0.87, 0.48)
    # legend for regional case
    #leg = root.TLegend(0.15, 0.75, 0.45, 0.88)
    leg.SetTextSize(18) 
    leg.SetBorderSize(0)


    graphs = []
    smooth_graphs = []
    splines = []
    oom_graphs = []
    for (gpu, path), color in zip(gpu_files.items(), colors):
        avg_hits, hits = dataset_hits(data_dir)
        data = preprocess_data(path)
        x, y, yerr = aggregate_by_threads(data, avg_hits)
        mask = x <= 25
        x = x[mask]
        y = y[mask]
        yerr = yerr[mask]

        graph = make_graph(x, y, yerr, f"graph_{gpu}")
        graph.SetMarkerStyle(21)
        graph.SetMarkerColor(color)
        graph.SetLineColor(color)
        graph.SetMarkerSize(1.0)
        graph.Draw("P same")

        spline = root.TSpline3(f"spline_{gpu}", graph)
        x_smooth = np.linspace(min(x), max(x), 200)
        y_smooth = np.array([spline.Eval(xx) for xx in x_smooth])

        graph_smooth = root.TGraph(len(x_smooth), x_smooth, y_smooth)
        graph_smooth.SetLineColor(color)
        graph_smooth.SetLineWidth(2)
        graph_smooth.Draw("L same")

        # Mark out-of-memory point (last point for Tesla and Ada)
        if 1==1:
            if gpu in ["NVIDIA Tesla T4", "NVIDIA RTX 5000 Ada"]:
                oom_graph = root.TGraph(1)
                oom_graph.SetPoint(0, x[-1], y[-1])
                oom_graph.SetMarkerStyle(20)   # filled circle
                oom_graph.SetMarkerColor(root.kRed + 2)
                oom_graph.SetMarkerSize(1.4)
                oom_graph.Draw("P same")
                oom_graphs.append(oom_graph)

        leg.AddEntry(graph, gpu, "EP")

        graphs.append(graph)
        smooth_graphs.append(graph_smooth)
        splines.append(spline)

    leg.Draw()

    # ATLAS-style annotation
    tl = root.TLatex()
    tl.SetNDC()
    tl.SetTextSize(22)
    tl.DrawLatex(0.60, 0.29, "#bf{Full Detector Data}")
    tl.SetTextSize(20)
    tl.DrawLatex(0.60, 0.25, "#sqrt{s} = 14 TeV, <#mu> = 200, t#bar{t}")
    tl.DrawLatex(0.60, 0.21, "ITk Layout: 03-00-01")

    c.SaveAs("/eos/user/t/tcostaes/traccc_outputs/plots/throughput/hits/full_3gpu_allThreads.pdf")


if __name__ == '__main__':
    root.gROOT.SetBatch()
    main()
