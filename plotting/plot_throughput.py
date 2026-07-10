import numpy as np
import ROOT as root
import atlasplots as aplt

def preprocess_data(path):
    data = np.genfromtxt(path, delimiter=',', names=True, dtype=None, encoding='utf-8')
    return data

def throughput_per_second(processed_events, processing_time_ns):
    return processed_events / (processing_time_ns * 1e-9)

def aggregate_by_threads(data):
    threads = data['threads'].astype(int)
    throughput = throughput_per_second(
        data['processed_events'].astype(float),
        data['processing_time'].astype(float),
    )

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
        "NVIDIA Tesla T4": "/eos/user/t/tcostaes/traccc_outputs/roi/lxplus.csv",
        "NVIDIA RTX 5000 Ada": "/eos/user/t/tcostaes/traccc_outputs/roi/v2_mldev02.csv",
        "NVIDIA RTX A5000": "/eos/user/t/tcostaes/traccc_outputs/roi/v2_mldev01.csv",
        "NVIDIA A100-PCIE-40GB": "/eos/user/t/tcostaes/traccc_outputs/roi/a100.csv",
        "NVIDIA H100L-2-24C MIG 2g.24gb": "/eos/user/t/tcostaes/traccc_outputs/roi/h100_24c.csv",
        "NVIDIA H100 NVL": "/eos/user/t/tcostaes/traccc_outputs/roi/h100_nvl.csv",
        "NVIDIA H100L-1-12C MIG 1g.12gb": "/eos/user/t/tcostaes/traccc_outputs/roi/h100_12c.csv"
        
    }

    c = root.TCanvas("c", "Throughput", 800, 600)
    c.SetGridx()
    c.SetGridy()
    c.SetLeftMargin(0.12)
    c.SetRightMargin(0.05)
    c.SetTopMargin(0.05)
    c.SetBottomMargin(0.15)

    colors = [
        root.kBlue,
        root.kRed,
        root.kGreen + 2,
        root.kMagenta + 1,
        root.kOrange + 7,
        root.kCyan + 2,
        root.kCyan - 7
    ]


    all_x = []
    all_y = []

    for path in gpu_files.values():
        data = preprocess_data(path)
        x, y, _ = aggregate_by_threads(data)
        mask = x <= 12
        x = x[mask]
        y = y[mask]
        all_x.extend(x)
        all_y.extend(y)

    ax_xmin = min(all_x) - 0.5
    ax_xmax = max(all_x) + 0.5
    ax_ymin = 0
    ax_ymax = max(all_y) * 1.1   # or 1.15 if you want a bit more headroom

    h_frame = c.DrawFrame(ax_xmin, ax_ymin, ax_xmax, ax_ymax)
    h_frame.GetXaxis().SetTitle("Number of threads")
    h_frame.GetYaxis().SetTitle("Throughput [events/s]")
    h_frame.GetYaxis().SetTitleOffset(1.6)

    leg = root.TLegend(0.15, 0.65, 0.40, 0.88)
    leg.SetTextSize(18)
    leg.SetBorderSize(0)

    graphs = []
    smooth_graphs = []
    splines = []
    for (gpu, path), color in zip(gpu_files.items(), colors):

        data = preprocess_data(path)
        x, y, yerr = aggregate_by_threads(data)
        x, y, _ = aggregate_by_threads(data)
        mask = x <= 12
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

        leg.AddEntry(graph, gpu, "EP")

        graphs.append(graph)
        smooth_graphs.append(graph_smooth)
        splines.append(spline)

    leg.Draw()

    c.SaveAs("/eos/user/t/tcostaes/traccc_outputs/plots/roi_gpu_comparison_12threads2.pdf")


if __name__ == '__main__':
    root.gROOT.SetBatch()
    main()
