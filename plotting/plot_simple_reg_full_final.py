import numpy as np
import ROOT as root
import atlasplots as aplt


def preprocess_data():
    reg_file_path = "/eos/user/t/tcostaes/traccc_outputs/roi/mldev02.csv"
    data_reg = np.genfromtxt(reg_file_path, delimiter=',', names=True, dtype=None, encoding='utf-8')

    full_file_path = "/eos/user/t/tcostaes/traccc_outputs/full/mldev02.csv"
    data_full = np.genfromtxt(full_file_path, delimiter=',', names=True, dtype=None, encoding='utf-8')
    return data_reg, data_full


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

    data_reg, data_full = preprocess_data()
    x_reg, y_reg, yerr_reg = aggregate_by_threads(data_reg)
    x_full, y_full, yerr_full = aggregate_by_threads(data_full)

    custom_blue = root.TColor.GetColor("#3f90da")
    custom_purple = root.TColor.GetColor("#94a4a2")

    # Filter to only include up to 13 threads
    mask_reg = x_reg <= 10
    x_reg = x_reg[mask_reg]
    y_reg = y_reg[mask_reg]
    yerr_reg = yerr_reg[mask_reg]

    mask_full = x_full <= 10
    x_full = x_full[mask_full]
    y_full = y_full[mask_full]
    yerr_full = yerr_full[mask_full]

    c = root.TCanvas("c", "Throughput", 800, 1000)

    # Main plot pad (65% of canvas)
    p1 = root.TPad("p1", "p1", 0, 0.30, 1, 1)
    p1.Draw()
    p1.cd()
    p1.SetGridx()
    p1.SetGridy()
    p1.SetBottomMargin(0.02)
    p1.SetLeftMargin(0.12)
    p1.SetRightMargin(0.05)
    p1.SetTopMargin(0.05)

    graph_reg = make_graph(x_reg, y_reg, yerr_reg, "graph_reg")
    graph_reg.SetMarkerStyle(21)
    graph_reg.SetMarkerColor(custom_blue)
    graph_reg.SetLineColor(custom_blue)
    graph_reg.SetMarkerSize(1.0)
    ax_xmin = min(x_reg) - 0.5
    ax_xmax = max(x_reg) + 0.5
    ax_ymin = 0
    ax_ymax = max(np.concatenate([y_reg, y_full])) * 1.2

    h_frame = p1.DrawFrame(ax_xmin, ax_ymin, ax_xmax, ax_ymax)
    h_frame.GetXaxis().SetTitle("")
    h_frame.GetXaxis().SetLabelSize(0)
    h_frame.GetYaxis().SetTitle("Throughput [events/s]")
    h_frame.GetYaxis().SetLabelSize(18)
    h_frame.Draw()

    graph_reg.Draw("P same")

    graph_full = make_graph(x_full, y_full, yerr_full, "graph_full")
    graph_full.SetMarkerStyle(22)
    graph_full.SetMarkerColor(custom_purple+1)
    graph_full.SetLineColor(custom_purple+1)
    graph_full.SetMarkerSize(1.0)
    graph_full.Draw("P same")

    spline_reg = root.TSpline3("spline_reg", graph_reg)
    x_smooth_reg = np.linspace(min(x_reg), max(x_reg), 200)
    y_smooth_reg = np.array([spline_reg.Eval(x) for x in x_smooth_reg])
    graph_smooth_reg = root.TGraph(len(x_smooth_reg), x_smooth_reg, y_smooth_reg)
    graph_smooth_reg.SetLineColor(custom_blue)
    graph_smooth_reg.SetLineWidth(2)
    graph_smooth_reg.Draw("L same")

    spline_full = root.TSpline3("spline_full", graph_full)
    x_smooth_full = np.linspace(min(x_full), max(x_full), 200)
    y_smooth_full = np.array([spline_full.Eval(x) for x in x_smooth_full])
    graph_smooth_full = root.TGraph(len(x_smooth_full), x_smooth_full, y_smooth_full)
    graph_smooth_full.SetLineColor(custom_purple+1)
    graph_smooth_full.SetLineWidth(2)
    graph_smooth_full.Draw("L same")

    p1.cd()
    tl = root.TLatex()
    tl.SetNDC()
    tl.SetTextSize(22)
    tl.DrawLatex(0.55, 0.21, "#bf{NVIDIA RTX 5000 Ada GPU}")
    tl.SetTextSize(20)
    tl.DrawLatex(0.55, 0.17, "#sqrt{s} = 14 TeV, <#mu> = 200, t#bar{t}")
    tl.DrawLatex(0.55, 0.13, "ITk Layout: 03-00-01")

    leg = root.TLegend(0.20, 0.80, 0.47, 0.88)
    leg.AddEntry(graph_reg, "Regional input", "EP")
    leg.AddEntry(graph_full, "Full input", "EP")
    leg.SetTextSize(20)
    leg.SetBorderSize(0)
    leg.Draw()

    # Ratio plot pad (30% of canvas)
    c.cd()
    p2 = root.TPad("p2", "p2", 0, 0, 1, 0.30)
    p2.Draw()
    p2.cd()
    p2.SetGridx()
    p2.SetGridy()
    p2.SetTopMargin(0.02)
    p2.SetBottomMargin(0.30)
    p2.SetLeftMargin(0.12)
    p2.SetRightMargin(0.05)

    y_ratio = y_reg / y_full
    yerr_ratio = y_ratio * np.sqrt((yerr_reg / y_reg)**2 + (yerr_full / y_full)**2)

    graph_ratio = make_graph(x_reg, y_ratio, yerr_ratio, "graph_ratio")
    graph_ratio.SetMarkerStyle(21)
    graph_ratio.SetMarkerColor(root.kBlack)
    graph_ratio.SetLineColor(root.kBlack)
    graph_ratio.SetMarkerSize(1.0)

    h_frame2 = p2.DrawFrame(ax_xmin, 0, ax_xmax, max(y_ratio) * 1.2)
    h_frame2.GetXaxis().SetTitle("Number of threads")
    h_frame2.GetYaxis().SetTitle("Regional / Fullscan")
    h_frame2.GetYaxis().SetLabelSize(15)
    h_frame2.Draw()

    graph_ratio.Draw("P same")

    p1.Update()
    p2.Update()

    c.SaveAs("throughput_reg_full_mldev02.pdf")


if __name__ == '__main__':
    root.gROOT.SetBatch()
    main()