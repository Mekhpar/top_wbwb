
import os
import argparse
from itertools import product
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import numpy as np
import awkward as ak
import hist.intervals as intervals
import hist as hi
import mplhep as hep
from pepper import HistCollection, Config
from pepper.config import ConfigError
from tqdm import tqdm
import matplotlib
import time
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def remove_datasets(hist, datasets):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="List indexing selection is experimental.")
        remaining = [ds for ds in hist.axes["dataset"] if ds not in datasets]
        return hist[{"dataset": remaining}]


def check_plot_group_config(config):
    datasets = []
    group_labels = []
    for group in config["plot_dataset_groups"]:
        datasets.extend(group["datasets"])
        group_labels.append(group["label"])
    if len(set(group_labels)) != len(group_labels):
        raise ConfigError(
            "Groups with duplicate labels in plot_dataset_groups")
    if len(datasets) != len(set(datasets)):
        raise ConfigError(
            "Some data sets occur more than once in one or more "
            "plot_dataset_groups")


def check_hist_for_missing_datasets(name, hist, config):
    ds_in_hist = set(hist.axes["dataset"])
    for group in config["plot_dataset_groups"]:
        ds_in_group = set(group["datasets"])
        diff = ds_in_group - ds_in_hist
        if len(diff) in (0, len(ds_in_group)):
            continue
        warnings.warn(f"Histogram {name} is missing the following datasets "
                      f"from group '{group['label']}': " + ", ".join(diff))


def fill_empty_sys(hist):
    # checks for systs filled with zero vals, replaces these variations with nominal hist
    if "sys" not in [ax.name for ax in hist.axes]:
        return
    nom = hist[{"sys": "nominal"}]
    for dataset in hist.axes["dataset"]:
        for sysname in hist.axes["sys"]:
            if not np.all(
                hist[{"sys": sysname, "dataset": dataset}].values() == 0
            ):
                continue
            hist[{"sys": sysname, "dataset": dataset}] =\
                nom[{"dataset": dataset}].view()


def calculate_sys_uncertainty(hist, config):
    scales = {}
    if "plot_scale_sysuncertainty" in config:
        scales = config["plot_scale_sysuncertainty"]
    # Keep only MC datasets and sum them
    mc_ds = []
    for ds in hist.axes["dataset"]:
        if ds not in config["mc_datasets"]:
            continue
        mc_ds.append(ds)
    if len(mc_ds) >=1: hist = hist.integrate("dataset", mc_ds)
    sysax = hist.axes["sys"]
    nom = hist[{"sys": "nominal"}]

    # Group systematics by their group. Same group means fully correlated
    sysvalues = defaultdict(list)
    for sysname in sysax:
        sysgroup = sysname.rsplit("_", 1)[0]
        scale = 1.
        if scales != None: 
            if sysgroup in scales:
                scale = scales.get(sysgroup, 1)
        diff = (hist[{"sys": sysname}].values() - nom.values()) * scale
        sysvalues[sysgroup].append(diff)
    # axis0: sysgroup, axis1: direction (e.g. up and down), axis..: hist axes
    sysvalues = ak.Array(list(sysvalues.values()))
    # Get largest up and largest down variation of each group
    upper = np.maximum(np.asarray(ak.max(sysvalues, axis=1)), 0)
    lower = np.minimum(np.asarray(ak.min(sysvalues, axis=1)), 0)

    return lower, upper


def stacked_bars(x, counts, labels, colors, order, ax):
    # for key in order:
    #     print(key, counts[key].shape,counts[key][-1].shape)
    counts = [np.r_[counts[key], counts[key][-1]] for key in order]
    labels = [labels[key] for key in order]
    colors = [colors[key] for key in order]
    return ax.stackplot(x, counts, labels=labels, colors=colors, step="post")


def hatched_area(x, ycenter, up, down, label, ax):
    ycenter = np.r_[ycenter, ycenter[-1]]
    up = np.r_[up, up[-1]]
    down = np.r_[down, down[-1]]
    return ax.fill_between(
        x,
        ycenter + up,
        ycenter - down,
        step="post",
        facecolor="none",
        hatch="////",
        edgecolor="black",
        alpha=0.5,
        linewidth=0,
        label=label
    )


def dots_with_bars(x, y, yerr, label, ax):
    return ax.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        markersize=3,
        color="black",
        linestyle="none",
        label=label
    )


def plot_counts_mc(init_time, cats, cat_labels, hist, hist_og, x, config, sysunc, ax):
    if "plot_dataset_groups" in config:
        groups = config["plot_dataset_groups"]
    else:
        groups = {}
    mc_datasets = []
    mc_counts = defaultdict(int)
    mc_labels = {}
    mc_colors = {}
    onshell_counts = np.zeros(1)
    offshell_counts = np.zeros(1)
    for dataset in hist.axes["dataset"]:
        if dataset not in config["mc_datasets"]:
            continue
        mc_datasets.append(dataset)
        counts = hist[{"dataset": dataset}].values()

        for group in groups:
            if dataset in group["datasets"]:
                break
        else:
            group = None

        if len(cats) == 5:
            if cats[3] == '[sum]':
                #print("Dataset",dataset)
                if dataset.lower().startswith("wbj"):
                    print("Signal dataset found")
                    print("cats",cats)
                    print("cat_labels",cat_labels)
                    #Cat_labels probably could have been hard coded - same for everybody
                    #['channel', 'jet_btag', 'sign_cat', 'top_cat', 'btag_veto_cat']
                    split_cats = ["top","antitop","offshell_top","offshell_antitop"]

                    onshell_counts = np.zeros(len(counts))
                    offshell_counts = np.zeros(len(counts))

                    print("Length of counts for dataset (group?)",len(counts))
                    for top_cat in split_cats:

                        top_cats_split_list = list(top_cat if i == 3 else list(cats)[i] for i in range(len(list(cats))))
                        top_cats_split = tuple(sum if top_i == '[sum]' else top_i for top_i in top_cats_split_list)
                        #top_cats_split = tuple(s.split())

                        print("top_cat",top_cat)
                        print("top_cats_split_list",top_cats_split_list)
                        print("top_cats_split",top_cats_split)

                        hist_top = hist_og[dict(zip(cat_labels, top_cats_split))]
                        counts_top = hist_top[{"dataset": dataset}].values()
                        print("Length of counts_top",len(counts_top))
                        print("type of counts_top",type(counts_top))
                        print(counts_top)

                        #Adding the onshell together and the offshell together (not keeping top and antitop separate)
                        if top_cat.startswith("offshell_"):
                            offshell_counts = np.add(offshell_counts, counts_top)
                        else:
                            onshell_counts = np.add(onshell_counts, counts_top)

                        print("Current onshell counts",onshell_counts)
                        print("Current offshell counts",offshell_counts)

                    mc_counts["group:" + "onshell"] += onshell_counts
                    mc_labels["group:" + "onshell"] = "signal_wbj_onshell"
                    mc_colors["group:" + "onshell"] = "Blue"

                    mc_counts["group:" + "offshell"] += offshell_counts
                    mc_labels["group:" + "offshell"] = "signal_wbj_offshell"
                    mc_colors["group:" + "offshell"] = "Magenta"

                    print("Group label inside dataset",group["label"])

                    #if group is not None:
                    #    group["label"].append("onshell_signal_wbj")
                    #    group["label"].append("offshell_signal_wbj")

                #Background dataset        
                else:
                    if group is None:
                        mc_counts["dataset:" + dataset] += counts
                        mc_labels["dataset:" + dataset] = dataset
                        mc_colors["dataset:" + dataset] = None
                    else:
                        mc_counts["group:" + group["label"]] += counts
                        mc_labels["group:" + group["label"]] = group["label"]
                        mc_colors["group:" + group["label"]] = group["color"]
                
        # print(counts.shape)
    '''
    mc_counts["group:" + "onshell"] += onshell_counts
    mc_labels["group:" + "onshell"] = "signal_wbj_onshell"
    mc_colors["group:" + "onshell"] = "Blue"

    mc_counts["group:" + "offshell"] += offshell_counts
    mc_labels["group:" + "offshell"] = "signal_wbj_offshell"
    mc_colors["group:" + "offshell"] = "Magenta"
    '''

    print("mc_labels",mc_labels)
    print("mc_colors",mc_colors)
    print("labels in group",group["label"])
    #print("Type of group labels",type(groups["label"]))
    if len(mc_datasets) == 0:
        return []
    order = []
    for group in groups:
        if "group:" + group["label"] in mc_counts:
            order.append("group:" + group["label"])
    for key in mc_counts.keys():
        if key.startswith("dataset:"):
            order.append(key)

    print("Order",order)
    print("Number of categories",cats)
    print("Cats in plot_counts_mc",len(cats))
    if len(cats) == 5:
        print(cats[0:3])
        if cats[3] != '[sum]':
            print("No extra stacks")
            bars = stacked_bars(x.edges, mc_counts, mc_labels, mc_colors, order, ax)
        elif cats[3] == '[sum]':
            print("Extra stacks required for signal (wbj)")
            order = []

            for group in groups:
                if "group:" + group["label"] in mc_counts:
                    print("Group label",group["label"])
                    #if ~((group["label"].startswith("onshell")) | (group["label"].startswith("offshell"))):
                    order.append("group:" + group["label"])
                elif group["label"] == "Signal":
                    order.append("group:" + "onshell")
                    order.append("group:" + "offshell")

            for key in mc_counts.keys():
                if key.startswith("dataset:"):
                    order.append(key)
                    #Not sure whether anything else needs to be added to this

            print("Order with extra stacks",order)
            for key in mc_counts.keys():
                print("Key",key)
                print("Values?",mc_counts.values())

            bars = stacked_bars(x.edges, mc_counts, mc_labels, mc_colors, order, ax)
    #hahaha
    nom_sum = np.sum(list(mc_counts.values()), axis=0)
    stat_var = hist.integrate("dataset", mc_datasets).variances()
    if sysunc is None:
        # If plotting stat only, try to approximate 68% interval
        print("Stats unc only")
        #Dataset is just name of the process - eg wbj
        print("mc_datasets")
        print(mc_datasets)
        print("Bin edges in x?")
        print(x.edges)
        print("Mc counts")
        print(mc_counts)
        print("mc_labels")
        print(mc_labels)
        b = time.time()
        print("Time before starting the cat loop",init_time)
        print("Time after printing the spew",b)
        print("Time from the initial start of the cats loop",b-init_time)
        stat_unc = np.abs(
            intervals.poisson_interval(nom_sum, stat_var) - nom_sum) ** .5
        hatch = hatched_area(
            x.edges, nom_sum, stat_unc[1], stat_unc[0], "Stat. unc.", ax)
    else:
        stat_unc = stat_var ** .5
        # Sum all uncertainties in quadrature
        lo = np.concatenate([sysunc[0], [stat_unc]], axis=0)
        up = np.concatenate([sysunc[1], [stat_unc]], axis=0)

        lo = np.asarray(ak.sum(lo ** 2, axis=0))
        up = np.asarray(ak.sum(up ** 2, axis=0))
        hatch = hatched_area(x.edges, nom_sum, up ** .5, lo ** .5, "Unc.", ax)

    return [bars, hatch]


def plot_counts_data(hist, x, config, ax):
    if "sys" in [ax.name for ax in hist.axes]:
        hist = hist[{"sys": "nominal"}]

    has_data = False
    data_counts = 0
    data_variances = 0
    for dataset in hist.axes["dataset"]:
        if dataset not in config["exp_datasets"]:
            continue
        has_data = True
        hist_ds = hist[{"dataset": dataset}]
        data_counts += hist_ds.values()
        data_variances += hist_ds.variances()
    # print(has_data,data_counts)
    if not has_data:
        return []
    return dots_with_bars(
        x.centers, data_counts, data_variances ** .5, "Data", ax)


def plot_counts(init_time, cats, cat_labels, hist, hist_og, x, config, sysunc, logarithmic, no_data, ax):
    """Create the upper part of the ratio plot."""
    ret = []
    #print("Reached plot_counts function")
    #print(plot_counts_mc(hist, x, config, sysunc, ax))

    ret += plot_counts_mc(init_time, cats, cat_labels, hist, hist_og, x, config, sysunc, ax)
    if not no_data:
        ret += plot_counts_data(hist, x, config, ax)

    if logarithmic:
        ax.set_yscale("log")

    ax.set_ylabel(hist.label)

    if len(ret) > 4:
        legend_columns = 2
    else:
        legend_columns = 1
    ax.legend(ncols=legend_columns)

    return ret


def plot_ratio(hist, x, config, sysunc, ax):
    """Create the lower part of the ratio plot."""
    if "sys" in [ax.name for ax in hist.axes]:
        hist = hist[{"sys": "nominal"}]

    mc_datasets = [
        ds for ds in hist.axes["dataset"]
        if ds in config["mc_datasets"].keys()
    ]
    exp_datasets = [
        ds for ds in hist.axes["dataset"]
        if ds in config["exp_datasets"].keys()
    ]
    mc_hist = hist.integrate("dataset", mc_datasets)
    exp_hist = hist.integrate("dataset", exp_datasets)
    mc_vals = mc_hist.values().astype(float)
    mc_vars = mc_hist.variances().astype(float)
    exp_vals = exp_hist.values().astype(float)
    exp_vars = exp_hist.variances().astype(float)

    # Plot the dots indicating the ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        r = exp_vals / mc_vals
        # Error bars on the dots are data uncertainty only. Using error
        # propagation:
        rvar = exp_vars / mc_vals ** 2
    exp_unc = np.abs(intervals.poisson_interval(r, rvar) - r)
    dots_with_bars(x.centers, r, exp_unc, None, ax)

    # Plot MC uncertainty
    ones = np.ones_like(mc_vals)
    with np.errstate(divide="ignore", invalid="ignore"):
        # MC variance is scaled as if every MC bin is of height 1
        scaled_mc_vars = mc_vars / mc_vals ** 2
    if sysunc is None:
        lo, up = np.abs(intervals.poisson_interval(
            ones, scaled_mc_vars) - ones)
    else:
        stat_unc = scaled_mc_vars ** .5
        with np.errstate(divide="ignore", invalid="ignore"):
            sysunc = sysunc / mc_vals
        # Add sys uncertainties in quadrature to the uncertainty interval
        lo = np.sum(np.concatenate(
            [sysunc[0], [stat_unc]], axis=0) ** 2, axis=0) ** 0.5
        up = np.sum(np.concatenate(
            [sysunc[1], [stat_unc]], axis=0) ** 2, axis=0) ** 0.5
    hatched_area(x.edges, ones, up, lo, None, ax)

    ax.axhline(1, linestyle="--", color="black", linewidth=0.5)

    ax.set_ylim(0.5, 1.5)
    ax.set_ylabel("Data / Pred.")


def plot(init_time, cats, cat_labels, hist, hist_og, config, fpath, logarithmic, stat_only, no_data):
    for x in hist.axes:
        if isinstance(x, (hi.axis.Regular, hi.axis.Variable, hi.axis.Integer)):
            break
    else:
        raise ValueError("Could not find an axis to use as x axis")
    has_data = not no_data and any(
        ds in hist.axes["dataset"] for ds in config["exp_datasets"].keys())
    has_mc = any(
        ds in hist.axes["dataset"] for ds in config["mc_datasets"].keys())
    has_sys = "sys" in [ax.name for ax in hist.axes]
    sysunc = None
    if has_sys:
        if not stat_only:
            sysunc = calculate_sys_uncertainty(hist, config)
        nom = hist[{"sys": "nominal"}]
    else:
        nom = hist
    if has_data and has_mc:
        fig, (ax1, ax2) = plt.subplots(
            nrows=2, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    else:
        fig, ax1 = plt.subplots()
    plot_counts(init_time, cats, cat_labels, nom, hist_og, x, config, sysunc, logarithmic, no_data, ax1)
    if has_data and has_mc:
        plot_ratio(hist, x, config, sysunc, ax2)
        ax2.set_xlabel(x.label)
    else:
        ax1.set_xlabel(x.label)

    ax1.margins(x=0)
    ax1.ticklabel_format(axis="y", scilimits=(-4, 4), useMathText=True)

    yaxis = ax1.get_yaxis()
    if isinstance(
            yaxis.get_major_formatter(),
            matplotlib.ticker.ScalarFormatter
    ):
        # Workaround so that the CMS label is not placed on top of the
        # offset label (if present). Happens if offset has not been
        # computed prior creating the label. Thus compute and set it:
        yaxis.get_majorticklabels()  # Triggers an update of the offset
        yaxis.get_offset_text().set_text(
            yaxis.get_major_formatter().get_offset())

    label_opts = {"label": "Work in progress", "ax": ax1}
    if "luminosity" in config:
        label_opts["lumi"] = round(config["luminosity"], 1)
    if "sqrt_s" in config:
        label_opts["com"] = config["sqrt_s"]
    if has_data:
        label_opts["data"] = True
    hep.cms.label(**label_opts)
    fig.tight_layout()
    fig.savefig(fpath)
    plt.close(fig)


def process(
    histcol, histcol_ttbar, key, config, name, cut, cutidx, output, logarithmic, stat_only,
    no_data, fmt
):
    print("Histcol inside process",histcol)
    print("Key for histcol loading",key)
    hist_other = histcol.load(key)

    print("Histcol inside process for ttbar",histcol_ttbar)
    print("Key for histcol ttbar loading",key)
    hist_ttbar = histcol_ttbar.load(key)

    ds_other = set(hist_other.axes["dataset"])
    ds_ttbar = set(hist_ttbar.axes["dataset"])

    print("Datasets",ds_other)
    print("ds_ttbar",ds_ttbar)

    hist = hist_other + hist_ttbar     

    ds = set(hist.axes["dataset"])
    print("ds total",ds)

    if "plot_dataset_groups" in config:
        check_hist_for_missing_datasets(key, hist, config)
    if "plot_datasets_ignore" in config:
        hist = remove_datasets(hist, config["plot_datasets_ignore"])
    fill_empty_sys(hist)
    cat_axes = []
    dense_axes = []
    for ax in hist.axes:
        if isinstance(ax, hi.axis.StrCategory):
            if ax.name not in ("dataset", "sys"):
                cat_axes.append(ax)
        else:
            dense_axes.append(ax)
    cat_labels = [ax.name for ax in cat_axes]
    cats = [list(cat_ax) + [sum] for cat_ax in cat_axes]
    a = time.time()
    print("Initial starting time before entering the cat loop",a)
    num_cats = 0
    for cats in product(*cats):
        hist_catsplit = hist[dict(zip(cat_labels, cats))]
        #cats_og = cats
        print("Cats before replacing with [sum]",cats)
        cats = tuple("[sum]" if c is sum else c for c in cats)
        
        #if all(cats)
        print()
        print("Chronological number",num_cats)
        print("cats",cats)
        #print("Cats to adjust for sum",cats_og)
        print("Number of categories in cats",)
        print(ak.Array(cats))
        num_cats +=1
        #if num_cats > 10:
        #    break

        len_cats = len(cats)

        if (len_cats >= 3):
            print("Number of categories is more than 3")
            if (cats[0] == 'mu') & (cats[1] == 'j2_b1'):
                print("First category is mu")
                print("Second category is j2_b1 (current signal region)")

            #if (cats[0] == 'mu') & (cats[1] == 'j2+_b1'):
            #    print("First category is mu")
            #    print("Second category is j2+_b1")

                for this_ax in dense_axes:
                    print("dense_axes",dense_axes)
                    print("this_ax",this_ax)
                    dense_axes_to_sum = [ax for ax in dense_axes if ax is not this_ax]
                    #Maybe because these are all one 1d histograms, this is empty, so not going to pass this as an argument
                    print("dense_axes_to_sum",dense_axes_to_sum) 
                    # hist_1d only has one dense axis, sys and dataset cat axes
                    hist_1d = hist_catsplit[{ax.name: sum for ax in dense_axes_to_sum}]
                    directory = os.path.join(output, name, *cats)
                    os.makedirs(directory, exist_ok=True)
                    fname = "_".join(
                        (f"Cut_{cutidx:03}_" + cut, name)
                        + cats
                        + (this_ax.name,)
                    ) + "." + fmt
                    fpath = os.path.join(directory, fname)
                    
                    print("name",name)
                    try:
                        plot(a, cats, cat_labels, hist_1d, hist, config, fpath, logarithmic, stat_only, no_data)
                    except Exception as e:
                        print(e)
                        print("Current cats which give the exception",cats)
                        print("Current hist which gives the exception",name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Plot histograms in a ratioplot style")
    parser.add_argument("config", help="Pepper configuration JSON")
    parser.add_argument("histsfile")
    parser.add_argument(
        "-o", "--output", default="plots",
        help="Output directory. Default: plots"
    )
    parser.add_argument(
        "-f", "--format", choices=["svg", "pdf", "png"], default="svg",
        help="Format to save the plots in. Default: svg"
    )
    parser.add_argument(
        "-c", "--cut", help="Plot only histograms for this specific cut "
                            "(can be cut name or number; -1 for last cut)"
    )
    parser.add_argument(
        "--histname", help="Plot only histograms with this name"
    )
    parser.add_argument(
        "--log", action="store_true", help="Make y axis log scale"
    )
    parser.add_argument(
        "--stat-only", action="store_true",
        help="Plot only statistical uncertainties"
    )
    parser.add_argument(
        "--no-data", action="store_true", help="Do not plot experimental data"
    )
    parser.add_argument(
        "-p", "--processes", type=int, default=10,
        help="Number of concurrent processes to use. Default: 10"
    )
    args = parser.parse_args()

    #with open(args.histsfile) as f:
    #    histcol = HistCollection.from_json(f)

    with open("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0316_ttbar_semileptonic/hists/hists.json") as f_ttbar:
        histcol_ttbar = HistCollection.from_json(f_ttbar)

    with open("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0316_no_ttbar_semileptonic_20000/hists/hists.json") as f:
        histcol = HistCollection.from_json(f)

    all_cuts = histcol.userdata["cuts"]
    if args.cut or args.histname:
        cut = args.cut
        if cut:
            try:
                cutidx = int(cut)
                cut = all_cuts[cutidx]
            except ValueError:
                pass
        histcol = histcol[{
            "cut": [cut] if cut else None,
            "hist": [args.histname] if args.histname else None
        }]
    config = Config(args.config)
    if "plot_dataset_groups" in config:
        check_plot_group_config(config)
    with ProcessPoolExecutor(max_workers=args.processes) as executor:
        futures = []
        for key in histcol.keys():
            cutidx = all_cuts.index(key[0])
            futures.append(executor.submit(
                process, histcol, histcol_ttbar, key, config, key[1], key[0], cutidx,
                args.output, args.log, args.stat_only, args.no_data,
                args.format
            ))
        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()
    ### non-parallelized version for debugging: 
    # for key in histcol.keys():
    #     print(key)
    #     cutidx = all_cuts.index(key[0])
    #     process(histcol, key, config, key[1], key[0], cutidx,
    #         args.output, args.log, args.stat_only, args.no_data,
    #         args.format
    #     )


