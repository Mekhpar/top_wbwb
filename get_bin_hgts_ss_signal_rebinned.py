
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
#import zfit
import ROOT

def poly_exp(x, p):
    m = p[0] # shape
    k = p[1]
    '''
    a3 = p[2] # location of minimum
    a2 = p[3] # scale
    a1 = p[4] # baseline
    '''
    a2 = p[2]
    a1 = p[3]

    x = x[0]
    #y = (a3*(x-m)**3 + a2*(x-m)**2 + a1*(x-m))*np.exp(-k*(x))
    y = (a2*(x-m)**2 + a1*(x-m))*np.exp(-k*(x))
    #y = (a1*(x-m))*np.exp(-k*(x))
    return y

def voigt(x, p):
    norm = p[0]
    mean = p[1]
    sigma = p[2]
    width_relation = p[3]
    x = x[0]

    y = norm*ROOT.TMath.Voigt(x-mean, sigma, width_relation)
    return y

def landau(x, p):
    norm = p[0]
    mean = p[1]
    sigma = p[2]
    x = x[0]

    y = norm*ROOT.TMath.Landau(x, mean, sigma)
    return y

#'''
def gauss(x, p):
    norm = p[0]
    mean = p[1]
    sigma = p[2]
    x = x[0]

    y = norm*ROOT.TMath.Gaus(x, mean, sigma)
    return y
#'''

'''
def gauss(x, p):
    norm_gauss_numerator = p[0]
    mean = p[1]
    sigma = p[2]
    alpha_non_crystalball = p[3]
    x = x[0]
    arg_alpha = (alpha_non_crystalball-mean)/sigma
    arg_x = (x-mean)/sigma
    #y = norm*ROOT.TMath.Gaus(x, mean, sigma)
    y = (norm_gauss_numerator/np.exp(-0.5*arg_alpha**2))*np.exp(-0.5*arg_x**2)
    return y
'''

def power_law(x,p):
    norm = p[0]
    mean = p[1]
    power = p[2]
    x = x[0]

    y = norm*(x - mean)**(-power)
    return y

def wald(x,p):
    norm = p[0]
    l = p[1]
    mean = p[2]
    shift = p[3]
    x = x[0]

    arg = x - shift
    y = norm*np.sqrt(l/(2*np.pi*arg**3))*ROOT.TMath.Exp(-(l*(arg-mean)**2)/(2*(mean**2)*arg))
    return y

def exponential(x,p):
    norm = p[0]
    decay_constant = p[1]
    shift = p[2]
    #power = p[3]
    #m = p[2]
    x = x[0]

    #y = norm*ROOT.TMath.Exp(-decay_constant*x) + shift
    #y = norm*ROOT.TMath.Exp(-decay_constant*(x**power)) + shift #Honestly the behavior here is probably not much different from a power law
    #y = norm*ROOT.TMath.Exp(-decay_constant*(x**power))

    #What Alberto suggested (decreasing part of parabola i.e. inverted Gaussian)
    #y = norm*ROOT.TMath.Exp(decay_constant*(x-m)**2) #Here decay constant needs to be positive
    y = norm*ROOT.TMath.Exp(-decay_constant*(x-shift))

    return y

def gauss_rev_gauss(x,p):
    norm = p[0]
    a = p[1]
    m = p[2]
    s = p[3]
    n = p[4]
    x = x[0]

    arg_normal = (x-m)/s
    arg_rev = (x-n)/s
    if x <= a:
        gauss_rev_gauss = ROOT.TMath.Exp(-0.5*arg_normal**2)
        
    elif x>a:
        gauss_rev_gauss = ROOT.TMath.Exp(-0.5*((a-m)*(n-m))/(s**2))*ROOT.TMath.Exp(-0.5*arg_rev**2*(a-m)/(a-n))
    
    return norm*gauss_rev_gauss


def crystalball(x,p):
    norm = p[0]
    mean = p[1]
    sigma = p[2]
    #mean_power_law = p[2]
    alpha = p[3]
    #alpha_x = p[3]
    n = p[4]

    x = x[0]

    #Sign reversed since we want positive tail
    arg = (x-mean)/sigma
    abs_alpha = abs(alpha)
    if arg <= -alpha:
        crystal_ball = np.exp(- 0.5 * arg * arg)
    elif arg > -alpha:
        nDivAlpha = n/abs_alpha
        AA =  np.exp(-0.5 * abs_alpha * abs_alpha)
        B = nDivAlpha - abs_alpha
        #arg_exp = nDivAlpha/(B-arg)
        arg_exp = nDivAlpha/(B+arg)
        crystal_ball = AA * (arg_exp**n)

    #'''
    return norm*crystal_ball
    
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
    print("Bin content in stacked_bars function")
    print(counts)
    print(type(counts))
    print(np.sum(counts,axis=0))

    x_lower_edges = x[:-1]

    #Crude rebinning - choosing 10 GeV (or maybe 20) since it is an exact multiple of 2 GeV (which is the current binning)
    bin_multiple = 10
    #current_binsize = 2
    current_binsize = (x[10] - x[0])/10 #Assuming that the bin sizes are all equal
    #last_upper_bin_edge = 1000
    last_upper_bin_edge = x[-1]
    print("Current binsize in stacked_bars",current_binsize)
    print("last_upper_bin_edge in stacked_bars", last_upper_bin_edge)

    new_x_edges = x_lower_edges[np.subtract(x_lower_edges,x_lower_edges[0])%(current_binsize*bin_multiple) == 0]
    print("Bins to be considered in the rebinning case",new_x_edges)
    
    new_x_edges = np.append(new_x_edges,last_upper_bin_edge)
    counts_rebinned = []
    for k in range(len(order)):
        current_key = order[k]
        print("Current key",current_key)
        print("Type of each element in counts",type(counts[k]))
        new_bincontent_array = [] #Defined initially as a list, and going to convert to a numpy array

        for i in range(0,len(x_lower_edges)):
            print("Number of bins in original histogram",i)
            if i%bin_multiple == 0:
                bin_content_current = 0
                for j in range(i,i+bin_multiple):
                    print("bin number for which content is to be added", j)
                    print("Current bin content",counts[k][j])
                    bin_content_current += counts[k][j]
                print("Total bin content",bin_content_current)
                new_bincontent_array.append(bin_content_current)

        #This is just to make the number of entries the same as the x edges - this includes the last upper bin limit as well
        new_bincontent_array.append(new_bincontent_array[-1])
        array_rebinned = np.array(new_bincontent_array)
        counts_rebinned.append(array_rebinned)
        #hahaha

    #Should be ideal to check for the condition of new_x_edges and new_bincontent_array

    #The stackplot is already called so there is no need (probably) to populate the pyroot histogram
    '''
    #net_mass = ROOT.TH1D("net_mass","net_mass",len(new_x_edges),new_x_edges)
    net_mass = ROOT.TH1D("net_mass","net_mass",len(new_x_edges)-1,new_x_edges) #Do not know why this is causing a problem because there was no problem for the original one

    if len(new_x_edges)-1 == len(new_bincontent_array):
        for i in range(0,len(new_x_edges)-1):
            net_mass.SetBinContent(i+1,new_bincontent_array[i])
        print(net_mass)        

    else:
        print("Error in rebinning - length of x and bin heights not the same")

    print("length_net_mass",net_mass.GetNbinsX())

    for i in range(0,len(new_x_edges)-1):
        print("Number of bins in rebinned histogram",i)
        print("Lower edge of bins",net_mass.GetXaxis().GetBinLowEdge(i+1))
        print("Bin content in rebinned histogram",net_mass.GetBinContent(i+1))
    '''

    print("length_new_x_edges",len(new_x_edges))

    #'''
    #return np.sum(counts,axis=0), ax.stackplot(x, counts, labels=labels, colors=colors, step="post")
    return new_x_edges, counts_rebinned, np.sum(counts_rebinned,axis=0), ax.stackplot(new_x_edges, counts_rebinned, labels=labels, colors=colors, step="post")

    #return np.sum(counts,axis=0), ax.stackplot(x, counts, labels=labels, colors=colors, step="post")



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


def plot_counts_mc(init_time, cats, cat_labels, hist, hist_og, x, config, sysunc, ax, plot_signal):
    print("Reached plot_counts_mc function")
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

        #===================================Added Apr 12 for non stack loop======================================================================
        '''
        if group is None:
            mc_counts["dataset:" + dataset] += counts
            mc_labels["dataset:" + dataset] = dataset
            mc_colors["dataset:" + dataset] = None
        else:
            mc_counts["group:" + group["label"]] += counts
            mc_labels["group:" + group["label"]] = group["label"]
            mc_colors["group:" + group["label"]] = group["color"]
        '''
        #========================================================================================================================================

        if len(cats) == 6:
            if plot_signal == 1:
                if dataset.lower().startswith("wbj"):
                    print("Signal dataset found")
                    print("cats",cats)
                    print("cat_labels",cat_labels)

                    if group is None:
                        mc_counts["dataset:" + dataset] += counts
                        mc_labels["dataset:" + dataset] = dataset
                        mc_colors["dataset:" + dataset] = None
                    else:
                        mc_counts["group:" + group["label"]] += counts
                        mc_labels["group:" + group["label"]] = group["label"]
                        mc_colors["group:" + group["label"]] = group["color"]

            elif plot_signal == 0:
                if dataset.lower().startswith("wbj") == False:
                    print("Bkg dataset found")
                    print("cats",cats)
                    print("cat_labels",cat_labels)

                    if group is None:
                        mc_counts["dataset:" + dataset] += counts
                        mc_labels["dataset:" + dataset] = dataset
                        mc_colors["dataset:" + dataset] = None
                    else:
                        mc_counts["group:" + group["label"]] += counts
                        mc_labels["group:" + group["label"]] = group["label"]
                        mc_colors["group:" + group["label"]] = group["color"]
                

            '''
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
            '''
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
    #This has to be brought in again because we do not want stacks
    '''
    for group in groups:
        print("Group in non stack loop",group["label"])
        if "group:" + group["label"] in mc_counts:
            order.append("group:" + group["label"])
    for key in mc_counts.keys():
        print("Key in non stack loop",key)
        if key.startswith("dataset:"):
            order.append(key)
    '''
    print("Order",order)
    print("Number of categories",cats)
    print("Cats in plot_counts_mc",len(cats))
    #if len(cats) == 5:
    if len(cats) == 6:
        print(cats[0:3])
        #bars = stacked_bars(x.edges, mc_counts, mc_labels, mc_colors, order, ax)

        if plot_signal == 1:
            for group in groups:
                #if "group:" + group["label"] in mc_counts:
                #    print("Group label",group["label"])
                if ("group:" + group["label"] in mc_counts) & (group["label"] == "Signal"):
                    print("Group label",group["label"])
                    order.append("group:" + group["label"])
                    #order.append("group:" + "offshell")

            for key in mc_counts.keys():
                if key.startswith("dataset:"):
                    print("Key in plot_signal",key)
                    order.append(key)
                #Not sure whether anything else needs to be added to this

            print("Order with signal plotting only",order)

        elif plot_signal == 0:
            for group in groups:
                #if "group:" + group["label"] in mc_counts:
                #    print("Group label",group["label"])
                if ("group:" + group["label"] in mc_counts) & (group["label"] != "Signal"):
                    print("Group label",group["label"])
                    order.append("group:" + group["label"])
                    #order.append("group:" + "offshell")

            for key in mc_counts.keys():
                if key.startswith("dataset:"):
                    print("Key in plot_signal",key)
                    order.append(key)
                #Not sure whether anything else needs to be added to this

            print("Order with bkg plotting only",order)

        for key in mc_counts.keys():
            print("Key",key)
            print("Values?",mc_counts.values())

        #bars = stacked_bars(x.edges, mc_counts, mc_labels, mc_colors, order, ax)
        #bar_content, bars = stacked_bars(x.edges, mc_counts, mc_labels, mc_colors, order, ax)

        #Returning rebinned counts as well
        new_x_edges, mc_counts_rebinned, bar_content, bars = stacked_bars(x.edges, mc_counts, mc_labels, mc_colors, order, ax)

        print("Current category after stacked_bars",cats)
        print("x bar_content")
        print(x.edges)
        print("y bar_content")
        print(bar_content)
        print(type(bar_content))
        #hahaha

        print("Original x length",len(x.edges))
        print("Length without last upper edge",len(x.edges[:-1]))
        print("Original y length",len(bar_content))

        print("Type of edges",type(x.edges))

        net_mass = ROOT.TH1D("net_mass","net_mass",len(new_x_edges[:-1]),new_x_edges[:-1])
        for i in range(0,len(new_x_edges[:-1])):
            print("Number of bins just before drawing and fitting (rebinned)",i)
            net_mass.SetBinContent(i+1,bar_content[i])
            print("Bin content just before drawing and fitting (rebinned)",net_mass.GetBinContent(i+1))
        print(net_mass)

        c1 = ROOT.TCanvas('c1', 'c1', 700, 500)
        print("Current set of categories just before drawing",cats)
        net_mass.Draw()
        print("Drew histogram")  

        alpha_non_crystalball = 210
        #alpha_non_crystalball = 192.5 #This is the designated upper boundary of the mass window at the moment
        #alpha_non_crystalball = 200 #This is the designated upper boundary of the mass window at the moment
        #alpha_non_crystalball = 210

        '''
        fit_power_law = ROOT.TF1("fit_power_law",power_law,alpha_non_crystalball,600,3)
        #fit_power_law.SetParNames("norm","mean","sigma")
        fit_power_law.SetParNames("norm","mean","power")
        #fit_power_law.SetParameters(10.0, 172.5, 20.0)
        #fit_power_law.SetParameters(10.0, 205, 20.0)
        #fit_power_law.SetParameters(10.0, alpha_non_crystalball, 10.0)
        #fit_power_law.SetParameters(10.0, 0, 10.0)

        fit_power_law.SetParameters(10.0, 209, 10.0)

        fit_power_law.SetParLimits(2,0,10000)
        fit_power_law.SetParLimits(1,-10000,alpha_non_crystalball)

        #So that it does not become exactly 0, surprisingly this error was not obtained in the previous case (maybe due to change in bin size)
        #fit_power_law.SetParLimits(1,-10000,alpha_non_crystalball-1) 
        #fit_power_law.SetParLimits(1,-10000,0) #Only for debugging purposes
        #fit_power_law.SetParLimits(1,204.95,205.05)
        #fit_power_law.FixParameter(1,205)
        '''

        '''
        fit_voigt = ROOT.TF1("fit_voigt",voigt,100,210,4)
        fit_voigt.SetParNames("norm","mean","sigma","width")
        fit_voigt.SetParameters(10.0, 172.5, 20.0, 1.3)
        '''

        '''
        print("Current set of categories just before pyroot fitting",cats)
        net_mass.Fit("fit_power_law", "R")

        
        p_power_law = fit_power_law.GetParameters()
        print("fitting result for power law: {parameter1}, {parameter2}, {parameter3}".format(
        parameter1 = p_power_law[0],
        parameter2 = p_power_law[1],
        parameter3 = p_power_law[2]
        ))
        '''

        #=============================================================Exponential=================================================================
        
        #'''
        #fit_exponential = ROOT.TF1("fit_exponential",exponential,alpha_non_crystalball,600,3)
        #fit_exponential = ROOT.TF1("fit_exponential",exponential,alpha_non_crystalball,1000,3)
        #fit_exponential = ROOT.TF1("fit_exponential",exponential,alpha_non_crystalball,1000,4)

        #fit_exponential = ROOT.TF1("fit_exponential",exponential,400,1000,4) 
        #fit_exponential = ROOT.TF1("fit_exponential",exponential,210,1000,3) 

        fit_exponential = ROOT.TF1("fit_exponential",exponential,325,1000,3) #Exponential that goes with wald
        fit_exponential.SetParNames("norm","decay_constant","shift") #This shift is in x, not the whole function
        #fit_exponential.SetParNames("norm","decay_constant","shift","power")

        #fit_exponential.SetParameters(10.0, 20, 1.0, 0.5)
        fit_exponential.SetParameters(20.0, 0.0085, 325.0)

        fit_exponential.SetParLimits(0,10,40)
        fit_exponential.SetParLimits(1,0.005,0.01) #Decay constant definitely has to be positive
        #fit_exponential.SetParLimits(2,1000,100000) #Shift has to be greater than equal to upper limit to capture the decreasing part
        fit_exponential.FixParameter(2,325) #Shift has to be greater than equal to upper limit to capture the decreasing part
        
        #fit_exponential.SetParLimits(3,0,100000) #Power definitely has to be positive

        '''
        net_mass.Fit("fit_exponential", "R")

        p_exponential = fit_exponential.GetParameters()
        print("fitting result for power law: {parameter1}, {parameter2}, {parameter3}".format(
        parameter1 = p_exponential[0],
        parameter2 = p_exponential[1],
        parameter3 = p_exponential[2],
        #parameter4 = p_exponential[3]
        ))
        '''
        #'''

        #=========================================================================================================================================

        #norm_gauss_numerator = (p_power_law[0]*(alpha_non_crystalball - p_power_law[1])**(-p_power_law[2]))

        #mean_gauss = alpha_non_crystalball - (p_power_law[2]/alpha_non_crystalball - p_power_law[1])
        #print("Mean gauss Parameter fixed from power law fit",mean_gauss)

        #=============================================================Gaussian=================================================================

        '''     
        fit_gauss = ROOT.TF1("fit_gauss",gauss,100,alpha_non_crystalball,3)
        #fit_gauss = ROOT.TF1("fit_gauss",gauss,100,alpha_non_crystalball,4)
        fit_gauss.SetParNames("norm","mean","sigma")
        #fit_gauss.SetParNames("norm_gauss_numerator","mean","sigma","alpha_non_crystalball")
        
        fit_gauss.SetParameters(10.0, 172.5, 20.0)
        fit_gauss.SetParLimits(2,0,100000) #Sigma definitely has to be positive
        #fit_gauss.SetParLimits(1,172.4,172.6)
        #fit_gauss.FixParameter(1,mean_gauss)
        #fit_gauss.FixParameter(0,norm_gauss_numerator)
        #fit_gauss.FixParameter(3,alpha_non_crystalball)


        #net_mass.Fit("fit_voigt", "R")

        net_mass.Fit("fit_gauss", "R")
        
        p_gauss = fit_gauss.GetParameters()
        print("fitting result for gauss: {parameter1}, {parameter2}, {parameter3}".format(
        parameter1 = p_gauss[0],
        parameter2 = p_gauss[1],
        parameter3 = p_gauss[2]
        ))        

        #fit_power_law.SetParameters(p_power_law[0], p_power_law[1], p_power_law[2])
        fit_exponential.SetParameters(p_exponential[0], p_exponential[1], p_exponential[2])
        #fit_exponential.SetParameters(p_exponential[0], p_exponential[1], p_exponential[2], p_exponential[3])
        fit_gauss.SetParameters(p_gauss[0], p_gauss[1], p_gauss[2])
        '''

        #===========================================================================================================================================


        #==============================================================Wald (Inverse Gaussian)======================================================

        #fit_wald = ROOT.TF1("fit_wald",wald,100,1000,4) 
        fit_wald = ROOT.TF1("fit_wald",wald,100,325,4) #Functional form of Wald is not going to cover the whole thing
        #fit_wald = ROOT.TF1("fit_wald",wald,100,370,4) #Functional form of Wald is not going to cover the whole thing
        fit_wald.SetParNames("norm","lambda","mean","shift") #This shift is in x, not the whole function

        #fit_wald.SetParameters(1000.0, 0.2, 170.0, 99)
        fit_wald.SetParameters(120000.0, 450, 170.0, 99)

        '''
        fit_wald.SetParLimits(0,0,100000)
        fit_wald.SetParLimits(1,0,100000) #lambda probably has to be positive
        fit_wald.SetParLimits(2,0,100000) #mean probably has to be positive
        fit_wald.SetParLimits(3,95,99.999999) #Shift has to be probably positive (also less than the lower limit)
        '''

        #Parameter limits set according to what I have observed in desmos
        #fit_wald.SetParLimits(0,100000,150000)
        fit_wald.SetParLimits(0,120000,150000)
        fit_wald.SetParLimits(1,400,500) #lambda probably has to be positive
        fit_wald.SetParLimits(2,0,1000) #mean probably has to be positive
        #fit_wald.SetParLimits(3,95,99.999999) #Shift has to be probably positive (also less than the lower limit)
        fit_wald.SetParLimits(3,92,99.999999) #Shift has to be probably positive (also less than the lower limit)

        #fit_wald.SetParLimits(3,0,100000) #Power definitely has to be positive
        '''
        net_mass.Fit("fit_wald", "R")

        p_wald = fit_wald.GetParameters()
        print("fitting result for power law: {parameter1}, {parameter2}, {parameter3}, {parameter4}".format(
        parameter1 = p_wald[0],
        parameter2 = p_wald[1],
        parameter3 = p_wald[2],
        parameter4 = p_wald[3]
        ))

        fit_wald.SetParameters(p_wald[0], p_wald[1], p_wald[2], p_wald[3])
        '''

        #===========================================================================================================================================



        '''
        #fit_crystalball = ROOT.TF1("fit_crystalball",crystalball,100,600,5)
        #fit_crystalball = ROOT.TF1("fit_crystalball",crystalball,100,1000,5)
        fit_crystalball = ROOT.TF1("fit_crystalball",crystalball,100,2000,5)
        #fit_crystalball = ROOT.TF1("fit_crystalball",crystalball,100,500,5)
        fit_crystalball.SetParNames("norm","mean","sigma","alpha","n")
        #fit_crystalball.SetParameters(10.0, 172.5, 20.0, -1.0, 2.0)
        #fit_crystalball.SetParameters(10.0, 172.5, 20.0, -4000.0, 2.0)
        fit_crystalball.SetParameters(10.0, 172.5, 20.0, -2.0, 2.8)
        #fit_crystalball.SetParameters(10.0, 172.5, 20.0, 1.0, 2.0)

        #Somehow the mean is negative for width_2 (at least) so constrain that as well
        #fit_crystalball.SetParLimits(1,170,175)

        #We need alpha to be negative to have a positive tail so the upper limit is 0 and the lower limit is a very large value (not sure whether there is an option for inf)
        fit_crystalball.SetParLimits(3,-100000,0)
        #n needs to be positive (or non negative at least?)
        fit_crystalball.SetParLimits(4,0,100000)

        print("Current set of categories just before pyroot fitting",cats)

        #net_mass.Fit("fit_voigt", "R")

        net_mass.Fit("fit_crystalball", "R")

        p = fit_crystalball.GetParameters()
        
        print("fitting result for crystalball: {parameter1}, {parameter2}, {parameter3}, {parameter4}, {parameter5}".format(
        parameter1 = p[0],
        parameter2 = p[1],
        parameter3 = p[2],
        parameter4 = p[3],
        parameter5 = p[4]
        ))        

        fit_crystalball.SetParameters(p[0], p[1], p[2], p[3], p[4])
        '''

        #===============================================================After trying to fix alpha=======================================================

        '''
        #fit_crystalball = ROOT.TF1("fit_crystalball",crystalball,100,1000,5)
        fit_crystalball = ROOT.TF1("fit_crystalball",crystalball,100,210,5)
        #fit_crystalball.SetParNames("norm","mean","sigma","alpha_x","n")
        fit_crystalball.SetParNames("norm","mean","mean_power_law","alpha_x","n")
        fit_crystalball.SetParameters(10.0, 172.5, 205.0, 210.0, 2.8)
        #fit_crystalball.SetParameters(10.0, 172.5, 20.0, 1.0, 2.0)

        #Somehow the mean is negative for width_2 (at least) so constrain that as well
        fit_crystalball.SetParLimits(1,170,175)


        #We need alpha to be negative to have a positive tail so the upper limit is 0 and the lower limit is a very large value (not sure whether there is an option for inf)
        #fit_crystalball.SetParLimits(3,-100000,0)
        fit_crystalball.SetParLimits(2,-10000,alpha_non_crystalball)
        #fit_crystalball.FixParameter(3,alpha_non_crystalball)
        fit_crystalball.SetParLimits(3,alpha_non_crystalball-5,alpha_non_crystalball+5)
        #n needs to be positive (or non negative at least?)
        fit_crystalball.SetParLimits(4,0,10000)

        print("Current set of categories just before pyroot fitting",cats)

        #net_mass.Fit("fit_voigt", "R")

        net_mass.Fit("fit_crystalball", "R")

        p = fit_crystalball.GetParameters()
        
        print("fitting result for crystalball: {parameter1}, {parameter2}, {parameter3}, {parameter4}, {parameter5}".format(
        parameter1 = p[0],
        parameter2 = p[1],
        parameter3 = p[2],
        parameter4 = p[3],
        parameter5 = p[4]
        ))        

        fit_crystalball.SetParameters(p[0], p[1], p[2], p[3], p[4])
        '''

        #'''
        #fit_power_law.Draw("same")
        #print("Drew fit_power_law")

        #fit_wald.Draw("same")
        #print("Drew fit_wald")

        #fit_wald.SetParameters(120000, 477, 86, 100)
        '''
        fit_wald.SetParameters(120000, 477, 86, 99)
        fit_wald.SetLineColor(3)

        fit_wald.Draw("same")
        print("Drew fit_wald with visual parameters")
        '''

        '''
        fit_exponential = ROOT.TF1("fit_exponential",exponential,325,1000,3)
        #fit_exponential.SetParameters(20, 0.004438, 325)
        #fit_exponential.SetParameters(20, 0.0047, 325)
        #fit_exponential.SetParameters(20, 0.0055, 325)
        #fit_exponential.SetParameters(20, 0.006, 325)
        #fit_exponential.SetParameters(20, 0.007, 325)
        #fit_exponential.SetParameters(18, 0.0075, 370)
        fit_exponential.SetParameters(20, 0.0085, 325)
        fit_exponential.SetLineColor(1)
        fit_exponential.Draw("same")
        print("Drew fit_exponential with visual parameters")
        '''

        '''
        fit_exponential.SetParameters(p_exponential[0], p_exponential[1], p_exponential[2])
        fit_exponential.SetLineColor(1)
        fit_exponential.Draw("same")
        print("Drew fit_exponential with wald")
        '''

        #====================================================================Gauss and reverse gauss==============================================================
        '''
        fit_gauss_rev_gauss = ROOT.TF1("fit_gauss_rev_gauss",gauss_rev_gauss,100,1000,5)
        #fit_gauss_rev_gauss.SetParameters(440, 210, 172.5, 42.7, 2470)
        #fit_gauss_rev_gauss.SetParameters(440, 210, 172.5, 25, 2470)
        fit_gauss_rev_gauss.SetParNames("norm", "alpha", "mean", "sigma", "rev_mean")
        fit_gauss_rev_gauss.SetParameters(440, 185, 172.5, 25, 2700)
        #fit_gauss_rev_gauss.FixParameter(0,440)

        #Since it is a Gaussian obtain the max of the histogram and then set normalization limits accordingly
        print("Max of histogram")
        print(net_mass.GetMaximum())

        #fit_gauss_rev_gauss.SetParLimits(0,1000,1700)
        fit_gauss_rev_gauss.SetParLimits(0,net_mass.GetMaximum()-10,net_mass.GetMaximum()+10)
        fit_gauss_rev_gauss.FixParameter(2,172.5)
        #fit_gauss_rev_gauss.FixParameter(3,25)

        fit_gauss_rev_gauss.SetParLimits(3,18,30)

        fit_gauss_rev_gauss.SetParLimits(1,174,270)
        #fit_gauss_rev_gauss.SetParLimits(4,800,20000)
        #fit_gauss_rev_gauss.SetParLimits(4,1000,20000)
        #fit_gauss_rev_gauss.SetParLimits(4,1500,20000)
        fit_gauss_rev_gauss.SetParLimits(4,1000,20000)

        net_mass.Fit("fit_gauss_rev_gauss", "R")

        p = fit_gauss_rev_gauss.GetParameters()
        
        print("fitting result for gauss_rev_gauss: {parameter1}, {parameter2}, {parameter3}, {parameter4}, {parameter5}".format(
        parameter1 = p[0],
        parameter2 = p[1],
        parameter3 = p[2],
        parameter4 = p[3],
        parameter5 = p[4]
        ))        

        fit_gauss_rev_gauss.SetParameters(p[0], p[1], p[2], p[3], p[4])

        fit_gauss_rev_gauss.SetLineColor(2)
        fit_gauss_rev_gauss.Draw("same")
        print("Drew fit_gauss_rev_gauss")
        '''

        #=========================================================================================================================================================

        #nom_sum = np.sum(list(mc_counts.values()), axis=0)
        print(type(mc_counts))
        print(type(mc_counts.values()))
        nom_sum = np.sum(mc_counts_rebinned, axis=0)
        nom_sum = nom_sum[:-1]

        stat_var = hist.integrate("dataset", mc_datasets).variances()

        #Since we are rebinning here it is probably ok to add the variances in quadrature (actually the variances will just add, no quadrature)

        x_lower_edges = x.edges[:-1]

        #Crude rebinning - choosing 10 GeV (or maybe 20) since it is an exact multiple of 2 GeV (which is the current binning)
        bin_multiple = 10
        #current_binsize = 2
        current_binsize = (x.edges[10] - x.edges[0])/10 #Assuming that the bin sizes are all equal
        #last_upper_bin_edge = 1000
        last_upper_bin_edge = x.edges[-1]
        print("Current binsize in plot_counts_mc",current_binsize)
        print("last_upper_bin_edge in plot_counts_mc", last_upper_bin_edge)

        new_x_edges = x_lower_edges[np.subtract(x_lower_edges,x_lower_edges[0])%(current_binsize*bin_multiple) == 0]
        print("Bins to be considered in the rebinning case",new_x_edges)
        
        #new_x_edges = np.append(new_x_edges,last_upper_bin_edge)

        stat_var_rebinned = [] #Defined initially as a list, and going to convert to a numpy array

        for i in range(0,len(x_lower_edges)):
            print("Number of bins in original histogram",i)
            if i%bin_multiple == 0:
                variance_current = 0
                for j in range(i,i+bin_multiple):
                    print("bin number for which content is to be added", j)
                    print("Current bin content",stat_var[j])
                    variance_current += stat_var[j]
                print("Total bin content",variance_current)
                stat_var_rebinned.append(variance_current)

        #This is just to make the number of entries the same as the x edges - this includes the last upper bin limit as well
        #stat_var_rebinned.append(stat_var_rebinned[-1])
        stat_var_rebinned = np.array(stat_var_rebinned)

        print("len_stat_var",len(stat_var))
        print(type(stat_var))
        print("len_stat_var",len(stat_var_rebinned))
        print(type(stat_var_rebinned))

        new_x_edges = np.append(new_x_edges,last_upper_bin_edge)

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
            '''
            stat_unc = np.abs(
                intervals.poisson_interval(nom_sum, stat_var) - nom_sum) ** .5
            hatch = hatched_area(
                x.edges, nom_sum, stat_unc[1], stat_unc[0], "Stat. unc.", ax)
            '''
            #stat_unc = np.abs(intervals.poisson_interval(nom_sum, stat_var_rebinned) - nom_sum) ** .5
            stat_unc = np.abs(intervals.poisson_interval(nom_sum, stat_var_rebinned) - nom_sum)
            '''
            for i in range(0,len(new_x_edges)):
                print()
                print("Bin number after calculating poisson_interval",i)
                print("Bin height rebinned after poisson interval",nom_sum[i])
                print("Bin variance rebinned after poisson interval",stat_var_rebinned[i])
                print("Lower limit of Interval calculated according to understanding",nom_sum[i] - (stat_var_rebinned[i])**0.5)
                print("Upper limit of Interval calculated according to understanding",nom_sum[i] + (stat_var_rebinned[i])**0.5)
                print("Lower interval from poisson_interval", intervals.poisson_interval(nom_sum, stat_var_rebinned)[0][i])
                print("Upper interval from poisson_interval", intervals.poisson_interval(nom_sum, stat_var_rebinned)[1][i])
                print("Lower Stat unc calculated",stat_unc[0][i])
                print("Lower Stat unc calculated",stat_unc[1][i])
            '''

            hatch = hatched_area(
                new_x_edges, nom_sum, stat_unc[1], stat_unc[0], "Stat. unc.", ax)

        '''
        else:
            stat_unc = stat_var ** .5
            # Sum all uncertainties in quadrature
            lo = np.concatenate([sysunc[0], [stat_unc]], axis=0)
            up = np.concatenate([sysunc[1], [stat_unc]], axis=0)

            lo = np.asarray(ak.sum(lo ** 2, axis=0))
            up = np.asarray(ak.sum(up ** 2, axis=0))
            hatch = hatched_area(x.edges, nom_sum, up ** .5, lo ** .5, "Unc.", ax)
        '''

        net_mass = ROOT.TH1D("net_mass","Same sign signal fit",len(new_x_edges[:-1]),new_x_edges[:-1])
        net_mass_error = ROOT.TH1D("net_mass","Same sign signal fit",len(new_x_edges[:-1]),new_x_edges[:-1])
        net_mass_var_binhgt = ROOT.TH1D("net_mass_var_binhgt","Same sign signal fit",len(new_x_edges[:-1]),new_x_edges[:-1])

        for i in range(0,len(new_x_edges[:-1])):
            print("Number of bins just before drawing and fitting (rebinned)",i)
            net_mass.SetBinContent(i+1,bar_content[i])
            if(bar_content[i]==0):
                net_mass_var_binhgt.SetBinContent(i+1,0.0)
            else:
                net_mass_var_binhgt.SetBinContent(i+1,stat_var_rebinned[i]/bar_content[i])
            #'''
            #Also setting only statistical error at the moment (have removed the extra sqrt which may have been a typo)
            #Do not think there is an option for setting two separate errors for upper and lower ones
            net_mass_error.SetBinContent(i+1,bar_content[i])
            net_mass_error.SetBinError(i+1,(max(stat_unc[0][i],stat_unc[1][i])))
            #'''
            print("Bin content just before drawing and fitting (rebinned)",net_mass.GetBinContent(i+1))
        #'''
        print(net_mass)

        ROOT.gROOT.SetBatch(True)

        c1 = ROOT.TCanvas('c1', 'c1', 700, 1000)
        print("Current set of categories just before drawing",cats)

        pad_upper = ROOT.TPad("pad_upper", "pad_upper", 0.00, 0.41, 1, 1)
        pad_upper.SetBottomMargin(0) 
        pad_upper.SetLeftMargin(0.1)
        pad_upper.SetRightMargin(0.05)
        pad_upper.SetTopMargin(0.15)
        pad_upper.Draw()
        c1.cd()  

        pad_lower = ROOT.TPad("pad_lower", "pad_lower", 0.00, 0.00, 1, 0.405)
        pad_lower.SetTopMargin(0.03) 
        pad_lower.SetBottomMargin(0.3)
        pad_lower.SetLeftMargin(0.1)
        pad_lower.SetRightMargin(0.05)
        pad_lower.Draw()

        pad_upper.cd()

        net_mass_error.SetFillStyle(3354)
        net_mass_error.SetFillColor(1)

        net_mass_error.Draw("E0E2")
        print("Drew statistical error bars")

        net_mass.SetFillStyle(0)
        net_mass.Draw("hist same")
        print("Drew histogram")  

        upper_limit = 1000
        lower_limit = 80

        #fit_crystalball = ROOT.TF1("fit_crystalball",crystalball,lower_limit,upper_limit,5)
        fit_crystalball = ROOT.TF1("fit_crystalball",crystalball,lower_limit,upper_limit,5)
        fit_crystalball.SetParNames("norm", "mean", "sigma", "alpha", "power")
        fit_crystalball.SetParameters(440*20, 172.5, 30, -1, 0.75)
        #fit_crystalball.FixParameter(0,440)

        #Since it is a Gaussian obtain the max of the histogram and then set normalization limits accordingly
        print("Max of histogram")
        print(net_mass.GetMaximum())

        #fit_crystalball.SetParLimits(0,1000,1700)
        fit_crystalball.SetParLimits(0,net_mass.GetMaximum()-10,net_mass.GetMaximum()+10)
        #fit_crystalball.SetParLimits(0,net_mass.GetMaximum()-100,net_mass.GetMaximum()+100)
        #fit_crystalball.SetParLimits(0,net_mass.GetMaximum()-20,net_mass.GetMaximum()+20)
        fit_crystalball.FixParameter(1,172.5)
        #fit_crystalball.FixParameter(2,30)

        #fit_crystalball.SetParLimits(2,18,40)
        fit_crystalball.SetParLimits(2,18,80)

        fit_crystalball.SetParLimits(3,-3.25,-0.05)
        #Changing the lower limit so that maybe the polynomial starts a little 'later' (higher x)
        #Actually this is even more terrible
        #fit_crystalball.SetParLimits(3,-3.25,-0.7)
        fit_crystalball.SetParLimits(4,0,10)
        #fit_crystalball.SetParLimits(4,4.5,10)
        #fit_crystalball.SetParLimits(4,5,10)
        

        net_mass.Fit("fit_crystalball", "R")

        p = fit_crystalball.GetParameters()
        
        print("fitting result for crystalball: {parameter1}, {parameter2}, {parameter3}, {parameter4}, {parameter5}".format(
        parameter1 = p[0],
        parameter2 = p[1],
        parameter3 = p[2],
        parameter4 = p[3],
        parameter5 = p[4]
        ))        

        fit_crystalball.SetParameters(p[0], p[1], p[2], p[3], p[4])
        initial_power = p[4]

        fit_crystalball.SetLineColor(2)
        fit_crystalball.Draw("same")
        print("Drew fit_crystalball")

        chi_2 = 0
        for i in range(0,len(new_x_edges[:-1])):
            print()
            print("Number of bins just before drawing and fitting (rebinned)",i)
            x_center = net_mass.GetBinCenter(i+1)
            print("Value of x at bin center",x_center)
            current_hist = net_mass.GetBinContent(i+1)
            print("Current hist value",current_hist)
            current_function = fit_crystalball.Eval(x_center)
            print("Current fit value", current_function)
            current_sigma_2 = stat_var_rebinned[i] #Technically this is defined only for the absence of sys uncs
            print("Current variance in chi_2 loop",current_sigma_2)
            if (current_sigma_2 > 0) & (x_center > lower_limit) & (x_center < upper_limit):
                print("Good statistics bin within the confines of the fit")
                chi_2_term = (current_hist - current_function)**2/current_sigma_2
                print("Current chi_2 term",chi_2_term)
                chi_2 += chi_2_term
        
        print("Total chi_2",chi_2)

        '''
        #Even more original fit that somehow cannot be replicated for width_1
        fit_crystalball.SetParameters(1.31190e+03, 1.72123e+02, 2.60617e+01, -9.42691e-01, 3.49194e+00)
        fit_crystalball.Draw("same")
        print("Drew fit_crystalball with very good fit")
        '''

        #latex = ROOT.TLatex(0., 0., '\chi_2 = ' + str(chi_2))
        latex = ROOT.TLatex(0., 0., '')
        latex.SetNDC(True)
        latex.SetTextFont(63)
        latex.SetTextSize(20)
        #latex.DrawLatex(700,20000,"#chi^{2} = " + str(chi_2))
        latex.DrawLatex(0.7,0.7,"#chi^{2} = " + str(round(chi_2,3)))
        latex.DrawLatex(0.7,0.6,"power = " + str(round(initial_power,3)))
        #latex.DrawLatex(0.7,0.7,"#chi^{2} = " + f"{chi_2:03}")

        pad_lower.cd()
        #net_mass_ratio = ROOT.TH1D("net_mass_ratio","net_mass_ratio",len(new_x_edges[:-1]),new_x_edges[:-1])
        net_mass_ratio = ROOT.TGraphErrors(len(new_x_edges[:-1]))
        net_mass_ratio.SetTitle("")
        net_mass_error_ratio = ROOT.TGraphErrors(len(new_x_edges[:-1]))
        #net_mass_error_ratio = ROOT.TH1D("net_mass_error_ratio","net_mass_error_ratio",len(new_x_edges[:-1]),new_x_edges[:-1])

        for i in range(0,len(new_x_edges[:-1])):
            print("Number of bins just before drawing and fitting (rebinned)",i)
            #net_mass_ratio.SetBinContent(i+1,bar_content[i])
            net_mass_ratio.SetPoint(i+1,net_mass.GetBinCenter(i+1),bar_content[i]/fit_crystalball.Eval(net_mass.GetBinCenter(i+1)))
            #net_mass_ratio.SetPointError(i+1,net_mass.GetBinCenter(i+1),(max(stat_unc[0][i],stat_unc[1][i]))/fit_crystalball.Eval(net_mass.GetBinCenter(i+1)))
            net_mass_ratio.SetPointError(i+1,0.0,(max(stat_unc[0][i],stat_unc[1][i]))/fit_crystalball.Eval(net_mass.GetBinCenter(i+1)))
            #'''
            #Also setting only statistical error at the moment (have removed the extra sqrt which may have been a typo)
            #Do not think there is an option for setting two separate errors for upper and lower ones
            '''
            net_mass_error_ratio.SetBinContent(i+1,bar_content[i]/fit_crystalball.Eval(net_mass.GetBinCenter(i+1)))
            #Little dicey about this expression - error/data or error/fit?
            net_mass_error_ratio.SetBinError(i+1,(max(stat_unc[0][i],stat_unc[1][i]))/fit_crystalball.Eval(net_mass.GetBinCenter(i+1)))
            '''
            #'''
            #print("Bin content just before drawing and fitting (rebinned)",net_mass.GetBinContent(i+1))

        '''
        net_mass_error_ratio.SetFillStyle(3354)
        net_mass_error_ratio.SetFillColor(1)
        net_mass_error_ratio.Draw("E0E2")
        '''

        net_mass_ratio.SetFillStyle(3354)
        net_mass_ratio.SetMarkerStyle(20)
        net_mass_ratio.SetMarkerSize(0.5)
        net_mass_ratio.GetXaxis().SetLimits(0.,upper_limit)
        net_mass_ratio.GetXaxis().SetTitle("Invariant mass of reconstructed top (in GeV)")

        #net_mass_ratio.Draw("AP same")
        net_mass_ratio.Draw("AP")

        newline_ratio_horizontal = ROOT.TLine(0,1,upper_limit,1)
        newline_ratio_horizontal.Draw("same")
        newline_ratio_horizontal.SetLineWidth(1)
        ROOT.gPad.Update()

        c1.Print("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/width_" + str(args.width_num) + "/net_mass_"+ cats[0] + cats[1] + cats[2] + cats[3] + cats[4] + cats[5] + "_nonnegative_power_crystalball_new_20GeV_bins_signal.png")     
        c1.Print("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/width_" + str(args.width_num) + "/net_mass_"+ cats[0] + cats[1] + cats[2] + cats[3] + cats[4] + cats[5] + "_nonnegative_power_crystalball_new_20GeV_bins_signal.C")     

        #Zoomed in - will be removed and added back as required for debugging
        #net_mass_error.GetXaxis().SetRangeUser(500,1000)
        #net_mass_error.GetYaxis().SetRangeUser(0,20)

        c1.Print("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/width_" + str(args.width_num) + "/tail_zoomed_net_mass_"+ cats[0] + cats[1] + cats[2] + cats[3] + cats[4] + cats[5] + "_nonnegative_power_crystalball_new_20GeV_bins_signal.png")     
        c1.Print("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/width_" + str(args.width_num) + "/tail_zoomed_net_mass_"+ cats[0] + cats[1] + cats[2] + cats[3] + cats[4] + cats[5] + "_nonnegative_power_crystalball_new_20GeV_bins_signal.C")     

        #Back to original
        #net_mass_error.GetXaxis().SetRangeUser(0,1000)

        pad_upper.SetLogy()
        #Added June 7 for width 21 to match the axis with width 14
        #pad_upper.RangeAxis(0,10000,1000,15000)
        #Universal range for all widths (taken from width 21)
        #Unfortunately this works better only for the log scale
        net_mass_error.GetYaxis().SetRangeUser(0.1,9000)


        c1.Print("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/width_" + str(args.width_num) + "/net_mass_"+ cats[0] + cats[1] + cats[2] + cats[3] + cats[4] + cats[5] + "_nonnegative_power_crystalball_new_log_y_20GeV_bins_signal.png")     
        c1.Print("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/width_" + str(args.width_num) + "/net_mass_"+ cats[0] + cats[1] + cats[2] + cats[3] + cats[4] + cats[5] + "_nonnegative_power_crystalball_new_log_y_20GeV_bins_signal.C")     

        c1.Close()

        c2 = ROOT.TCanvas('c2', 'c2', 700, 700)
        net_mass_var_binhgt.Draw()
        net_mass_var_binhgt.SetStats(0)
        print("Drew variance per bin height comparison")

        c2.Print("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/width_" + str(args.width_num) + "/net_mass_"+ cats[0] + cats[1] + cats[2] + cats[3] + cats[4] + cats[5] + "_var_binhgt_20GeV_bins_signal.png")     
        c2.Print("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/width_" + str(args.width_num) + "/net_mass_"+ cats[0] + cats[1] + cats[2] + cats[3] + cats[4] + cats[5] + "_var_binhgt_20GeV_bins_signal.C")     

        c2.SetLogy()

        c2.Print("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/width_" + str(args.width_num) + "/net_mass_"+ cats[0] + cats[1] + cats[2] + cats[3] + cats[4] + cats[5] + "_var_binhgt_log_y_20GeV_bins_signal.png")     
        c2.Print("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/width_" + str(args.width_num) + "/net_mass_"+ cats[0] + cats[1] + cats[2] + cats[3] + cats[4] + cats[5] + "_var_binhgt_log_y_20GeV_bins_signal.C")     

        c2.Close()

        #'''
        #for power_index in range(1,6):
        for power_index in range(1,51):
        #for power_index in range(1,5): #Only for debugging
        #for power_index in range(1,101): #Specifically for width_21
        #for power_index in range(1,2):
            c1 = ROOT.TCanvas('c1', 'c1', 700, 1000)

            pad_upper = ROOT.TPad("pad_upper", "pad_upper", 0.00, 0.41, 1, 1)
            pad_upper.SetBottomMargin(0) 
            pad_upper.SetLeftMargin(0.1)
            pad_upper.SetRightMargin(0.05)
            pad_upper.SetTopMargin(0.15)
            pad_upper.Draw()
            c1.cd()  

            pad_lower = ROOT.TPad("pad_lower", "pad_lower", 0.00, 0.00, 1, 0.405)
            pad_lower.SetTopMargin(0.03) 
            pad_lower.SetBottomMargin(0.3)
            pad_lower.SetLeftMargin(0.1)
            pad_lower.SetRightMargin(0.05)
            pad_lower.Draw()

            pad_upper.cd()

            net_mass_error.SetFillStyle(3354)
            net_mass_error.SetFillColor(1)

            #net_mass_error.GetYaxis().SetRangeUser(0,4500)
            net_mass_error.GetYaxis().SetRangeUser(0.001,4500)

            net_mass_error.Draw("E0E2")
            print("Drew statistical error bars")

            net_mass.SetFillStyle(0)
            net_mass.Draw("hist same")

            print()
            print("Drew histogram")  
            #current_power = initial_power+power_index*0.2
            current_power = initial_power+power_index*0.1
            fit_crystalball.SetParameters(440*20, 172.5, 30, -1, initial_power)
            fit_crystalball.SetParLimits(0,net_mass.GetMaximum()-10,net_mass.GetMaximum()+10)
            fit_crystalball.FixParameter(1,172.5)
            fit_crystalball.SetParLimits(2,18,80)
            fit_crystalball.SetParLimits(3,-3.25,-0.05)
            #fit_crystalball.FixParameter(4,initial_power+power_index*0.2)
            fit_crystalball.FixParameter(4,current_power)

            #Only for debugging
            #fit_crystalball.FixParameter(4,initial_power)
            net_mass.Fit("fit_crystalball", "R")

            p_loop = fit_crystalball.GetParameters()
            print("Power index",power_index)
            print("Current value of p_4",initial_power)
            #print("Power value",initial_power+power_index*0.2)
            print("Power value",current_power)
            print("fitting result for crystalball in loop: {parameter1}, {parameter2}, {parameter3}, {parameter4}, {parameter5}".format(
            parameter1 = p_loop[0],
            parameter2 = p_loop[1],
            parameter3 = p_loop[2],
            parameter4 = p_loop[3],
            parameter5 = p_loop[4]
            ))        

            
            fit_crystalball.SetParameters(p_loop[0], p_loop[1], p_loop[2], p_loop[3], p_loop[4])

            fit_crystalball.SetLineColor(2)
            fit_crystalball.Draw("same")
            print("Drew fit_crystalball")
            

            chi_2 = 0
            for i in range(0,len(new_x_edges[:-1])):
                print()
                print("Number of bins just before drawing and fitting (rebinned)",i)
                x_center = net_mass.GetBinCenter(i+1)
                print("Value of x at bin center",x_center)
                current_hist = net_mass.GetBinContent(i+1)
                print("Current hist value",current_hist)
                current_function = fit_crystalball.Eval(x_center)
                print("Current fit value", current_function)
                current_sigma_2 = stat_var_rebinned[i] #Technically this is defined only for the absence of sys uncs
                print("Current variance in chi_2 loop",current_sigma_2)
                if (current_sigma_2 > 0) & (x_center > lower_limit) & (x_center < upper_limit):
                    print("Good statistics bin within the confines of the fit")
                    chi_2_term = (current_hist - current_function)**2/current_sigma_2
                    print("Current chi_2 term",chi_2_term)
                    chi_2 += chi_2_term
            
            #print("Power value",initial_power+power_index*0.2)
            print("Power value",current_power)
            print("Total chi_2 for power value in steps",chi_2)

            latex = ROOT.TLatex(0., 0., '')
            latex.SetNDC(True)
            latex.SetTextFont(63)
            latex.SetTextSize(20)
            #latex.DrawLatex(700,20000,"#chi^{2} = " + str(chi_2))
            latex.DrawLatex(0.7,0.7,"#chi^{2} = " + str(round(chi_2,3)))
            #latex.DrawLatex(0.7,0.6,"power = " + str(round(initial_power+power_index*0.2,3)))
            latex.DrawLatex(0.7,0.6,"power = " + str(round(current_power,3)))
            #latex.DrawLatex(0.7,0.7,"#chi^{2} = " + f"{chi_2:03}")

            pad_lower.cd()

            #Since there is a new set of fit parameters obviously new values of ratio have to be set
            for i in range(0,len(new_x_edges[:-1])):
                print("Number of bins just before drawing and fitting (rebinned)",i)
                net_mass_ratio.SetPoint(i+1,net_mass.GetBinCenter(i+1),bar_content[i]/fit_crystalball.Eval(net_mass.GetBinCenter(i+1)))
                net_mass_ratio.SetPointError(i+1,0.0,(max(stat_unc[0][i],stat_unc[1][i]))/fit_crystalball.Eval(net_mass.GetBinCenter(i+1)))


            net_mass_ratio.SetFillStyle(3354)
            net_mass_ratio.SetMarkerStyle(20)
            net_mass_ratio.SetMarkerSize(0.5)
            net_mass_ratio.GetXaxis().SetLimits(0.,upper_limit)
            net_mass_ratio.GetXaxis().SetTitle("Invariant mass of reconstructed top (in GeV)")

            #net_mass_ratio.Draw("AP same")
            net_mass_ratio.Draw("AP")

            newline_ratio_horizontal = ROOT.TLine(0,1,upper_limit,1)
            newline_ratio_horizontal.Draw("same")
            newline_ratio_horizontal.SetLineWidth(1)
            ROOT.gPad.Update()

            c1.Print("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/width_" + str(args.width_num) + "/net_mass_"+ cats[0] + cats[1] + cats[2] + cats[3] + cats[4] + cats[5] + "_loop_power_index_"+ str(power_index) + "_crystalball_new_20GeV_bins.png")     

            pad_upper.SetLogy()

            net_mass_error.GetYaxis().SetRangeUser(0.1,9000)

            c1.Print("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/width_" + str(args.width_num) + "/net_mass_"+ cats[0] + cats[1] + cats[2] + cats[3] + cats[4] + cats[5] + "_loop_power_index_"+ str(power_index) + "_crystalball_new_log_y_20GeV_bins.png")     

            c1.Close()
        #'''

        '''
        if cats[3] != '[sum]':
            print("No extra stacks")
            bars = stacked_bars(x.edges, mc_counts, mc_labels, mc_colors, order, ax)
        elif cats[3] == '[sum]':
            print("Extra stacks required for signal (wbj)")
            order = []

        '''
    #hahaha

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


def plot_counts(init_time, cats, cat_labels, hist, hist_og, x, config, sysunc, logarithmic, no_data, ax, plot_signal):
    """Create the upper part of the ratio plot."""
    ret = []
    #print("Reached plot_counts function")
    #print(plot_counts_mc(hist, x, config, sysunc, ax))

    #Boolean for whether the signal is to be plotted or the background
    ret += plot_counts_mc(init_time, cats, cat_labels, hist, hist_og, x, config, sysunc, ax, plot_signal)
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


def plot(init_time, cats, cat_labels, hist, hist_og, config, fpath, logarithmic, stat_only, no_data, plot_signal):
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
    plot_counts(init_time, cats, cat_labels, nom, hist_og, x, config, sysunc, logarithmic, no_data, ax1, plot_signal)
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
    histcol, histcol_ttbar, histcol_qcd, key, config, name, cut, cutidx, output, logarithmic, stat_only,
    no_data, fmt
):
    print("Histcol inside process",histcol)
    print("Key for histcol loading",key)
    hist_other = histcol.load(key)

    print("Histcol inside process for ttbar",histcol_ttbar)
    print("Key for histcol ttbar loading",key)
    hist_ttbar = histcol_ttbar.load(key)

    hist_qcd = histcol_qcd.load(key)

    ds_other = set(hist_other.axes["dataset"])
    ds_ttbar = set(hist_ttbar.axes["dataset"])
    ds_qcd = set(hist_qcd.axes["dataset"])

    print("Datasets",ds_other)
    print("ds_ttbar",ds_ttbar)
    print("ds_qcd",ds_qcd)

    print("Type of dataset entries to be ignored")
    print(config["plot_datasets_ignore"])
    print(type(config["plot_datasets_ignore"]))
    #Removing qcd from the hist_other
    qcd_remove = ["QCD_Pt-1000_MuEnrichedPt5_TuneCP5_13TeV-pythia8","QCD_Pt-120To170_MuEnrichedPt5_TuneCP5_13TeV-pythia8","QCD_Pt-120to170_EMEnriched_TuneCP5_13TeV-pythia8",
                "QCD_Pt-15To20_MuEnrichedPt5_TuneCP5_13TeV-pythia8","QCD_Pt-15to20_EMEnriched_TuneCP5_13TeV-pythia8","QCD_Pt-170To300_MuEnrichedPt5_TuneCP5_13TeV-pythia8",
                "QCD_Pt-170to300_EMEnriched_TuneCP5_13TeV-pythia8","QCD_Pt-20To30_MuEnrichedPt5_TuneCP5_13TeV-pythia8","QCD_Pt-20to30_EMEnriched_TuneCP5_13TeV-pythia8",
                "QCD_Pt-300To470_MuEnrichedPt5_TuneCP5_13TeV-pythia8","QCD_Pt-300toInf_EMEnriched_TuneCP5_13TeV-pythia8","QCD_Pt-30To50_MuEnrichedPt5_TuneCP5_13TeV-pythia8",
                "QCD_Pt-30to50_EMEnriched_TuneCP5_13TeV-pythia8","QCD_Pt-470To600_MuEnrichedPt5_TuneCP5_13TeV-pythia8","QCD_Pt-50to80_EMEnriched_TuneCP5_13TeV-pythia8",
                "QCD_Pt-600To800_MuEnrichedPt5_TuneCP5_13TeV-pythia8","QCD_Pt-800To1000_MuEnrichedPt5_TuneCP5_13TeV-pythia8","QCD_Pt-80To120_MuEnrichedPt5_TuneCP5_13TeV-pythia8",
                "QCD_Pt-80to120_EMEnriched_TuneCP5_13TeV-pythia8"]
                
    #hist_other = remove_datasets(hist_other, qcd_remove)
    #hist = hist_other + hist_ttbar + hist_qcd   

    hist = hist_other #Temporarily, only for debugging

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

        #if (len_cats >= 3):
        if (len_cats == 6):
            print("Number of categories is 6")
            #if (cats[0] == 'mu') & (cats[1] == 'j2_b1'):
            #New category constraints added because we really need only those fits
            #if (cats[0] == 'mu') & (cats[1] == 'j2_b1') & ((cats[2] == 'opp_sign') | (cats[2] == 'same_sign')) & (cats[3] == '[sum]'):
            #if (cats[0] == 'mu') & (cats[1] == 'j2_b1') & ((cats[2] == 'opp_sign') | (cats[2] == 'same_sign')) & (cats[3] == '[sum]') & ((cats[4] == 'offshell') | (cats[4] == 'onshell')) & (cats[5] == '[sum]'):
            
            #This combo is for mass histograms specifically
            #if (cats[0] == 'mu') & (cats[1] == 'j2_b1') & ((cats[2] == 'opp_sign') | (cats[2] == 'same_sign')) & (cats[3] == '[sum]') & ((cats[4] == '[sum]')) & (cats[5] == '[sum]'):
            if (cats[0] == 'mu') & (cats[1] == 'j2_b1') & (cats[2] == 'same_sign') & (cats[3] == '[sum]') & ((cats[4] == '[sum]')) & (cats[5] == '[sum]'):
            #Only one of the 4 categories for debugging
            #if (cats[0] == 'mu') & (cats[1] == 'j2_b1') & ((cats[2] == 'opp_sign')) & (cats[3] == '[sum]') & ((cats[4] == 'offshell')) & (cats[5] == '[sum]'):

                print("First category is mu")
                print("Second category is j2_b1 (current signal region)")

            #if (cats[0] == 'mu') & (cats[1] == 'j2_b1'):
            #    print("First category is mu")
            #    print("Second category is j2_b1")

                for this_ax in dense_axes:
                    print("dense_axes",dense_axes)
                    print("Label of histogram",this_ax.label)
                    #hahaha
                    print("this_ax",this_ax)
                    dense_axes_to_sum = [ax for ax in dense_axes if ax is not this_ax]
                    #Maybe because these are all one 1d histograms, this is empty, so not going to pass this as an argument
                    print("dense_axes_to_sum",dense_axes_to_sum) 
                    # hist_1d only has one dense axis, sys and dataset cat axes
                    hist_1d = hist_catsplit[{ax.name: sum for ax in dense_axes_to_sum}]
                    directory = os.path.join(output, name, *cats)
                    
                    print("name",name)
                    #if this_ax.label == "leading Jet $p_{\mathrm{T}}$":
                    if this_ax.label == "Top mass with leading bjet absolute min (GeV)":
                        print("Plotting only net invariant mass")

                        os.makedirs(directory, exist_ok=True)
                        fname = "_".join(
                            (f"Cut_{cutidx:03}_" + cut, name)
                            + cats
                            + (this_ax.name,)
                        ) + "_signal." + fmt
                        fpath = os.path.join(directory, fname)

                        try:
                            plot(a, cats, cat_labels, hist_1d, hist, config, fpath, logarithmic, stat_only, no_data, 1)
                            #plot(a, cats, cat_labels, hist_1d, hist, config, fpath, logarithmic, stat_only, no_data, 0)
                        except Exception as e:
                            print("An error occurred:")
                            print(type(e).__name__)
                            print("Error message")
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
        "-w", "--width_num", help="Chronological width number"
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

    #with open("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0316_ttbar_semileptonic/hists/hists.json") as f_ttbar:
    #with open("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0411_mc_ttbarsemilep/hists/hists.json") as f_ttbar:
    with open("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_ttbar_tw_wjets_0508/hists/hists.json") as f_ttbar:
    #with open("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_ttbar_tw_wjets_0523/hists/hists.json") as f_ttbar:
        histcol_ttbar = HistCollection.from_json(f_ttbar)

    #with open("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/full_0402_qcd/hists/hists.json") as f_qcd:
    #with open("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0411_mc_qcd/hists/hists.json") as f_qcd:
    with open("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/full_rest_bkg_0508/hists/hists.json") as f_qcd:
    #with open("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/full_rest_bkg_0523/hists/hists.json") as f_qcd:
        histcol_qcd = HistCollection.from_json(f_qcd)
    
    #with open("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0316_no_ttbar_semileptonic_20000/hists/hists.json") as f:
    #Pretty sure this one was with chunksize 50000
    #with open("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0316_no_ttbar_semileptonic/hists/hists.json") as f:
    #with open("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0411_mc_noqcd_nottbarsemilep/hists/hists.json") as f:
    with open("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/width_" + str(args.width_num) + "/hists/hists.json") as f:

    #with open("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0528_wbj/width_" + str(args.width_num) + "/hists/hists.json") as f:
    #with open("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0523_wbj/width_1/hists/hists.json") as f:
        
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
                process, histcol, histcol_ttbar, histcol_qcd, key, config, key[1], key[0], cutidx,
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


