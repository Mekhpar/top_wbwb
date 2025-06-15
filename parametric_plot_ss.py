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
from typing import List
import yaml

def float_flag_calc(word:str):
    try:
        float(word)
        float_flag = True
    except ValueError:
        float_flag = False

    return float_flag

def get_formula_par_vals(width, txt_file:str):
    fit_file = open(txt_file, "r")
    #fit_file = open("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/0612_norm_vs_width_with_errors_same_sign_root_original.txt", "r")
    lines_chi2 = fit_file.readlines()
    #print(lines_chi2)

    par_args = [width]

    ROOT.gROOT.SetBatch(True)

    fit_formula_string = "Fitting formula"

    par_name_list = []
    par_index_list = []

    for line in lines_chi2:
        print("current line")
        print(line)
        if line == '\n':
            pass
        elif line.startswith(fit_formula_string):
            formula_split = line.split()[2:]
            formula = " ".join(formula_split)
            formula = formula.replace("x[0]", "@0")
            formula = formula.replace("TMath::Exp","exp")
            print("Formula from txt file",formula)

        else:
            first_string = line.split()[0]
            print("First string in line")
            print(first_string)
            
            #Here it is assuming a lot of things - that there will be only one fit, and that every line starting with a number represents a fit parameter
            float_flag = float_flag_calc(first_string)
            #print("Float flag",float_flag)
            if float_flag == True:
                first_par = float(first_string)
                par_name = line.split()[1]
                par_value = float(line.split()[2])
                print("Parameter name",par_name)
                print("Parameter central value",par_value)

                variable = ROOT.RooRealVar(par_name,par_name, par_value)
                par_args.append(variable)
                par_index_list.append(first_string)
                par_name_list.append(par_name)

    for par_index in range(len(par_index_list)):
        formula = formula.replace(par_name_list[par_index], "@"+par_index_list[par_index])
    print("Final formula",formula)

    return par_args, formula


width = ROOT.RooRealVar("width","width", 0.5, 10)
c1 = ROOT.TCanvas('c1', 'c1', 700, 500)
plot_vs_width = width.frame()
#norm_ss = ROOT.RooFormulaVar("norm_same_sign", "norm_same_sign", "@1 - @2*(@0-@4) - @3/((@0-@4)**@5)", ROOT.RooArgList(width,shift,lin_coeff,neg_power_coeff, mean, neg_power))
#norm_ss = ROOT.RooFormulaVar("norm_same_sign", "norm_same_sign", "@1 - @2*(@0-@4) - @3/((@0-@4)**@5)", ROOT.RooArgList(args_norm))

#norm_ss = ROOT.RooFormulaVar("norm_same_sign", "norm_same_sign", formula_str_final, ROOT.RooArgList(args_norm))

args_norm, formula_norm = get_formula_par_vals(width, "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/0612_norm_vs_width_with_errors_same_sign_root_original.txt")
print("args_norm outside get_formula_par_vals",args_norm)
print("formula_norm outside get_formula_par_vals",formula_norm)
norm_ss = ROOT.RooFormulaVar("norm_same_sign", "norm_same_sign", formula_norm, ROOT.RooArgList(args_norm))
norm_ss.plotOn(plot_vs_width)
plot_vs_width.Draw()
c1.Update()
c1.Draw()
c1.SaveAs("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/norm_vs_width_ss_parametric.png")
#c1.SaveAs("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/norm_vs_width_ss_parametric_dummy.png")

