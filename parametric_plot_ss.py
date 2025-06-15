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

def bulk_comment_flag_calc(line_num, comment_start_points, comment_end_points):
    bulk_comment_flag = 0
    if len(comment_start_points) == len(comment_end_points):
        for comment_num in range(len(comment_start_points)):
            #This means that it is not an individual but a bulk comment
            if (line_num > comment_start_points[comment_num]) & (line_num < comment_end_points[comment_num]): 
                bulk_comment_flag = 1

    return bulk_comment_flag

def formula_c_file(line_num, return_variable, comment_start_points, comment_end_points):
    formula_string = ""
    split_current_line = lines_normc[line_num].split()

    #if lines_normc[line_num].strip().startswith(return_variable): #Where the return variable is defined (hopefully only once) and it is not individually commented
    #Where the return variable is defined (hopefully only once) and it is not individually commented
    if (len(split_current_line) >= 3) & (lines_normc[line_num].strip().startswith("//") == False):
        if ((split_current_line[0] == 'float') | (split_current_line[0] == 'double')) & (split_current_line[1] == return_variable) & (split_current_line[2] == '='): 
            return_bulk_comment_flag = bulk_comment_flag_calc(line_num, comment_start_points, comment_end_points)
            #print("return_bulk_comment_flag",return_bulk_comment_flag)

            if return_bulk_comment_flag == 0:
                formula_string = lines_normc[line_num].strip().replace(split_current_line[0] + " " + split_current_line[1] + " " + split_current_line[2] + " ","")
                formula_string = formula_string[:-1]
                print(return_bulk_comment_flag)
                #print("formula_string in function",formula_string)
    return formula_string

def par_val_calc(line_num, return_variable, comment_start_points, comment_end_points):
    var_str = ""
    var_index = -1
    par_str = "p["
    find_start_par_str = lines_normc[line_num].find(par_str)
    find_end_par_str = lines_normc[line_num].find("]")
    split_current_line = lines_normc[line_num].split()

    #if lines_normc[line_num].strip().startswith(return_variable): #Where the return variable is defined (hopefully only once) and it is not individually commented
    #Where the return variable is defined (hopefully only once) and it is not individually commented
    if (len(split_current_line) >= 3) & (lines_normc[line_num].strip().startswith("//") == False) & (find_start_par_str >= 0) & (find_start_par_str < len(lines_normc[line_num])):
        if ((split_current_line[0] == 'float') | (split_current_line[0] == 'double')) & (split_current_line[2] == '='): 
            return_bulk_comment_flag = bulk_comment_flag_calc(line_num, comment_start_points, comment_end_points)
            #print("return_bulk_comment_flag",return_bulk_comment_flag)

            if return_bulk_comment_flag == 0:
                #var_str = lines_normc[line_num].strip().replace(split_current_line[0] + " " + split_current_line[1] + " " + split_current_line[2] + " ","")
                var_str = split_current_line[1]
                print(return_bulk_comment_flag)
                #print("var_str in function",var_str)
                var_index = lines_normc[line_num][find_start_par_str+2:find_end_par_str] #This is also a string though
                print(var_index) #Again this assumes that there is only one instance each of [ and ]
    return var_str, var_index



norm_fit_file = open("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/0612_norm_vs_width_with_errors_same_sign_root_original.txt", "r")
lines_chi2 = norm_fit_file.readlines()
#print(lines_chi2)

width = ROOT.RooRealVar("width","width", 0.5, 10)
args_norm = [width]

ROOT.gROOT.SetBatch(True)

for line in lines_chi2:
    print("current line")
    print(line)
    if line == '\n':
        pass
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
            args_norm.append(variable)

norm_c_file = open("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/norm_vs_width_with_errors_same_sign_root_original.C", "r")
#norm_c_file = open("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/power_vs_width_with_errors_same_sign_root.C", "r")
lines_normc = norm_c_file.readlines()
string_args = "(double * x, double *p)" #This is the string to search for to indicate the start of the function

function_list = []
function_start_points = []
function_end_points = []
comment_start_points = []
comment_end_points = []
function_use_tf1 = []

for line_num in range(len(lines_normc)):
    print()
    current_line = lines_normc[line_num]
    print("Lines in norm c file",current_line)

    current_line_strip = current_line.strip()

    if current_line_strip == "{":
        function_start_points.append(line_num)

    elif current_line_strip == "}":
        function_end_points.append(line_num)

    if current_line_strip == "/*":
        comment_start_points.append(line_num)

    elif current_line_strip == "*/":
        comment_end_points.append(line_num)

print("function_start_points",function_start_points)
print("function_end_points",function_end_points)
print("comment_start_points",comment_start_points)
print("comment_end_points",comment_end_points)

for line_num in range(len(lines_normc)):
    #print()
    current_line = lines_normc[line_num]
    current_line_strip = current_line.strip()

    #Last condition is checking for whether it is individually commented
    if (current_line.find("new TF1") >= 0) & (current_line.find("new TF1") < len(current_line)) & (current_line_strip.startswith("//")==False):

        bulk_comment_flag = bulk_comment_flag_calc(line_num,comment_start_points,comment_end_points)
        '''
        bulk_comment_flag = 0
        if len(comment_start_points) == len(comment_end_points):
            for comment_num in range(len(comment_start_points)):
                #This means that it is not an individual but a bulk comment
                if (line_num > comment_start_points[comment_num]) & (line_num < comment_end_points[comment_num]): 
                    bulk_comment_flag = 1
        '''
        print("TF1 line",current_line)
        print("Bulk comment flag",bulk_comment_flag)  
        if bulk_comment_flag == 0:
            function_use_tf1.append(line_num)

print("function_use_tf1",function_use_tf1) #With all the comments this should contain only one entry since there is only one successful fit

if len(function_use_tf1) == 1:
    for tf1_inst in function_use_tf1:
        print("tf1 lines",lines_normc[tf1_inst])
        print(lines_normc[tf1_inst].split())
        function_term = lines_normc[tf1_inst].split()[5]
        print(function_term)
        print(function_term[:-1])

        function_def = function_term[:-1] + string_args
        print("function_def")
        print(function_def)
        for i in range(len(function_start_points)):
            parenthesis_line = function_start_points[i]
            find_string_args = lines_normc[parenthesis_line-1].find(function_def)
            print(lines_normc[parenthesis_line-1])
            print("find_string_args",find_string_args)

            return_variable = ""
            if (find_string_args >= 0) & (find_string_args < len(lines_normc[parenthesis_line-1])):
                for line_fcn in range(function_start_points[i]+1,function_end_points[i]): #Checking every line between consecutive opening and closing parentheses
                    print("Line in function call",line_fcn)
                    print(lines_normc[line_fcn])

                    if lines_normc[line_fcn].strip().startswith("return"):
                        return_variable = lines_normc[line_fcn].split()[1][:-1] #Dropping the semicolon

                print("return_variable",return_variable)

                formula_str_final = ""
                var_string_list = []
                var_index_list = []
                for line_fcn in range(function_start_points[i]+1,function_end_points[i]-1): #The last one is for excluding the return line
                    formula_string = formula_c_file(line_fcn, return_variable, comment_start_points, comment_end_points)
                    var_str, var_index = par_val_calc(line_fcn, return_variable, comment_start_points, comment_end_points)
                    print("formula_string",formula_string)
                    print("var_string",var_str)
                    print("var_index",var_index)

                    if (var_str == "") | (var_index == -1):
                        pass
                    else:
                        var_string_list.append(var_str)
                        var_index_list.append(var_index)

                    if formula_string != "":
                        formula_str_final += formula_string
                    elif formula_string == "":
                        pass

                print("var_string_list",var_string_list)
                print("var_index_list",var_index_list)
                print(formula_str_final)
                
                if (formula_str_final.find('x[0]') >=0) & (formula_str_final.find('x[0]') < len(formula_str_final)):
                    formula_str_final = formula_str_final.replace("x[0]", "@0")
                for par_num in range(len(var_index_list)):
                    formula_str_final = formula_str_final.replace(var_string_list[par_num],"@"+str(int(var_index_list[par_num])+1))

                #formula_str_final = formula_str_final.replace("TMath::Exp","ROOT.TMath.Exp")
                #formula_str_final = formula_str_final.replace("TMath::Exp","math.exp")
                formula_str_final = formula_str_final.replace("TMath::Exp","exp")
                print("formula_str_final",formula_str_final)


else:
    print("More or less than one instance of function use found")

#hahaha

c1 = ROOT.TCanvas('c1', 'c1', 700, 500)
plot_vs_width = width.frame()
#norm_ss = ROOT.RooFormulaVar("norm_same_sign", "norm_same_sign", "@1 - @2*(@0-@4) - @3/((@0-@4)**@5)", ROOT.RooArgList(width,shift,lin_coeff,neg_power_coeff, mean, neg_power))
#norm_ss = ROOT.RooFormulaVar("norm_same_sign", "norm_same_sign", "@1 - @2*(@0-@4) - @3/((@0-@4)**@5)", ROOT.RooArgList(args_norm))

norm_ss = ROOT.RooFormulaVar("norm_same_sign", "norm_same_sign", formula_str_final, ROOT.RooArgList(args_norm))
norm_ss.plotOn(plot_vs_width)
plot_vs_width.Draw()
c1.Update()
c1.Draw()
c1.SaveAs("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/norm_vs_width_ss_parametric.png")
#c1.SaveAs("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/norm_vs_width_ss_parametric_dummy.png")

