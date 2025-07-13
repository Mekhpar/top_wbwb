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

odir = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/width_"

def float_flag_calc(word:str):
    try:
        float(word)
        float_flag = True
    except ValueError:
        float_flag = False

    return float_flag

def grid_tick_plot(xlims, ylims, xtick_lims, ytick_lims, x_major_grid_lines:List[str], y_major_grid_lines:List[str],
    x_minor_grid_lines:List[str], y_minor_grid_lines:List[str], major_tick_pars:List[int], minor_tick_pars:List[int],
    tick_label_type):

    print("xlims in grid tick plot",xlims[0],xlims[1])
    ax.set_xlim(xlims[0],xlims[1])
    ax.set_ylim(ylims[0],ylims[1])

    ax.set_xticks(np.arange(xtick_lims[0],xtick_lims[1],xtick_lims[2]), minor=True)
    ax.set_yticks(np.arange(ytick_lims[0],ytick_lims[1],ytick_lims[2]), minor=True)

    axes.xaxis.grid(which = x_major_grid_lines[0],color = x_major_grid_lines[1], linestyle=x_major_grid_lines[2])
    axes.yaxis.grid(which = y_major_grid_lines[0],color = y_major_grid_lines[1], linestyle=y_major_grid_lines[2])

    axes.xaxis.grid(which = x_minor_grid_lines[0],color = x_minor_grid_lines[1], linestyle=x_minor_grid_lines[2])
    axes.yaxis.grid(which = y_minor_grid_lines[0],color = y_minor_grid_lines[1], linestyle=y_minor_grid_lines[2])

    ax.tick_params(which='major', length=major_tick_pars[0])
    ax.tick_params(which='major', labelsize=major_tick_pars[1])

    ax.tick_params(which='minor', length=minor_tick_pars[0])
    ax.tick_params(which='minor', labelsize=minor_tick_pars[1])

    ax.xaxis.set_minor_formatter(tick_label_type[0])
    ax.yaxis.set_minor_formatter(tick_label_type[1])

def fcn_parse(lines_chi2, par_num):
    #Getting the rest of fit parameters for that particular power index
    fcn_lines = 0
    print("Type of lines_chi2",type(lines_chi2))
    fcn_list = []
    for line_num in range(len(lines_chi2)):
        lines = lines_chi2[line_num]
        #print("FCN line",lines)
        #print(lines.find("FCN"))
        fcn_dummy_list = []

        if lines.find("FCN") == 1:
            '''
            print("FCN line",lines)
            print("Words in FCN lines",lines.split())
            print("Status",lines.split()[3])
            '''

            for dummy_lines in range(0,4+par_num):
                fcn_dummy_list.append(lines_chi2[line_num+dummy_lines])

            '''
            if fcn_lines == index_chi2_min:
                print("Index at min chi2",fcn_lines)
                print("Status of fit at min chi2",lines.split()[3])
            '''
            fcn_lines+=1

            #print('fcn_dummy_list',fcn_dummy_list)
            fcn_list.append(fcn_dummy_list)

    #print("fcn_lines",fcn_lines)
    #print("fcn_list",fcn_list)

    return fcn_list #Probably could do this with some sort of grouping instead of appending but this is not the whole content of the file

def power_select(fcn_list, index_chi2):
    if index_chi2 > 0: #Length of array - 1
        print("Fits for all entries have failed")
        return power_select(fcn_list, 0) #If all fits have failed, might as well return the minimum chi2 one

    print("power_select",fcn_list[index_chi2])
    status_string = fcn_list[index_chi2][0].split()[3]
    print("Status of fit at min chi2 in power_select function",status_string)
    if (status_string!="STATUS=CONVERGED"):
        print("index_chi2",index_chi2)
        print("Chi 2 fit configuration has not worked")
        #fcn_list, index_chi2 = power_select(fcn_list, index_chi2+1)
        index_chi2 = power_select(fcn_list, index_chi2+1)

    else:
        print("index_chi2",index_chi2)
        print("Chi 2 fit configuration has converged")
        '''
        print(fcn_list[index_chi2])
        print("Type of fcn_list",type(fcn_list))
        print("index_chi2",index_chi2)
        '''
        #return fcn_list, index_chi2

    return index_chi2


chi2_min_values = []
'''
norm_values = []
sigma_values = []
alpha_values = []
power_index_min_values = []
'''
power_min_values = []
width_values = []

fit_par_values = []
fit_par_error_values = []
par_names = []
par_nums = 5
for parameters in range(par_nums):
    par_vals = []
    par_errors = []
    fit_par_values.append(par_vals)
    fit_par_error_values.append(par_errors)

start_width = 1
end_width = 22
#for i in range(14,15):

width_weight_file = "/afs/desy.de/user/p/paranjpe/top_wbwb/pepper/inputs/common/width_weights.yaml"
f = open(width_weight_file)
width_weights = yaml.safe_load(f)
dsname = "WbjToLNu_4f_TuneCP5_13TeV-madgraph-pythia8"

for i in range(start_width,end_width):
    line_number = 0
    line_ch = 0    
    chi_2_file = open(odir + str(i) + "/20GeV_binsize_0605_signal_crystalball_new_j2_b1_net_mass_os_width_" + str(i) + ".txt", "r")
    lines_chi2 = chi_2_file.readlines()
    #print(len(lines_chi2))

    
    #Could have created a list in the script itself but this is ok too
    power_values = []
    chi2_values = []

    for lines in lines_chi2:
        line_number+=1
        #print(lines)
        #hahaha
        chi_2_flag = 0
        for words in range(len(lines.split())):
            current_word = lines.split()[words]
            #print("Current word",current_word)
            if words < len(lines.split())-1:
                next_word = lines.split()[words+1]
                #print("Next word",next_word)
            
                if (current_word == "Total") & (next_word == "chi_2"):
                    print("Line with chi 2",lines)
                    chi_2_flag = 1

        if chi_2_flag == 1:
            print("Chi 2 line outside word loop",lines)
            prev_line = lines_chi2[line_number-2]
            print("Previous line (might contain power)")
            #print(lines_chi2[line_number])
            #print(lines_chi2[line_number-1])
            print(prev_line)

            chi_2_current = 0
            power = -1 #Because obviously this power value is always non negative
            for words in lines.split():
                #print("Words",words)
                float_flag = float_flag_calc(words)
                #print("Float flag",float_flag)
                if float_flag == True:
                    chi_2_current = float(words)

                    #chi2_values.append(chi_2_current)
                    chi2_min_values.append(chi_2_current) #Since there is only one value put it directly in the width chi squared list

    '''
            if prev_line.find("Power") == 0:
                for words in prev_line.split():
                    #print("Words",words)
                    float_flag = float_flag_calc(words)
                    #print("Float flag",float_flag)
                    if float_flag == True:
                        power = float(words)
                        power_values.append(power)

            print()
            print("Chi 2 value for all lines",chi_2_current)
            print("Power value for all lines",power)
                    
    print()
    step_value = 0.1
    power_values.insert(0,power_values[0]-step_value) #Original power when every parameter is left sort of free
    print("Chi 2 values for all powers",chi2_values)
    print(len(chi2_values))
    print("Power values for all powers", power_values)
    print(len(power_values))
    #print("Line containing injected channels", line_ch)

    fig, axes = plt.subplots(1,1,figsize=(10,10),sharey=False)
    axes.set_ylabel("Chi squared value")
    axes.set_xlabel("Power value fixed for each step in the fit")
    #axes.xaxis.grid(True)    

    ax = axes
    ax.grid(which='both')
    x_leeway = 0.1
    y_leeway = 5

    grid_tick_plot([power_values[0]-x_leeway,power_values[-1]+x_leeway],    #xlims
    [min(chi2_values)-y_leeway,max(chi2_values)+y_leeway],                  #ylims
    [power_values[0],power_values[-1],0.5],                                 #xtick lims and spacing
    [min(chi2_values),max(chi2_values),50],                                 #ytick lims and spacing
    ['major','r','--'],                                                     #x major grid line attributes (which axis, color and linestyle)
    ['major','r','--'],                                                     #y major grid line attributes (which axis, color and linestyle)
    ['minor','k','-'],                                                      #x minor grid line attributes (which axis, color and linestyle)
    ['minor','k','-'],                                                      #y minor grid line attributes (which axis, color and linestyle)
    [4,0],                                                                  #Major axes ticks - length and label size
    [7,10],                                                                 #Minor axes ticks - length and label size
    ["{x:.2f}","{x:.0f}"]                                                   #Format for x and y minor tick labels
    )

    print("Applied grid settings for the plot just before plotting")
    index_chi2_min = np.argmin(np.array(chi2_values))
    print("Index of arg_chi2_min",index_chi2_min)

    print("arg_chi2_min",chi2_values[index_chi2_min]) #So technically this is also the power index
    print("arg_power_chi2_min",power_values[index_chi2_min])

    chi2_min_values.append(chi2_values[index_chi2_min])
    power_min_values.append(power_values[index_chi2_min])
    '''

    #fcn_list = fcn_parse(lines_chi2, index_chi2_min, 5)
    #fcn_list = fcn_parse(lines_chi2, 5)
    fcn_list = fcn_parse(lines_chi2, par_nums)
    print("fcn_list after calling the function")
    print(fcn_list)

    '''
    #Maybe sorting in order of increasing chi^2 would help before selecting the power by recursion
    sorted_fcn_list = sorted(zip(chi2_values, fcn_list))
    sorted_power_index = sorted(zip(chi2_values, power_values))
    

    print("sorted_power_index",sorted_power_index)
    print("sorted_fcn_list",sorted_fcn_list)
    '''
    #hahaha

    #fcn_list_final = []
    #index_chi2_final = -1
    #fcn_list_final, index_chi2_final = power_select(sorted_fcn_list, 0)
    #sorted_fcn_list, index_chi2_final = power_select(sorted_fcn_list, 0)
    #print(power_select(sorted_fcn_list, 0))
    #index_chi2_final = power_select(sorted_fcn_list, 0)
    index_chi2_final = power_select(fcn_list, 0)
    print()
    print("Width num",i)
    print("index_chi2_final",index_chi2_final)
    #print(sorted_fcn_list[index_chi2_final])
    print(fcn_list[index_chi2_final])

    for pars in range(par_nums):
        parname = fcn_list[index_chi2_final][4+pars].split()[1]
        #print(parname,fcn_list[index_chi2_final][4+pars])
        par_error = fcn_list[index_chi2_final][4+pars].split()[3]

        #'''
        if float_flag_calc(par_error) == True:
            fit_par_error_values[pars].append(float(par_error))
        
        elif float_flag_calc(par_error) == False:
            if par_error == "fixed":
                print("Fixed parameter",parname)
            else:
                print("Errors not able to be calculated") #Actually this condition probably will not appear at all since only taking cases of converged fit
            
            fit_par_error_values[pars].append(0.0)
        #'''
        #print("Only errors populated")
        #print("fit_par_error_values[pars]",fit_par_error_values[pars])
        #hahaha

    for pars in range(par_nums):
        parname = fcn_list[index_chi2_final][4+pars].split()[1]
        par_val = float(fcn_list[index_chi2_final][4+pars].split()[2])
        fit_par_values[pars].append(par_val)

        if i == start_width:
            par_names.append(parname)

        print("Current parameter",parname)
        print("Both errors and values populated")
        print("fit_par_error_values[pars]",fit_par_error_values[pars])
        print("fit_par_values[pars]",fit_par_values[pars])
        print(par_val)

        print(len(fit_par_values[pars]))
    #hahaha

    for pars in range(par_nums):
        print(len(fit_par_values[pars]))
    #hahaha
    width_value = width_weights[dsname]["Width"][str(i)]["Value"]
    width_values.append(float(width_value))

    '''
    ax.scatter(power_values, chi2_values,marker='o', label="Width_"+str(i))
    h,l=ax.get_legend_handles_labels()
    ax.legend(handles=h,labels=l,loc='upper right',ncol=2)
    #plt.setp(visible=True)
    plt.savefig(odir+str(i)+'/Power_steps_chi_2_width_'+str(i)+'.png', format='png', bbox_inches='tight')         
    print("Saved image for width_"+str(i))
    plt.close()
    '''

#y_leeway = []

width_values.reverse()
chi2_min_values.reverse()
for pars in range(par_nums):
    #Reversing the order of lists because it is stored in decreasing order
    fit_par_values[pars].reverse()
    fit_par_error_values[pars].reverse()
    print("Reversed list")
    print(fit_par_values[pars])
    print(fit_par_error_values[pars])
    print(width_values)

#=============================================================================Matplotlib format=======================================================================

'''
for pars in range(par_nums):
    fig, axes = plt.subplots(1,1,figsize=(15,10),sharey=False)
    axes.set_ylabel(par_names[pars] + " values")
    axes.set_xlabel("Width set in MC (in GeV)")
    ax = axes
    ax.grid(which='both')
    x_leeway = 0.5
    #y_leeway = 5
    y_leeway = 0.1*(max(fit_par_values[pars]) - min(fit_par_values[pars]))

    print("Parameter number",pars)
    print(par_names[pars])
    print(width_values)
    print(fit_par_values[pars])
    #ax.scatter(width_values, fit_par_values[pars],marker='o', label="Width_"+str(i))

    print("Values of xlims before entering grid_tick_plot")
    grid_tick_plot([width_values[0]-x_leeway,width_values[-1]+x_leeway],     #xlims
    [min(np.subtract(np.array(fit_par_values[pars]),np.array(fit_par_error_values[pars])))-y_leeway,
    max(np.add(np.array(fit_par_values[pars]),np.array(fit_par_error_values[pars])))+y_leeway], #ylims
    [width_values[0],width_values[-1],0.5],                                  #xtick lims and spacing
    [min(fit_par_values[pars]),max(fit_par_values[pars]),50],                #ytick lims and spacing
    ['major','r','--'],                                                      #x major grid line attributes (which axis, color and linestyle)
    ['major','r','--'],                                                      #y major grid line attributes (which axis, color and linestyle)
    ['minor','k','-'],                                                       #x minor grid line attributes (which axis, color and linestyle)
    ['minor','k','-'],                                                       #y minor grid line attributes (which axis, color and linestyle)
    [7,10],                                                                  #Major axes ticks - length and label size
    [7,10],                                                                  #Minor axes ticks - length and label size
    ["{x:.1f}","{x:.2f}"]                                                    #Format for x and y minor tick labels
    )

    print(len(width_values))
    print(len(fit_par_values[pars]))
    print(len(fit_par_error_values[pars]))

    #plt.scatter(width_values, fit_par_values[pars],marker='o')
    #plt.errorbar(width_values, fit_par_values[pars], yerr=fit_par_error_values[pars],fmt='none')
    ax.scatter(width_values, fit_par_values[pars],marker='o')
    ax.errorbar(width_values, fit_par_values[pars], yerr=fit_par_error_values[pars],fmt='none')
    h,l=ax.get_legend_handles_labels()
    ax.legend(handles=h,labels=l,loc='upper right',ncol=2)
    #plt.setp(visible=True)
    plt.savefig('/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/'+ par_names[pars] +'_vs_width_with_errors_opp_sign.png', format='png', bbox_inches='tight')         
    print("Saved " + par_names[pars])
    plt.close()
'''

#======================================================================================================================================================================

#==========================================================Root format because we kind of need the .C files============================================================

zero_error_list = []
for i in range(start_width,end_width):
    zero_error_list.append(0.0)

ROOT.gROOT.SetBatch(True)

for pars in range(par_nums):

    canvas = ROOT.TCanvas()
    canvas.SetCanvasSize(1600, 1200)
    ROOT.gStyle.SetOptStat(0000)


    widths_total = end_width-start_width
    #par_vs_width = ROOT.TGraphErrors(widths_total, width_values, fit_par_values[pars], zero_error_list, fit_par_error_values[pars])
    par_vs_width = ROOT.TGraphErrors(widths_total, np.array(width_values), np.array(fit_par_values[pars]), np.array(zero_error_list), np.array(fit_par_error_values[pars]))
    '''
    par_vs_width = ROOT.TGraphErrors(widths_total)

    for i in range(0,widths_total):
        print("Number of bins just before drawing and fitting (rebinned)",i)
        par_vs_width.SetPoint(i, width_values[i], fit_par_values[pars][i])
        par_vs_width.SetPointError(i,0.0,fit_par_error_values[pars][i])
        #par_vs_width.SetBinError(i+1,(max(stat_unc[0][i],stat_unc[1][i])))
        #print("Bin content just before drawing and fitting (rebinned)",par_vs_width.GetBinContent(i+1))   
    '''

    par_vs_width.Draw("AP") 
    par_vs_width.SetMarkerStyle(20)
    par_vs_width.SetMarkerSize(1)

    canvas.Print('/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/'+ par_names[pars] +'_vs_width_with_errors_opp_sign_root.C')
    canvas.Print('/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/'+ par_names[pars] +'_vs_width_with_errors_opp_sign_root.png')
    canvas.Close() 

#======================================================================================================================================================================

#=======================================================================Plotting roughly the min chi_squared==========================================================

fig, axes = plt.subplots(1,1,figsize=(10,10),sharey=False)
axes.set_ylabel("Min chi_squared values")
axes.set_xlabel("Width set in MC (in GeV)")
ax = axes
ax.grid(which='both')
x_leeway = 0.5
#y_leeway = 5
y_leeway = 0.1*(max(chi2_min_values) - min(chi2_min_values))

print(width_values)
print(chi2_min_values) #Technically we could have just taken the min, not even sure why this array is theres

#ax.scatter(width_values, chi2_min_values,marker='o', label="Width_"+str(i))

print("Values of xlims before entering grid_tick_plot")
grid_tick_plot([width_values[0]-x_leeway,width_values[-1]+x_leeway],     #xlims
[min(chi2_min_values)-y_leeway,max(chi2_min_values)+y_leeway],           #ylims
[width_values[0],width_values[-1],0.5],                                  #xtick lims and spacing
[min(chi2_min_values),max(chi2_min_values),50],                          #ytick lims and spacing
['major','r','--'],                                                      #x major grid line attributes (which axis, color and linestyle)
['major','r','--'],                                                      #y major grid line attributes (which axis, color and linestyle)
['minor','k','-'],                                                       #x minor grid line attributes (which axis, color and linestyle)
['minor','k','-'],                                                       #y minor grid line attributes (which axis, color and linestyle)
[7,10],                                                                  #Major axes ticks - length and label size
[7,0],                                                                  #Minor axes ticks - length and label size
["{x:.1f}","{x:.2f}"]                                                    #Format for x and y minor tick labels
)

print(len(width_values))
print(len(chi2_min_values))

ax.scatter(width_values, chi2_min_values,marker='o')
#ax.errorbar(width_values, chi2_min_values, yerr=fit_par_error_values[pars],fmt='none')
h,l=ax.get_legend_handles_labels()
ax.legend(handles=h,labels=l,loc='upper right',ncol=2)
#plt.setp(visible=True)
plt.savefig('/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/Chi_squared_vs_width_opp_sign.png', format='png', bbox_inches='tight')         
print("Saved " + par_names[pars])
plt.close()

#======================================================================================================================================================================
