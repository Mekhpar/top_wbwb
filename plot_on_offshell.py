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
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import string
import ROOT
from typing import List
import time

import json
import glob
import yaml

def read_key_dict(data:dict, key_list:List[str], pos):
    if isinstance(data, dict): #Try to find keys by looping over, only if it is still a dict
        print()
        cur_key = key_list[pos]
        for key in data.keys():
            print("Current Key name")
            print(key)
            print("Key to be found",cur_key)
            string_start = key.lower().find(cur_key)
            print("string_start",string_start)
            #Cancelled this because while asking for top_same_sign, it was giving both_top_same_sign counts instead
            #if string_start>=0 & string_start <len(key): #This means that the search string is somewhere inside the main string
            if string_start==0: #This means that the search string is at the beginning of the main string
                if (isinstance(data[key], dict)) & (pos < len(key_list)-1):          
                    data[key] = read_key_dict(data[key], key_list,pos+1)
                return data[key]

def read_val_dict(data_internal:dict,key_list:List[str],num_list:List[int],final_key:str):
    key_list_plot = []
    cat_num = []
    key_val_plot = []
    for key in data_internal.keys():
        print("Current Key name")
        print(key)
        for i in range(len(key_list)):
            string_start = key.lower().find(key_list[i])
            print("string_start",string_start)
            #if string_start>=0 & string_start <len(key): #This means that the search string is somewhere inside the main string
            if string_start==0: #This means that the search string is at the beginning of the main string
                print(data_internal[key])

                for key_final in data_internal[key].keys():
                    string_final = key_final.lower().find(final_key)
                    if string_final>=0 & string_final <len(key_final): #This means that the search string is somewhere inside the main string
                        key_list_plot.append(key)
                        cat_num.append(num_list[i])
                        key_val_plot.append(data_internal[key][key_final])

    return key_list_plot,cat_num,key_val_plot

jet_cats = ["j2+_b1"]
#sign_cats = ["opp_sign"]
sign_cats = ["opp_sign","same_sign"]
charge_cats = ["lep_pos","lep_neg"]
mass_window_cats = ["onshell","offshell"]
#eta_cats = ["l_fwd","no_bjet","l_be_sl_be","sl_fwd","l_be_sl_fwd","l_be","l_fwd_sl_fwd","l_fwd_sl_be","sl_be"]

btag_veto_cats = ["b_loose_0","b_loose_1+_no_effect","b_loose_1+_leading"]
btag_veto_labels = [1,2,3]

#for i in range(len(btag_veto_cats)):
#    print()
#    print("btag_veto_cats",i)

fig1, axes1 = plt.subplots(len(sign_cats),len(charge_cats),figsize=(16,16),sharey=False)
fig2, axes2 = plt.subplots(len(sign_cats),len(charge_cats),figsize=(16,16),sharey=False)
fig3, axes3 = plt.subplots(1,len(sign_cats),figsize=(16,8),sharey=False)
odir = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths/"

#with open('/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0218_wbj_1/cutflows.json', 'r') as file:
for k in range(0,len(sign_cats)):
    lep_pos_total = []
    lep_neg_total = []
    for j in range(0,len(charge_cats)): #Loops over lepton charge (charge_cats)
        onshell_counts = []
        offshell_counts = []
        width_num = []
        width_values = []

        width_folders = glob.glob(odir + "width_*/")
        width_weight_file = "/afs/desy.de/user/p/paranjpe/top_wbwb/pepper/inputs/common/width_weights.yaml"
        f = open(width_weight_file)
        width_weights = yaml.safe_load(f)

        for width_val in width_folders:
            print("width_val")
            print(width_val)
            string_start = width_val.find("/width_")
            print("string_start for width_val",string_start)
            if string_start>=0 & string_start <len(width_val):
                width_current = width_val[string_start+7:-1]
                width_num.append(width_current)
                print(width_current)
                width_value = width_weights["WbjToLNu_4f_TuneCP5_13TeV-madgraph-pythia8"]["Width"][width_current]["Value"]
                width_values.append(width_value)
            for i in range(0,len(mass_window_cats)): #Loops over mass_window_cats
                #with open('/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths/width_1/cutflows.json', 'r') as file:
                with open(width_val + 'cutflows.json', 'r') as file:
                    data = json.load(file)

                a = time.time()
                jet_cat_vals_new = read_key_dict(data,["wbj","mu",jet_cats[0],sign_cats[k],charge_cats[j],mass_window_cats[i]],0)
                print("jet_cat_vals_new",jet_cat_vals_new)
                b = time.time()
                key_list, cat_num, val_list = read_val_dict(jet_cat_vals_new,btag_veto_cats,btag_veto_labels,"btagsf")
                c = time.time()

                print("Time taken for read_key_dict",b-a)
                print("Time taken for read_val_dict",c-b)

                print("key_list",key_list)
                print("val_list",val_list)
                print("cat_num",cat_num)
                #Because at the moment we might need the whole thing
                print("mass_window_cats",mass_window_cats[i])
                print("Sum over all btag veto cats",sum(val_list))

                if i==0:
                    onshell_counts.append(sum(val_list))
                elif i==1:
                    offshell_counts.append(sum(val_list))

        print("width nums",width_num)
        print("width_values",width_values)
        print("onshell_counts",onshell_counts)
        print("offshell_counts",offshell_counts)

        index_list_plot = np.argsort(np.array(width_values))
        print("index_list_plot",index_list_plot)
        width_num_plot = np.array(width_num)[index_list_plot]
        width_values_plot = np.array(width_values)[index_list_plot]
        onshell_counts_plot = np.array(onshell_counts)[index_list_plot]
        offshell_counts_plot = np.array(offshell_counts)[index_list_plot]

        on_off_sum_plot = np.add(onshell_counts_plot,offshell_counts_plot)
        if charge_cats[j] == "lep_pos":
            lep_pos_total = on_off_sum_plot
        elif charge_cats[j] == "lep_neg":
            lep_neg_total = on_off_sum_plot

        print("10 GeV value (last element) for the onshell and offshell sum",on_off_sum_plot[-1])
        on_off_sum_plot_percent = (on_off_sum_plot - on_off_sum_plot[-1]) / on_off_sum_plot[-1]
        on_off_ratio_plot = np.array(onshell_counts_plot)/np.array(offshell_counts_plot)

        print("Sign of bjet and lepton charge",sign_cats[k])
        print("Lepton charge",charge_cats[j])
        print("width nums_plot",width_num_plot)
        print("width_values_plot",width_values_plot)
        print("onshell_counts_plot",onshell_counts_plot)
        print("offshell_counts_plot",offshell_counts_plot)
        print("on_off_sum_plot",on_off_sum_plot)
        print(type(on_off_sum_plot))
        print("on_off_ratio_plot",on_off_ratio_plot)
        print(type(on_off_ratio_plot))
        print("on_off_sum_plot_percent",on_off_sum_plot_percent)

        ax1 = axes1[k][j]
        ax2 = axes2[k][j]
        #ax1.xaxis.grid(True)
        ax1.grid(which='both')
        ax1.set_xticks(range(0,int(width_values_plot.max()),int(width_values_plot.max())*2), minor=True)
        ax1.minorticks_on()
        ax1.grid(which='minor', alpha=0.2)
        ax1.set_ylim(0.0,1.0)
        #ax1.set_xticks(range(0,int(width_values_plot.max()),int(width_values_plot.max())*2))
        ax1.scatter(width_values_plot,on_off_ratio_plot,label=sign_cats[k] + '_' + charge_cats[j])
        h,l=ax1.get_legend_handles_labels()
        ax1.legend(handles=h,labels=l,loc='upper right',ncol=2)        
        ax1.set_xlabel(r'Width value set in MC (GeV)')
        ax1.set_ylabel(r'Fraction of onshell events to offshell events (40 GeV mass window)')

        ax2.grid(which='both')
        ax2.set_xticks(range(0,int(width_values_plot.max()),int(width_values_plot.max())*2), minor=True)
        ax2.minorticks_on()
        ax2.grid(which='minor', alpha=0.2)
        ax2.set_ylim(-0.1,0.1)
        #ax1.set_xticks(range(0,int(width_values_plot.max()),int(width_values_plot.max())*2))
        #ax2.scatter(width_values_plot,on_off_sum_plot,label=sign_cats[k] + '_' + charge_cats[j])
        ax2.scatter(width_values_plot,on_off_sum_plot_percent,label=sign_cats[k] + '_' + charge_cats[j])
        h,l=ax2.get_legend_handles_labels()
        ax2.legend(handles=h,labels=l,loc='upper right',ncol=2)        
        ax2.set_xlabel(r'Width value set in MC (GeV)')
        ax2.set_ylabel(r'Sum of onshell and offshell events (40 GeV mass window)')

    print("Sign of bjet and lepton charge",sign_cats[k])
    print("lep_neg_total",lep_neg_total)
    print("lep_pos_total",lep_pos_total)
    lep_neg_pos_ratio_plot = np.array(lep_neg_total)/np.array(lep_pos_total)
    ax3 = axes3[k]

    ax3.grid(which='both')
    ax3.set_xticks(range(0,int(width_values_plot.max()),int(width_values_plot.max())*2), minor=True)
    ax3.minorticks_on()
    ax3.grid(which='minor', alpha=0.2)
    ax3.set_ylim(0.0,1.0)
    #ax1.set_xticks(range(0,int(width_values_plot.max()),int(width_values_plot.max())*2))
    #ax3.scatter(width_values_plot,on_off_sum_plot,label=sign_cats[k] + '_' + charge_cats[j])
    ax3.scatter(width_values_plot,lep_neg_pos_ratio_plot,label=sign_cats[k])
    h,l=ax3.get_legend_handles_labels()
    ax3.legend(handles=h,labels=l,loc='upper right',ncol=2)        
    ax3.set_xlabel(r'Width value set in MC (GeV)')
    ax3.set_ylabel(r'Fraction of lep_neg events to lep_pos events')


#plt.savefig(f'{odir}/On_off_ratio_width.pdf', format='pdf', bbox_inches='tight')
fig1.savefig(f'{odir}/On_off_ratio_width.pdf', format='pdf', bbox_inches='tight')
fig2.savefig(f'{odir}/On_off_sum_width.pdf', format='pdf', bbox_inches='tight')
fig3.savefig(f'{odir}/Lep_neg_pos_ratio.pdf', format='pdf', bbox_inches='tight')
'''
btag_ordered = np.stack((key_list, cat_num), axis=1)
print("btag_ordered")
print(btag_ordered)

#index_list_plot = np.argsort(cat_num, axis=0)
index_list_plot = np.argsort(np.array(cat_num))
print("index_list_plot",index_list_plot)
cat_num_plot = np.array(cat_num)[index_list_plot]
val_list_plot = np.array(val_list)[index_list_plot]

print("cat_num_plot",cat_num_plot)
print("val_list_plot",val_list_plot)

#hahaha

fig = plt.figure(figsize = (7, 7))

# creating the bar plot

#plt.bar(key_list, val_list, color ='blue', width = 0.2)
plt.bar(cat_num_plot, val_list_plot, color ='blue', width = 0.2)
#plt.ylim(bottom=100)
plt.xlabel("Categories of events based on extra loose bjets")
plt.ylabel("Number of events")
#plt.yscale("log")
#plt.savefig('/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0218_wbj_1/'+charge[0]+ '_'+top_and_mom_cats[0]+'.pdf')
plt.savefig('/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths/width_1/' + jet_cats[0] + '_' + sign_cats[0] + '_' + charge_cats[0] + '_' + mass_window_cats[0] +'.pdf')
'''
