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

import json

def read_key_dict(data:dict, key_list:List[str], pos):
    if isinstance(data, dict): #Try to find keys by looping over, only if it is still a dict
        cur_key = key_list[pos]
        for key in data.keys():
            string_start = key.lower().find(cur_key)
            if string_start>=0 & string_start <len(key): #This means that the search string is somewhere inside the main string
                if (isinstance(data[key], dict)) & (pos < len(key_list)-1):          
                    data[key] = read_key_dict(data[key], key_list,pos+1)
                return data[key]

def read_val_dict(data_internal:dict,key_list:List[str],final_key:str):
    key_list_plot = []
    key_val_plot = []
    for key in data_internal.keys():
        print("Current Key name")
        print(key)
        for i in range(len(key_list)):
            string_start = key.lower().find(key_list[i])
            print("string_start",string_start)
            if string_start>=0 & string_start <len(key): #This means that the search string is somewhere inside the main string
                print(data_internal[key])

                for key_final in data_internal[key].keys():
                    string_final = key_final.lower().find(final_key)
                    if string_final>=0 & string_final <len(key_final): #This means that the search string is somewhere inside the main string
                        key_list_plot.append(key)
                        key_val_plot.append(data_internal[key][key_final])

    return key_list_plot,key_val_plot

jet_cats = ["j2_b1"]
charge = ["bneg"]
charge_cats = ["id_"+charge[0]+"_rec_"+charge[0]]
score_cats = ["conclusive_50"]
top_and_mom_cats = ["no_top_l_r_lneg"]
#eta_cats = ["l_fwd","no_bjet","l_be_sl_be","sl_fwd","l_be_sl_fwd","l_be","l_fwd_sl_fwd","l_fwd_sl_be","sl_be"]
btag_veto_cats = ["b_loose_0","b_loose_1+_no_effect","b_loose_1+_leading_inconclusive_50","b_loose_1+_leading_conclusive_50_right","b_loose_1+_leading_conclusive_50_wrong"]
btag_veto_labels = [1,2,3,4,5]

#for i in range(len(btag_veto_cats)):
#    print()
#    print("btag_veto_cats",i)

with open('/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0218_wbj_1/cutflows.json', 'r') as file:
    data = json.load(file)

jet_cat_vals_new = read_key_dict(data,["wbj","mu",jet_cats[0],charge_cats[0],score_cats[0],top_and_mom_cats[0]],0)
key_list, val_list = read_val_dict(jet_cat_vals_new,btag_veto_cats,"btagsf")

print("key_list",key_list)
print("val_list",val_list)
hahaha

fig = plt.figure(figsize = (13, 7))

# creating the bar plot

plt.bar(key_list, val_list, color ='blue', width = 0.2)
#plt.ylim(bottom=100)
plt.xlabel("Categories of events based on extra loose bjets")
plt.ylabel("Number of events")
#plt.yscale("log")
plt.savefig('/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0218_wbj_1/'+charge[0]+ '_'+top_and_mom_cats[0]+'.png')
