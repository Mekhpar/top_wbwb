import pdb
import numpy as np
import pepper
import awkward as ak
from functools import partial
from copy import copy
import logging
#import tensorflow as tf
import uproot
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

#import testModel_mk
from scipy.special import softmax

from featureDict import featureDict

import json

#with open('/afs/desy.de/user/p/paranjpe/outputs_wbwb/chunk_1113_ttbar_plus_signal/cutflows_og.json', 'r') as file:
with open('/afs/desy.de/user/p/paranjpe/outputs_wbwb/chunk_1122_ttbar_had/cutflows.json', 'r') as file:
    data = json.load(file)

jet_cats = ["j2_b0","j2_b1","j2_b2","j2+_b0","j2+_b1","j2+_b2"]

b_meson_cats = ["no_b_jet"]
meson_cat = ["bneg","b0bar","b0","bpos"]
meson_names = ["B-","B0bar","B0","B+"]
for gen_i in range(len(meson_cat)):
    for rec_i in range(len(meson_cat)):
        string_mask = "id_" + meson_cat[gen_i] + "_rec_" + meson_cat[rec_i]
        b_meson_cats.append(string_mask)

b_meson_cats.append("l_proper_tag")
b_meson_cats.append("l_sl_proper_tag")
print("List of categories",b_meson_cats)

jet_cat_vals_t = [0]*len(jet_cats)
jet_cat_vals_s = [0]*len(jet_cats)
#meson_cat_vals = [0]*len(b_meson_cats)
jet_cat_vals = [0]*len(jet_cats)

#score_cats = ["inconclusive","conclusive"]
threshold_list = [0.3,0.4,0.5,0.6,0.7,0.8]
threshold_cats = ["conclusive_" + str(int(i * 100)) for i in threshold_list]
print("Dict containing percentages corresponding to lower limit probability scores")
print(threshold_cats)

print(data.keys())
#'''
for proc in data.keys():
    #str = "st_t-channel_antitop"
    str = "tttohadronic"
    #str = "ttto2l2nu"
    #str = "st_s-"
    #str = "st_t-channel_top"
    #if proc.lower().find("st_t-")==0 | proc.lower().find("st_s-")==0: #Expect substring to be at the beginning of the string
    if proc.lower().find(str)==0: #Expect substring to be at the beginning of the string
        meson_cat_vals = {}
        meson_cat_vals = dict.fromkeys(b_meson_cats,0) #This is now only for conclusively categorized events
        for key_meson in b_meson_cats:
            meson_cat_vals[key_meson] = dict.fromkeys(threshold_cats,0)

        print("Threshold category dict")
        print(meson_cat_vals)

        print("signal process keys",proc)
        for key_ch in data[proc].keys():
            if key_ch.lower().find("mu")==0:
                for key_cut in data[proc][key_ch].keys():
                    #print("Cut name",key_cut)
                    for i in range(len(jet_cats)):
                        print("Jet category",jet_cats[i])

                        #if key_cut.find(jet_cats[i])==0: #Expect substring to be at the beginning of the string
                        if key_cut==jet_cats[i]: #Expect substring to be at the beginning of the string
                            for key_meson in data[proc][key_ch][key_cut].keys():
                                for j in range(len(b_meson_cats)):
                                    if key_meson==b_meson_cats[j]:
                                        
                                        for key_score in data[proc][key_ch][key_cut][key_meson].keys():
                                            for k in range(len(threshold_cats)):
                                                if key_score==threshold_cats[k]:
                                                    
                                                    meson_cat_vals[key_meson][key_score] += data[proc][key_ch][key_cut][key_meson][key_score]["btagSF"]
                                                    #jet_cat_vals_s[i] += data[proc][key_ch][key_cut][key_meson]["JetPtReq"]
                                                    print("Current value added", data[proc][key_ch][key_cut][key_meson][key_score]["btagSF"], key_ch, key_cut,key_meson,key_score)

        print("Final values of events for signal at each cut",meson_cat_vals)
        print()
#'''
        mistag_rate = np.zeros((len(meson_cat), len(threshold_cats)))
        cor_vals = np.zeros((len(meson_cat), len(threshold_cats)))
        print(mistag_rate)
        threshold_entry = 0

        for key_score in threshold_cats:
            print()
            print("Threshold value",key_score)
            category, label = key_score.split("_")
            fig1, axes1 = plt.subplots(2,2,figsize=(20,20),sharey=False) #This is for overall deviation from expected value for each point (histogram for all data points for all channels)

            for gen_i in range(len(meson_cat)):
                print("Gen meson id",meson_names[gen_i])
                fig, axes = plt.subplots(1,1,figsize=(10,12),sharey=False) #This is for overall deviation from expected value for each point (histogram for all data points for all channels)
                ax = axes
                b_cat = []
                b_vals = []
                print(int(gen_i/2),gen_i%2)
                ax1 = axes1[int(gen_i/2)][gen_i%2]
                for rec_i in range(len(meson_cat)):
                    string_mask = "id_" + meson_cat[gen_i] + "_rec_" + meson_cat[rec_i]
                    b_cat.append(meson_names[rec_i])
                    b_vals.append(meson_cat_vals[string_mask][key_score])

                
                ax1.bar(b_cat, b_vals, edgecolor ='blue', fill=False, width = 0.2, linewidth = 3)
                #plt.bar(b_cat, b_vals, edgecolor ='blue', fill=False, width = 0.2, linewidth = 3)
                ax1.text(1.5,0.95*max(b_vals),meson_names[gen_i],fontsize = 'xx-large')
                #fig.savefig('/afs/desy.de/user/p/paranjpe/top_wbwb/pepper/Plots/' + str + "_" + meson_names[gen_i] + '_rates.png', format='png', bbox_inches='tight') 

            fig1.savefig('/afs/desy.de/user/p/paranjpe/top_wbwb/pepper/Plots/' + str + "_" + 'gen_rates_'+ label + '.png', format='png', bbox_inches='tight') 

            fig2, axes2 = plt.subplots(2,2,figsize=(20,20),sharey=False) #This is for overall deviation from expected value for each point (histogram for all data points for all channels)

            for rec_i in range(len(meson_cat)):
                print("Rec meson id",meson_names[rec_i])
                fig, axes = plt.subplots(1,1,figsize=(10,12),sharey=False) #This is for overall deviation from expected value for each point (histogram for all data points for all channels)
                ax = axes
                b_cat = []
                b_vals = []
                print(int(rec_i/2),rec_i%2)
                ax2 = axes2[int(rec_i/2)][rec_i%2]
                b_right_vals = 0
                b_wrong_vals = 0
                for gen_i in range(len(meson_cat)):
                    string_mask = "id_" + meson_cat[gen_i] + "_rec_" + meson_cat[rec_i]
                    b_cat.append(meson_names[gen_i])
                    b_vals.append(meson_cat_vals[string_mask][key_score])

                    #Calculating mistag efficiency
                    if meson_cat[gen_i] == meson_cat[rec_i]:
                        b_right_vals += meson_cat_vals[string_mask][key_score]
                    elif meson_cat[gen_i] != meson_cat[rec_i]:
                        b_wrong_vals += meson_cat_vals[string_mask][key_score]

                print("Rec category",meson_cat[rec_i])
                print("Correctly tagged",b_right_vals)
                print("Mistagged",b_wrong_vals)
                mistag_vals = b_wrong_vals/(b_wrong_vals + b_right_vals)
                print("Mistag rate",mistag_vals)

                #To be plotted vs values of probability threshold later
                mistag_rate[rec_i][threshold_entry] = mistag_vals
                cor_vals[rec_i][threshold_entry] = b_right_vals
                
                ax2.bar(b_cat, b_vals, edgecolor ='blue', fill=False, width = 0.2, linewidth = 3)
                #plt.bar(b_cat, b_vals, edgecolor ='blue', fill=False, width = 0.2, linewidth = 3)
                ax2.text(1.5,0.95*max(b_vals),meson_names[rec_i],fontsize = 'xx-large')
                #fig.savefig('/afs/desy.de/user/p/paranjpe/top_wbwb/pepper/Plots/' + str + "_" + meson_names[rec_i] + '_reco_rates.png', format='png', bbox_inches='tight') 

            fig2.savefig('/afs/desy.de/user/p/paranjpe/top_wbwb/pepper/Plots/' + str + "_" + 'rec_rates_'+ label + '.png', format='png', bbox_inches='tight') 
            threshold_entry +=1

        print(mistag_rate)
        print(cor_vals)

        fig, axes = plt.subplots(1,2,figsize=(16,9),sharey=False)

        ax=axes[0]
        ax.set_xlabel('Threshold probability value')
        ax.set_ylabel('Mistag rate')
        ax.xaxis.grid(True)

        cmap = matplotlib.colormaps['YlOrBr']
        for rec_cat in range(len(meson_cat)):
            ax.scatter(threshold_list, mistag_rate[rec_cat].tolist(), label=meson_names[rec_cat])
            ax.plot(threshold_list, mistag_rate[rec_cat].tolist())

        h,l=ax.get_legend_handles_labels()
        ax.legend(handles=h,labels=l,loc='upper right')

        ax=axes[1]
        ax.set_xlabel('Threshold probability value')
        ax.set_ylabel('Correctly identified b jets')
        ax.xaxis.grid(True)

        cmap = matplotlib.colormaps['YlOrBr']
        for rec_cat in range(len(meson_cat)):
            ax.scatter(threshold_list, cor_vals[rec_cat].tolist(), label=meson_names[rec_cat])
            ax.plot(threshold_list, cor_vals[rec_cat].tolist())

        h,l=ax.get_legend_handles_labels()
        ax.legend(handles=h,labels=l,loc='upper right')

        fig.savefig('/afs/desy.de/user/p/paranjpe/top_wbwb/pepper/Plots/' + 'mistag_rates.png', format='png', bbox_inches='tight') 
