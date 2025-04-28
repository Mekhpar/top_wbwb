import uproot
import awkward as ak
import pandas as pd
#import math
import time
import glob
import ROOT

wbj_files = glob.glob("/pnfs/desy.de/cms/tier2/store/user/eranken/WbChargeReco/test/mc_2017/WbjToLNu_4f_TuneCP5_13TeV-madgraph-pythia8/WbChargeReco_mc_2017_test/230523_132632/0000/nanox_*.root")
file_num = 0
print("First 3 files",wbj_files[0:4])
#file_3 = wbj_files[0:4]
#df = ROOT.RDataFrame('Events', file_3)
df = ROOT.RDataFrame('Events', wbj_files)
#print(df)
eventCountHandle = df.Count()
print(f"Event count: {eventCountHandle.GetValue()}")

a = time.time()
for i in range(1,22):
    print()
    print("Width number",i)
    sumOfWeightsHandle = df.Sum("LHEWeight_width_"+str(i))
    print(f"Weight sum: {sumOfWeightsHandle.GetValue()}")

b = time.time()
print("Time taken with Rdata fram for all 21 widths",b-a)

'''
tree = uproot.open(wbj_file)["Events"]
print("branches",tree.keys())
a = time.time()
width_1_wghts = tree.arrays(library='ak')["LHEWeight_width_1"]
b = time.time()
print("Time taken with tree arrays and awkward library",b-a)
print("Width 1 (10 GeV) weights", width_1_wghts)
'''
