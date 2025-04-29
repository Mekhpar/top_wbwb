#=================================================================================================================================

#This is going to remove all the chain making parts, so there will be no mother condition or categories made even for MC
#This script will eventually have categories suitable only for data

#=================================================================================================================================


import pdb
import numpy as np
import pepper
import awkward as ak
from functools import partial
from copy import copy
import logging
import tensorflow as tf
import uproot
#import uproot4

import testModel_mk
from scipy.special import softmax
import time

from featureDict import featureDict

import matplotlib.pyplot as plt
import networkx as nx
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
import math

import json
import yaml

#This is for extra branches related to the b charge tagger
#These are jet constituents that are not usually stored in nano aod files

logger = logging.getLogger(__name__)

# All processors inherit from pepper.ProcessorBasicPhysics
class Processor(pepper.ProcessorBasicPhysics):

    config_class = pepper.ConfigBasicPhysics
    '''
    def __init__(self, config, eventdir):

        # Initialize the class, maybe overwrite some config variables and
        # load additional files if needed
        # Need to call parent init to make histograms and such ready
        super().__init__(config, eventdir)

        # It is not recommended to put anything as member variable into a
        # a Processor because the Processor instance is sent as raw bytes
        # between nodes when running on HTCondor.
    '''

    def __init__(self, config, eventdir,outdir):

        # Initialize the class, maybe overwrite some config variables and
        # load additional files if needed
        # Need to call parent init to make histograms and such ready
        super().__init__(config, eventdir,outdir)

        # It is not recommended to put anything as member variable into a
        # a Processor because the Processor instance is sent as raw bytes
        # between nodes when running on HTCondor.

    def process_selection(self, selector, dsname, is_mc, filler):
        # Implement the selection steps

        #Apparently there are 105 weights related to width in total, but printing just a few
        #Turns out that the other 84 are for different masses and we probably do not need that because we are not measuring mass
        '''
        for i in range(1,22):
            print()
            print("Width number outside event loop",i)
            print("Weight values for all events for this width")
            for j in range(len(selector.data)):
                print()
                print("Event number",j)
                print("Width number inside event loop",i)
                print("Weights",selector.data["LHEWeight"][j]["width_" +str(i)])
                #print("Weights",selector.data["LHEWeight_width_" +str(i)][j])
        '''
        width_num = 1
        #width_num = 5
        #width_num = 10
        #width_num = 15
        #width_num = 8
        #width_num = 3
        #width_num = 18
        #width_num = 13
        #width_num = 21
        if is_mc:
            selector.add_cut("Width_"+str(width_num),partial(self.width_reweighting,is_mc,width_num,dsname))
        era = self.get_era(selector.data, is_mc)
        # lumi mask
        if not is_mc:
            selector.add_cut("Lumi", partial(
                self.good_lumimask, is_mc, dsname))
        if is_mc:
            print("Number of events before cross section scaling",len(selector.data))
            selector.add_cut(
                "CrossSection", partial(self.crosssection_scale, dsname))
            print("Number of events after cross section scaling",len(selector.data))

            #hahaha

        if is_mc and "pileup_reweighting" in self.config:
            selector.add_cut("Reweighting", partial(
                self.do_pileup_reweighting, dsname))

        # Only allow events that pass triggers specified in config
        # This also takes into account a trigger order to avoid triggering
        # the same event if it's in two different data datasets.
        pos_triggers, neg_triggers = pepper.misc.get_trigger_paths_for(
            dsname, is_mc, self.config["dataset_trigger_map"],
            self.config["dataset_trigger_order"])
        selector.add_cut("Trigger", partial(
            self.passing_trigger, pos_triggers, neg_triggers))
        print("Number of events after passing trigger",len(selector.data))        

        # lepton columns
        selector.set_multiple_columns(self.pick_electrons)
        selector.set_multiple_columns(self.pick_muons)
        selector.set_column("Muon",partial(self.apply_rochester_corr,selector.rng,is_mc))
        #selector.set_column("Lepton", partial(self.build_lepton_column, is_mc, selector.rng))
        selector.set_column("Lepton", partial(self.build_lepton_column, is_mc, selector.rng,""))
        selector.add_cut("HasLepton", partial(self.require_lepton_nums, is_mc),
                    no_callback=True)

        print("Number of events before OneGoodLep",len(selector.data))

        # Require 1 muon, no veto muons
        selector.add_cut("OneGoodLep", self.one_good_lepton)
        selector.add_cut("NoVetoLeps", self.no_veto_muon)

        print("Number of events after OneGoodLep",len(selector.data))

        #selector.set_cat("channel", {"mu", "el" })
        selector.set_cat("channel", {"mu"})
        selector.set_multiple_columns(self.channel_masks)


        selector.set_column("mTW", self.w_mT)
        '''
        for i in range(len(selector.data["Jet"])):
            if is_mc:
                print()
                print("MC Rec set")
                
                print("Relative event number",i)
                print("Different types of nominal weights")
                print("L1 nominal prefiring weight",selector.data["L1PreFiringWeight"][i].Nom)
                print("L1 ecal nominal prefiring weight",selector.data["L1PreFiringWeight"][i].ECAL_Nom)
                print("L1 mjon nominal prefiring weight",selector.data["L1PreFiringWeight"][i].Muon_Nom)

                #print("LHE Reweighting weight",selector.data["LHEReweightingWeight"][i])

                print("Gen weight for event",selector.data["genWeight"][i])
                print("MC Generator weight for event",selector.data["Generator"][i].weight)
                #print("LHE weight",selector.data["LHEWeight"][i])

                #This is only to compare with previous spew files (of course the event number could also have been printed)
                print("Lepton rec pt",selector.data["Lepton"][i].pt)
                print("Rec transverse mass of W",selector.data["mTW"][i])
        '''

        if "trigger_sf_sl" in self.config and is_mc:
            selector.add_cut("trigSF", self.do_trigger_sf_sl)

        # # gen jet info
        # if is_mc: 
        #     selector.set_column("jetorigin", self.build_jet_geninfo)
        # selector.set_column("OrigJet", selector.data["Jet"])

        #JEC variations 
        if (is_mc and self.config["compute_systematics"]
            and dsname not in self.config["dataset_for_systematics"]):
            if hasattr(filler, "sys_overwrite"):
                assert filler.sys_overwrite is None
            for variarg in self.get_jetmet_variation_args():
                print("doing variation", variarg.name)
                selector_copy = copy(selector)
                filler.sys_overwrite = variarg.name
                self.process_selection_jet_part(selector_copy, is_mc,
                                                variarg, dsname, filler, era)
                if self.eventdir is not None:
                    logger.debug(f"Saving per event info for variation"
                                 f" {variarg.name}")
                    self.save_per_event_info(
                        dsname + "_" + variarg.name, selector_copy, False)
            filler.sys_overwrite = None

        #nominal JEC
        self.process_selection_jet_part(selector, is_mc,
                                self.get_jetmet_nominal_arg(),
                                dsname, filler, era)
        logger.debug("Selection done")


        # apply btag sf, define categories



        # Current end of selection steps. After this, define relevant methods

    #Copied from Matthias' code (unpack.py) and modified to include the selector.data array elements instead of the branches from the nano aod file
    #============================================================================================================================================
    def make1DArray(self,data,ievent,ijet,features):
        arr = np.zeros((len(features),),dtype=np.float32)
        for i,feature in enumerate(features):
            #Since the rdataframe (?) has elements that way, the feature string needs to be split at the first instance of an underscore
            #for example global_numberElectron will be global + numberElectron
            us_pos = feature.find("_")
            #print(us_pos)
            if us_pos != -1:
                const_type = feature[:us_pos]
                attr_const = feature[us_pos+1:]

                arr[i] = data[const_type][ievent][attr_const][ijet]
                #print("Branch names for 1d array (global?)",const_type,attr_const)
                #print(arr[i])

            np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return arr
        
    def make2DArray(self,data,ievent,indices,nmax,features):
        arr = np.zeros((nmax,len(features)),dtype=np.float32)
        for i,feature in enumerate(features):
            #Since the rdataframe (?) has elements that way, the feature string needs to be split at the first instance of an underscore
            #for example global_numberElectron will be global + numberElectron
            us_pos = feature.find("_")
            #print(us_pos)
            if us_pos != -1:
                const_type = feature[:us_pos]
                attr_const = feature[us_pos+1:]

                for j,idx in enumerate(indices[:nmax]): #Here it seems like it is just truncating to the max number of 'constituents' of a particular type allowed
                    arr[j,i] = data[const_type][ievent][attr_const][idx]
                    #print("Branch names for 2d array",const_type,attr_const)
                    #print(arr[j,i])
                np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            
        return arr
    #============================================================================================================================================

    def zip(self,*args):
        #for i in range(len(args)):
            #print(args[i])

        tot = ak.from_numpy(np.array(list(zip(*args))).flatten())
        return tot

    '''
    def create_tensor_file(self,*args):
        for i in range(len(args)):
            print(args[i])

        with uproot.recreate("tensor_list.root") as file:
            #file["jet_attr"] = uproot.newtree(*args)
            file["jet_attr"] = uproot.newtree({"global","cpf",})

        return file
    '''

    def process_selection_jet_part(self, selector, is_mc, variation, dsname,
                                   filler, era):

        a = time.time()
        tfdata = {}
        #idx = selector.data["jetorigin"]["jetIdx"]
        #jetorigin = selector.data["jetorigin"]
        #jetorigin["eta"] = selector.data["Jet"][idx]["eta"]
        #jetorigin["pt"] = selector.data["Jet"][idx]["pt"]
        #print("Fields in reco jets",selector.data["Jet"].fields)
        #print("Fields in gen jets",selector.data["GenJet"].fields)

        #This is the info that will be used for get the scores/probability values from the frozen function (tensorflow)

        #===================================================================================================================================================
        selector.set_column("Jet", partial(self.build_jet_column, is_mc, tfdata, 0))
        print("Used build_jet_column for the first time")
        '''
        for i in range(len(selector.data["Jet"])):
            print("Relative event number",i)
            if i>1:
                break

            print("Btag flag for all the rec jets (?)",ak.to_numpy(selector.data["Jet"][i]["btagged"]))
            print("Global jet pt (?)",ak.to_numpy(selector.data["global"][i]["pt"])) #Just to check whether this actually matches with the jet origin order
            print("Uncut Jet rec pt",ak.to_numpy(selector.data["Jet"][i].pt))
            print("Number of secondary vertices in the reco jet",ak.to_numpy(selector.data["length"][i]["sv"]))
        '''

        #'''
        #hahaha
        #Not using this because it was either flattened the wrong way or the column (event) length is not enough to accommodate all the jets
        #if is_mc:
        #    selector.set_column("Gen_Jet_Flavor", ak.unflatten(jetorigin["hadronFlavor"],1))
        #    selector.set_column("Gen_Hadron_Id", ak.unflatten(jetorigin["bHadronId"],1))
            #reco_flat = ak.flatten(selector.data["Jet"]["btagged"])*5.
            #print("Number of reco jets before cuts",len(reco_flat))
        #    selector.set_column("Reco_Jet_Flavor", ak.unflatten(selector.data["Jet"]["btagged"],1))
            
        #    print("Lengths of the jet number wise unflattened columns")
        #    print(len(selector.data["Gen_Jet_Flavor"]),len(selector.data["Gen_Hadron_Id"]),len(selector.data["Reco_Jet_Flavor"]))

        #    print("Lengths of the actual jet number wise unflattened arrays")
        #    print(len(ak.unflatten(jetorigin["hadronFlavor"],1)),len(ak.unflatten(jetorigin["bHadronId"],1)),len(ak.unflatten(selector.data["Jet"]["btagged"],1)))
        #    for i in range(len(selector.data["Gen_Jet_Flavor"])):
        #        print("Gen flavor array")
        #        print(selector.data["Gen_Jet_Flavor"][i])

        #Flattening, zipping and unflattening the gen hadron flavor and (original) b hadron id and also the reco b tag flag since that will be used for plotting and categorizing for each jet later
        '''
        gen_had_flavor = ak.to_numpy(ak.flatten(jetorigin["hadronFlavor"]))
        gen_bhad_id = ak.to_numpy(ak.flatten(jetorigin["bHadronId"]))
        reco_btag = ak.to_numpy(ak.flatten(selector.data["Jet"]["btagged"])*5.) #This is a boolean array which is converted to b hadron flavor - that is why the 5

        print("Number of entries in the gen and reco id arays",len(gen_had_flavor),len(gen_bhad_id),len(reco_btag))
        rec_gen_bhad = ak.from_numpy(np.array(list(zip(gen_had_flavor,gen_bhad_id,reco_btag))).flatten())
        '''
        #Putting back only for trying to replicate the error
        #flag_jets = ak.unflatten(rec_gen_bhad, 3)
        #print("Flag jets added back again")
        #Probably have to include it in selector.data 

        #===================================================================================================================================================
        #This is a list of lists of split tensors, so not exactly much of an improvement over the original since it cannot be converted into an awkward array of split tensors 
        tfdata_split = {} 
        #'''
        #with uproot.recreate("tensor_list.root") as file:

        for name,featureGroup in featureDict.items():
            feature_loop_start_time = time.time()
            #print("Feature group",name)
            #if name == 'cpf':

            attr_array = []
            type_array = []
            count = 0
            for feature in featureGroup['branches']:
                count+=1
                #if count>2: #Only 2 features from each feature group for speeding up processing time during debugging
                #    break
                #print("feature name", feature)
                us_pos = feature.find("_")
                #print(us_pos)
                if us_pos != -1:
                    const_type = feature[:us_pos]
                    attr_const = feature[us_pos+1:]
                    type_array.append(const_type)
                    attr_array.append(attr_const)

            #print(type_array)
            #print(attr_array)
            args = []
            #print("Length of type_array",len(type_array))
            #print("Length of attr_array",len(attr_array))
            if len(type_array) == len(attr_array):
                #print("Reached if condition")
                for i in range(len(type_array)):
                    #print(selector.data[type_array[i]][attr_array[i]])
                    arr = ak.to_numpy(ak.flatten(selector.data[type_array[i]][attr_array[i]]))
                    if len(arr) == 0:
                        #Insert dummy array here
                        #arr_dummy = [[None]] * len(selector.data)

                        #Making length equal to number of jets rather than number of events
                        #arr_dummy = [[0.]] * len(selector.data)
                        #print("Number of events in case of zero arr length",len(selector.data["Jet"]))
                        if len(selector.data["Jet"]) > 0:
                            arr_dummy = [[0.]] * len(ak.flatten(selector.data["Jet"]["pt"]))
                        elif len(selector.data["Jet"]) == 0:
                            #This is full dummy - obviously there is no jet but we will add one
                            arr_dummy = [[0.]]
                        #print("arr_dummy",arr_dummy)
                        arr_flat = ak.flatten(arr_dummy)
                        #print("arr_flat",arr_flat)
                        arr = ak.to_numpy(arr_flat)
                        
                    #print("Feature name",attr_array[i],"flattened array",arr)
                    #print("Length",attr_array[i],"flattened array",len(arr))

                    #data_array.append(arr)
                    args.append(arr)
            
            #print(args)
            #print(len(args))

            array_zipped = self.zip(*args)
            #print("Zipped array from args",array_zipped)
            #print("Length of zipped array",len(array_zipped))
            feat_num = int(len(array_zipped)/len(arr))
            #print("Number of feature names in the group",feat_num)

            array_constituent = ak.unflatten(array_zipped, feat_num)
            #print("Unflattened array", array_constituent)

            if 'max' in featureGroup.keys():
                len_us_pos = featureGroup["length"].find("_")
                #print(len_us_pos)
                len_type = featureGroup["length"][:len_us_pos]
                len_attr = featureGroup["length"][len_us_pos+1:]
                #offset = selector.data[len_type][i][len_attr + "_offset"][j] #Not using this since this is only calculated wrt the starting jet for each event

                jet_length_constituent = ak.flatten(selector.data[len_type][len_attr])

                #This is for arrays that might turn up empty in a particular chunk
                #if jet_length_constituent == [0] * len(ak.flatten(selector.data["Jet"]["pt"])):
                if ak.any(jet_length_constituent) > 0:
                    pass
                else:
                    #print("Empty - no jet constituents anywhere in the array")
                    if len(selector.data["Jet"]) > 0:
                        jet_length_constituent = ak.Array([1] * len(ak.flatten(selector.data["Jet"]["pt"])))

                    elif len(selector.data["Jet"]) == 0:
                        #This is full dummy - obviously there is no jet but we will add one
                        jet_length_constituent = ak.Array([1])

                #jet_offset_constituent = ak.from_numpy(np.zeros((len(jet_length_constituent))))
                #print(jet_offset_constituent)
                #print("Indices",ak.local_index(jet_length_constituent))
                #jet_offset_constituent = ak.where(jet_length_constituent,ak.sum(jet_length_constituent[:ak.local_index(jet_length_constituent)]))
                #print("Lengths i.e. number of constituents in each jet",ak.to_numpy(jet_length_constituent))
                #print("Number of jets (?)",len(jet_length_constituent))

                array_jet = ak.unflatten(array_zipped, feat_num*jet_length_constituent)
                #print("Unflattened array with one line for each jet",array_jet)
                #print("Number of jets (?)",ak.num(array_jet,axis=0))
                #print("Number of constituents",ak.num(array_jet,axis=1)/feat_num)
                
                
                #for i in range(len(array_jet)):
                #    print(array_jet[i])
                #    print("Number of constituents",len(array_jet[i])/feat_num)
                
                #print("Max number of constituents allowed",featureGroup["max"])
                array_padded_jet = ak.pad_none(array_jet, feat_num*featureGroup["max"], clip=True)

                array_padded_jet_zeros = ak.fill_none(array_padded_jet, 0.)
                #default_array = np.zeros((ak.num(array_padded_jet,axis=0),feat_num*featureGroup["max"]),dtype=np.float32)
                #array_padded_jet_zeros = ak.where(ak.is_none(array_padded_jet), default_array, array_padded_jet)

                #print("Padded array with one line for each jet",array_padded_jet_zeros)
                #print("Number of jets (?)",ak.num(array_padded_jet_zeros,axis=0))
                #print("Number of padded/truncated constituents",ak.num(array_padded_jet_zeros,axis=1)/feat_num)

                
                #for i in range(len(array_padded_jet_zeros)):
                #    print(array_padded_jet_zeros[i])
                #    print("Number of constituents",ak.num(array_padded_jet_zeros,axis=1)/feat_num)
                
                #tfdata[name] = array_padded_jet_zeros
                #print("Original Depth (?)",array_padded_jet_zeros.layout.minmax_depth)

                jet_unfl = ak.unflatten(array_padded_jet_zeros, feat_num, axis=1)

                tfdata[name] = jet_unfl #This will take the shaping into account right here rather than inside the jet loop
                #print("Unflattened array along axis one",jet_unfl)
                #print("Length along axis zero",ak.num(jet_unfl,axis=0))
                #print("Length along axis one",ak.num(jet_unfl,axis=1))
                #print("Depth (?)",jet_unfl.layout.minmax_depth)

                jet_unfl_np = ak.to_numpy(jet_unfl)
                #print(jet_unfl_np)
                #print("Shape of numpy array", jet_unfl_np.shape)
                '''
                for i in range(ak.num(jet_unfl,axis=0)):
                    print(jet_unfl[i])
                    print(ak.num(jet_unfl[i],axis=0))
                    print(ak.num(jet_unfl[i],axis=1))
                '''
            else:
                tfdata[name] = array_constituent
            '''
            print()
            print("tfdata")
            print(name)
            print(tfdata[name])
            print(tfdata[name].layout.minmax_depth)
            print(ak.num(tfdata[name],axis=0))
            print(ak.num(tfdata[name],axis=1))
            '''
        #'''
            #print("Just before tfdata",ak.num(selector.data["Jet"],axis=1))
            #print(name,tfdata[name])
            #if name == "global":
            #    print(name,ak.to_numpy(tfdata[name]))
            
            if len(selector.data["Jet"]) > 0:
                tfdata[name] = ak.unflatten(tfdata[name],ak.num(selector.data["Jet"],axis=1))
            elif len(selector.data["Jet"]) == 0:
                tfdata[name] = ak.unflatten(tfdata[name],1)
            #print("Just after tfdata")
            #print(name,tfdata[name])
            '''
            print("Unflattened tf data")
            print(name)
            print(tfdata[name])
            print(tfdata[name].layout.minmax_depth)
            print(ak.num(tfdata[name],axis=0))
            print(ak.num(tfdata[name],axis=1))
            print(ak.num(tfdata[name],axis=2))
            '''
            feature_loop_inter_time = time.time()

            #This is somehow adding a lot more time, especially for the feature group cpf
            '''
            jet_unfl_tf = tf.convert_to_tensor(tfdata[name],dtype=tf.float32)
            #Trying to convert from regular eager tensor rather than from ragged tensor
            #tf.cast(jet_unfl_tf, tf.float32)

            tfdata_split[name] = tf.split(jet_unfl_tf, num_or_size_splits=jet_unfl_tf.shape[0], axis=0) 
            print()
            print("Name of feature group just after converting to tensor",name)
            print("Whole tensor in float form")
            print(jet_unfl_tf)

            print("Split tensor")
            print(tfdata_split[name])
            print("Type of split tensor",type(tfdata_split[name]))
            '''

            #This is not working
            '''
            #Converting the list of split tensors into awkward array of split tensors
            jet_unfl_split_ak = ak.Array(tfdata_split[name])
            print("Awkward array of split tensors")
            print(jet_unfl_split_ak)
            print("Type of ak array of split tensors",type(jet_unfl_split_ak))
            '''
            #feature_loop_end_time = time.time()
            #print("Feature group for calculating time taken",name)
            #print("feature_loop_start_time",feature_loop_start_time)
            #print("feature_loop_inter_time",feature_loop_inter_time)
            #print("feature_loop_end_time",feature_loop_end_time)
            #print("'Original amount of time",feature_loop_inter_time - feature_loop_start_time)
            #print("Time for tensor conversion and splitting",feature_loop_end_time - feature_loop_inter_time)

        b = time.time() #This is the time it takes to populate all the feature group arrays

        #===========================Copied from testModel_mk.py in nfs dust space========================================
        with tf.io.gfile.GFile("/afs/desy.de/user/p/paranjpe/top_wbwb/pepper/frozenModel.pb", "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            loaded = graph_def.ParseFromString(f.read())

        #Guess this should be retained since it is a different order of feature groups
        featureGroups = ["cpf","cpf_charge","muon","muon_charge","electron","electron_charge","npf","sv","global"]

        # Wrap frozen graph to ConcreteFunctions
        frozen_func = testModel_mk.wrap_frozen_graph(
            graph_def=graph_def,
            inputs=[x+":0" for x in featureGroups],
            outputs=["prediction:0"],
            print_graph=True
        )

        #==================================================================================================================
                
        
        '''
        for i in range(len(prob_jet)):
            print("Jet number",i)
            print("Probability values",prob_jet[i])
            print("Reco flag",rec_bflag_jet[i])
            #print(len(rec_meson_cat[i]))
            print("Category assigned",rec_meson_cat[i])
        '''
        """Part of the selection that needs to be repeated for
        every systematic variation done for the jet energy correction,
        resultion and for MET"""
        reapply_jec = ("reapply_jec" in self.config and self.config["reapply_jec"])
        selector.set_multiple_columns(partial(
            self.compute_jet_factors, is_mc, reapply_jec, variation.junc,
            variation.jer, selector.rng))
        selector.set_column("OrigJet", selector.data["Jet"])

        #selector.set_column("Jet", partial(self.build_jet_column, is_mc, 1))

        #jet, meson_tag_spew = self.build_jet_column(is_mc, 1, selector.data)
        #selector.data["tfdata"] = tfdata - do not use this, gives an extra layer of depth
        '''
        for feature_group in selector.data["tfdata"].fields:
            print()
            print()
            print("Checking correct assignment of tfdata to selector.data")
            print(feature_group)
            print(tfdata[feature_group])
            print(tfdata[feature_group].layout.minmax_depth)
            print(ak.num(tfdata[feature_group],axis=0))
            print(ak.num(tfdata[feature_group],axis=1))
            print(ak.num(tfdata[feature_group],axis=2))
            print()
            print(selector.data["tfdata"][feature_group])
            print(selector.data["tfdata"][feature_group].layout.minmax_depth)
            print(ak.num(selector.data["tfdata"][feature_group],axis=0))
            print(ak.num(selector.data["tfdata"][feature_group],axis=1))
            print(ak.num(selector.data["tfdata"][feature_group],axis=2))

        print("Type of tfdata",type(selector.data["tfdata"]))

        hahaha
        '''
        #print("Type of tfdata before jet cuts",type(tfdata))
        #for feature_group in tfdata.keys():
        #    print(feature_group)
        #    print("Type of tfdata feature group before cuts",type(tfdata[feature_group]))
        print("Number of jets before build_reco_jet_column",len(selector.data["Jet"]))
        #It is a little confusing - do we just use 'not is_mc' for data?
        selector.data["Jet"], tfdata = self.build_reco_jet_column(is_mc, tfdata, 1, selector.data)
        print("Number of jets after build_reco_jet_column",len(selector.data["Jet"]))
        b_jet_cut = time.time()
        #selector.data["Jet"], selector.data["b_meson_tag"],selector.data["b_gen_meson"], selector.data["prob_neg"], selector.data["prob_zerobar"], selector.data["prob_zero"], selector.data["prob_pos"] = self.build_jet_column(is_mc, 1, selector.data)

        #=========================================================================================================================================================================================

        #At least to check whether cuts are consistent
        #Flatten the tfdata again
        #print("Type of tfdata after jet cuts",type(tfdata))
        for feature_group in tfdata.keys():
            #print(feature_group)
            #print("Type of tfdata feature group after cuts",type(tfdata[feature_group]))
            tfdata[feature_group] = ak.flatten(tfdata[feature_group], axis=1)

        if len(selector.data) > 0:
            jet_total = ak.num(tfdata["global"],axis=0)
        elif len(selector.data) == 0:
            jet_total = 0 #Obviously don't want the dummy to appear in here
        print("Total number of jets from all the events after pt and eta cuts and sorting",jet_total)
        flag_final = []
        logit_final = []
        prob_final = []

        #Testing out batch mode suggested by Yi-Mu here
        batch = {}
        #print("inputs to frozen func",frozen_func.inputs)
        for featureGroup in featureGroups:
            #print("Feature name for frozen function (different order)",featureGroup)
            batch[featureGroup] = tf.convert_to_tensor(tfdata[featureGroup],dtype=tf.float32)
            #print("Shape to inputted to function",batch[featureGroup].shape)
        
        #Very dumb method of ensuring that it does not give error for zero events
        if jet_total > 0:
            logit_jet = frozen_func(**batch)
            prob = softmax(logit_jet,axis=2)
            prob_neg = ak.flatten(prob[:,:,0],axis=1)
            prob_zerobar = ak.flatten(prob[:,:,1],axis=1)
            prob_zero = ak.flatten(prob[:,:,2],axis=1)
            prob_pos = ak.flatten(prob[:,:,3],axis=1)
            
            selector.data["prob_neg"] = ak.unflatten(prob_neg,ak.num(selector.data["Jet"]))
            selector.data["prob_zerobar"] = ak.unflatten(prob_zerobar,ak.num(selector.data["Jet"]))
            selector.data["prob_zero"] = ak.unflatten(prob_zero,ak.num(selector.data["Jet"]))
            selector.data["prob_pos"] = ak.unflatten(prob_pos,ak.num(selector.data["Jet"]))

            #print("Probability values in group",prob)
            #print(type(prob))

        elif jet_total == 0:
            #Of course this is not normalized and also it doesn't need to be, based on threshold applied it will mostly land in the inconclusive category
            #and since the cutflow is processed (apparently) for all the chunks, it should not affect those counts either
            #prob = ak.to_numpy(ak.Array([[[0., 0., 0., 0.]]]))
            #prob = ak.to_numpy(ak.Array([[[[],[],[],[]]]]))
            selector.data["prob_neg"] = [[]]*len(selector.data["Jet"])
            selector.data["prob_zerobar"] = [[]]*len(selector.data["Jet"])
            selector.data["prob_zero"] = [[]]*len(selector.data["Jet"])
            selector.data["prob_pos"] = [[]]*len(selector.data["Jet"])

        #print("Number of events")
        #print(ak.num(selector.data["Jet"],axis=0))
        #print("Number of jets (to be replaced with dummy if the actual number is zero)")
        #print(ak.num(selector.data["Jet"],axis=1))

        c = time.time() #This is the time it takes to loop through all the jets and apply the frozen func using the populated arrays
        #print("Time taken to populate the arrays by looping over all the feature groups",b-a)
        #print("Time taken to call the frozen function by looping over all the jets",c-b_jet_cut) #This is just after putting the first pt and sorting cuts, before flattening

        '''
        for i in range(len(selector.data["Jet"])):
            print()
            print("Event number for probability",i)
            print("Jet pt",selector.data["Jet"][i]["pt"])
            #print("B jet pt",selector.data["bjets_tight"][i]["pt"])
            print(selector.data["prob_neg"][i])
            print(selector.data["prob_zerobar"][i])
            print(selector.data["prob_zero"][i])
            print(selector.data["prob_pos"][i])
        '''    
            
        #hahaha

        #hahaha

        #selector.set_column("NN_prob",ak.unflatten((ak.from_numpy(prob_final)),4))
        #Assigning the meson categories here - they will only be assigned to jets having the reco btag flag as 1 (since that will be the case for data as well)
        meson_cat = ["bneg","b0bar","b0","bpos"]
        #prob_jet = ak.unflatten((ak.from_numpy(prob_final)),4)

        '''
        selector.data["prob_neg"] = ak.unflatten(prob_jet[:,0],ak.num(selector.data["Jet"],axis=1))
        selector.data["prob_zerobar"] = ak.unflatten(prob_jet[:,1],ak.num(selector.data["Jet"],axis=1))
        selector.data["prob_zero"] = ak.unflatten(prob_jet[:,2],ak.num(selector.data["Jet"],axis=1))
        selector.data["prob_pos"] = ak.unflatten(prob_jet[:,3],ak.num(selector.data["Jet"],axis=1))
        '''

        if jet_total > 0:
            #print("Prob for rec_meson_cat",ak.to_numpy(ak.flatten(prob,axis=1)))
        
            #rec_meson_cat = ak.argmax(prob_jet,axis=1)
            rec_meson_cat = ak.argmax(ak.flatten(prob,axis=1),axis=1)
            #print("rec_meson_cat",rec_meson_cat)
            #print("Length of category assigned array",len(rec_meson_cat))
            selector.set_column("b_meson_tag", ak.unflatten(rec_meson_cat,ak.num(selector.data["Jet"],axis=1)))
            
        elif jet_total == 0:
            #Obviously since this is not a real index and when putting it in categories actually it will not be classified anywhere
            #But actually we do not need to be worried about that either because this is j0 (actually having zero jets) and they will not appear in j2 or j2+
            if len(selector.data["Jet"]) > 0:
                rec_meson_cat = [-1] * len(selector.data["Jet"]) 
            elif len(selector.data["Jet"]) == 0:
                rec_meson_cat = [-1]
            #print("rec_meson_cat",rec_meson_cat)
            selector.set_column("b_meson_tag", ak.unflatten(rec_meson_cat,len(selector.data["Jet"])))
        #hahaha
        
        
        #b_meson_tag is indeed being used for build_btag_column - March 7 2025
        #print(ak.unflatten(rec_meson_cat,ak.num(selector.data["Jet"],axis=1)))
        #print(selector.data["b_meson_tag"])

        #=========================================================================================================================================================================================

        print("Number of jets before applying JetPUIdSFs (replace if everything is zero)",len(selector.data))
        '''
        for i in range(len(selector.data)):
            print()
            print("Event number before JetPUIdSFs cut",i)
            print("Jet pt",selector.data["Jet"][i].pt)
        '''

        #selector.set_multiple_columns({"Jet","b_meson_tag"},partial(self.build_jet_column, is_mc, 1)) #added meson tag #Wrong syntax

        #==========================================Mar 19 2025 - removing this for debugging to apply a lower bjet pt cut==============================
        
        if "jet_puid_sf" in self.config and is_mc:
            selector.add_cut("JetPUIdSFs", self.jet_puid_sfs)

        #==============================================================================================================================================

        #selector.set_column("Jet", self.jets_with_puid) 
        
        #Again defining a new function, sure that this could be done in a better way
        selector.data["Jet"], selector.data["b_meson_tag"],selector.data["prob_neg"], selector.data["prob_zerobar"], selector.data["prob_zero"], selector.data["prob_pos"] = self.reco_jets_with_puid(selector.data) 
        #selector.set_multiple_columns({"Jet","b_meson_tag"}, self.jets_with_puid) #added meson tag

        smear_met = "smear_met" in self.config and self.config["smear_met"]
        selector.set_column(
            "MET", partial(self.build_met_column, is_mc, variation.junc,
                           variation.jer if smear_met else None, selector.rng,
                           era, variation=variation.met))
        
        #selector.add_cut("JetPtReq", self.jet_pt_requirement)
        print("Number of jets before applying JetPtReq (replace if everything is zero)",len(selector.data))
        '''
        for i in range(len(selector.data)):
            print()
            print("Event number before JetPtReq cut",i)
            print("Jet pt",selector.data["Jet"][i].pt)
        '''
        selector.add_cut("JetPtReq", partial(self.jet_pt_requirement,"Jet"))

        print("Number of events after JetPtReq cut",len(selector.data))


        #March 8 - this is where 6 for the last chunk in wjets 0j 331.root becomes zero
        #print("Number of jets after applying JetPtReq (replace if everything is zero)",len(selector.data["Jet"]))
        if is_mc and self.config["compute_systematics"]:
            self.scale_systematics_for_btag(selector, variation, dsname)
        
        #selector.set_column("nbtag", self.num_btags)
        selector.set_column("nbtag", partial(self.num_btags,""))
        #selector.set_column("bjets_tight", partial(self.build_btag_column,is_mc))  
        #selector.set_multiple_columns({"bjets_tight","meson_tag_tight"}, partial(self.build_btag_column,is_mc))  #added meson tag

        #=================Probably have to be changed due to the sorting according to gen pt required in wbwb_reduced_gen_chains.py=====================================
        selector.data["flag_tight"], selector.data["btag_tight"], selector.data["bjets_tight"], selector.data["meson_tag_tight"], selector.data["prob_neg_tight"], selector.data["prob_zerobar_tight"], selector.data["prob_zero_tight"], selector.data["prob_pos_tight"] = self.build_btag_reco_column(is_mc,"btagged", "tight", selector.data)
        selector.data["flag_loose"], selector.data["btag_loose"], selector.data["bjets_loose"], selector.data["meson_tag_loose"], selector.data["prob_neg_loose"], selector.data["prob_zerobar_loose"], selector.data["prob_zero_loose"], selector.data["prob_pos_loose"] = self.build_btag_reco_column(is_mc,"btagged_loose", "loose", selector.data)
        #================================================================================================================================================================
        
        '''
        for i in range(len(selector.data["Jet"])):
            print()
            print("Event number",i)
            print("Jet pt",selector.data["Jet"][i]["pt"])
            print("Btag flavor overall scores",selector.data["Jet"][i]["btag"])
            print("Tight jets")
            print("Bjet meson tag reshaped",selector.data["meson_tag_tight"][i])
            print("Btag flavor scores",selector.data["btag_tight"][i])
            print("Real flag tight",selector.data["flag_tight"][i])
            print("Bjet tight pt",selector.data["bjets_tight"][i]["pt"])
            print("Real Probability values")
            print("B-",selector.data["prob_neg_tight"][i])
            print("B0bar",selector.data["prob_zerobar_tight"][i])
            print("B0",selector.data["prob_zero_tight"][i])
            print("B+",selector.data["prob_pos_tight"][i])

            print("Loose (medium) jets")
            print("Bjet meson tag reshaped",selector.data["meson_tag_loose"][i])
            print("Bjet gen meson tag",selector.data["gen_meson_loose"][i])
            print("Btag flavor scores",selector.data["btag_loose"][i])
            print("Bjet tight pt",selector.data["bjets_loose"][i]["pt"])
            print("Real flag loose",selector.data["flag_loose"][i])

            print("Real loose Probability values")
            print("B-",selector.data["prob_neg_loose"][i])
            print("B0bar",selector.data["prob_zerobar_loose"][i])
            print("B0",selector.data["prob_zero_loose"][i])
            print("B+",selector.data["prob_pos_loose"][i])

            print("Number of tight jets",len(selector.data["bjets_tight"][i]))
            print("Number of loose (medium) jets",len(selector.data["bjets_loose"][i]))

        hahaha
        '''

        selector.set_cat("jet_btag", {"j2_b0", "j2_b1","j2_b2","j2+_b0", "j2+_b1","j2+_b2"})
        selector.set_multiple_columns(self.btag_categories)

        #Removed for debugging purposes
        #selector.set_multiple_columns(partial(self.eta_cat,is_mc))
        cat_dict = ["no_b_jet"]
        meson_cat = ["bneg","b0bar","b0","bpos"]
        for gen_i in range(len(meson_cat)):
            for rec_i in range(len(meson_cat)):
                string_mask = "id_" + meson_cat[gen_i] + "_rec_" + meson_cat[rec_i]
                cat_dict.append(string_mask)

        #This is for plotting the scores
        for gen_i in range(len(meson_cat)):
            string_mask = "id_" + meson_cat[gen_i]
            cat_dict.append(string_mask)

        cat_dict.append("l_proper_tag")
        cat_dict.append("l_sl_proper_tag")
        #print("List of categories",cat_dict)
        #selector.set_cat("b_meson_cat",set(cat_dict))

        #==========================================Categories that are going to be replaced with a cut at least temporarily======================================
        '''
        cat_dict = ["id_bneg_rec_bneg","id_bpos_rec_bpos"]
        selector.set_cat("b_meson_cat",set(cat_dict)) #Small number of categories for debugging

        selector.set_multiple_columns(partial(self.charge_tag_cat,is_mc)) #Again, only for mc, because data has no chance of having a gen hadron id

        threshold_list = [0.5] #New temporary threshold that could be roughly near the optimum of the ROC curve
        threshold_dict = []
        for lowlim in threshold_list:
            threshold_dict.append("inconclusive_"+str(int(lowlim*100)))
            threshold_dict.append("conclusive_"+str(int(lowlim*100)))
        selector.set_cat("score_cat", set(threshold_dict))
        selector.set_multiple_columns(partial(self.score_categories,is_mc))
        '''
        #=========================================================================================================================================================

        #'''
        '''
        print("Overall mask values")
        print("no_b_jet",selector.data["no_b_jet"])
        print("Depth for no_b_jet",selector.data["no_b_jet"].layout.minmax_depth)
        print("Number of events for no_b_jet",len(selector.data["no_b_jet"]))
        print(type(selector.data["no_b_jet"]))

        print("id_bneg_rec_bneg",selector.data["id_bneg_rec_bneg"])
        print("Depth for id_bneg_rec_bneg",selector.data["id_bneg_rec_bneg"].layout.minmax_depth)
        print("Number of events for id_bneg_rec_bneg",len(selector.data["id_bneg_rec_bneg"]))
        print(type(selector.data["id_bneg_rec_bneg"]))
        '''

        print("Number of events after charged meson cut",len(selector.data))
        print("Length of bjets_tight",len(selector.data["bjets_tight"]))
        print("Length of Lepton",len(selector.data["Lepton"]))
        selector.set_column("mlb",partial(self.mass_lepton_bjet,"Lepton","bjets_tight"))    
        selector.set_column("dR_lb",partial(self.dR_lepton_bjet,"Lepton","bjets_tight"))    
        selector.set_column("dphi_lb",partial(self.dphi_lepton_bjet,"Lepton","bjets_tight"))    

        #It is probably more useful to have the absolute value of the eta of the leading bjet
        selector.set_column("abs_eta_b",partial(self.eta_leading_bjet,"bjets_tight"))    
        selector.set_column("abs_eta_b2",partial(self.eta_subleading_bjet,"bjets_tight"))
        #selector.set_column("q_net",partial(self.charge_lepton_bjet,"Lepton","bjets_tight"))

        #====================================================This is taken from top_reco_wbj.py and then from wbwb_reduced_gen_chains.py=============================================================

        #Trying out new chain making here - starting from last to first (suggestion by Alberto)
        #because it is much easier to find the mother index rather than compare a mother index (finding the daughters essentially)
        #'''
        index_gen = ak.local_index(selector.data["GenPart"],axis=1)
        abs_pdgid = abs(selector.data["GenPart"]["pdgId"])

        #mask = abs_pdgid == 5
        #mask = abs_pdgid == 24 #Chains are for W now, not b
        mask = (abs_pdgid == 12) | (abs_pdgid == 14) | (abs_pdgid == 16) #Chains are for gen_nu
        part_index = index_gen[mask]
        object = selector.data["GenPart"][mask]
        mother_index = object["genPartIdxMother"] #This is nothing but a set of indices
        #part_index = ak.sort(part_index,axis=1,ascending=True)   

        #Converting to numpy array hopefully to take intersection and complement

        npart_max_pad = ak.max(ak.num(part_index,axis=1))
        #print("Maximum number of parts",npart_max_pad)
        part_ak = ak.pad_none(part_index, npart_max_pad, axis=1)
        mom_ak = ak.pad_none(mother_index, npart_max_pad, axis=1)

        part_dummy = ak.from_numpy(np.full((len(selector.data["GenPart"]), npart_max_pad), -2))#Since -2 cannot be the mother index of anything
        #print(ak.num(part_dummy,axis=0))
        #print(ak.num(part_dummy,axis=1))

        part_ak_non_none = ak.where(ak.is_none(part_ak,axis=1), part_dummy, part_ak)
        mom_ak_non_none = ak.where(ak.is_none(mom_ak,axis=1), part_dummy, mom_ak)

        ngenpart_for_cumulative = ak.to_numpy(ak.num(selector.data["GenPart"],axis=1))
        cumulative_ngenpart = np.cumsum(ngenpart_for_cumulative)
        cumulative_ngenpart_shift = np.roll(cumulative_ngenpart, 1)
        cumulative_ngenpart_shift[0] = 0
        cumulative_ngenpart_ak = ak.from_numpy(cumulative_ngenpart_shift)

        #print("ngenpart_for_cumulative",ngenpart_for_cumulative)
        #print("cumulative_ngenpart_ak",cumulative_ngenpart_ak)

        actual_part_index = self.cumulative_sum(cumulative_ngenpart_ak,part_index)
        actual_mother_index = self.cumulative_sum(cumulative_ngenpart_ak,mother_index)
        actual_last_index = np.setdiff1d(actual_part_index,actual_mother_index)
        #print("Part index after subtraction",actual_last_index)

        actual_index_2d = (np.array([actual_last_index]* len(selector.data["GenPart"])))
        unflatten_flag = ak.mask(ak.from_numpy(actual_index_2d),(actual_index_2d < cumulative_ngenpart[:, None]) & (actual_index_2d >= cumulative_ngenpart_shift[:, None]))
        #
        #index_unflatten_cts = ak.unflatten(ak.count(unflatten_flag,axis=1),1)
        index_unflatten_cts = ak.count(unflatten_flag,axis=1)

        #Very important to mention that these two are the same, but not ideal to print the loops everytime
        #print("Length of actual_last_index",len(actual_last_index))
        #print("total_last_index",total_last_index)
        actual_last_index_unflat = ak.unflatten(actual_last_index,index_unflatten_cts)
        last_index = ak.unflatten(self.cumulative_sum([-1]*cumulative_ngenpart_ak,actual_last_index_unflat),index_unflatten_cts)
        #print("Length of last_index",len(last_index))
        '''
        for i in range(len(selector.data["Jet"])):
            print()
            print("Event number in new chain making",i)
            print(ak.to_numpy(selector.data["GenPart"][i]["pdgId"]))
            print("Part index",part_index[i])
            print("Part mother index",mother_index[i])
            print("Part index padded with none",part_ak[i])
            print("Part mother index padded with none",mom_ak[i])

            print("Part index padded with -2",part_ak_non_none[i])
            print("Intersection for Part mother index padded with -2",mom_ak_non_none[i])
            print("event wise set subtraction (only for comparison)",np.setdiff1d(part_index[i],mother_index[i]))
            print("actual_last_index_unflat",actual_last_index_unflat[i])
            print("last_index",last_index[i])
        '''
        recursion_level = 1
        #last_index_object = [last_index]
        #last_index_object = self.chain_last(recursion_level,last_index_object,selector.data)          
        chain_current = last_index
        counts_chain = [1]*len(ak.flatten(last_index))
        #counts_chain, chain_current, last_index = self.chain_last(5, counts_chain, recursion_level, chain_current, last_index, selector.data) 
        #counts_chain, chain_current, last_index = self.chain_last(24, counts_chain, recursion_level, chain_current, last_index, selector.data) 
        #Neutrino chains
        counts_chain, chain_current, last_index = self.chain_last([12,14,16], counts_chain, recursion_level, chain_current, last_index, selector.data) 
        counts_chain_final = ak.unflatten(counts_chain,ak.num(last_index))
        #'''
        #'''            
        print(chain_current)
        chain_flat_np = ak.to_numpy(ak.flatten(chain_current,axis=1))
        index_chain_flat = ak.to_numpy(ak.local_index(ak.flatten(chain_current,axis=1)))

        #print(ak.flatten(chain_current,axis=1))
        #print(chain_flat_np)
        print(index_chain_flat)
        print(counts_chain)
        chain_flag = index_chain_flat < counts_chain[:, None]
        #hahaha
        #print(chain_flag)
        #The chain_flag gave fully flat array, so first unflatten according to the number of elements in each chain (counts_chain) and then wrt last_index
        chain_final = ak.unflatten(ak.unflatten((ak.flatten(chain_current,axis=1)[chain_flag]),counts_chain),ak.num(last_index))
        print(chain_final)

        selector.data["nu_fin_final"] = chain_final #Only used for sign comparison at gen level and invariant mass at gen level

        selector.data["nu_first"] = selector.data["nu_fin_final"][:,:,-1]
        selector.data["nu_last"] = selector.data["nu_fin_final"][:,:,0]

        nu_first_mother_id = abs(selector.data["GenPart"][selector.data["GenPart"][selector.data["nu_first"]]["genPartIdxMother"]].pdgId)
        selector.data["nu_first_resonant"] = selector.data["nu_first"][nu_first_mother_id == 24]

        selector.data["Gen_nu"], selector.data["Gen_nu_mask"], nu_muon_mass_mask = self.build_genpart_column(is_mc, selector.data["nu_first_resonant"], selector.data)
        selector.data["Gen_nu_pz"] = selector.data["Gen_nu"].pt[:,0]*np.sinh(selector.data["Gen_nu"].eta[:,0])
        #wmother_id = abs(selector.data["GenPart"][selector.data["GenPart"][nu_first]["genPartIdxMother"]].pdgId)

        #print("Time taken with b_recursion method",b_mom_time_after - b_mom_time_before)
        #Changing this to be consistent with the script from top_reco_nochains to avoid the spans error
        #final_array = [[]]*len(selector.data["Jet"])        
        final_array = [-1]*len(selector.data["Jet"])     
        final_array_sl = [-1]*len(selector.data["Jet"])   
        #Trying out Matthias' pz for MET calculation method after the top reco has been done at GenPart (bW and blnu level) and nu and GenMET pt, phi have been compared
        m_w = 80.4 #in units of GeV, might have to make an array out of this
        half_mw_2 = pow(m_w,2)/2
        mw_arr = [[half_mw_2]]*len(selector.data["Jet"])
        zero_discr = [0]*len(selector.data["Jet"])

        print("Length of number of events before entering the top mass reconstruction block",len(selector.data["Jet"]))
        if len(selector.data["Jet"])>0:
            xi_indpt_w_recmet = ak.unflatten(selector.data["Lepton"].pt[:,0]*selector.data["MET"].pt*np.cos(selector.data["MET"].phi-selector.data["Lepton"].phi[:,0]),1)
            xi_recmet_sum = ak.concatenate([xi_indpt_w_recmet,mw_arr],axis=1)
            xi_recmet = ak.sum(xi_recmet_sum,axis=1)

            mt_l_2_rec = (pow(selector.data["Lepton"].pt[:,0],2) + pow(selector.data["Lepton"].mass[:,0],2))
            lep_energy = np.sqrt(pow((selector.data["Lepton"].pt[:,0]*np.cosh(selector.data["Lepton"].eta[:,0])),2) + pow(selector.data["Lepton"].mass[:,0],2))
            lep_pz = selector.data["Lepton"].pt[:,0]*np.sinh(selector.data["Lepton"].eta[:,0])

            #discr_rec = pow(xi_recmet,2) - pow(selector.data["MET"].pt,2)*mt_l_2_rec
            discr_rec = pow(xi_recmet,2) - pow(selector.data["MET"].pt,2)*(pow(selector.data["Lepton"].pt[:,0],2) + pow(selector.data["Lepton"].mass[:,0],2))
            discr_rec_sign = discr_rec/abs(discr_rec)
            selector.data["neg_discr_rec_mask"] = ak.mask(discr_rec,discr_rec<0)
            neg_discr_rec_num = ak.count(selector.data["neg_discr_rec_mask"], axis=0)        
            print("neg_discr_rec_num",neg_discr_rec_num)

            discr_rec_new = ak.where(ak.is_none(selector.data["neg_discr_rec_mask"]), discr_rec, zero_discr) #Populating this before finding the sqrt obviously

            selector.data["min_pz_recmet"] = ((xi_recmet*lep_pz) - (lep_energy*np.sqrt(discr_rec_new)))/mt_l_2_rec
            selector.data["max_pz_recmet"] = ((xi_recmet*lep_pz) + (lep_energy*np.sqrt(discr_rec_new)))/mt_l_2_rec

            selector.data["MET_nu_min"] = self.build_met_eta_column(is_mc, "min_pz_recmet", "MET", selector.data)
            selector.data["MET_nu_max"] = self.build_met_eta_column(is_mc, "max_pz_recmet", "MET", selector.data)

            '''
            selector.data["Rec_bjets"], selector.data["bjet_mask"] = self.build_lead_bjet_column(is_mc, "bjets_tight", selector.data)
            selector.data["b_l_recMET_min_mass"] = (selector.data["Rec_bjets"][:,0] + selector.data["Lepton"][:,0] + selector.data["MET_nu_min"]).mass
            selector.data["b_l_recMET_max_mass"] = (selector.data["Rec_bjets"][:,0] + selector.data["Lepton"][:,0] + selector.data["MET_nu_max"]).mass

            selector.data["b_l_recMET_min_mass_final"] = ak.where(ak.is_none(selector.data["bjet_mask"]), final_array, selector.data["b_l_recMET_min_mass"])
            selector.data["b_l_recMET_max_mass_final"] = ak.where(ak.is_none(selector.data["bjet_mask"]), final_array, selector.data["b_l_recMET_max_mass"]) 
            '''

            #================================================This is the mass for the leading bjets============================================================

            #selector.data["Rec_bjets"], selector.data["bjet_mask"] = self.build_lead_bjet_column(is_mc, "bjets_tight", selector.data)
            #selector.data["Rec_bjets"], selector.data["leading_bjet_mask"], selector.data["subleading_bjet_mask"] = self.build_bjet_column(is_mc, "bjets_tight", selector.data)

            #After the new function definition was added to include options for using matched_gen columns as well
            selector.data["Rec_bjets"], selector.data["leading_bjet_mask"], selector.data["subleading_bjet_mask"] = self.build_bjet_column(is_mc, "bjets_tight", 0, selector.data)

            selector.data["leading_b_l_recMET_min_mass"] = (selector.data["Rec_bjets"][:,0] + selector.data["Lepton"][:,0] + selector.data["MET_nu_min"]).mass
            selector.data["leading_b_l_recMET_max_mass"] = (selector.data["Rec_bjets"][:,0] + selector.data["Lepton"][:,0] + selector.data["MET_nu_max"]).mass

            selector.data["leading_b_l_recMET_min_mass_final"] = ak.where(ak.is_none(selector.data["leading_bjet_mask"]), final_array, selector.data["leading_b_l_recMET_min_mass"])
            selector.data["leading_b_l_recMET_max_mass_final"] = ak.where(ak.is_none(selector.data["leading_bjet_mask"]), final_array, selector.data["leading_b_l_recMET_max_mass"]) 

            #==================================================================================================================================================

            #=============This is the mass for the subleading bjets (only the bjet is changed, obviously the lepton and MET are still the same)================

            #Trying this for the spans must have compatible lengths error

            selector.data["subleading_b_l_recMET_min_mass"] = (selector.data["Rec_bjets"][:,1] + selector.data["Lepton"][:,0] + selector.data["MET_nu_min"]).mass
            selector.data["subleading_b_l_recMET_max_mass"] = (selector.data["Rec_bjets"][:,1] + selector.data["Lepton"][:,0] + selector.data["MET_nu_max"]).mass

            selector.data["subleading_b_l_recMET_min_mass_final"] = ak.where(ak.is_none(selector.data["subleading_bjet_mask"]), final_array_sl, selector.data["subleading_b_l_recMET_min_mass"])
            selector.data["subleading_b_l_recMET_max_mass_final"] = ak.where(ak.is_none(selector.data["subleading_bjet_mask"]), final_array_sl, selector.data["subleading_b_l_recMET_max_mass"]) 

            #==================================================================================================================================================

        elif len(selector.data["Jet"])==0:
            selector.data["leading_b_l_recMET_min_mass_final"] = final_array
            selector.data["leading_b_l_recMET_max_mass_final"] = final_array

            selector.data["subleading_b_l_recMET_min_mass_final"] = final_array_sl
            selector.data["subleading_b_l_recMET_max_mass_final"] = final_array_sl

        '''
        for i in range(len(selector.data)):
            print()
            print("Event number",i)
            print("Jet pt",selector.data["Jet"][i]["pt"])
            print("Btagged flag",selector.data["Jet"][i]["btagged"])
            print("b_l_recMET_min_mass_final",selector.data["b_l_recMET_min_mass_final"][i])
            print("b_l_recMET_max_mass_final",selector.data["b_l_recMET_max_mass_final"][i])
        '''

        abs_min_val_mask = abs(selector.data["min_pz_recmet"]) <= abs(selector.data["max_pz_recmet"])
        abs_max_val_mask = abs(selector.data["min_pz_recmet"]) > abs(selector.data["max_pz_recmet"])

        selector.data["abs_min_pz"] = ak.where((abs_min_val_mask==True)&(abs_max_val_mask==False), selector.data["min_pz_recmet"], selector.data["max_pz_recmet"])
        selector.data["abs_max_pz"] = ak.where((abs_min_val_mask==True)&(abs_max_val_mask==False), selector.data["max_pz_recmet"], selector.data["min_pz_recmet"])

        selector.data["diff_min_real"] = abs(selector.data["min_pz_recmet"] - selector.data["Gen_nu_pz"])
        selector.data["diff_max_real"] = abs(selector.data["max_pz_recmet"] - selector.data["Gen_nu_pz"])

        selector.data["diff_abs_min_real"] = abs(selector.data["abs_min_pz"] - selector.data["Gen_nu_pz"])
        selector.data["diff_abs_max_real"] = abs(selector.data["abs_max_pz"] - selector.data["Gen_nu_pz"])

        #hahaha
        '''
        for i in range(len(selector.data["Jet"])):
            print()
            print("Relative event number",i)
            print("pdg id list")
            print(ak.to_numpy(selector.data["GenPart"][i]["pdgId"]))    

            #==============================================Part for neutrino chains added - April 28========================================================
            print("Gen neutrino chains",chain_final[i])
            print("Resonant neutrino first",selector.data["nu_first_resonant"][i])

            print("Lepton pt",selector.data["Lepton"][i]["pt"])
            print("Lepton phi",selector.data["Lepton"][i]["phi"])
            print("MET pt",selector.data["MET"][i]["pt"])
            print("MET phi",selector.data["MET"][i]["phi"])
            print("xi_indpt_w_recmet",xi_indpt_w_recmet[i])
            print("xi_recmet_sum",xi_recmet_sum[i])
            print("xi_recmet",xi_recmet[i])

            print("Discr_rec",discr_rec[i])
            print("Discr_rec sign",discr_rec_sign[i])
            print("Discr_rec negative mask",(discr_rec<0)[i])
            print("New positive bound discr_rec",discr_rec_new[i])

            print("mt_l_rec",np.sqrt(mt_l_2_rec)[i])
            print("mt_l_rec from lep_energy and lep_pz", np.sqrt(pow(lep_energy,2) - pow(lep_pz,2))[i])

            print("MET_nu_min pt",selector.data["MET_nu_min"][i]["pt"])
            print("MET_nu_min phi",selector.data["MET_nu_min"][i]["phi"])
            print("MET_nu_min eta",selector.data["MET_nu_min"][i]["eta"])
            print("MET_nu_min mass",selector.data["MET_nu_min"][i]["mass"])

            print("MET_nu_max pt",selector.data["MET_nu_max"][i]["pt"])
            print("MET_nu_max phi",selector.data["MET_nu_max"][i]["phi"])
            print("MET_nu_max eta",selector.data["MET_nu_max"][i]["eta"])
            print("MET_nu_max mass",selector.data["MET_nu_max"][i]["mass"])

            print("min_pz_recmet",selector.data["min_pz_recmet"][i])
            print("max_pz_recmet",selector.data["max_pz_recmet"][i])

            #==============================================Part for neutrino chains added - April 28========================================================
            print("gen nu pz",selector.data["Gen_nu_pz"][i])

            print("abs_min_val_mask",abs_min_val_mask[i])
            print("abs_max_val_mask",abs_max_val_mask[i])

            print("abs_min_pz",selector.data["abs_min_pz"][i])
            print("abs_max_pz",selector.data["abs_max_pz"][i])

            print("diff_min_real",selector.data["diff_min_real"][i])
            print("diff_max_real",selector.data["diff_max_real"][i])
            print("diff_abs_min_real",selector.data["diff_abs_min_real"][i])
            print("diff_abs_max_real",selector.data["diff_abs_max_real"][i])

            print("leading_b_l_recMET_min_mass",selector.data["leading_b_l_recMET_min_mass"][i])
            print("leading_b_l_recMET_max_mass",selector.data["leading_b_l_recMET_max_mass"][i])

            print("leading_bjet_mask",selector.data["leading_bjet_mask"][i])
            print("leading_b_l_recMET_min_mass_final",selector.data["leading_b_l_recMET_min_mass_final"][i])
            print("leading_b_l_recMET_max_mass_final",selector.data["leading_b_l_recMET_max_mass_final"][i])

        '''

        #'''
        sign_dict = ["no_bjet","same_sign", "opp_sign", "neutral"]
        #sign_dict = ["no_bjet","same_sign", "opp_sign"]
        selector.set_cat("sign_cat", set(sign_dict))
        selector.set_multiple_columns(partial(self.sign_cats,is_mc))

        #top_dict = ["top", "antitop", "no_top","both_top"]
        #The problem in naming the offshell categories this way is that it will likely not hold true for other processes like w + jets
        #'''
        #top_dict = ["top", "antitop", "offshell_top","offshell_antitop","both_top"]
        #top_dict = ["onshell_lep_pos","onshell_lep_neg","offshell_lep_pos","offshell_lep_neg"]
        top_dict = ["lep_pos","lep_neg"]

        selector.set_cat("top_cat", set(top_dict))
        selector.set_multiple_columns(partial(self.top_cats,is_mc))
        #'''
        #'''

        pz_dict = ["complex", "real"]
        selector.set_cat("pz_cat", set(pz_dict))
        selector.set_multiple_columns(partial(self.pz_categories,is_mc))


        d = time.time()

        #'''

        #print("Shape of prob array", np.shape(prob_final))
        #print("Shape of prob array", len(ak.from_numpy(prob_final)))
        
        #selector.data["top_max_pt"] = ak.max(selector.data["GenPart"][(abs(selector.data["GenPart"].pdgId)==6)]["pt"],axis=1)

        top_neg = 0
        top_pos = 0
        antitop_neg = 0
        antitop_pos = 0

        btag_veto_dict = ["b_loose_0","b_loose_1+_no_effect","b_loose_1+_leading"]
        selector.set_cat("btag_veto_cat", set(btag_veto_dict))
        selector.set_multiple_columns(partial(self.btag_veto_cats,is_mc))

        '''
        for i in range(len(selector.data["Jet"])):
            print()
            print("Event number",i)
            print("New order of reco pt",selector.data["Jet"][i]["pt"])
            print("Jet GenJetidx",selector.data["Jet"][i]["genJetIdx"])
            print("New Matching gen pt",selector.data["Jet"][i].matched_gen.pt) #This is kept as a control because it is unchanged by any scale/correction factor
            print("Assigned meson tag reshaped",selector.data["b_meson_tag"][i])
            print("Assigned gen meson tag",selector.data["b_gen_meson"][i])
            print("Number of jets in each event",ak.num(selector.data["Jet"],axis=1)[i])
            print("Probability values")
            print("B-",selector.data["prob_neg"][i])
            print("B0bar",selector.data["prob_zerobar"][i])
            print("B0",selector.data["prob_zero"][i])
            print("B+",selector.data["prob_pos"][i])

            print("B jet pt",selector.data["bjets_tight"][i]["pt"])
            #print("b particle mother indices?",selector.data[abs(selector.data["GenPart"].pdgId) == 5]["GenPart"][i]["genPartIdxMother"])
            print("b particle mother indices?",selector.data["GenPart"][abs(selector.data["GenPart"].pdgId) == 5][i].genPartIdxMother)
            print("b particle pt?",selector.data["GenPart"][abs(selector.data["GenPart"].pdgId) == 5][i].pt)

            print("Lepton pt",selector.data["Lepton"][i].pt)
            print("Matched attributes from gen")
            print("Matched pt",selector.data["Lepton"][i].matched_gen_pt)
            print("Matched eta",selector.data["Lepton"][i].matched_gen_eta)
            print("Matched phi",selector.data["Lepton"][i].matched_gen_phi)
            gen_lepton_match = selector.data["GenPart"][(selector.data["GenPart"].pdgId==selector.data["Lepton"][i].pdgId)&(selector.data["GenPart"].pt==selector.data["Lepton"][i].matched_gen_pt)
            &(selector.data["GenPart"].eta==selector.data["Lepton"][i].matched_gen_eta)
            &(selector.data["GenPart"].phi==selector.data["Lepton"][i].matched_gen_phi)][i]
            print("Matched gen particle",gen_lepton_match.genPartIdxMother)
            print("Matched gen pt",gen_lepton_match.pt)
            print("Matched gen eta",gen_lepton_match.eta)
            print("Matched gen phi",gen_lepton_match.phi)
            print("Matched gen pdgid",gen_lepton_match.pdgId)
            print("Mother particles pdg id",selector.data["GenPart"][i].pdgId[gen_lepton_match.genPartIdxMother])

            #print("Final lepton mothers outside the recursion function",selector.data["lep_mother_final"].pdgId[i])
            #print("Lepton mother particle pdg id",selector.data["Lepton"][i].pt)

            print("Bjet Matching gen pt",selector.data["bjets_tight"][i].matched_gen.pt) #This is kept as a control because it is unchanged by any scale/correction factor
            b_bbar_object = selector.data["GenPart"][(abs(selector.data["GenPart"].pdgId)==5)]

            print("B gen particles pt",b_bbar_object[i].pt)
            print("B gen particles mother particle index",b_bbar_object[i].genPartIdxMother)
            print("B gen particles mother particle pdg id",selector.data["GenPart"][i].pdgId[b_bbar_object[i].genPartIdxMother])

            #print("Initial b mother particles",selector.data["b_mother_final"][i])

            #print("Final non b mother particles",selector.data["non_b_mother_final"][i])
            #print("Number of total b mother particles",ak.num(selector.data["non_b_mother_final"],axis=1))[i]

            print("Dr between chosen leading lepton and b jet",selector.data["dR_lb"][i])
            print("Dphi between chosen leading lepton and b jet",selector.data["dphi_lb"][i])
            print("Leading B jet eta",selector.data["abs_eta_b"][i])
            
            print("Invariant mass of chosen leading lepton and b jet (never equal to top mass due to MET)",selector.data["mlb"][i])

            print("Top pt (to potentially check correlation with dR_lb)",selector.data["GenPart"][(abs(selector.data["GenPart"].pdgId)==6)][i]["pt"])
            print("Max pt of top",selector.data["top_max_pt"][i])
            #print("Net charge of chosen leading lepton and b jet",selector.data["q_net"][i])

            print("Bjet meson tag reshaped",selector.data["meson_tag_tight"][i])
            print("Bjet gen meson tag",selector.data["gen_meson_tight"][i])
            print("Real Probability values")
            print("B-",selector.data["prob_neg_tight"][i])
            print("B0bar",selector.data["prob_zerobar_tight"][i])
            print("B0",selector.data["prob_zero_tight"][i])
            print("B+",selector.data["prob_pos_tight"][i])

            print("Hadron id of all jets?",jetorigin["bHadronId"][i]) 

            if is_mc:
                #Missing
                print("B missing jets pt",selector.data["bjets_miss"][i]["pt"])
                print("B Missing jet Matching gen pt",selector.data["bjets_miss"][i].matched_gen.pt) #This is kept as a control because it is unchanged by any scale/correction factor
                print("B missing jet gen meson tag",selector.data["gen_meson_miss"][i])
                print("missing Probability values")
                print("B-",selector.data["prob_neg_miss"][i])
                print("B0bar",selector.data["prob_zerobar_miss"][i])
                print("B0",selector.data["prob_zero_miss"][i])
                print("B+",selector.data["prob_pos_miss"][i])

                #Fake
                print("B fake jet pt",selector.data["bjets_fake"][i]["pt"])
                print("B fake jet Matching gen pt",selector.data["bjets_fake"][i].matched_gen.pt) #This is kept as a control because it is unchanged by any scale/correction factor
                print("B fake jet meson tag reshaped",selector.data["meson_tag_fake"][i])
                print("B fake jet gen meson tag",selector.data["gen_meson_fake"][i])
                print("fake Probability values")
                print("B-",selector.data["prob_neg_fake"][i])
                print("B0bar",selector.data["prob_zerobar_fake"][i])
                print("B0",selector.data["prob_zero_fake"][i])
                print("B+",selector.data["prob_pos_fake"][i])
                
                print("Values of b meson score rejection masks")
                for key in threshold_dict:
                    print(key,selector.data[key][i])

                print("Values of b and l top and mom combination masks")
                for key in top_and_mom_dict:
                    print(key,selector.data[key][i])

                print("min_pz_recmet",selector.data["min_pz_recmet"][i])
                print("max_pz_recmet",selector.data["max_pz_recmet"][i])

                print("b_l_recMET_min_mass_final",selector.data["b_l_recMET_min_mass_final"][i])
                print("b_l_recMET_max_mass_final",selector.data["b_l_recMET_max_mass_final"][i])

                print("Values of b meson categorization masks")
                for key in cat_dict:
                    print(key,selector.data[key][i])

                print("Values of b veto masks")
                for key in btag_veto_dict:
                    print(key,selector.data[key][i])
                
                #hahaha            

                #print("Inconclusive events",selector.data["inconclusive"][i])
                #print("Events successfully tagged",selector.data["conclusive"][i])
        '''

        e = time.time()

        #'''
        '''
        print("Time a",a)
        print("Time b",b)
        print("Time b_jet_cut",b_jet_cut)
        print("Time c",c)
        print("Time d",d)
        print("Time e",e)

        print("Time taken to populate the arrays by looping over all the feature groups",b-a)
        print("Time taken to call the frozen function by looping over all the jets",c-b_jet_cut) #This is just after putting the first pt and sorting cuts, before flattening
        print("Time taken for all the mask assignments and almost all cuts etc (just before printing)",d-c)
        print("Time taken for printing event wise",e-d)
        '''

        if "btag_sf" in self.config and len(self.config["btag_sf"]) != 0:
            selector.add_cut("btagSF", partial(self.btag_sf, is_mc))

        #'''

        #'''

        #feat_brnch = list(set(feat_brnch)) #remove duplicates        
        #'''
        '''
        for i in range(len(selector.data["Jet"])):
            if is_mc:
                print()
                print()
                print("MC Rec set")
                
                print("Relative event number",i)
                if i>1:
                    break

                print("Different types of nominal weights")
                print("L1 nominal prefiring weight",selector.data["L1PreFiringWeight"][i].Nom)
                print("L1 ecal nominal prefiring weight",selector.data["L1PreFiringWeight"][i].ECAL_Nom)
                print("L1 mjon nominal prefiring weight",selector.data["L1PreFiringWeight"][i].Muon_Nom)

                #print("LHE Reweighting weight",selector.data["LHEReweightingWeight"][i])

                print("Gen weight for event",selector.data["genWeight"][i])
                print("MC Generator weight for event",selector.data["Generator"][i].weight)
                #print("LHE weight",selector.data["LHEWeight"][i])

                #This is only to compare with previous spew files (of course the event number could also have been printed)
                print("Lepton rec pt",selector.data["Lepton"][i].pt)
                print("Rec transverse mass of W",selector.data["mTW"][i])

                print("Jet rec pt",selector.data["Jet"][i].pt)
                print("Jet rec eta",selector.data["Jet"][i].eta)

                print()
                print("Cpf per event for rec (or gen?) jets?",selector.data["cpf"][i]["charge"])
                print("Number of charged pf constituents?",len(selector.data["cpf"][i]))
                print("Npf per event for rec (or gen?) jets?",selector.data["npf"][i]["ptrel"])
                print("Number of neutral pf constituents?",len(selector.data["npf"][i]))

                print("Length of cpf",selector.data["length"][i]["cpf"])
                print("Length of npf",selector.data["length"][i]["npf"])
                print("Length offset of cpf",selector.data["length"][i]["cpf_offset"])
                print("Length offset of npf",selector.data["length"][i]["npf_offset"])

                print("Number of secondary vertices",selector.data["length"][i]["sv"])
                print("Global jet pt (?)",selector.data["global"][i]["pt"]) #Just to check whether this actually matches with the jet origin order
                print("Number of jets for global jet pt",len(selector.data["global"][i]["pt"]))
                print("Global jet id (?)",selector.data["global"][i]["jetIdx"]) #Just to check whether this actually matches with the jet origin order
                print("Number of jets for global jet pt",len(selector.data["global"][i]["jetIdx"]))

                print()
                print("Hadron flavor for jetorigin",jetorigin["hadronFlavor"][i])
                print("Parton flavor  for jetorigin (with charge)",jetorigin["partonFlavor"][i])
                print("Number of jetorigin jets in flavor",len(jetorigin["hadronFlavor"][i]))
                print()
                print("Btagged gen flag",selector.data["GenJet"]["hadronFlavour"][i])
                print("Btagged (and charge?) gen flag",selector.data["GenJet"]["partonFlavour"][i])
                print("Number of gen jets",len(selector.data["GenJet"]["hadronFlavour"][i]))

                #for j in range(len(selector.data["Jet"][i])): #This is actually from the rec and the constituents are stored for all the jet origin jets so this is not used
                print("csv values global",selector.data["csv"][i]["trackSumJetEtRatio"])
                print("csv values global",selector.data["csv"][i]["vertexCategory"])
                print("Constituents of rec jet?",selector.data["Jet"][i].nConstituents)
                print("Muon ids in jets",selector.data["Jet"][i].muonIdx1,selector.data["Jet"][i].muonIdx2)
                print("Electron ids in jets",selector.data["Jet"][i].electronIdx1,selector.data["Jet"][i].electronIdx2)

                print()
                print("Jet gen pt",selector.data["GenJet"][i].pt)
                print("Jet gen eta",selector.data["GenJet"][i].eta)

                print("Btag flag",selector.data["Jet"][i]["btagged"])
                print("Number of btagged rec jets",selector.data["nbtag"][i])
                #If rec_jet_mask and gen_jet_mask are calculated before applying the actual cut (which they will be) this indexing will be wrong so it is better to get rid of it
                
                print("Matching gen pt",selector.data["Jet"][i].matched_gen.pt)
                #print("Muon gen pt?",selector.data["GenPart"][abs(selector.data["GenPart"].pdgId) == 13][i].pt)
                print("Matching gen eta",selector.data["Jet"][i].matched_gen.eta)

                print("Jet idx?",idx[i])   
                #print("Jet origin array?",jetorigin[i].fields)        
                print(jetorigin[i])
                for key_origin in jetorigin[i].fields:
                    print(key_origin,jetorigin[i][key_origin])

                #print("Jet origin eta?",selector.data["GenJet"][idx])
                #print("Jet origin pt?",selector.data["GenJet"][idx]["pt"][i])
                print("Hadron id of all jets?",jetorigin["bHadronId"][i]) 

        '''

    def require_lepton_nums(self, is_mc, data):
        """Select events that contain at least one lepton."""
        accept = np.asarray(ak.num(data["Muon"]) >= 1) #At least one lepton or at least one muon?
        if is_mc:
            weight, systematics = self.compute_lepton_sf(data[accept])
            accept = accept.astype(float)
            accept[accept.astype(bool)] *= np.asarray(weight)
            return accept, systematics
        else:
            return accept

    def pick_electrons(self, data):
        good_id, good_pt, good_iso = self.config[[
                "good_ele_id", "good_ele_pt_min", "good_ele_iso"]]
        veto_id, veto_pt, veto_iso = self.config[[
                "veto_ele_id", "veto_ele_pt_min", "veto_ele_iso"]]
        eta_min, eta_max = self.config[["ele_eta_min", "ele_eta_max"]]

        ele = data["Electron"]

        # remove barrel-endcap transreg
        sc_eta_abs = abs(ele.eta + ele.deltaEtaSC)
        is_in_transreg = (1.444 < sc_eta_abs) & (sc_eta_abs < 1.566)
        etacuts = (eta_min < ele.eta) & (eta_max > ele.eta)

        # Electron ID, as an example we use the MVA one here
        has_id = self.electron_id(self.config["good_ele_id"], ele)

        # Finally combine all the requirements
        # to-do add iso
        is_good = (
            self.electron_id(good_id, ele)
            & (~is_in_transreg)
            & etacuts
            & (good_pt < ele.pt))

        is_veto = (
            ~is_good
            & self.electron_id(veto_id, ele)
            & (~is_in_transreg)
            & etacuts
            & (veto_pt < ele.pt))

        # Return all electrons with are deemed to be good
        return {"Electron": ele[is_good], "VetoElectron": ele[is_veto]}

    def pick_muons(self, data):
        good_id, good_pt, good_iso = self.config[[
                "good_muon_id", "good_muon_pt_min", "good_muon_iso"]]
        veto_id, veto_pt, veto_iso = self.config[[
                "veto_muon_id", "veto_muon_pt_min", "veto_muon_iso"]]
        eta_min, eta_max = self.config[["muon_eta_min", "muon_eta_max"]]

        muon = data["Muon"]

        etacuts = (eta_min < muon.eta) & (eta_max > muon.eta)

        is_good = (
            self.muon_id(good_id, muon)
            & self.muon_iso(good_iso, muon)
            & etacuts
            & (good_pt < muon.pt))
        
        is_veto = (
            ~is_good
            & self.muon_id(veto_id, muon)
            & self.muon_iso(veto_iso, muon)
            & etacuts
            & (veto_pt < muon.pt))

        return {"Muon": muon[is_good], "VetoMuon": muon[is_veto]}

    def one_good_lepton(self, data):
        '''
        print()
        #This will probably leave out missing reconstructed muons completely, although fakes might be a bit more obvious
        print("Muon rec pt",data["Muon"].pt)
        print("Matching gen pt",data["Muon"].matched_gen.pt)
        print("Muon rec eta",data["Muon"].eta)
        print("Matching gen eta",data["Muon"].matched_gen.eta)
        '''
        return ak.num(data["Electron"]) + ak.num(data["Muon"]) == 1 #Why not just use the data["Lepton"] column? Maybe because it is after the selections from pick_electrons and pick_muons

    def one_fresh_gen_lepton(self, is_mc, data):
        '''
        print()
        #This will probably leave out missing reconstructed muons completely, although fakes might be a bit more obvious
        print("Muon rec pt",data["Muon"].pt)
        print("Matching gen pt",data["Muon"].matched_gen.pt)
        print("Muon rec eta",data["Muon"].eta)
        print("Matching gen eta",data["Muon"].matched_gen.eta)
        '''
        '''
        for i in range(len(data["Jet"])):
            print()
            print("Event number in one_fresh_gen_lepton function",i)
            print("Number of fresh leptons having W mother")
            print(ak.num(data["GenPart"][data["e_mu_tau_fresh_resonant"]],axis=1)[i])

        hahaha
        '''
        return ak.num(data["GenPart"][data["e_mu_tau_fresh_resonant"]]) <= 1 #This is to include those which have zero leptons as well, those are getting replaced with a mask and then [] later for mass
        #return ak.num(data["Electron"]) + ak.num(data["Muon"]) == 1 #Why not just use the data["Lepton"] column? Maybe because it is after the selections from pick_electrons and pick_muons


    def no_veto_muon(self, data):
        return ak.num(data["VetoMuon"]) + ak.num(data["VetoElectron"]) == 0
    
    def channel_masks(self, data):
        channels = {}
        channels["mu"] = (ak.num(data["Muon"]) == 1)
        #channels["el"] = (ak.num(data["Electron"]) == 1)
        # channels["fake_mu"] 

        return channels
    
    def build_jet_geninfo(self,data):
        idx = data["jetorigin"]["jetIdx"]
        jetorigin = data["jetorigin"]
        jetorigin["eta"] = data["Jet"][idx]["eta"]
        jetorigin["pt"] = data["Jet"][idx]["pt"]

        jetorigin["isB0"] = jetorigin["bHadronId"]==511
        jetorigin["isB0bar"] = jetorigin["bHadronId"]==-511
        jetorigin["isBplus"] = jetorigin["bHadronId"]==521
        jetorigin["isBminus"] = jetorigin["bHadronId"]==-521

        return jetorigin
    
    def num_btags(self, prefix, data):
        jets = data[prefix+"Jet"]
        nbtags = ak.sum(jets["btagged"], axis=1)
        return ak.where(ak.num(jets) > 0, nbtags, 0)

    def has_bjet(self,data):
        return data["nbtag"] > 0

    def btag_categories(self,data):
        
        cats = {}
        
        num_btagged = data["nbtag"]
        njet = ak.num(data["Jet"])

        cats["j2_b0"] = (num_btagged == 0) & (njet == 2)
        cats["j2_b1"] = (num_btagged == 1) & (njet == 2)
        cats["j2_b2"] = (num_btagged == 2) & (njet == 2)
        cats["j2+_b0"] = (num_btagged == 0) & (njet >= 2)
        cats["j2+_b1"] = (num_btagged == 1) & (njet >= 2)
        cats["j2+_b2"] = (num_btagged == 2) & (njet >= 2)


        return cats

    def eta_cat(self,is_mc,data):
        
        cats = {}
        cats["no_bjet"] = [False] * len(data["Jet"]) 
        #Technically this was not required, but the j2_b0 and j2+_b0 events need somewhere to go and appear in the final sum histogram without being cut
        cats["l_be"] = [False] * len(data["Jet"])
        cats["l_fwd"] = [False] * len(data["Jet"])
        cats["sl_be"] = [False] * len(data["Jet"])
        cats["sl_fwd"] = [False] * len(data["Jet"])

        cats["l_be_sl_be"] = [False] * len(data["Jet"])
        cats["l_fwd_sl_be"] = [False] * len(data["Jet"])
        cats["l_be_sl_fwd"] = [False] * len(data["Jet"])
        cats["l_fwd_sl_fwd"] = [False] * len(data["Jet"])

        btagged = data["nbtag"]
        njet = ak.num(data["Jet"])          

        no_b_cut = (btagged == 0) & (njet >= 2)
        leading_cut = (btagged>=1) & (njet >= 2)
        subleading_cut = (btagged>=2) & (njet >= 2)

        fwd_cut = self.config["btag_jet_eta_max"]

        for i in range(len(data["Jet"])):
            #print("Entry number",i,"Leading and subleading Cut values",leading_cut[i],subleading_cut[i])
            cats["no_bjet"][i] = no_b_cut[i]

            if subleading_cut[i]:
                abs_eta_l = abs(data["bjets_tight"][i][0].eta)
                abs_eta_sl = abs(data["bjets_tight"][i][1].eta)

                #print("Jet etas",abs_eta_l,abs_eta_sl)
                #print("Entry number", i, ": two or more b jets",len(data["bjets_gen"][i].eta) > 1)
                #eta_leading = data["bjets_gen"][i][2].eta
                cats["sl_be"][i] = abs_eta_sl < fwd_cut
                cats["sl_fwd"][i] = abs_eta_sl >= fwd_cut

                cats["l_be_sl_be"][i] = (abs_eta_l < fwd_cut) & (abs_eta_sl < fwd_cut)
                cats["l_fwd_sl_be"][i] = (abs_eta_l >= fwd_cut) & (abs_eta_sl < fwd_cut)
                cats["l_be_sl_fwd"][i] = (abs_eta_l < fwd_cut) & (abs_eta_sl >= fwd_cut)
                cats["l_fwd_sl_fwd"][i] = (abs_eta_l >= fwd_cut) & (abs_eta_sl >= fwd_cut)

            if leading_cut[i]:
                abs_eta_l = abs(data["bjets_tight"][i][0].eta)

                cats["l_be"][i] = abs_eta_l < fwd_cut
                cats["l_fwd"][i] = abs_eta_l >= fwd_cut
            

        #cats["l_fwd"] = ak.Array(cats["l_fwd"])
    
        return cats

    #Probably not going to use this as the masks and categories are applied for events and not jets
    def tag_match(self,is_mc,data): #This data has each entry not as events but as jets
        cats = {}
        id_flag = data["Gen_Hadron_Id"]!=0
        reco_b_flag = data["Reco_Jet_Flavor"] == True
        
        cats["no_id_no_rec"] = (~id_flag) & (~reco_b_flag)
        cats["id_no_rec"] = (id_flag) & (~reco_b_flag)
        cats["no_id_rec"] = (~id_flag) & (reco_b_flag)
        cats["id_rec"] = (id_flag) & (reco_b_flag)

        return cats

    def charge_tag_cat(self,is_mc,data):
        cats = {}

        leading_cut = ak.mask(data,(ak.num(data["bjets_tight"].eta)>=1) & (ak.num(data["Jet"]) >= 2))
        subleading_cut = ak.mask(data,(ak.num(data["bjets_tight"].eta)>=2) & (ak.num(data["Jet"]) >= 2))

        meson_cat = ["bneg","b0bar","b0","bpos"]
        hadron_id = [-521, -511, 511, 521]

        #rec_meson = ak.argmax(data["NN_prob"],axis=1)
        #This is the meson type chosen from the probabilities from the nn evaluation, and is a 1 entry per jet awkward array
        cut_array = [False] * len(data["bjets_tight"]) 

        #Using this instead of the above (this was also used for reshaping the arrays for the max number of constituents of the jets)
        data_gen_meson = ak.pad_none(data["gen_meson_tight"], 2, clip=True)
        data_gen_meson = ak.fill_none(data_gen_meson, 0)
        data_meson_tag = ak.pad_none(data["meson_tag_tight"], 2, clip=True)
        data_meson_tag = ak.fill_none(data_meson_tag, 0)

        #=====================================================================================================

        for gen_i in range(len(meson_cat)):
            string_mask = "id_" + meson_cat[gen_i] + "_rec_" + meson_cat[gen_i]
            mask_value = (data_gen_meson[:,0] == hadron_id[gen_i]) & (data_meson_tag[:,0] == gen_i)
            cats[string_mask] = ak.where(ak.is_none(leading_cut), cut_array, mask_value)
            #print("Name of gen and rec charge tagging mask", string_mask)

        #=============================================================================================================================================

        return cats


    def score_categories(self,is_mc,data):
        cats = {}
        #score_lowlim = 0.5 #The event will be rejected if even the max score is less than this value
        #threshold_list = [0.3,0.4,0.5,0.6,0.7,0.8]
        #threshold_list = [0.025,0.05,0.075,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8] #New threshold list for completing the roc curve

        threshold_list = [0.5] #Guess threshold applied
        leading_cut = ak.mask(data,(ak.num(data["bjets_tight"].eta)>=1) & (ak.num(data["Jet"]) >= 2))

        prob = ["prob_neg","prob_zerobar","prob_zero","prob_pos"]

        if len(data) > 0:
            data_meson_tag = ak.pad_none(data["meson_tag_tight"], 2, clip=True)
            data_meson_tag = ak.fill_none(data_meson_tag, 0)

            data_prob_neg = ak.pad_none(data["prob_neg_tight"], 2, clip=True)
            data_prob_neg = ak.fill_none(data_prob_neg, 0)

            data_prob_zerobar = ak.pad_none(data["prob_zerobar_tight"], 2, clip=True)
            data_prob_zerobar = ak.fill_none(data_prob_zerobar, 0)

            data_prob_zero = ak.pad_none(data["prob_zero_tight"], 2, clip=True)
            data_prob_zero = ak.fill_none(data_prob_zero, 0)

            data_prob_pos = ak.pad_none(data["prob_pos_tight"], 2, clip=True)
            data_prob_pos = ak.fill_none(data_prob_pos, 0)

            #Since even the no bjet events have been padded with zeros, this condition will automatically hold for them too, irrespective of whether it is used or not
            #Checking all scores because it seems easier than checking where the max score is and comparing only that
            for score_lowlim in threshold_list:
                neg_mask = data_prob_neg[:,0] <= score_lowlim
                zerobar_mask = data_prob_zerobar[:,0] <= score_lowlim
                zero_mask = data_prob_zero[:,0] <= score_lowlim
                pos_mask = data_prob_pos[:,0] <= score_lowlim

                inconclusive_mask = (neg_mask) & (zerobar_mask) & (zero_mask) & (pos_mask)
                conclusive_mask = ~((neg_mask) & (zerobar_mask) & (zero_mask) & (pos_mask))

                cut_array_incon = [True] * len(data["bjets_tight"]) 
                cut_array_con = [False] * len(data["bjets_tight"]) 
                cats["inconclusive_"+str(int(score_lowlim*100))] = ak.where(ak.is_none(leading_cut), cut_array_incon, inconclusive_mask)
                cats["conclusive_"+str(int(score_lowlim*100))] = ak.where(ak.is_none(leading_cut), cut_array_con, conclusive_mask)
        
        elif len(data) == 0:
            for score_lowlim in threshold_list:
                cut_array_incon = [True] * len(data["bjets_tight"]) 
                cut_array_con = [False] * len(data["bjets_tight"]) 
                cats["inconclusive_"+str(int(score_lowlim*100))] = cut_array_incon
                cats["conclusive_"+str(int(score_lowlim*100))] = cut_array_con

        return cats

    def top_categories(self,is_mc,data):
        cats = {}
        #Making the assumption that there is only a single top in each process (s or t channel chosen)
        top_pdgid = data["GenPart"][abs(data["GenPart"].pdgId) == 6].pdgId[:,0]

        '''
        for i in range(len(data["Jet"])):
            print("Event number",i)
            print("Top charge", top_pdgid[i])
        '''

        #'''
        cats["top"] = (top_pdgid==6)
        cats["antitop"] = (top_pdgid==-6)

        return cats
        #'''

    def daughter_categories(self,is_mc,data):
        cats = {}
        cats["zero_daughters"] = data["n_daughters"] == 0
        cats["non_zero_daughters"] = data["n_daughters"] > 0

        return cats

    #No need to put any len(data)==0 condition because this is going to be used only for signal
    def pz_categories(self,is_mc,data):
        cats = {}
        #cats["complex"] = ~ak.is_none(data["neg_discr_mask"])
        #cats["real"] = ak.is_none(data["neg_discr_mask"])
        
        #Changing criteria based on discr mask in rec instead of gen
        cats["complex"] = ~ak.is_none(data["neg_discr_rec_mask"])
        cats["real"] = ak.is_none(data["neg_discr_rec_mask"])

        '''
        for i in range(len(data["Jet"])):
            print("Event number in pz_categories",i)
            print("min_pz_met",data["min_pz_met"][i])
            print("max_pz_met",data["max_pz_met"][i])
            print("gen nu pz",data["Gen_nu_pz"][i])
            print("complex mask",cats["complex"][i])
            print("real mask",cats["real"][i])
        '''
        return cats

    def top_and_mom_cats(self,is_mc,data):
        cats = {}
        w_id = 24
        top_id = 6

        #When mother information was available from chains, we could just check that the sign of the top and the W were the same
        #Now we need to check that the sign of the lepton and b charges are opposite
        #sign_b_mom = data["non_b_mother_final"].pdgId/abs(data["non_b_mother_final"].pdgId)
        #sign_l_mom = data["lep_mother_final"].pdgId/abs(data["lep_mother_final"].pdgId)

        #Using this instead of the above (this was also used for reshaping the arrays for the max number of constituents of the jets)
        #leading_cut_tight = ak.mask(data,(ak.num(data["bjets_tight"].eta)>=1) & (ak.num(data["Jet"]) >= 2))
        data_meson_tag = ak.pad_none(data["meson_tag_tight"], 1, clip=True)
        data_meson_tag = ak.fill_none(data_meson_tag, 1) #This one is the index of the array so it could give wrong values, but will be replaced later

        #Since we do not have pdg id for jets, using the charge tagger directly, probably could have still used the min dR matching (no non b mother or chains)
        meson_cat = ["bneg","b0bar","b0","bpos"]
        pdgid_cat = [1,0,0,-1]
        pdgid_cat_full = (np.array([pdgid_cat]* len(data["Lepton"])))
        pdgid_index_full = (np.array([ak.local_index(pdgid_cat)]* len(data["Lepton"])))

        print("Lepton length in top_mom_cats",len(data["Lepton"]))
        print("Length of data_meson_tag",len(data_meson_tag))
        print("Length of pdgid_index_full",len(pdgid_index_full))
        print("Length of pdgid_cat_full",len(pdgid_cat_full))
        #sign_b = pdgid_cat_full[:,data_meson_tag] #data_meson_tag is a mix of real with dummy indices

        false_array = [False]*len(data["Lepton"])

        if len(data) > 0:
            sign_b = pdgid_cat_full[pdgid_index_full==data_meson_tag] #data_meson_tag is a mix of real with dummy indices
            sign_l = data["Lepton"].pdgId/abs(data["Lepton"].pdgId) #Already exactly one lepton cut applied

            #Since this is always going to be used in combination with b_mask and l_mask, using [:,0] is ok since there is only one element
            #Turns out this is not ok, with the non recursive method it can have two mothers (weird but probable)
            #sign_mask = (sign_b[:,0] == -sign_l[:,0]) 

            sign_mask = (sign_b == -sign_l[:,0]) 
            same_sign_mask = (sign_b == sign_l[:,0]) 
            neutral_mask = (sign_b == 0) #Just for bookkeeping, to be fair, this should be 0 (false) for id_bpos_rec_bpos and id_bneg_rec_bneg

            #Slightly different conditions, if both top and antitop counts are non zero it can also be used as a category for ttbar
            top_num = ak.num(data["GenPart"][data["GenPart"].pdgId == 6], axis=1)        
            antitop_num = ak.num(data["GenPart"][data["GenPart"].pdgId == -6], axis=1)        

            top_mask = (top_num>=1) & (antitop_num==0)
            antitop_mask = (top_num==0) & (antitop_num>=1)
            no_top_mask = (top_num==0) & (antitop_num==0)
            both_top_mask = (top_num>=1) & (antitop_num>=1) #This wont happen for wbj, mostly only for ttbar

            #no_bl_mask = (ak.is_none(data["bjets_tight"].pt)) | (ak.is_none(data["Lepton"].pt))
            no_bl_mask = (ak.num(data["bjets_tight"].pt,axis=1)==0) | (ak.num(data["Lepton"].pt,axis=1)==0)

            cats["no_bjet"] = no_bl_mask

            cats["top_opp_sign"] = ~no_bl_mask & top_mask & sign_mask
            cats["top_same_sign"] = ~no_bl_mask & top_mask & same_sign_mask
            cats["top_neutral"] = ~no_bl_mask & top_mask & neutral_mask

            cats["antitop_opp_sign"] = ~no_bl_mask & antitop_mask & sign_mask
            cats["antitop_same_sign"] = ~no_bl_mask & antitop_mask & same_sign_mask
            cats["antitop_neutral"] = ~no_bl_mask & antitop_mask & neutral_mask
            
            #This means that even though there is no top, the sign of the top can be inferred from the sign of the W
            #What will still remain unclear is whether it is from resonance or scattering
            #Since we are comparing b jet charges anyway, might as well change this too

            #If the lepton sign is not explicitly being checked we will not know whether it is a top or antitop in case the no_top_neutral mask is true
            #But maybe then we don't even need to know
            cats["no_top_opp_sign"] = ~no_bl_mask & no_top_mask & sign_mask #top
            cats["no_top_same_sign"] = ~no_bl_mask & no_top_mask & same_sign_mask #antitop
            cats["no_top_neutral"] = ~no_bl_mask & no_top_mask & neutral_mask

            #====================================================================================================================

            cats["both_top_opp_sign"] = ~no_bl_mask & both_top_mask & sign_mask
            cats["both_top_same_sign"] = ~no_bl_mask & both_top_mask & same_sign_mask
            cats["both_top_neutral"] = ~no_bl_mask & both_top_mask & neutral_mask

            for key in cats.keys():
                cats[key] = ak.where(ak.is_none(cats[key]), false_array, cats[key])

        elif len(data) == 0:
            key_list = ["no_bjet","top_opp_sign","top_same_sign","top_neutral","antitop_opp_sign","antitop_same_sign","antitop_neutral","no_top_opp_sign","no_top_same_sign","no_top_neutral","both_top_opp_sign","both_top_same_sign","both_top_neutral"]
            for key in key_list:
                cats[key] = false_array
        '''
        for i in range(len(data["Lepton"])):
            print()
            print("Event number in top_and_mom_cats function",i)
            print("pdg id list")
            print(ak.to_numpy(data["GenPart"][i]["pdgId"]))
            print("top_num",top_num[i])
            print("antitop_num",antitop_num[i])
            print("Bjet pt",data["bjets_tight"][i].pt)
            print("Lepton pdgid",data["Lepton"][i].pdgId)

            print("no_bjet",cats["no_bjet"][i])

            print("Top onshell Mask values")
            for key in cats.keys():
                if key.find("top")==0:
                    print(key,cats[key][i])
            #print("top_opp_sign",cats["top_opp_sign"][i])
            #print("top_same_sign",cats["top_same_sign"][i])
            print("Antitop onshell Mask values")
            for key in cats.keys():
                if key.find("antitop")==0:
                    print(key,cats[key][i])

            print("No top (offshell) mask values")            
            for key in cats.keys():
                if key.find("no_top")==0:
                    print(key,cats[key][i])

            print("Both top (mainly for ttbar) mask values")            
            for key in cats.keys():
                if key.find("both_top")==0:
                    print(key,cats[key][i])

            print("id_bpos_rec_bpos",data["id_bpos_rec_bpos"][i])
            print("id_bneg_rec_bneg",data["id_bneg_rec_bneg"][i])
            #Just for debugging, does not provide much more information at truth level (especially for onshell), compare with id_bpos_rec_bpos and id_bneg_rec_bneg
            print("data_meson_tag",data_meson_tag[i])
            print("sign_b correct",pdgid_cat_full[i][data_meson_tag[i]])
            print("sign_b",sign_b[i])
            print("sign_l",ak.unflatten(sign_l[:,0],1)[i])

        '''
        
        return cats

    def sign_cats(self,is_mc,data):
        cats = {}
        w_id = 24
        top_id = 6

        print("Number of events",len(data))
        print("Length of bjets_tight",len(data["bjets_tight"]))
        #print("Length of bjets_fake",len(data["bjets_fake"]))
        leading_cut = ak.mask(data,(ak.num(data["bjets_tight"].eta)>=1) & (ak.num(data["Jet"]) >= 2))

        meson_cat = ["bneg","b0bar","b0","bpos"]
        hadron_id = [-521, -511, 511, 521]

        #rec_meson = ak.argmax(data["NN_prob"],axis=1)
        #This is the meson type chosen from the probabilities from the nn evaluation, and is a 1 entry per jet awkward array
        cut_array = [False] * len(data["bjets_tight"]) 

        #Using this instead of the above (this was also used for reshaping the arrays for the max number of constituents of the jets)
        #data_meson_tag = ak.pad_none(data["meson_tag_tight"], 2, clip=True)
        data_meson_tag = ak.pad_none(data["meson_tag_tight"], 1, clip=True)
        data_meson_tag = ak.fill_none(data_meson_tag, 0)

        #=====================================================================================================

        score_lowlim = 0.5 #Guess threshold applied

        prob = ["prob_neg","prob_zerobar","prob_zero","prob_pos"]

        if len(data) > 0:
            #data_meson_tag = ak.pad_none(data["meson_tag_tight"], 2, clip=True)
            #data_meson_tag = ak.fill_none(data_meson_tag, 0)

            data_prob_neg = ak.pad_none(data["prob_neg_tight"], 1, clip=True)
            data_prob_neg = ak.fill_none(data_prob_neg, 0)

            data_prob_zerobar = ak.pad_none(data["prob_zerobar_tight"], 1, clip=True)
            data_prob_zerobar = ak.fill_none(data_prob_zerobar, 0)

            data_prob_zero = ak.pad_none(data["prob_zero_tight"], 1, clip=True)
            data_prob_zero = ak.fill_none(data_prob_zero, 0)

            data_prob_pos = ak.pad_none(data["prob_pos_tight"], 1, clip=True)
            data_prob_pos = ak.fill_none(data_prob_pos, 0)

            #Since even the no bjet events have been padded with zeros, this condition will automatically hold for them too, irrespective of whether it is used or not
            #Checking all scores because it seems easier than checking where the max score is and comparing only that
            neg_mask = data_prob_neg[:,0] <= score_lowlim
            zerobar_mask = data_prob_zerobar[:,0] <= score_lowlim
            zero_mask = data_prob_zero[:,0] <= score_lowlim
            pos_mask = data_prob_pos[:,0] <= score_lowlim

            conclusive_mask = ~((neg_mask) & (zerobar_mask) & (zero_mask) & (pos_mask)) 
            conclusive_mask = ak.where(ak.is_none(leading_cut), cut_array, conclusive_mask)
        
        elif len(data) == 0:
            conclusive_mask = cut_array

        #Using this instead of the above (this was also used for reshaping the arrays for the max number of constituents of the jets)
        #leading_cut_tight = ak.mask(data,(ak.num(data["bjets_tight"].eta)>=1) & (ak.num(data["Jet"]) >= 2))
        #data_meson_tag = ak.pad_none(data["meson_tag_tight"], 1, clip=True)
        #data_meson_tag = ak.fill_none(data_meson_tag, 1) #This one is the index of the array so it could give wrong values, but will be replaced later

        #Since we do not have pdg id for jets, using the charge tagger directly, probably could have still used the min dR matching (no non b mother or chains)
        meson_cat = ["bneg","b0bar","b0","bpos"]
        pdgid_cat = [1,0,0,-1]
        pdgid_cat_full = (np.array([pdgid_cat]* len(data["Lepton"])))
        pdgid_index_full = (np.array([ak.local_index(pdgid_cat)]* len(data["Lepton"])))

        print("Lepton length in top_mom_cats",len(data["Lepton"]))
        print("Length of data_meson_tag",len(data_meson_tag))
        print("Length of pdgid_index_full",len(pdgid_index_full))
        print("Length of pdgid_cat_full",len(pdgid_cat_full))
        #sign_b = pdgid_cat_full[:,data_meson_tag] #data_meson_tag is a mix of real with dummy indices

        false_array = [False]*len(data["Lepton"])

        key_list = ["no_bjet","opp_sign","same_sign","neutral"]

        if len(data) > 0:
            sign_b = pdgid_cat_full[pdgid_index_full==data_meson_tag] #data_meson_tag is a mix of real with dummy indices
            sign_l = data["Lepton"].pdgId/abs(data["Lepton"].pdgId) #Already exactly one lepton cut applied

            #Since this is always going to be used in combination with b_mask and l_mask, using [:,0] is ok since there is only one element
            #Turns out this is not ok, with the non recursive method it can have two mothers (weird but probable)
            #sign_mask = (sign_b[:,0] == -sign_l[:,0]) 

            sign_mask = (sign_b == -sign_l[:,0]) 
            same_sign_mask = (sign_b == sign_l[:,0]) 
            neutral_mask = (sign_b == 0) #Just for bookkeeping, to be fair, this should be 0 (false) for id_bpos_rec_bpos and id_bneg_rec_bneg

            #no_bl_mask = (ak.is_none(data["bjets_tight"].pt)) | (ak.is_none(data["Lepton"].pt))
            no_bl_mask = (ak.num(data["bjets_tight"].pt,axis=1)==0) | (ak.num(data["Lepton"].pt,axis=1)==0)

            cats["no_bjet"] = no_bl_mask
            cats["opp_sign"] = ~no_bl_mask & conclusive_mask & sign_mask
            cats["same_sign"] = ~no_bl_mask & conclusive_mask & same_sign_mask
            cats["neutral"] = ~no_bl_mask & conclusive_mask & neutral_mask

            for key in cats.keys():
                cats[key] = ak.where(ak.is_none(cats[key]), false_array, cats[key])

        elif len(data) == 0:
            #key_list = ["no_bjet","opp_sign","same_sign"]
            for key in key_list:
                cats[key] = false_array
        '''
        for i in range(len(data)):
            print()
            print("Event number in sign_cats",i)
            print("no_bl_mask",no_bl_mask[i])
            print("conclusive_mask",conclusive_mask[i])
            for key in key_list:
                print(key,cats[key][i])
        
        hahaha
        '''
        return cats

    def top_cats(self,is_mc,data):
        cats = {}
        w_id = 24
        top_id = 6

        #Using this instead of the above (this was also used for reshaping the arrays for the max number of constituents of the jets)
        #leading_cut_tight = ak.mask(data,(ak.num(data["bjets_tight"].eta)>=1) & (ak.num(data["Jet"]) >= 2))
        data_meson_tag = ak.pad_none(data["meson_tag_tight"], 1, clip=True)
        data_meson_tag = ak.fill_none(data_meson_tag, 1) #This one is the index of the array so it could give wrong values, but will be replaced later

        #Since we do not have pdg id for jets, using the charge tagger directly, probably could have still used the min dR matching (no non b mother or chains)
        meson_cat = ["bneg","b0bar","b0","bpos"]
        pdgid_cat = [1,0,0,-1]
        pdgid_cat_full = (np.array([pdgid_cat]* len(data["Lepton"])))
        pdgid_index_full = (np.array([ak.local_index(pdgid_cat)]* len(data["Lepton"])))

        #print("Lepton length in top_mom_cats",len(data["Lepton"]))
        #print("Length of data_meson_tag",len(data_meson_tag))
        #print("Length of pdgid_index_full",len(pdgid_index_full))
        #print("Length of pdgid_cat_full",len(pdgid_cat_full))
        #sign_b = pdgid_cat_full[:,data_meson_tag] #data_meson_tag is a mix of real with dummy indices

        false_array = [False]*len(data["Lepton"])

        if len(data) > 0:
            '''
            top_num = ak.num(data["GenPart"][data["GenPart"].pdgId == 6], axis=1)        
            antitop_num = ak.num(data["GenPart"][data["GenPart"].pdgId == -6], axis=1)        
            no_top_mask = (top_num==0) & (antitop_num==0)
            '''

            #Extra added specifically for offshell (no top) now that id_bpos_rec_bpos and id_bneg_rec_bneg have been sort of merged
            sign_l = data["Lepton"].pdgId/abs(data["Lepton"].pdgId) #Already exactly one lepton cut applied
            '''
            cats["onshell_lep_pos"] = ~no_top_mask & (sign_l[:,0] == -1) #This is the sign of the pdg id, not the actual lepton
            cats["onshell_lep_neg"] = ~no_top_mask & (sign_l[:,0] == 1) #This is the sign of the pdg id, not the actual lepton

            #Criteria for this have not been changed, so these two should remain exactly the same in the cutflow files
            cats["offshell_lep_pos"] = no_top_mask & (sign_l[:,0] == -1) #This is the sign of the pdg id, not the actual lepton
            cats["offshell_lep_neg"] = no_top_mask & (sign_l[:,0] == 1) #This is the sign of the pdg id, not the actual lepton
            '''

            #Removed onshell and offshell at truth level altogether (although the mass window cut will be there at some point)   
            #'''         
            cats["lep_pos"] = (sign_l[:,0] == -1) #This is the sign of the pdg id, not the actual lepton
            cats["lep_neg"] = (sign_l[:,0] == 1) #This is the sign of the pdg id, not the actual lepton
            #'''
            for key in cats.keys():
                cats[key] = ak.where(ak.is_none(cats[key]), false_array, cats[key])

        elif len(data) == 0:
            #key_list = ["onshell_lep_pos","onshell_lep_neg","offshell_lep_pos","offshell_lep_neg"]
            key_list = ["lep_pos","lep_neg"]
            for key in key_list:
                cats[key] = false_array

        '''
        for key in cats.keys():
            print(key,len(cats[key]))
        for i in range(len(data)):
            print()
            print("Event number in top_cats",i)
            for key in cats.keys():
                print(key,cats[key][i])
        '''
        return cats


    def btag_veto_cats(self,is_mc,data):
        cats = {}
        #Based on this parameter (roughly) the categories have already been made
        #This is not the number of tight bjets, but the number of 'correct' tight bjets
        tight_jets = ak.num(data["bjets_tight"],axis=1)
        loose_jets = ak.num(data["bjets_loose"])

        #============================should really have made a function for this=====================================================

        #threshold_list = [0.5] #Guess threshold applied
        '''
        score_lowlim = 0.5 #Guess threshold applied
        leading_cut_loose = ak.mask(data,(ak.num(data["bjets_loose"].eta)>=1) & (ak.num(data["Jet"]) >= 2))
        prob = ["prob_neg","prob_zerobar","prob_zero","prob_pos"]
        '''

        #This is purely for pt comparison, to see whether the loose bjet has a higher pt than this
        leading_cut_tight = ak.mask(data,(ak.num(data["bjets_tight"].eta)>=1) & (ak.num(data["Jet"]) >= 2))
        cut_array = [False] * len(data["bjets_loose"]) 
        if len(data) > 0:
            data_bjet_pt_tight = ak.pad_none(data["bjets_tight"]["pt"], 1, clip=True)
            data_bjet_pt_tight = ak.fill_none(data_bjet_pt_tight, 0)

            data_bjet_pt_loose = ak.pad_none(data["bjets_loose"]["pt"], 1, clip=True)
            data_bjet_pt_loose = ak.fill_none(data_bjet_pt_loose, 0)
            
            cats["b_loose_0"] = (loose_jets==0)
            cats["b_loose_1+_no_effect"] = (loose_jets>=1) & (data_bjet_pt_loose[:,0] <= data_bjet_pt_tight[:,0])
            cats["b_loose_1+_leading"] = (loose_jets>=1) & (data_bjet_pt_loose[:,0] > data_bjet_pt_tight[:,0])

        elif len(data) == 0:
            cats["b_loose_0"] = cut_array
            cats["b_loose_1+_no_effect"] = cut_array
            cats["b_loose_1+_leading"] = cut_array

        '''
        for i in range(len(data["Jet"])):
            print()
            print("Event number in btag_veto_cats",i)
            print("Jet pt",data["Jet"][i]["pt"])
            print("Btag flavor overall scores",data["Jet"][i]["btag"])

            print("tight_jets",tight_jets[i])
            #print("loose_jets",loose_jets[i])
            print("data_bjet_pt_tight",data_bjet_pt_tight[i])
            print("Real Probability values")
            print("B-",data["prob_neg_tight"][i])
            print("B0bar",data["prob_zerobar_tight"][i])
            print("B0",data["prob_zero_tight"][i])
            print("B+",data["prob_pos_tight"][i])

            print("data_bjet_pt_loose",data_bjet_pt_loose[i])
            print("Real loose Probability values")
            print("B-",data["prob_neg_loose"][i])
            print("B0bar",data["prob_zerobar_loose"][i])
            print("B0",data["prob_zero_loose"][i])
            print("B+",data["prob_pos_loose"][i])          

            print("id_bpos_rec_bpos",data["id_bpos_rec_bpos"][i])
            print("id_bneg_rec_bneg",data["id_bneg_rec_bneg"][i])

            print("inconclusive_mask",inconclusive_mask[i])
            print("conclusive_mask",conclusive_mask[i])
            print("opposite_charge_mask",opposite_charge_mask[i])

            print("b_loose_0",cats["b_loose_0"][i])
            print("b_loose_1+_no_effect",cats["b_loose_1+_no_effect"][i])
            print("b_loose_1+_leading_inconclusive_50",cats["b_loose_1+_leading_inconclusive_50"][i])
            print("b_loose_1+_leading_conclusive_50_wrong",cats["b_loose_1+_leading_conclusive_50_wrong"][i])
            print("b_loose_1+_leading_conclusive_50_right",cats["b_loose_1+_leading_conclusive_50_right"][i])
        '''

        return cats

    def btag_sf(self, is_mc, data):
        """Apply btag scale factors."""
        if is_mc and (
                "btag_sf" in self.config and len(self.config["btag_sf"]) != 0):
            weight, systematics = self.compute_weight_btag(data)
            return weight, systematics
        else:
            return np.ones(len(data))

    def mass_lepton_bjet(self, l_col, b_col, data):
        """Return invariant mass of lepton and leading b jet."""
        mlb_default = [0]*len(data[l_col])
        mlb_final = [0]*len(data[l_col])

        #empty_array = np.empty(len(data[l_col]))
        empty_array = [None]*len(data[l_col])
        have_bjets = ak.where(ak.num(data[b_col].eta)>0)
        #have_lepton = ak.where(ak.num(data[b_col].eta)>0) #This is required especially for data, because the exact one lepton cut is applied for gen only
        #print("Events having bjets",have_bjets)

        mlb_cut = data[ak.num(data[b_col].eta)>0]
        #print("Size of events satisfying the cut (having bjets)",len(mlb_cut))

        mlb_mask = ak.mask(data,ak.num(data[b_col].eta)>0)
        #print("Size of events satisfying the mask (having bjets)",len(mlb_mask))

        default_array = np.zeros((len(data[b_col]), 2))
        #Since this is being calculated after the at least one lepton cut is put, there is no need to ensure whether the lepton exists or not

        data_bjet_pt = ak.where(ak.is_none(mlb_mask), default_array, data[b_col].pt)
        data_bjet_mass = ak.where(ak.is_none(mlb_mask), default_array, data[b_col].mass) #What exactly is this?
        data_bjet_eta = ak.where(ak.is_none(mlb_mask), default_array, data[b_col].eta)
        data_bjet_phi = ak.where(ak.is_none(mlb_mask), default_array, data[b_col].phi)

        print("type of pt array",type(data[b_col].pt))
        print("Type of copied pt array",type(data_bjet_pt))

        cosh_1 = np.cosh(data[l_col].eta[:,0])
        sinh_1 = np.sinh(data[l_col].eta[:,0])

        cosh_2 = np.cosh(data_bjet_eta[:,0])
        sinh_2 = np.sinh(data_bjet_eta[:,0])

        cosh = np.cosh(data[l_col].eta[:,0]-data_bjet_eta[:,0])
        #cosh = np.cosh(-data_bjet_eta[:,0])
        cos = np.cos(data[l_col].phi[:,0]-data_bjet_phi[:,0])
        #
        #mlb_dummy = np.sqrt(2*data[l_col].pt[:,0]*data_bjet_pt[:,0]*(cosh - cos))
        #dr_dummy = np.sqrt(pow(data[l_col].eta[:,0]-data_bjet_eta[:,0],2) + pow(data[l_col].phi[:,0]-data_bjet_phi[:,0],2))

        lep_pt = data[l_col].pt[:,0]
        b_l_pt = data_bjet_pt[:,0]
        m_l_2 = pow(data[l_col].mass[:,0],2)
        m_b_2 = pow(data_bjet_mass[:,0],2)

        lep_p = np.sqrt(pow(lep_pt*cosh_1,2)+m_l_2)

        b_p = np.sqrt(pow(b_l_pt*cosh_2,2)+m_b_2)

        mlb_massive = np.sqrt(m_l_2 + m_b_2 + 2*lep_p*b_p - 2*lep_pt*b_l_pt*(sinh_1*sinh_2 + cos))
        print("final event size",len(mlb_massive))
        print("Size of array to be populated",len(mlb_final))
        #mlb_final = ak.where(ak.is_none(mlb_mask), [-1]*len(data[l_col]), mlb_massive)
        mlb_final = ak.where(ak.is_none(mlb_mask), empty_array, mlb_massive)
        '''
        for i in range(len(l_col)):
            print()
            print("MC Gen set")
            print("Event number",i)            

            #This is only to compare with previous spew files (of course the event number could also have been printed)
            print("B gen jet pt",data["bjets_gen"][i].pt) 
            print("Dummy B gen jet pt",data_bjet_pt[i]) 

            print("B gen jet eta",data["bjets_gen"][i].eta) 
            print("Dummy B gen jet eta",data_bjet_eta[i]) 

            print("B gen jet phi",data["bjets_gen"][i].phi) 
            print("Dummy B gen jet phi",data_bjet_phi[i]) 

            print("Cosh",cosh[i])
            print("Cos",cos[i])
            print("mlb_dummy",mlb_dummy[i])
            print("dr_dummy",dr_dummy[i])

            print("Individual masses")
            print("Lepton mass",data[l_col].mass[i])
            print("B quark mass?",data[b_col].mass[i])
            print("Invariant mass calculated considering particle masses",mlb_massive[i])
            #This is JUST for debugging, to check whether the inbuilt mass function gives the same value as the one used here
            if i in have_bjets[0]:
                #print("Invariant mass calculated here",np.sqrt(2*lep_pt*b_l_pt*(cosh_1*cosh_2 - sinh_1*sinh_2 -cos))[i])
                
                
                #print()
                print("Mass from inbuilt function",(data[l_col][i, 0] + data[b_col][i, 0]).mass)
                print("Delta r from inbuilt function",data[b_col][i, 0].delta_r(data[l_col][i, 0]))


            print("Lepton gen pt",data["GenLepton"][i].pt)
            print("Gen transverse mass of W",data["GenmTW"][i])
            print("Matching gen pt",data["Jet"][i].matched_gen.pt)
            #print("Muon gen pt?",data["GenPart"][abs(data["GenPart"].pdgId) == 13][i].pt)
            print("Jet gen pt",data["GenJet"][i].pt)
            print("Number of btagged gen jets",data["nbtag_gen"][i])
            print("Invariant gen lepton and b jet mass (excluding MET)", mlb_mask[i])

            print("Type of object of bjets",type(data["bjets_gen"][i]))
            #print("Invariant gen lepton and b jet mass (excluding MET)", mlb[i])

        '''    

        '''
        for i in range(len(have_bjets)):
            for j in range(len(have_bjets[i])): #Weird syntax, i is also an array somehow, so have to do this
                #print("Entry number",have_bjets[i][j])
                #print("Lepton object",data[l_col][have_bjets[i][j], 0])

                #Instead of printing in a text file, have this appear in the terminal window

                logger.info("Absolute iteration number=%s",j)
                logger.info("Entry number=%s",have_bjets[i][j])
                #logger.info("Lepton object",data[l_col][have_bjets[i][j], 0])
                
                mlb[have_bjets[i][j]] = (data[l_col][have_bjets[i][j], 0] + data[b_col][have_bjets[i][j], 0]).mass
                #print("Invariant mass excluding MET",mlb[have_bjets[i][j]])
                logger.info("Invariant mass excluding MET=%s",mlb[have_bjets[i][j]])

        '''       
        return mlb_final
        #return mlb_mask

    def dR_lepton_bjet(self, l_col, b_col, data):
        """Return dR between lepton and leading b jet."""

        dr_mask = ak.mask(data,ak.num(data[b_col].eta)>0)

        #of course lengths of l_col and b_col are supposed to be the same, that is why they are used interchangeably
        default_array = np.zeros((len(data[b_col]), 2))
        #empty_array = [None]*len(data[l_col])
        empty_array = [[]]*len(data[l_col])
        #Since this is being calculated after the at least one lepton cut is put, there is no need to ensure whether the lepton exists or not

        data_bjet_pt = ak.where(ak.is_none(dr_mask), default_array, data[b_col].pt)
        data_bjet_mass = ak.where(ak.is_none(dr_mask), default_array, data[b_col].mass) #What exactly is this?
        data_bjet_eta = ak.where(ak.is_none(dr_mask), default_array, data[b_col].eta)
        data_bjet_phi = ak.where(ak.is_none(dr_mask), default_array, data[b_col].phi)

        dr_dummy_old = np.sqrt(pow(data[l_col].eta[:,0]-data_bjet_eta[:,0],2) + pow(data[l_col].phi[:,0]-data_bjet_phi[:,0],2))

        pi = 4*math.atan(1)
        #This is added because of the error found at the time of the mother correction script - this is not working directly, so flatten, compute and apply the mask, and then unflatten
        delta_phi = abs(data[l_col].phi[:,0]-data_bjet_phi[:,0])
        pi_array = [2*pi]*len(data)

        large_phi_mask = ak.mask(delta_phi,delta_phi>pi)
        delta_phi_final = ak.where(ak.is_none(large_phi_mask), delta_phi, pi_array-delta_phi)

        #Only for debugging
        #count_pi = ak.count(ak.mask(delta_phi,delta_phi>pi),axis=1)

        dr_dummy = np.sqrt(pow(data[l_col].eta[:,0]-data_bjet_eta[:,0],2) + pow(delta_phi_final,2))
        dr_final = ak.where(ak.is_none(dr_mask), empty_array, dr_dummy)

        '''
        for i in range(len(data["Jet"])):
            print()
            print("Event number in dR_lepton_bjet",i)
            print("Leading B jet pt",data[b_col][i].pt)
            print("dR dummy not adjusted",dr_dummy_old[i])

            print("Wrong phi (not adjusted for greater than pi values)",data[l_col].phi[:,0][i]-data_bjet_phi[:,0][i])
            print("Absolute value of unadjusted delta phi",abs(data[l_col].phi[:,0][i]-data_bjet_phi[:,0][i]))
            #print("Adjusted value of delta phi",delta_phi[i])
            #print("Number of b quarks in event (counts for unflattening delta_phi_new)",ak.num(data_bpart_phi,axis=1)[i])
            print("Adjusted value of delta phi after flattening, applying mask and unflattening (delta_phi_final)",delta_phi_final[i])
            #Only for debugging
            #print("Values of delta phi greater than pi",count_pi[i])
            print("dr_final",dr_final[i])
            #Comparing with delta_r from the vector.py
            if (len(data[b_col][i])>0):
                dR_function = data[l_col][i][0].delta_r(data[b_col][i][0])
                print("dR dummy adjusted",dr_dummy[i])
                print("dR function",dR_function)
           
        '''    
        return dr_final

    def dphi_lepton_bjet(self, l_col, b_col, data):
        """Return dphi between lepton and leading b jet."""

        dr_mask = ak.mask(data,ak.num(data[b_col].eta)>0)
        #print("Size of events satisfying the mask (having bjets)",len(dr_mask))

        #of course lengths of l_col and b_col are supposed to be the same, that is why they are used interchangeably
        default_array = np.zeros((len(data[b_col]), 2))
        #empty_array = [None]*len(data[l_col])
        empty_array = [[]]*len(data[l_col])
        #Since this is being calculated after the at least one lepton cut is put, there is no need to ensure whether the lepton exists or not

        data_bjet_phi = ak.where(ak.is_none(dr_mask), default_array, data[b_col].phi)

        pi = 4*math.atan(1)
        #print("value of pi",pi)

        #This is added because of the error found at the time of the mother correction script - this is not working directly, so flatten, compute and apply the mask, and then unflatten
        delta_phi = abs(data[l_col].phi[:,0]-data_bjet_phi[:,0])
        pi_array = [2*pi]*len(data)

        large_phi_mask = ak.mask(delta_phi,delta_phi>pi)
        delta_phi_new = ak.where(ak.is_none(large_phi_mask), delta_phi, pi_array-delta_phi)
        delta_phi_final = ak.where(ak.is_none(dr_mask), empty_array, delta_phi_new)

        '''
        for i in range(len(data["Jet"])):
            print()
            print("Event number in dphi_lepton_bjet",i)
            print("Leading B jet pt",data[b_col][i].pt)

            print("Wrong phi (not adjusted for greater than pi values)",data[l_col].phi[:,0][i]-data_bjet_phi[:,0][i])
            print("Absolute value of unadjusted delta phi",abs(data[l_col].phi[:,0][i]-data_bjet_phi[:,0][i]))
            print("Adjusted value of delta phi after flattening, applying mask and unflattening (delta_phi_new)",delta_phi_new[i])
            print("delta_phi_final",delta_phi_final[i])
        '''
        #hahaha
        return delta_phi_final

    def eta_leading_bjet(self, b_col, data):
        dr_mask = ak.mask(data,ak.num(data[b_col].eta)>0)
        #print("Size of events satisfying the mask (having bjets)",len(dr_mask))

        #of course lengths of l_col and b_col are supposed to be the same, that is why they are used interchangeably
        default_array = np.zeros((len(data[b_col]), 2))
        #empty_array = [None]*len(data[l_col])
        #empty_array = [[]]*len(data[b_col])
        empty_array = [-1]*len(data[b_col])

        data_bjet_eta = ak.where(ak.is_none(dr_mask), default_array, data[b_col].eta)
        abs_eta_bjet = abs(data_bjet_eta[:,0])
        bjet_eta_final = ak.where(ak.is_none(dr_mask), empty_array, abs_eta_bjet)

        '''
        for i in range(len(data["Jet"])):
            print()
            print("Event number in eta_leading_bjet",i)
            print("Leading B jet pt",data[b_col][i].pt)
            print("Leading B jet eta",data[b_col][i].eta)
            print("Absolute value of leading B jet eta",abs_eta_bjet[i])
            print("bjet_eta_final",bjet_eta_final[i])
        '''
        #hahaha
        return bjet_eta_final

    def eta_subleading_bjet(self, b_col, data):
        dr_mask = ak.mask(data,ak.num(data[b_col].eta)>1) #Obviously we need at least 2 here
        #print("Size of events satisfying the mask (having bjets)",len(dr_mask))

        #of course lengths of l_col and b_col are supposed to be the same, that is why they are used interchangeably
        default_array = np.zeros((len(data[b_col]), 2))
        #empty_array = [[]]*len(data[b_col])
        empty_array = [-1]*len(data[b_col])

        data_bjet_eta = ak.where(ak.is_none(dr_mask), default_array, data[b_col].eta)
        abs_eta_bjet = abs(data_bjet_eta[:,1])
        bjet_eta_final = ak.where(ak.is_none(dr_mask), empty_array, abs_eta_bjet)

        '''
        for i in range(len(data["Jet"])):
            print()
            print("Event number in eta_subleading_bjet",i)
            print("SubLeading B jet pt",data[b_col][i].pt)
            print("SubLeading B jet eta",data[b_col][i].eta)
            print("Absolute value of subleading B jet eta",abs_eta_bjet[i])
            print("Subleading bjet_eta_final",bjet_eta_final[i])
            print("Eta of leading bjet from the other function",data["abs_eta_b"][i])
        '''
        #hahaha
        return bjet_eta_final

    #This method is not quite right, but even taking them from the charge tagger, not sure whether anything new will actually be obtained from this,
    #because the rec (highest probability score) meson categories have already been made
    def charge_lepton_bjet(self, l_col, b_col, data):
        """Return absolute total charge of lepton and leading b jet - used to try to gauge whether it is coming from a resonance"""

        pdgid_mask = ak.mask(data,ak.num(data[b_col].eta)>0)
        #print("Size of events satisfying the mask (having bjets)",len(mlb_mask))

        default_array = np.zeros((len(data[b_col]), 2))
        empty_array = [None]*len(data[l_col])
        #Since this is being calculated after the at least one lepton cut is put, there is no need to ensure whether the lepton exists or not

        mlb_debug = self.mass_lepton_bjet(l_col,b_col,data)

        #data_bjet_q = ak.where(ak.is_none(pdgid_mask), default_array, data[b_col].pdgId)
        data_bjet_eta = ak.where(ak.is_none(pdgid_mask), default_array, data[b_col].eta)
        data_bjet_phi = ak.where(ak.is_none(pdgid_mask), default_array, data[b_col].phi)

        b_part_col = data["GenPart"][abs(data["GenPart"].pdgId) == 5]
        
        dr_part = np.sqrt(pow(data_bjet_eta[:,0]-b_part_col.eta,2) + pow(data_bjet_phi[:,0]-b_part_col.phi,2))

        b_part_col = b_part_col[ak.argsort(dr_part, ascending=True)]
        dr_part = dr_part[ak.argsort(dr_part, ascending=True)]

        data_bjet_q = ak.where(ak.is_none(pdgid_mask), default_array, b_part_col.pdgId)
        q_lep = -data[l_col].pdgId[:,0]/abs(data[l_col].pdgId[:,0]) #Since the particles being assigned positive pdg id are negatively charged
        
        #'''
        q_b = (1/3)*(-data_bjet_q[:,0]/abs(data_bjet_q[:,0])) #Here also the b is negatively charged and bbar is positively charged, and extra 1/3 for the actual charge

        #q_abs_net = abs(q_lep+q_b)
        q_net = q_lep+q_b

        q_final = ak.where(ak.is_none(pdgid_mask), empty_array, q_net)
        #'''

        return q_final
        #return q_lep

    def cumulative_sum(self,cumulative_ngenpart_ak,rel_index):
        #Sum for the cumulative indices
        sum_combo = ak.cartesian([cumulative_ngenpart_ak, rel_index], axis=1)
        print("sum_combo",sum_combo)
        sum_combo_flat = ak.flatten(sum_combo)
        genpart_size, relative_index = ak.unzip(sum_combo_flat) #Trying this out inside of the 3 index labeling
        genpart_size = ak.unflatten(genpart_size,1)
        relative_index = ak.unflatten(relative_index,1)
        print("genpart_size",genpart_size)
        print("relative_index",relative_index)
        concat_sum = ak.concatenate([genpart_size,relative_index],axis=1)
        print("concat_sum",concat_sum)
        actual_index = ak.to_numpy(ak.sum(concat_sum,axis=1))
        print("actual_index",actual_index)
        return actual_index

    def do_trigger_sf_sl(self, data):
        """Compute identification and isolation scale factors for
           leptons (electrons and muons)."""
        lead_muons = data["Muon"][:,0]
        weight = np.ones(len(data))
        # Electron trig efficiency
        # TBA 
        # Muon trig efficiency, systematics TBA
        sffunc =self.config["trigger_sf_sl"]
        params = {}
        for dimlabel in sffunc.dimlabels:
            if dimlabel == "abseta":
                params["abseta"] = abs(lead_muons.eta)
            else:
                params[dimlabel] = getattr(lead_muons, dimlabel)
        central = sffunc(**params)

        weight = weight * central
        return weight

    def w_mT(self, data):
        """Get the pt of the four vector difference of the MET and the
        neutrinos"""
        met = data["MET"]
        lep = data["Lepton"]
        mt = np.sqrt(2*(met.pt*lep.pt-met.x*lep.x-met.y*lep.y))
        return mt

    def chain_last(self,pdgId_req,counts_chain,recursion_level,chain_current, last_index,data):
        #print("Reached level ", str(recursion_level)," of mothers")

        end_index = -1
        #end_index = 10 #Only for debugging and testing since no event at this level actually seems to have the index -1

        #fresh_mask = abs(data["GenPart"][data["GenPart"][last_index].genPartIdxMother].pdgId) != pdgId_req

        #Assuming exactly 3 arguments in pdgid_arr
        fresh_mask = (abs(data["GenPart"][data["GenPart"][last_index].genPartIdxMother].pdgId) != pdgId_req[0]) & (abs(data["GenPart"][data["GenPart"][last_index].genPartIdxMother].pdgId) != pdgId_req[1]) & (abs(data["GenPart"][data["GenPart"][last_index].genPartIdxMother].pdgId) != pdgId_req[2])
        #mask = (abs_pdgid == pdgid_arr[0]) | (abs_pdgid == pdgid_arr[1]) | (abs_pdgid == pdgid_arr[2])

        #These are the number of events that have only repeat mothers and do not any mothers at the next level (i.e. something like mother)
        repeat_end = ak.num(last_index[fresh_mask],axis=1)
        repeat_end_num = ak.num(repeat_end[repeat_end==ak.num(last_index,axis=1)],axis=0)
        #print("Number of events having only mothers that are also at the end (beginning)",repeat_end_num)

        repeat_mask = ak.mask(last_index,fresh_mask)
        index_gen = ak.local_index(data["GenPart"],axis=1)
        #index_gen_pdgid = index_gen[abs(data["GenPart"]["pdgId"]) == pdgId_req]
        #Assuming exactly 3 arguments in pdgid_arr
        index_gen_pdgid = index_gen[(abs(data["GenPart"]["pdgId"]) == pdgId_req[0]) | (abs(data["GenPart"]["pdgId"]) == pdgId_req[1]) | (abs(data["GenPart"]["pdgId"]) == pdgId_req[2])]

        '''
        for i in range(len(data["Jet"])):
            print()
            print("Event number",i)
            print("pdg id list")
            print(ak.to_numpy(data["GenPart"][i]["pdgId"]))    

            #print("Lepton pdg id",data["Lepton"].pdgId[:,0][i])
            #print("Non repeating non lepton mothers",repeat_only_pdg[i])
            print("Mother indices at level",str(recursion_level),last_index[i])
            print("Mother pdgids at level",str(recursion_level),data["GenPart"][last_index].pdgId[i])
            print("Repeat mask",repeat_mask[i])
            print("Number of mothers for each event at first level",ak.num(last_index,axis=1)[i])
            print("Number of particles having mother index as ", str(end_index),"for each event",repeat_end[i])
            print("Mother indices at second level",data["GenPart"][last_index].genPartIdxMother[i])
            print("chain_current",chain_current[i])
            print("counts_chain",ak.unflatten(counts_chain,ak.num(last_index))[i])
            print("index_gen_pdgid",index_gen_pdgid[i])
        '''    

        if repeat_end_num < len(data["Jet"]): #Go the next level of mothers
            last_index_2 = data["GenPart"][last_index].genPartIdxMother
            last_index = ak.unflatten(ak.where(ak.is_none(ak.flatten(repeat_mask)), ak.flatten(last_index_2), ak.flatten(last_index)),ak.num(last_index))

            '''
            print("chain_current before concatenation",chain_current)
            print(ak.flatten(chain_current))
            print(ak.unflatten(ak.flatten(chain_current),recursion_level))
            print("last_index before concatenation",last_index)
            print(ak.flatten(last_index))
            print(ak.unflatten(ak.flatten(last_index),1))
            '''

            if recursion_level==1:
                chain_inter = ak.unflatten(ak.flatten(chain_current),recursion_level)
            elif recursion_level>1:
                chain_inter = ak.flatten(chain_current)

            chain_flat = ak.concatenate([chain_inter,ak.unflatten(ak.flatten(last_index),1)],axis=1)
            #print("chain_flat",chain_flat)
            recursion_level +=1

            counts_recursion = [recursion_level]*len(ak.flatten(last_index))
            counts_chain = ak.where(ak.is_none(ak.flatten(repeat_mask)), counts_recursion, counts_chain)

            #print("counts_recursion",counts_recursion)
            print("counts_chain",counts_chain)

            #chain_current = ak.unflatten(chain_flat,recursion_level*ak.num(last_index))
            chain_current = ak.unflatten(chain_flat,ak.num(last_index))
            #print("chain_current",chain_current)

            counts_chain, chain_current, last_index = self.chain_last(pdgId_req,counts_chain,recursion_level,chain_current,last_index,data)
            
        return counts_chain, chain_current, last_index


    def width_reweighting(self,is_mc,width_num,dsname,data):
        #print("Weights",selector.data["LHEWeight"][j]["width_" +str(i)])
        if "width_weights" in self.config:
            #width_weight_file = "/afs/desy.de/user/p/paranjpe/top_wbwb/pepper/inputs/common/width_weights.hjson"
            width_weight_file = "/afs/desy.de/user/p/paranjpe/top_wbwb/pepper/inputs/common/width_weights.yaml"
            #print("Keys in File used for mc_lumifactors (just for comparing)",self.config["mc_lumifactors"].keys())
            #print("File used for width_weights",self.config["width_weights"])
            
            #print("Keys in File used for width_weights",self.config["width_weights"].keys())
            #width_weights = json.load(open(self.config["width_weights"]))
            
            f = open(width_weight_file)
            width_weights = yaml.safe_load(f)
            #print("Contents of width_weights file?",width_weights)
            #print("Dataset name",dsname)
            event_counts = width_weights[dsname]["Event_counts"]
            width_value = width_weights[dsname]["Width"][str(width_num)]["Value"]
            weight_sum = width_weights[dsname]["Width"][str(width_num)]["Weight_sum"]

            #print("Event counts from yaml file", event_counts)
            #print("Width value", width_value)
            #print("Sum of weights", weight_sum)
            #weight = (data["LHEWeight"]["width_" +str(width_num)])*(21300000.0/373959295.4848724) #denominator is wght_sum for width_21
            weight = (data["LHEWeight"]["width_" +str(width_num)])*(event_counts/weight_sum)

            
        '''
        print("Width number in the width_reweighting function",width_num)
        print("Weight values for all events for this width")
        for j in range(len(data)):
            print()
            print("Event number",j)
            print("Width number inside event loop",width_num)
            print("Weights",data["LHEWeight"][j]["width_" +str(width_num)])
            print("Normalized weights",weight[j])
            #print("Weights",selector.data["LHEWeight_width_" +str(i)][j])
        '''
        return weight
