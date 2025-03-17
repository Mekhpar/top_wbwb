#=================================================================================================================================

#This is going to remove all the chain making parts, so there will be no mother condition or categories made even for MC

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

#This is for extra branches related to the b charge tagger
#These are jet constituents that are not usually stored in nano aod files

logger = logging.getLogger(__name__)

# All processors inherit from pepper.ProcessorBasicPhysics
class Processor(pepper.ProcessorBasicPhysics):

    config_class = pepper.ConfigBasicPhysics

    def __init__(self, config, eventdir):

        # Initialize the class, maybe overwrite some config variables and
        # load additional files if needed
        # Need to call parent init to make histograms and such ready
        super().__init__(config, eventdir)

        # It is not recommended to put anything as member variable into a
        # a Processor because the Processor instance is sent as raw bytes
        # between nodes when running on HTCondor.

    def process_selection(self, selector, dsname, is_mc, filler):
        # Implement the selection steps
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
        idx = selector.data["jetorigin"]["jetIdx"]
        jetorigin = selector.data["jetorigin"]
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
        gen_had_flavor = ak.to_numpy(ak.flatten(jetorigin["hadronFlavor"]))
        gen_bhad_id = ak.to_numpy(ak.flatten(jetorigin["bHadronId"]))
        reco_btag = ak.to_numpy(ak.flatten(selector.data["Jet"]["btagged"])*5.) #This is a boolean array which is converted to b hadron flavor - that is why the 5

        print("Number of entries in the gen and reco id arays",len(gen_had_flavor),len(gen_bhad_id),len(reco_btag))
        rec_gen_bhad = ak.from_numpy(np.array(list(zip(gen_had_flavor,gen_bhad_id,reco_btag))).flatten())
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
                
        
        if is_mc:
            selector.set_column("b_gen_meson", jetorigin["bHadronId"])
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
        #print("Number of jets before build_jet_column",len(selector.data["Jet"]))
        selector.data["Jet"], selector.data["b_gen_meson"], tfdata = self.build_jet_column(is_mc, tfdata, 1, selector.data)
        #print("Number of jets after build_jet_column",len(selector.data["Jet"]))
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


        #selector.set_multiple_columns({"Jet","b_meson_tag"},partial(self.build_jet_column, is_mc, 1)) #added meson tag #Wrong syntax

        if "jet_puid_sf" in self.config and is_mc:
            selector.add_cut("JetPUIdSFs", self.jet_puid_sfs)

        #selector.set_column("Jet", self.jets_with_puid) 
        
        selector.data["Jet"], selector.data["b_meson_tag"],selector.data["b_gen_meson"], selector.data["prob_neg"], selector.data["prob_zerobar"], selector.data["prob_zero"], selector.data["prob_pos"] = self.jets_with_puid(selector.data) 
        #selector.set_multiple_columns({"Jet","b_meson_tag"}, self.jets_with_puid) #added meson tag

        smear_met = "smear_met" in self.config and self.config["smear_met"]
        selector.set_column(
            "MET", partial(self.build_met_column, is_mc, variation.junc,
                           variation.jer if smear_met else None, selector.rng,
                           era, variation=variation.met))
        
        #selector.add_cut("JetPtReq", self.jet_pt_requirement)
        #print("Number of jets before applying JetPtReq (replace if everything is zero)",len(selector.data["Jet"]))
        selector.add_cut("JetPtReq", partial(self.jet_pt_requirement,"Jet"))
        #March 8 - this is where 6 for the last chunk in wjets 0j 331.root becomes zero
        #print("Number of jets after applying JetPtReq (replace if everything is zero)",len(selector.data["Jet"]))
        if is_mc and self.config["compute_systematics"]:
            self.scale_systematics_for_btag(selector, variation, dsname)
        
        #selector.set_column("nbtag", self.num_btags)
        selector.set_column("nbtag", partial(self.num_btags,""))
        #selector.set_column("bjets_tight", partial(self.build_btag_column,is_mc))  
        #selector.set_multiple_columns({"bjets_tight","meson_tag_real_tight"}, partial(self.build_btag_column,is_mc))  #added meson tag

        selector.data["real_flag_tight"], selector.data["btag_tight"], selector.data["bjets_tight"], selector.data["meson_tag_real_tight"],selector.data["gen_meson_real_tight"], selector.data["prob_neg_real_tight"], selector.data["prob_zerobar_real_tight"], selector.data["prob_zero_real_tight"], selector.data["prob_pos_real_tight"] = self.build_btag_column(is_mc,"btagged", "tight", selector.data)
        selector.data["real_flag_loose"], selector.data["btag_loose"], selector.data["bjets_loose"], selector.data["meson_tag_real_loose"],selector.data["gen_meson_real_loose"], selector.data["prob_neg_real_loose"], selector.data["prob_zerobar_real_loose"], selector.data["prob_zero_real_loose"], selector.data["prob_pos_real_loose"] = self.build_btag_column(is_mc,"btagged_loose", "loose", selector.data)
        '''
        for i in range(len(selector.data["Jet"])):
            print()
            print("Event number",i)
            print("Jet pt",selector.data["Jet"][i]["pt"])
            print("Btag flavor overall scores",selector.data["Jet"][i]["btag"])
            print("Tight jets")
            print("Bjet meson tag reshaped",selector.data["meson_tag_real_tight"][i])
            print("Bjet gen meson tag",selector.data["gen_meson_real_tight"][i])
            print("Btag flavor scores",selector.data["btag_tight"][i])
            print("Real flag tight",selector.data["real_flag_tight"][i])
            print("Bjet tight pt",selector.data["bjets_tight"][i]["pt"])
            print("Real Probability values")
            print("B-",selector.data["prob_neg_real_tight"][i])
            print("B0bar",selector.data["prob_zerobar_real_tight"][i])
            print("B0",selector.data["prob_zero_real_tight"][i])
            print("B+",selector.data["prob_pos_real_tight"][i])

            print("Loose (medium) jets")
            print("Bjet meson tag reshaped",selector.data["meson_tag_real_loose"][i])
            print("Bjet gen meson tag",selector.data["gen_meson_real_loose"][i])
            print("Btag flavor scores",selector.data["btag_loose"][i])
            print("Bjet tight pt",selector.data["bjets_loose"][i]["pt"])
            print("Real flag loose",selector.data["real_flag_loose"][i])

            print("Real loose Probability values")
            print("B-",selector.data["prob_neg_real_loose"][i])
            print("B0bar",selector.data["prob_zerobar_real_loose"][i])
            print("B0",selector.data["prob_zero_real_loose"][i])
            print("B+",selector.data["prob_pos_real_loose"][i])

            print("Number of tight jets",len(selector.data["bjets_tight"][i]))
            print("Number of loose (medium) jets",len(selector.data["bjets_loose"][i]))

        hahaha
        '''
        #==================================Fake and missing, not quite sure what to do with them at the moment==============================
        if is_mc:
            selector.data["bjets_miss"], selector.data["gen_meson_miss"], selector.data["prob_neg_miss"], selector.data["prob_zerobar_miss"], selector.data["prob_zero_miss"], selector.data["prob_pos_miss"] = self.build_missing_btag_column(is_mc,selector.data)
            selector.data["bjets_fake"], selector.data["meson_tag_fake"],selector.data["gen_meson_fake"], selector.data["prob_neg_fake"], selector.data["prob_zerobar_fake"], selector.data["prob_zero_fake"], selector.data["prob_pos_fake"] = self.build_fake_btag_column(is_mc,selector.data)

        selector.set_cat("jet_btag", {"j2_b0", "j2_b1","j2_b2","j2+_b0", "j2+_b1","j2+_b2"})
        selector.set_multiple_columns(self.btag_categories)

        #selector.set_cat("bjet_eta", {"no_bjet","l_be","l_fwd","sl_be","sl_fwd","l_be_sl_be","l_fwd_sl_be","l_be_sl_fwd","l_fwd_sl_fwd"})


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

        print("Number of events before charged meson cut",len(selector.data))
        #selector.data = self.Bmeson_charge_score_cut(is_mc,selector.data)
        #Might remove for debugging
        selector.add_cut("Charged_Bmeson", partial(self.Bmeson_charge_score_cut,is_mc))
        
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

        '''
        data,lepton_mother_1 = self.lepton_mother(selector.data)
        lep_recursion_level = 1
        selector.data["lep_mother_final"] = self.lepton_recursion(lep_recursion_level,lepton_mother_1,selector.data)
        #Not put none condition for lepton at the moment

        b_mom_time_before = time.time()
        #print("b_mom_time_before",b_mom_time_before)
        '''
        '''
        index_gen = ak.local_index(selector.data["GenPart"],axis=1)
        abs_pdgid = abs(selector.data["GenPart"]["pdgId"])

        #==================================================================b quark chains================================================================

        b_mask = abs_pdgid == 5
        bquark_index = index_gen[b_mask]
        b_object = selector.data["GenPart"][b_mask]
        b_mother = b_object["genPartIdxMother"] #This is nothing but a set of indices
        b_fresh = bquark_index[abs(selector.data["GenPart"][b_mother]["pdgId"])!=5]

        bquark_index = ak.sort(bquark_index,axis=1,ascending=True)   
        b_1 = ak.unflatten(b_fresh,1,axis=1)

        recursion_level = 1
        b_init_final = [[]]*len(selector.data["Jet"])
        selector.data["b_fin_final"] = self.make_chain(b_fresh,bquark_index,b_1,b_init_final,recursion_level,selector.data)

        #===================================================================================================================================================
        '''

        #Trying out new chain making here - starting from last to first (suggestion by Alberto)
        #because it is much easier to find the mother index rather than compare a mother index (finding the daughters essentially)
        '''
        index_gen = ak.local_index(selector.data["GenPart"],axis=1)
        abs_pdgid = abs(selector.data["GenPart"]["pdgId"])

        mask = abs_pdgid == 5
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
        
        recursion_level = 1
        #last_index_object = [last_index]
        #last_index_object = self.chain_last(recursion_level,last_index_object,selector.data)          
        chain_current = last_index
        counts_chain = [1]*len(ak.flatten(last_index))
        counts_chain, chain_current, last_index = self.chain_last(5, counts_chain, recursion_level, chain_current, last_index, selector.data) 
        counts_chain_final = ak.unflatten(counts_chain,ak.num(last_index))
        print(chain_current)
        chain_flat_np = ak.to_numpy(ak.flatten(chain_current,axis=1))
        index_chain_flat = ak.to_numpy(ak.local_index(ak.flatten(chain_current,axis=1)))

        print(ak.flatten(chain_current,axis=1))
        print(chain_flat_np)
        print(index_chain_flat)

        chain_flag = index_chain_flat < counts_chain[:, None]
        #print(chain_flag)
        #The chain_flag gave fully flat array, so first unflatten according to the number of elements in each chain (counts_chain) and then wrt last_index
        chain_final = ak.unflatten(ak.unflatten((ak.flatten(chain_current,axis=1)[chain_flag]),counts_chain),ak.num(last_index))
        print(chain_final)

        for i in range(len(data["Jet"])):
            print()
            print("Event number after chain_last function",i)
            print("chain_current",chain_current[i])
            print("counts_chain",counts_chain_final[i])
            print("chain_final",chain_final[i])
            #print("b_fin_final",selector.data["b_fin_final"][i])
        '''

        '''
        selector.data["b_fin_final"] = chain_final
        #This was causing the script in debug mode to terminate, and really we do not need it, so it is commented out
        #selector.data["w_fresh"], selector.data["w_fin_final"] = self.chain_full(24,data)

        selector.data["e_mu_tau_fresh"] = self.lepton_chain_full(0,[11,13,15],data)
        selector.data["nu_fresh"] = self.lepton_chain_full(0,[12,14,16],data)

        data, min_delta_r, selector.data["bjet_part_dr"] = self.dR_bjet_chain_match(selector.data)
        b_mom_time_after = time.time()
        #print("Time taken with make_chain method",b_mom_time_after - b_mom_time_before)

        b_mask = ak.mask(data,(ak.num(selector.data["bjets_tight"].eta)>0)&(ak.num(selector.data["bjets_tight"].matched_gen.eta)>0)) #So it should have reco pt and also matched gen pt
        empty_b_array = [None]*len(selector.data["bjets_tight"])
        #selector.data["non_b_mother_final"] = ak.where(ak.is_none(b_mask), empty_b_array, non_b_mother)
        selector.data["non_b_mother_final"] = ak.where(ak.is_none(b_mask), empty_b_array, min_delta_r)

        #==================Picking only resonant leptons for gen level first step of W reconstruction (two particle invariant mass only)=========================

        e_mu_tau_fresh_mother_id = abs(selector.data["GenPart"][selector.data["GenPart"][selector.data["e_mu_tau_fresh"]]["genPartIdxMother"]].pdgId)
        selector.data["e_mu_tau_fresh_resonant"] = selector.data["e_mu_tau_fresh"][e_mu_tau_fresh_mother_id == 24]
        nu_fresh_mother_id = abs(selector.data["GenPart"][selector.data["GenPart"][selector.data["nu_fresh"]]["genPartIdxMother"]].pdgId)
        selector.data["nu_fresh_resonant"] = selector.data["nu_fresh"][nu_fresh_mother_id == 24]

        #one_fresh_gen_lepton_mask = self.one_fresh_gen_lepton(is_mc, selector.data)
        selector.add_cut("HasOneGenLepton", partial(self.one_fresh_gen_lepton, is_mc), no_callback=True)

        w_mother_object_fresh = selector.data["GenPart"][selector.data["GenPart"][selector.data["e_mu_tau_fresh_resonant"]]["genPartIdxMother"]]
        #w_mother_object = selector.data["GenPart"][selector.data["GenPart"][selector.data["e_mu_tau_first_resonant"]]["genPartIdxMother"]]

        #==========================================================================================================================================================

        selector.data["Gen_emutau"], selector.data["Gen_emutau_mask"], lep_muon_mass_mask = self.build_genpart_column(is_mc, selector.data["e_mu_tau_fresh_resonant"], selector.data)
        #selector.data["Gen_nu"], selector.data["Gen_nu_mask"], nu_muon_mass_mask = self.build_genpart_column(is_mc, selector.data["nu_first_resonant"], selector.data)
        selector.data["Gen_nu"], selector.data["Gen_nu_mask"], nu_muon_mass_mask = self.build_genpart_column(is_mc, selector.data["nu_fresh_resonant"], selector.data)
        selector.data["W_mass"] = (selector.data["Gen_emutau"][:,0] + selector.data["Gen_nu"][:,0]).mass

        lnu_none_mask = ak.is_none(selector.data["Gen_emutau_mask"])|ak.is_none(selector.data["Gen_nu_mask"])
        selector.data["W_mass_final"] = ak.where(lnu_none_mask, final_array, selector.data["W_mass"]) 
        '''

        '''
        #==================Picking only resonant b and W for gen level first step of top reconstruction (two particle invariant mass only)=========================

        b_first = selector.data["b_fin_final"][:,:,0]
        b_last = selector.data["b_fin_final"][:,:,-1]
        bmother_id = abs(selector.data["GenPart"][selector.data["GenPart"][b_first]["genPartIdxMother"]].pdgId)
        selector.data["b_resonant"] = selector.data["b_fin_final"][bmother_id == 6]
        selector.data["b_first_resonant"] = selector.data["b_resonant"][:,:,0]

        w_first = selector.data["w_fin_final"][:,:,0]
        w_last = selector.data["w_fin_final"][:,:,-1]
        wmother_id = abs(selector.data["GenPart"][selector.data["GenPart"][w_first]["genPartIdxMother"]].pdgId)
        selector.data["w_resonant"] = selector.data["w_fin_final"][wmother_id == 6]
        selector.data["w_first_resonant"] = selector.data["w_resonant"][:,:,0]
        selector.data["w_last_resonant"] = selector.data["w_resonant"][:,:,-1]

        #==========================================================================================================================================================

        '''
        #Only print for leptons (muons) because the nested list problem is occuring after that for the first time
        '''
        for i in range(len(selector.data["Jet"])):
            print()
            print("Event number in build_genpart_column function",i)
            print("Pdg id of fresh leptons",selector.data["GenPart"][selector.data["e_mu_tau_fresh_resonant"]].pdgId[i])
            print("Checking whether the fresh mother is indeed W",w_mother_object_fresh.pdgId[i])
            print("Number of fresh leptons that supposedly have a W mother",len(w_mother_object_fresh.pdgId[i]))
            print("one_fresh_gen_lepton_mask",one_fresh_gen_lepton_mask[i])

            if len(w_mother_object.pdgId[i]) > 1:
                print("Problematic multiple leptons having W as a mother")
                #self.event_display(i, "/afs/desy.de/user/p/paranjpe/top_wbwb/pepper/top_tchannel_weird_leptons/", selector.data)
        '''
        #hahaha

        #==========================================================================================================================================================


        #Now build a set of nice columns like in build_lepton_column (from the processor basic script) and also account for the events that have these missing
        #(using ak.mask maybe)
        #Is there really a need to put a mask on this? Maybe not for this specific (single top) sample
        #selector.data["Gen_emutau"], selector.data["Gen_emutau_mask"], lep_muon_mass_mask = self.build_genpart_column(is_mc, selector.data["e_mu_tau_first_resonant"], selector.data)
        '''

        selector.data["Gen_W"], selector.data["Gen_W_mask"], W_muon_mass_mask = self.build_genpart_column(is_mc, selector.data["w_first_resonant"], selector.data)
        selector.data["Gen_b"], selector.data["Gen_b_mask"], b_muon_mass_mask = self.build_genpart_column(is_mc, selector.data["b_first_resonant"], selector.data)

        selector.data["bW_mass"] = (selector.data["Gen_b"][:,0] + selector.data["Gen_W"][:,0]).mass
        selector.data["b_lnu_mass"] = (selector.data["Gen_b"][:,0] + selector.data["Gen_emutau"][:,0] + selector.data["Gen_nu"][:,0]).mass

        bW_none_mask = ak.is_none(selector.data["Gen_b_mask"])|ak.is_none(selector.data["Gen_W_mask"])
        blnu_none_mask = ak.is_none(selector.data["Gen_emutau_mask"])|ak.is_none(selector.data["Gen_nu_mask"])|ak.is_none(selector.data["Gen_b_mask"])

        selector.data["bW_mass_final"] = ak.where(bW_none_mask, final_array, selector.data["bW_mass"]) 
        selector.data["b_lnu_mass_final"] = ak.where(blnu_none_mask, final_array, selector.data["b_lnu_mass"]) 
        '''

        '''
        for i in range(len(selector.data["Jet"])):
            print()
            print("Relative event number",i)
            print("pdg id list")
            print(ak.to_numpy(selector.data["GenPart"][i]["pdgId"]))    

            print("pt list")
            print(ak.to_numpy(selector.data["GenPart"][i]["pt"]))    

            print("phi list")
            print(ak.to_numpy(selector.data["GenPart"][i]["phi"]))    

            print("eta list")
            print(ak.to_numpy(selector.data["GenPart"][i]["eta"]))    

            print("mass list")
            print(ak.to_numpy(selector.data["GenPart"][i]["mass"]))    

            print("b chains",selector.data["b_fin_final"][i])
            print("b first quark",b_first[i])
            print("bmother_id",bmother_id[i])
            print("Resonant top b chains only",selector.data["b_resonant"][i])
            print("First b of resonant b chains",selector.data["b_first_resonant"][i])

            print("w chains",selector.data["w_fin_final"][i])
            print("w first",w_first[i])
            print("wmother_id",wmother_id[i])
            print("Resonant top w chains only",selector.data["w_resonant"][i])
            print("First w of resonant w chains",selector.data["w_first_resonant"][i])

            print("Combined charged lepton chains",selector.data["e_mu_tau_fin_final"][i])
            print("e_mu_tau first",e_mu_tau_first[i])
            print("e_mu_tau_mother_id",e_mu_tau_mother_id[i])
            print("Resonant top e_mu_tau chains only",selector.data["e_mu_tau_resonant"][i])
            print("First e_mu_tau of resonant e_mu_tau chains",selector.data["e_mu_tau_first_resonant"][i])
            print("w_last_e_mu_tau",w_last_e_mu_tau[i])

            print("Combined neutral lepton chains",selector.data["nu_fin_final"][i])
            print("nu_first",nu_first[i])
            print("nu_mother_id",nu_mother_id[i])
            print("Resonant top nu chains only",selector.data["nu_resonant"][i])
            print("First nu of resonant nu chains",selector.data["nu_first_resonant"][i])

            #hahaha

            #b and W from resonant attributes to check whether the columns for calculating invariant mass have been built correctly
            print("b pt",selector.data["Gen_b"][i]["pt"])
            print("b phi",selector.data["Gen_b"][i]["phi"])
            print("b eta",selector.data["Gen_b"][i]["eta"])
            print("b mass",selector.data["Gen_b"][i]["mass"])
            print("b pdgid",selector.data["Gen_b"][i]["pdgId"])

            print("W pt",selector.data["Gen_W"][i]["pt"])
            print("W phi",selector.data["Gen_W"][i]["phi"])
            print("W eta",selector.data["Gen_W"][i]["eta"])
            print("W mass",selector.data["Gen_W"][i]["mass"])
            print("W pdgid",selector.data["Gen_W"][i]["pdgId"])

            print("Gen_emutau mask",selector.data["Gen_emutau_mask"][i])
            print("Gen_emutau pt",selector.data["Gen_emutau"][i]["pt"])
            print("Gen_emutau phi",selector.data["Gen_emutau"][i]["phi"])
            print("Gen_emutau eta",selector.data["Gen_emutau"][i]["eta"])
            print("Gen_emutau mass",selector.data["Gen_emutau"][i]["mass"])
            print("Gen_emutau pdgid",selector.data["Gen_emutau"][i]["pdgId"])

            print("Gen_nu mask",selector.data["Gen_nu_mask"][i])
            print("Gen_nu pt",selector.data["Gen_nu"][i]["pt"])
            print("Gen_nu phi",selector.data["Gen_nu"][i]["phi"])
            print("Gen_nu eta",selector.data["Gen_nu"][i]["eta"])
            print("Gen_nu mass",selector.data["Gen_nu"][i]["mass"])
            print("Gen_nu pdgid",selector.data["Gen_nu"][i]["pdgId"])

            print("Invariant mass of W and b",selector.data["bW_mass"][i])
            print("Final Invariant mass of W and b",selector.data["bW_mass_final"][i])

            print("Invariant mass of W (from charged and neutral leptons)",selector.data["W_mass"][i])
            print("Final Invariant mass of W (from charged and neutral leptons)",selector.data["W_mass_final"][i])

            print("Invariant mass of top from b and charged and neutral leptons (not using W directly)",selector.data["b_lnu_mass"][i])
            print("Final Invariant mass of top from b and charged and neutral leptons (not using W directly)",selector.data["b_lnu_mass_final"][i])

            #hahaha

            if (len(selector.data["Gen_emutau"][i]["pt"])==0):
                print("Weird chains being made")
                self.event_display(i, "/afs/desy.de/user/p/paranjpe/top_wbwb/pepper/top_tchannel_weird_chains/", selector.data)

            #if i>10:
            #    break
        '''    
        #hahaha

        '''
        for i in range(len(selector.data["Jet"])):
            #if i>2:
            #    break
            print()
            print("Relative event number",i)
            print("pdg id list")
            print(ak.to_numpy(selector.data["GenPart"][i]["pdgId"]))    
            print("Respective indices")
            print(ak.to_numpy(index_gen[i]))
            
            print(bquark_index[i])
            
            #==================================================================b quark chains================================================================
            print()
            #Skipping the step that groups into pt and counts the 'unique' B hadrons according to status 2, since it does not seem to be required
            print("b mothers",b_mother[i])
            print("b fresh",b_fresh[i])            
            print("Total number of b quarks",len(bquark_index[i]))

            if (len(bquark_index[i])%len(b_fresh[i])!=0):
                print("Not equal number of b quarks in all chains")

            #New idea - first take only the level 0 and level 1 of b quarks, not level 2, and then take that and 'combine' with the level 3 of quarks
            #So what we really need here is ak.cartesian, not ak.combinations
            print("Unflattened b_fresh",b_1[i])
            print("Non b Mother from matching jet to last b in chain",min_delta_r.pdgId[i])
            print("Final Non b Mother from matching jet to last b in chain",selector.data["non_b_mother_final"].pdgId[i])
            
            recursion_level = 1
            b_init_final = []
            b_fin_final = self.make_chain(i,b_fresh,bquark_index[i],b_1,b_init_final,recursion_level,selector.data)
            print("b_fin_final outside the function make_chain",b_fin_final)
            b_fin_final = b_fin_final[ak.argsort(b_fin_final[:,0],axis=0,ascending=True)]  
            print("b_fin_final after sorting the chains by first element",b_fin_final)
            
            #===================================================================================================================================================
        '''
        #hahaha

        #b_mom_time_after = time.time()

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
        #selector.data["Gen_nu_pz"] = selector.data["Gen_nu"].pt[:,0]*np.sinh(selector.data["Gen_nu"].eta[:,0])
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
            selector.data["Rec_bjets"], selector.data["leading_bjet_mask"], selector.data["subleading_bjet_mask"] = self.build_bjet_column(is_mc, "bjets_tight", selector.data)
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
        #hahaha
        '''
        for i in range(len(selector.data["Jet"])):
            print()
            print("Relative event number",i)
            print("pdg id list")
            print(ak.to_numpy(selector.data["GenPart"][i]["pdgId"]))    

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

            print("min_pz_recmet",selector.data["min_pz_recmet"][i])
            print("max_pz_recmet",selector.data["max_pz_recmet"][i])
            print("Nu pz",selector.data["Gen_nu_pz"][i])

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

            print("gen nu pz",selector.data["Gen_nu_pz"][i])

            print("b_l_recMET_min_mass",selector.data["b_l_recMET_min_mass"][i])
            print("b_l_recMET_max_mass",selector.data["b_l_recMET_max_mass"][i])

            print("bjet_mask",selector.data["bjet_mask"][i])
            print("b_l_recMET_min_mass_final",selector.data["b_l_recMET_min_mass_final"][i])
            print("b_l_recMET_max_mass_final",selector.data["b_l_recMET_max_mass_final"][i])

        '''

        #'''
        #sign_dict = ["no_mask","same_sign", "opp_sign", "neutral"]
        sign_dict = ["no_mask","same_sign", "opp_sign"]

        selector.set_cat("sign_cat", set(sign_dict))
        selector.set_multiple_columns(partial(self.sign_cats,is_mc))

        #top_dict = ["top", "antitop", "no_top","both_top"]
        #The problem in naming the offshell categories this way is that it will likely not hold true for other processes like w + jets
        #'''
        top_dict = ["top", "antitop", "offshell_top","offshell_antitop","both_top"]

        selector.set_cat("top_cat", set(top_dict))
        selector.set_multiple_columns(partial(self.top_cats,is_mc))
        #'''
        #'''

        d = time.time()

        #'''

        #print("Shape of prob array", np.shape(prob_final))
        #print("Shape of prob array", len(ak.from_numpy(prob_final)))
        
        selector.data["top_max_pt"] = ak.max(selector.data["GenPart"][(abs(selector.data["GenPart"].pdgId)==6)]["pt"],axis=1)

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

            print("Bjet meson tag reshaped",selector.data["meson_tag_real_tight"][i])
            print("Bjet gen meson tag",selector.data["gen_meson_real_tight"][i])
            print("Real Probability values")
            print("B-",selector.data["prob_neg_real_tight"][i])
            print("B0bar",selector.data["prob_zerobar_real_tight"][i])
            print("B0",selector.data["prob_zero_real_tight"][i])
            print("B+",selector.data["prob_pos_real_tight"][i])

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

    def Bmeson_charge_score_cut(self,is_mc,data):
        leading_cut = ak.mask(data,(ak.num(data["bjets_tight"].eta)>=1) & (ak.num(data["Jet"]) >= 2))
        subleading_cut = ak.mask(data,(ak.num(data["bjets_tight"].eta)>=2) & (ak.num(data["Jet"]) >= 2))

        meson_cat = ["bneg","b0bar","b0","bpos"]
        hadron_id = [-521, -511, 511, 521]

        #rec_meson = ak.argmax(data["NN_prob"],axis=1)
        #This is the meson type chosen from the probabilities from the nn evaluation, and is a 1 entry per jet awkward array
        cut_array = [False] * len(data["bjets_tight"]) 

        #Using this instead of the above (this was also used for reshaping the arrays for the max number of constituents of the jets)
        data_gen_meson = ak.pad_none(data["gen_meson_real_tight"], 2, clip=True)
        data_gen_meson = ak.fill_none(data_gen_meson, 0)

        data_meson_tag = ak.pad_none(data["meson_tag_real_tight"], 2, clip=True)
        data_meson_tag = ak.fill_none(data_meson_tag, 0)

        #=====================================================================================================

        score_lowlim = 0.5 #Guess threshold applied

        prob = ["prob_neg","prob_zerobar","prob_zero","prob_pos"]

        if len(data) > 0:
            #data_meson_tag = ak.pad_none(data["meson_tag_real_tight"], 2, clip=True)
            #data_meson_tag = ak.fill_none(data_meson_tag, 0)

            data_prob_neg = ak.pad_none(data["prob_neg_real_tight"], 2, clip=True)
            data_prob_neg = ak.fill_none(data_prob_neg, 0)

            data_prob_zerobar = ak.pad_none(data["prob_zerobar_real_tight"], 2, clip=True)
            data_prob_zerobar = ak.fill_none(data_prob_zerobar, 0)

            data_prob_zero = ak.pad_none(data["prob_zero_real_tight"], 2, clip=True)
            data_prob_zero = ak.fill_none(data_prob_zero, 0)

            data_prob_pos = ak.pad_none(data["prob_pos_real_tight"], 2, clip=True)
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

        match_pos_mask = (data_gen_meson[:,0] == hadron_id[3]) & (data_meson_tag[:,0] == 3)
        match_neg_mask = (data_gen_meson[:,0] == hadron_id[0]) & (data_meson_tag[:,0] == 0)

        charge_mask = ak.where(ak.is_none(leading_cut), cut_array, match_pos_mask|match_neg_mask)

        '''
        print("Length of conclusive_mask",len(conclusive_mask))
        print("Length of match_pos_mask",len(match_pos_mask))
        print("Length of match_neg_mask",len(match_neg_mask))
        print("Length of charge_mask",len(charge_mask))
        for i in range(len(data)):
            print()
            print("Event in Bmeson_charge_score_cut (before actually cutting)",i)
            print("Conclusive mask",conclusive_mask[i])
            print("Pos charge mask",match_pos_mask[i])
            print("Neg charge mask",match_neg_mask[i])
            print("Final charge mask",charge_mask[i])
         '''   
        #hahaha

        return conclusive_mask & charge_mask

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
        data_gen_meson = ak.pad_none(data["gen_meson_real_tight"], 2, clip=True)
        data_gen_meson = ak.fill_none(data_gen_meson, 0)
        data_meson_tag = ak.pad_none(data["meson_tag_real_tight"], 2, clip=True)
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
            data_meson_tag = ak.pad_none(data["meson_tag_real_tight"], 2, clip=True)
            data_meson_tag = ak.fill_none(data_meson_tag, 0)

            data_prob_neg = ak.pad_none(data["prob_neg_real_tight"], 2, clip=True)
            data_prob_neg = ak.fill_none(data_prob_neg, 0)

            data_prob_zerobar = ak.pad_none(data["prob_zerobar_real_tight"], 2, clip=True)
            data_prob_zerobar = ak.fill_none(data_prob_zerobar, 0)

            data_prob_zero = ak.pad_none(data["prob_zero_real_tight"], 2, clip=True)
            data_prob_zero = ak.fill_none(data_prob_zero, 0)

            data_prob_pos = ak.pad_none(data["prob_pos_real_tight"], 2, clip=True)
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

    def mother_categories(self,is_mc,data):
        cats = {}
        #data["lep_mother_final"]
        #data["non_b_mother_final"]
        #Now there is no specific condition for whether the sign is actually right since that is process dependent, so there is only a mismatch condition

        w_id = 24
        top_id = 6
        b_mask = ak.num(data["non_b_mother_final"][abs(data["non_b_mother_final"].pdgId) == top_id],axis=1)==1
        l_mask = ak.num(data["lep_mother_final"][abs(data["lep_mother_final"].pdgId) == w_id],axis=1)==1

        sign_b_mom = data["non_b_mother_final"].pdgId/abs(data["non_b_mother_final"].pdgId)
        sign_l_mom = data["lep_mother_final"].pdgId/abs(data["lep_mother_final"].pdgId)

        #Since this is always going to be used in combination with b_mask and l_mask, using [:,0] is ok since there is only one element
        #Turns out this is not ok, with the non recursive method it can have two mothers (weird but probable)
        sign_mask = (sign_b_mom[:,0] == sign_l_mom[:,0]) 
        false_array = [False]*len(data["Lepton"])

        b_w_l_w = ~b_mask & ~l_mask
        b_w_l_r = ~b_mask & l_mask
        b_r_l_w = b_mask & ~l_mask
        b_r_l_r = b_mask & l_mask & sign_mask
        b_l_sign_mismatch = b_mask & l_mask & ~sign_mask #This sign mask is technically only for ttbar, not single top
        #Removed for debugging
        #cats["no_mask"] = (data["non_b_mother_final"] == None) | (data["lep_mother_final"].pdgId == None)
        #cats["no_mask"] = (data["non_b_mother_final"] is None) | (data["lep_mother_final"].pdgId is None)
        cats["no_mask"] = (ak.is_none(data["non_b_mother_final"].pdgId)) | (ak.is_none(data["lep_mother_final"].pdgId))

        cats["b_w_l_w"] = ak.where(ak.is_none(b_w_l_w), false_array, b_w_l_w)
        cats["b_w_l_r"] = ak.where(ak.is_none(b_w_l_r), false_array, b_w_l_r)
        cats["b_r_l_w"] = ak.where(ak.is_none(b_r_l_w), false_array, b_r_l_w)
        cats["b_r_l_r"] = ak.where(ak.is_none(b_r_l_r), false_array, b_r_l_r)
        cats["b_l_sign_mismatch"] = ak.where(ak.is_none(b_l_sign_mismatch), false_array, b_l_sign_mismatch)

        '''
        for i in range(len(data["Lepton"])):
            print("Event number",i)
            print("B jet part dr",data["bjet_part_dr"][i])
            print("B mask",b_mask[i])
            print("Lepton mask",l_mask[i])
            print("B mom sign",sign_b_mom[i])
            print("Lepton mom sign",sign_l_mom[i])
            print("Sign mask",sign_mask[i])
            print("Mom mask values outside the loop")
            print("b_w_l_w",b_w_l_w[i])
            print("b_w_l_r",b_w_l_r[i])
            print("b_r_l_w",b_r_l_w[i])
            print("b_r_l_r",b_r_l_r[i])
            print("b_l_sign_mismatch",b_l_sign_mismatch[i])
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
        data_meson_tag = ak.pad_none(data["meson_tag_real_tight"], 1, clip=True)
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

            cats["no_mask"] = no_bl_mask

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
            key_list = ["no_mask","top_opp_sign","top_same_sign","top_neutral","antitop_opp_sign","antitop_same_sign","antitop_neutral","no_top_opp_sign","no_top_same_sign","no_top_neutral","both_top_opp_sign","both_top_same_sign","both_top_neutral"]
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

            print("no_mask",cats["no_mask"][i])

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

        #Using this instead of the above (this was also used for reshaping the arrays for the max number of constituents of the jets)
        #leading_cut_tight = ak.mask(data,(ak.num(data["bjets_tight"].eta)>=1) & (ak.num(data["Jet"]) >= 2))
        data_meson_tag = ak.pad_none(data["meson_tag_real_tight"], 1, clip=True)
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

            #no_bl_mask = (ak.is_none(data["bjets_tight"].pt)) | (ak.is_none(data["Lepton"].pt))
            no_bl_mask = (ak.num(data["bjets_tight"].pt,axis=1)==0) | (ak.num(data["Lepton"].pt,axis=1)==0)

            cats["no_mask"] = no_bl_mask
            cats["opp_sign"] = ~no_bl_mask & sign_mask
            cats["same_sign"] = ~no_bl_mask & same_sign_mask
            #cats["neutral"] = ~no_bl_mask & neutral_mask

            for key in cats.keys():
                cats[key] = ak.where(ak.is_none(cats[key]), false_array, cats[key])

        elif len(data) == 0:
            #key_list = ["no_mask","opp_sign","same_sign","neutral"]
            key_list = ["no_mask","opp_sign","same_sign"]
            for key in key_list:
                cats[key] = false_array

        return cats

    def top_cats(self,is_mc,data):
        cats = {}
        w_id = 24
        top_id = 6

        #Using this instead of the above (this was also used for reshaping the arrays for the max number of constituents of the jets)
        #leading_cut_tight = ak.mask(data,(ak.num(data["bjets_tight"].eta)>=1) & (ak.num(data["Jet"]) >= 2))
        data_meson_tag = ak.pad_none(data["meson_tag_real_tight"], 1, clip=True)
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
            #Slightly different conditions, if both top and antitop counts are non zero it can also be used as a category for ttbar
            top_num = ak.num(data["GenPart"][data["GenPart"].pdgId == 6], axis=1)        
            antitop_num = ak.num(data["GenPart"][data["GenPart"].pdgId == -6], axis=1)        

            #Extra added specifically for offshell (no top) now that id_bpos_rec_bpos and id_bneg_rec_bneg have been sort of merged
            sign_l = data["Lepton"].pdgId/abs(data["Lepton"].pdgId) #Already exactly one lepton cut applied

            top_mask = (top_num>=1) & (antitop_num==0)
            antitop_mask = (top_num==0) & (antitop_num>=1)
            no_top_mask = (top_num==0) & (antitop_num==0)
            both_top_mask = (top_num>=1) & (antitop_num>=1) #This wont happen for wbj, mostly only for ttbar

            cats["top"] = top_mask
            cats["antitop"] = antitop_mask
            #cats["no_top"] = no_top_mask
            cats["offshell_top"] = no_top_mask & (sign_l[:,0] == -1) #This is the sign of the pdg id, not the actual lepton
            cats["offshell_antitop"] = no_top_mask & (sign_l[:,0] == 1) #This is the sign of the pdg id, not the actual lepton
            cats["both_top"] = both_top_mask

            for key in cats.keys():
                cats[key] = ak.where(ak.is_none(cats[key]), false_array, cats[key])

        elif len(data) == 0:
            key_list = ["top","antitop","offshell_top","offshell_antitop","both_top"]
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

            '''
            data_meson_tag = ak.pad_none(data["meson_tag_real_loose"], 2, clip=True)
            data_meson_tag = ak.fill_none(data_meson_tag, 0)

            data_prob_neg = ak.pad_none(data["prob_neg_real_loose"], 2, clip=True)
            data_prob_neg = ak.fill_none(data_prob_neg, 0)

            data_prob_zerobar = ak.pad_none(data["prob_zerobar_real_loose"], 2, clip=True)
            data_prob_zerobar = ak.fill_none(data_prob_zerobar, 0)

            data_prob_zero = ak.pad_none(data["prob_zero_real_loose"], 2, clip=True)
            data_prob_zero = ak.fill_none(data_prob_zero, 0)

            data_prob_pos = ak.pad_none(data["prob_pos_real_loose"], 2, clip=True)
            data_prob_pos = ak.fill_none(data_prob_pos, 0)

            #Since even the no bjet events have been padded with zeros, this condition will automatically hold for them too, irrespective of whether it is used or not
            #Checking all scores because it seems easier than checking where the max score is and comparing only that
            neg_mask = data_prob_neg[:,0] <= score_lowlim
            zerobar_mask = data_prob_zerobar[:,0] <= score_lowlim
            zero_mask = data_prob_zero[:,0] <= score_lowlim
            pos_mask = data_prob_pos[:,0] <= score_lowlim

            inconclusive_mask = (neg_mask) & (zerobar_mask) & (zero_mask) & (pos_mask)
            conclusive_mask = ~((neg_mask) & (zerobar_mask) & (zero_mask) & (pos_mask))

            #============================================================================================================================================

            #============================================================================================================================================

            meson_cat = ["bneg","b0bar","b0","bpos"]
            hadron_id = [-521, -511, 511, 521]

            #rec_meson = ak.argmax(data["NN_prob"],axis=1)
            #This is the meson type chosen from the probabilities from the nn evaluation, and is a 1 entry per jet awkward array

            #hadron_id_net = ak.to_numpy(ak.unflatten(hadron_id* len(data["bjets_loose"]),4))

            #print("Length of hadron id mask",len(hadron_id_net))

            #================All this is copied from the gen code proc_wbwb_debug================================
            #no_b_mask = ak.mask(data,(ak.num(data["bjets_loose"].eta) > 0) & (ak.num(data["Jet"]) >= 2))

            #default_array = np.zeros((len(data["bjets_loose"]), 2)) #2 since we could have subleading jets too

            #data_gen_meson = ak.where(ak.is_none(no_b_mask), default_array, data["gen_meson_real_loose"])
            #data_meson_tag = ak.where(ak.is_none(no_b_mask), default_array, data["meson_tag_real_loose"])

            #Using this instead of the above (this was also used for reshaping the arrays for the max number of constituents of the jets)
            data_gen_meson = ak.pad_none(data["gen_meson_real_loose"], 2, clip=True)
            data_gen_meson = ak.fill_none(data_gen_meson, 0)
            data_meson_tag = ak.pad_none(data["meson_tag_real_loose"], 2, clip=True)
            data_meson_tag = ak.fill_none(data_meson_tag, 0)

            #=============================================================================================================================================

            match_charge_cut = (data["id_bpos_rec_bpos"]==1)|(data["id_bneg_rec_bneg"]==1)
            opposite_charge_mask = ((data["id_bpos_rec_bpos"]==1) & (data_gen_meson[:,0] == hadron_id[0]) & (data_meson_tag[:,0] == 0)) | ((data["id_bneg_rec_bneg"]==1) & (data_gen_meson[:,0] == hadron_id[3]) & (data_meson_tag[:,0] == 3))

            '''
            
            cats["b_loose_0"] = (loose_jets==0)
            cats["b_loose_1+_no_effect"] = (loose_jets>=1) & (data_bjet_pt_loose[:,0] <= data_bjet_pt_tight[:,0])
            cats["b_loose_1+_leading"] = (loose_jets>=1) & (data_bjet_pt_loose[:,0] > data_bjet_pt_tight[:,0])
            '''
            cats["b_loose_1+_leading_inconclusive_50"] = (loose_jets>=1) & (data_bjet_pt_loose[:,0] > data_bjet_pt_tight[:,0]) & inconclusive_mask
            cats["b_loose_1+_leading_conclusive_50_wrong"] = (loose_jets>=1) & (data_bjet_pt_loose[:,0] > data_bjet_pt_tight[:,0]) & conclusive_mask & ~opposite_charge_mask
            cats["b_loose_1+_leading_conclusive_50_right"] = (loose_jets>=1) & (data_bjet_pt_loose[:,0] > data_bjet_pt_tight[:,0]) & conclusive_mask & opposite_charge_mask
            '''

        elif len(data) == 0:
            cats["b_loose_0"] = cut_array
            cats["b_loose_1+_no_effect"] = cut_array
            cats["b_loose_1+_leading"] = cut_array

            '''
            cats["b_loose_1+_leading_inconclusive_50"] = cut_array
            cats["b_loose_1+_leading_conclusive_50_wrong"] = cut_array
            cats["b_loose_1+_leading_conclusive_50_right"] = cut_array
            '''

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
            print("B-",data["prob_neg_real_tight"][i])
            print("B0bar",data["prob_zerobar_real_tight"][i])
            print("B0",data["prob_zero_real_tight"][i])
            print("B+",data["prob_pos_real_tight"][i])

            print("data_bjet_pt_loose",data_bjet_pt_loose[i])
            print("Real loose Probability values")
            print("B-",data["prob_neg_real_loose"][i])
            print("B0bar",data["prob_zerobar_real_loose"][i])
            print("B0",data["prob_zero_real_loose"][i])
            print("B+",data["prob_pos_real_loose"][i])          

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


    def lepton_mother(self, data):
        #No mask needed here because the one lepton cut has already been applied, or maybe for the matched part is still required

        lepton_object = data["GenPart"][(data["GenPart"].pdgId==data["Lepton"].pdgId[:,0])] #Can do this since there is only one lepton, makes it easier

        dr_mask = ak.mask(data,(ak.num(data["Lepton"].eta)>0)&(ak.num(data["Lepton"].matched_gen_eta)>0)) #So it should have reco pt and also matched gen pt
        #dr_gen_mask = ak.mask(data,ak.num(data["Lepton"].matched_gen.eta)>0)
        dr_part_mask = ak.mask(data,ak.num(lepton_object.eta)>0)
        #print("Size of events satisfying the mask (having bjets)",len(dr_mask))

        #of course lengths of l_col and "Lepton" are supposed to be the same, that is why they are used interchangeably
        default_array = np.zeros((len(data["Lepton"]), 2))
        default_part_array = np.zeros((len(data["Lepton"]), 1))
        empty_array = [None]*len(data["Lepton"])

        data_lep_pt = ak.where(ak.is_none(dr_mask), default_array, data["Lepton"].matched_gen_pt)
        data_lep_eta = ak.where(ak.is_none(dr_mask), default_array, data["Lepton"].matched_gen_eta)
        data_lep_phi = ak.where(ak.is_none(dr_mask), default_array, data["Lepton"].matched_gen_phi)

        data_lep_part_pt = ak.where(ak.is_none(dr_part_mask), default_part_array, lepton_object.pt)
        data_lep_part_eta = ak.where(ak.is_none(dr_part_mask), default_part_array, lepton_object.eta)
        data_lep_part_phi = ak.where(ak.is_none(dr_part_mask), default_part_array, lepton_object.phi)

        gen_lepton_match = lepton_object[(data_lep_part_pt==data_lep_pt[:,0]) & (data_lep_part_eta==data_lep_eta[:,0]) & (data_lep_part_phi==data_lep_phi[:,0])]
        lepton_mother_1 = data["GenPart"][gen_lepton_match.genPartIdxMother]

        return data,lepton_mother_1 #This is the mother objects at the first level

        '''
        no_lepton_id = ~((abs(lepton_mother.pdgId) >= 11) & (abs(lepton_mother.pdgId) <= 18))

        repeat_only_pdg = ak.num(lepton_mother[no_lepton_id],axis=1)
        repeat_num = ak.num(repeat_only_pdg[repeat_only_pdg==0],axis=0)
        print("Number of events having only repeat lepton mothers",repeat_num)

        repeat_mask = ak.mask(data,repeat_only_pdg==0)

        if repeat_num > 0: #Go the next level of mothers
            print("Reached second level of mothers")
            lepton_mother_2 = data["GenPart"][data["GenPart"][gen_lepton_match.genPartIdxMother].genPartIdxMother]
            lepton_mother_2_final = ak.where(ak.is_none(repeat_mask), lepton_mother[no_lepton_id], lepton_mother_2)

            no_lepton_id_2 = ~((abs(lepton_mother_2_final.pdgId) >= 11) & (abs(lepton_mother_2_final.pdgId) <= 18))
            repeat_only_pdg_2 = ak.num(lepton_mother_2_final[no_lepton_id_2],axis=1)
            repeat_num_2 = ak.num(repeat_only_pdg_2[repeat_only_pdg_2==0],axis=0)
            print("Number of events having only repeat lepton mothers at the second level",repeat_num_2)

            repeat_mask_2 = ak.mask(data,repeat_only_pdg_2==0)

            if repeat_num_2 > 0: #Go the next level of mothers
                print("Reached third level of mothers")
                lepton_mother_3 = data["GenPart"][data["GenPart"][data["GenPart"][gen_lepton_match.genPartIdxMother].genPartIdxMother].genPartIdxMother]
                lepton_mother_3_final = ak.where(ak.is_none(repeat_mask_2), lepton_mother_2_final[no_lepton_id_2], lepton_mother_3)

                no_lepton_id_3 = ~((abs(lepton_mother_3_final.pdgId) >= 11) & (abs(lepton_mother_3_final.pdgId) <= 18))
                repeat_only_pdg_3 = ak.num(lepton_mother_3_final[no_lepton_id_3],axis=1)
                repeat_num_3 = ak.num(repeat_only_pdg_3[repeat_only_pdg_3==0],axis=0)
                print("Number of events having only repeat lepton mothers at the third level",repeat_num_3)

                #repeat_mask_3 = ak.mask(data,repeat_only_pdg_3==0)


        for i in range(len(data["Lepton"])):
            print("Event number",i)
            #print("Lepton pdg id",data["Lepton"][i].pdgId)
            print("Lepton pdg id",data["Lepton"].pdgId[:,0][i])
            print("Matched pt",data_lep_pt[i])
            print("Matched eta",data_lep_eta[i])
            print("Matched phi",data_lep_phi[i])
            print("Lepton Mother particles pdg id",lepton_mother.pdgId[i])
            #print("Lepton Mother particles pdg id final",lepton_mother_final[i])
            print("Non repeating non lepton mothers",repeat_only_pdg[i])

            print("Lepton mothers at second level",lepton_mother_2.pdgId[i])
            print("Lepton mothers at second level final",lepton_mother_2_final.pdgId[i])
            print("Repeat mask",repeat_mask[i])

            print("Lepton mothers at third level",lepton_mother_3.pdgId[i])
            print("Lepton mothers at third level final",lepton_mother_3_final.pdgId[i])
            print("Repeat mask 2",repeat_mask_2[i])

            print("DR mask",dr_mask[i])
        '''
        #lepton_mother_final = ak.where(ak.is_none(dr_mask), empty_array, lepton_mother.pdgId)

    def lepton_recursion(self,lep_recursion_level,lep_mother,data):
        print("Reached level ", str(lep_recursion_level)," of mothers")
        no_lepton_id = ~((abs(lep_mother.pdgId) >= 11) & (abs(lep_mother.pdgId) <= 18))

        repeat_only_pdg = ak.num(lep_mother[no_lepton_id],axis=1)
        repeat_num = ak.num(repeat_only_pdg[repeat_only_pdg==0],axis=0)

        end_index = -1
        #end_index = 10 #Only for debugging and testing since no event at this level actually seems to have the index -1

        #These are the number of events that have only repeat mothers and do not any mothers at the next level (i.e. something like mother)
        repeat_end = ak.num(lep_mother[lep_mother.genPartIdxMother == end_index],axis=1)
        repeat_end_num = ak.num(repeat_end[(repeat_end==ak.num(lep_mother,axis=1)) & (repeat_only_pdg==0)],axis=0)
        print("Number of events having only repeat lepton mothers",repeat_num)
        print("Number of events having only repeat lepton mothers that are also at the end (beginning)",repeat_end_num)

        repeat_mask = ak.mask(data,repeat_only_pdg==0)
        '''
        for i in range(len(data["Lepton"])):
            print("Event number",i)
            print("Lepton pdg id",data["Lepton"].pdgId[:,0][i])
            print("Non repeating non lepton mothers",repeat_only_pdg[i])
            print("Lepton mothers at level",str(lep_recursion_level),lep_mother.pdgId[i])
            print("Repeat mask",repeat_mask[i])
            print("Number of lepton mothers for each event at first level",ak.num(lep_mother,axis=1)[i])
            print("Number of particles having mother index as ", str(end_index),"for each event",repeat_end[i])
            print("Lepton mother indices at second level",lep_mother.genPartIdxMother[i])
        '''    

        if (repeat_num > 0) & (repeat_end_num < repeat_num): #Go the next level of mothers
            lep_recursion_level +=1
            lep_mother_2 = data["GenPart"][lep_mother.genPartIdxMother]
            #lep_mother_2_final = ak.where(ak.is_none(repeat_mask), lep_mother[no_lepton_id], lep_mother_2)
            #lep_mother_2_final = self.lepton_recursion(lep_recursion_level,lep_mother_2_final,data)

            lep_mother = ak.where(ak.is_none(repeat_mask), lep_mother[no_lepton_id], lep_mother_2)
            lep_mother = self.lepton_recursion(lep_recursion_level,lep_mother,data)
            
            '''
            for i in range(len(data["Lepton"])):
                print("Event number",i)
                print("Lepton mothers at level",str(lep_recursion_level),lep_mother_2.pdgId[i])
                print("Final Lepton mothers at level",str(lep_recursion_level),lep_mother.pdgId[i])
            '''

        return lep_mother

    def chain_last(self,pdgId_req,counts_chain,recursion_level,chain_current, last_index,data):
        #print("Reached level ", str(recursion_level)," of mothers")

        end_index = -1
        #end_index = 10 #Only for debugging and testing since no event at this level actually seems to have the index -1

        fresh_mask = abs(data["GenPart"][data["GenPart"][last_index].genPartIdxMother].pdgId) != pdgId_req
        #These are the number of events that have only repeat mothers and do not any mothers at the next level (i.e. something like mother)
        repeat_end = ak.num(last_index[fresh_mask],axis=1)
        repeat_end_num = ak.num(repeat_end[repeat_end==ak.num(last_index,axis=1)],axis=0)
        #print("Number of events having only mothers that are also at the end (beginning)",repeat_end_num)

        repeat_mask = ak.mask(last_index,fresh_mask)
        index_gen = ak.local_index(data["GenPart"],axis=1)
        index_gen_pdgid = index_gen[abs(data["GenPart"]["pdgId"]) == pdgId_req]

        '''
        for i in range(len(data["Jet"])):
            print()
            print("Event number",i)
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
            #print("counts_chain",counts_chain)

            #chain_current = ak.unflatten(chain_flat,recursion_level*ak.num(last_index))
            chain_current = ak.unflatten(chain_flat,ak.num(last_index))
            #print("chain_current",chain_current)

            counts_chain, chain_current, last_index = self.chain_last(pdgId_req,counts_chain,recursion_level,chain_current,last_index,data)
            
        return counts_chain, chain_current, last_index


    '''
    def make_chain(self,i,fresh,part_index,init,init_final,recursion_level,data):
        #print()
        #print("Current recursion level",recursion_level)
        pair_cross_final = ak.cartesian([init, part_index], axis=0)
        pair_cross = ak.cartesian([init[:,-1], part_index], axis=0)
        #print("Combination of b at 0 level with other bs",pair_cross)

        #print("First element of each pair (from init)",pair_cross[:,"0"])
        #print("Second element of each pair",pair_cross[:,"1"])
        
        combo_mom_mask = data["GenPart"][i][pair_cross[:,"1"]]["genPartIdxMother"] == pair_cross[:,"0"]
        #print("Mom mask for all pairs",ak.to_numpy(combo_mom_mask))
        #print("Successful pairs of mothers and daughters",pair_cross[combo_mom_mask])

        num_comp = int(len(pair_cross)/len(init))
        #print("Length of unflattening for the pair_comp",num_comp)
        pair_comp = ak.unflatten(pair_cross,num_comp)
        #print("Array for comparing whether any bs are not daughtered",pair_comp)

        mom_comp_mask = data["GenPart"][i][pair_comp[:,"1"]]["genPartIdxMother"] == pair_comp[:,"0"]
        #print("New comparison mom mask",mom_comp_mask)
        unflat_cut = ak.mask(mom_comp_mask,(data["GenPart"][i][pair_comp[:,"1"]]["genPartIdxMother"] == pair_comp[:,"0"]))
        no_mom_match = ak.count(unflat_cut, axis=1)
        #print("Count for unflat cut",no_mom_match) #This only counts the non - none entries so they will exclude all false entries

        if len(pair_cross[combo_mom_mask]) > 0:
            args = []
            for first_length in range(0,recursion_level): #The 1 here is the recursion level, vary it accordingly
                #print("Level of bs",first_length)
                arr = pair_cross_final[combo_mom_mask][:,"0"][:,first_length]
                args.append(ak.unflatten(arr,1))
            args.append(ak.unflatten(pair_cross_final[combo_mom_mask][:,"1"],1))

            fin = ak.concatenate(args,axis=1)
            #print("New array of level 0 and 1 bs",fin)
            #print("b quarks at 0th level that did not get matched",init[no_mom_match==0])
            fin_final = ak.concatenate([init[no_mom_match==0],fin],axis=0)
            #print("Final b chains at 0 and 1 level including the ones that did not get matched",fin_final)

            recursion_level+=1
            fin_final = self.make_chain(i,fresh,part_index,fin,fin_final,recursion_level,data)

        else:
            if recursion_level>1:
                fin_final = init_final
            elif recursion_level==1:
                fin_final = init #None in the original fresh array is actually getting matched to daughters of the same category

        return fin_final
    '''

    def lepton_chain_full(self,make_full_chain,pdgid_arr,data):
        #This consists of the rest of the part used to make chains
        index_gen = ak.local_index(data["GenPart"],axis=1)
        abs_pdgid = abs(data["GenPart"]["pdgId"])
        print("Pdg id in lepton_chain_full function",pdgid_arr)

        #==================================================================b quark chains================================================================

        #Assuming exactly 3 arguments in pdgid_arr
        mask = (abs_pdgid == pdgid_arr[0]) | (abs_pdgid == pdgid_arr[1]) | (abs_pdgid == pdgid_arr[2])
        part_index = index_gen[mask]
        object = data["GenPart"][mask]
        mother = object["genPartIdxMother"] #This is nothing but a set of indices
        #fresh = part_index[abs(data["GenPart"][mother]["pdgId"])!=pdgid_arr]
        fresh = part_index[~((abs(data["GenPart"][mother]["pdgId"])==pdgid_arr[0])|(abs(data["GenPart"][mother]["pdgId"])==pdgid_arr[1])|(abs(data["GenPart"][mother]["pdgId"])==pdgid_arr[2]))]
        #print("Fresh b s for event 95",fresh[95])

        if make_full_chain == 1:
            part_index = ak.sort(part_index,axis=1,ascending=True)   
            one = ak.unflatten(fresh,1,axis=1)

            recursion_level = 1
            #init_final = [[]]*len(data["Jet"])
            init_final = one
            fin_final = self.make_chain(fresh,part_index,one,init_final,recursion_level,data)
            #print("fin_final for event 95",fin_final[95])
            fin_final = fin_final[ak.argsort(fin_final[:,:,0],axis=1,ascending=True)]  
            return fresh, fin_final
        
        elif make_full_chain == 0:
            return fresh

    def chain_full(self,pdgid_req,data):
        #This consists of the rest of the part used to make chains
        index_gen = ak.local_index(data["GenPart"],axis=1)
        abs_pdgid = abs(data["GenPart"]["pdgId"])
        print("Pdg id in chain_full function",pdgid_req)

        #==================================================================b quark chains================================================================

        mask = abs_pdgid == pdgid_req
        part_index = index_gen[mask]
        object = data["GenPart"][mask]
        mother = object["genPartIdxMother"] #This is nothing but a set of indices
        fresh = part_index[abs(data["GenPart"][mother]["pdgId"])!=pdgid_req]
        #print("Fresh b s for event 95",fresh[95])
        part_index = ak.sort(part_index,axis=1,ascending=True)   
        one = ak.unflatten(fresh,1,axis=1)

        recursion_level = 1
        #init_final = [[]]*len(data["Jet"])
        init_final = one
        fin_final = self.make_chain(fresh,part_index,one,init_final,recursion_level,data)
        #print("fin_final for event 95",fin_final[95])
        fin_final = fin_final[ak.argsort(fin_final[:,:,0],axis=1,ascending=True)]  
        #return fin_final
        #Fresh is being returned because we most likely need it for calculating invariant mass (and checking original mother for that)
        return fresh,fin_final


    #Convert to function without looping through events
    def make_chain(self,fresh,part_index,init,init_final,recursion_level,data):
        print()
        print("Current recursion level",recursion_level)
        #pair_cross_final = ak.cartesian([init, part_index], axis=1)
        pair_cross_final = ak.cartesian([init_final, part_index], axis=1)
        #pair_cross = ak.cartesian([init[:,:,-1], part_index], axis=1)
        pair_cross = ak.cartesian([init_final[:,:,-1], part_index], axis=1)

        combo_mom_mask = data["GenPart"][pair_cross[:,:,"1"]]["genPartIdxMother"] == pair_cross[:,:,"0"]

        print("init",init)
        print("last part of init for making the pair_cross unflattening",init[:,:,-1])
        print("unflat step 2",ak.num(init[:,:,-1],axis=1))
        #init_comp = ak.unflatten(ak.unflatten(ak.flatten(init[:,:,-1]),1),ak.num(init[:,:,-1])) #This has to be done to make it appear like the init from the first recursion level
        init_comp = ak.unflatten(ak.unflatten(ak.flatten(init_final[:,:,-1]),1),ak.num(init_final[:,:,-1])) #This has to be done to make it appear like the init from the first recursion level
        print("init_comp",init_comp)
        num_comp = ak.num(part_index,axis=1)*ak.num(init_comp,axis=2)
        print(num_comp)
        num_comp_flat = ak.flatten(num_comp)
        
        print(num_comp_flat)
        pair_comp = ak.unflatten(pair_cross,num_comp_flat,axis=1)

        #print("Depth of pair_cross",pair_cross.layout.minmax_depth)
        #print("Depth of pair_comp",pair_comp.layout.minmax_depth)
        mom_comp_mask = ak.unflatten(combo_mom_mask,num_comp_flat,axis=1)
        #mom_comp_mask = data["GenPart"][pair_comp[:,"1"]]["genPartIdxMother"] == pair_comp[:,"0"]
        unflat_cut = ak.mask(mom_comp_mask,mom_comp_mask)
        no_mom_match = ak.count(unflat_cut, axis=2)

        num_match = ak.num(pair_cross[combo_mom_mask],axis=1)
        level_match_mask = ak.mask(pair_cross, num_match > 0)

        #==========================Making dummy array of 'mother' indices to fill for those events which have no matches========================

        #Alternative method - make an array with the max of the ngenpart
        ngenpart = ak.num(data["GenPart"],axis=1)
        ngenpart_max = ak.max(ngenpart)
        #print("ngenpart_max",ngenpart_max)

        args_axis = []
        if recursion_level>1:
            for first_length in range(0,recursion_level): #The 1 here is the recursion level, vary it accordingly
                args_axis.append(ngenpart_max)
            #ngen_one_entry = ak.concatenate(args_axis)
            dummy_ngen = ak.Array([[(args_axis, ngenpart_max)]]*len(data["GenPart"]))

        elif recursion_level==1:
            dummy_ngen = ak.Array([[([ngenpart_max], ngenpart_max)]]*len(data["GenPart"]))

        #print("dummy_ngen",dummy_ngen)
        #print("len(dummy_ngen)",len(dummy_ngen))
        #print(type(dummy_ngen))

        #=======================================================================================================================================

        pair_combo_concat = ak.where(ak.is_none(level_match_mask), dummy_ngen, pair_cross_final[combo_mom_mask])
        #print("pair_cross_final net",pair_cross_final[combo_mom_mask])
        args = []
        for first_length in range(0,recursion_level): #The 1 here is the recursion level, vary it accordingly
            #print("Level of bs",first_length)
            #print("pair_combo_concat",pair_combo_concat[:,"0"][:,first_length])
            #print("pair_cross_final",pair_cross_final[combo_mom_mask][:,:,"0"][:,:,first_length])
            #print("pair_combo_concat",pair_combo_concat[:,:,"0"][:,:,first_length])
            arr = pair_combo_concat[:,:,"0"][:,:,first_length]
            arr_unflat = ak.unflatten(ak.flatten(arr),1)
            args.append(arr_unflat)
        args.append(ak.unflatten(ak.flatten(pair_combo_concat[:,:,"1"]),1))
        #print(args)
        fin_mask = ak.concatenate(args,axis=1)

        #print("Args with zero match mask")
        #print(args)
        #print(fin_mask)
        #print(ak.num(pair_combo_concat,axis=1))
        fin_mask_unflat = ak.unflatten(fin_mask,ak.num(pair_combo_concat))
        
        empty_array = [[]]*len(data["GenPart"])
        fin_mask_final = ak.where(ak.is_none(level_match_mask), empty_array, fin_mask_unflat)

        if recursion_level>1:
            #fin_final = ak.where(ak.is_none(level_match_mask), init_final, ak.concatenate([init[no_mom_match==0],fin_mask_final],axis=1))
            fin_final = ak.where(ak.is_none(level_match_mask), init_final, ak.concatenate([init_final[no_mom_match==0],fin_mask_final],axis=1))
            problem_evt = 95
        elif recursion_level==1:
            fin_final = ak.where(ak.is_none(level_match_mask), init_final, ak.concatenate([init_final[no_mom_match==0],fin_mask_final],axis=1))

        '''
        for problem_evt in range(len(data["Jet"])):
            print()
            print("Event number at recursion level",recursion_level,problem_evt)
            print("Recursion level",recursion_level)
            print("init_final",init_final[problem_evt])
            print("level_match_mask",level_match_mask[problem_evt])
            print("fin_mask_final",fin_mask_final[problem_evt])
            print("no_mom_match",no_mom_match[problem_evt])
            print("init",init[problem_evt])
            #print("init[no_mom_match==0]",init[no_mom_match==0][problem_evt])
            #print("init_final[no_mom_match==0]",init_final[no_mom_match==0][problem_evt])
            print("init[no_mom_match==0]",init_final[no_mom_match==0][problem_evt]) #Spew only same for ease of comparing with previous text files
            print("fin_final",fin_final[problem_evt])
        '''

        '''
        for i in range(len(data["Jet"])):
            #if i>9:
            #    break
            print()
            print("Relative event number",i)
            print("Combination of b at 0 level with other bs",pair_cross[i])
            print("Whole Combination of existing b chain with other bs",pair_cross_final[i])

            print("First element of each pair (from init)",pair_cross[:,:,"0"][i])
            print("Second element of each pair",pair_cross[:,:,"1"][i])
        
            print("Mom mask for all pairs",ak.to_numpy(combo_mom_mask[i]))
            print("Successful pairs of mothers and daughters",pair_cross[combo_mom_mask][i])

            print("Length of unflattening for the pair_comp",num_comp[i])
            print("Number of b fresh quarks in each event",ak.num(init,axis=2)[i])
            print("Array for comparing whether any bs are not daughtered",pair_comp[i])

            print("First element of each pair (from init) for pair_comp",pair_comp[:,:,"0"][i])
            print("Second element of each pair for pair_comp",pair_comp[:,:,"1"][i])
            #print("Mother index of second element",data["GenPart"][pair_comp[:,:,"1"]]["genPartIdxMother"])

            print("New comparison mom mask",mom_comp_mask[i])
            print("Count for unflat cut",no_mom_match[i]) #This only counts the non - none entries so they will exclude all false entries
            
            print("Number of matches found at this level",num_match[i])
            print("Mask value for no matches found",level_match_mask[i])

            print("Dummy value to be inserted for concatenating to produce the fin array (nGenpart)",ngenpart[i])
            print("Dummy cross product",dummy_ngen[i])
            print(dummy_ngen[:,:,"0"][:,:,0][i])

            print("Whole Combination of existing b chain with other bs",pair_cross_final[combo_mom_mask][i])
            print("pair_combo_concat",pair_combo_concat[i])
            #print("fin_nomask_final",fin_nomask_final[i])
            print("fin_mask_unflat",fin_mask_unflat[i])
            print("fin_mask_final",fin_mask_final[i]) #After replacing the dummy array with an empty array before concatenating with the initial recursion level
            print("Final b chains at 0 and 1 level including the ones that did not get matched",fin_final[i])
        '''

        #print("Successful pairs of mothers and daughters for event 95",pair_cross[combo_mom_mask][95])

        #hahaha
        recursion_level+=1
        event_mom_match = ak.mask(ngenpart,num_match > 0)
        repeat_recursion = ak.count(event_mom_match, axis=0)

        #repeat_recursion = ak.num(num_match > 0,axis=0)
        print("repeat_recursion",repeat_recursion)
        #hahaha
        #Temporary condition
        '''
        if recursion_level>3:
            pass
        else:
            fin_final = self.make_chain(fresh,part_index,fin_mask_final,fin_final,recursion_level,data)
        '''
        if repeat_recursion == 0:
            pass
        elif repeat_recursion > 0:
            fin_final = self.make_chain(fresh,part_index,fin_mask_final,fin_final,recursion_level,data)

        return fin_final

    #Checking dR matching only for the last elements (b) now that the chains have been made
    def dR_bjet_chain_match(self, data):

        #b_last_object = data["b_fin_final"]
        #b_last_object = data["b_fin_final"][:,:,-1]
        #b_first_object = data["b_fin_final"][:,:,0]

        #The new method chain_last produces the chains in reverse order
        b_last_object = data["b_fin_final"][:,:,0]
        b_first_object = data["b_fin_final"][:,:,-1]

        b_bbar_object = data["GenPart"][b_last_object] 
        b_last_pdgid = b_bbar_object.pdgId

        #dr_mask = ak.mask(data,(ak.num(data["bjets_tight"].eta)>0)&(ak.num(data["bjets_tight"].matched_gen.eta)>0)) #So it should have reco pt and also matched gen pt
        dr_mask = ak.mask(data,(ak.num(data["bjets_tight"].eta)>0) & (ak.count(data["bjets_tight"].matched_gen.eta,axis=1)>0)) #So it should have reco pt and also matched gen pt
        #dr_gen_mask = ak.mask(data,ak.num(data["bjets_tight"].matched_gen.eta)>0)
        dr_part_mask = ak.mask(data,ak.num(b_bbar_object.eta)>0)
        #print("Size of events satisfying the mask (having bjets)",len(dr_mask))

        #of course lengths of l_col and "bjets_tight" are supposed to be the same, that is why they are used interchangeably
        default_array = np.zeros((len(data["bjets_tight"]), 2))
        default_part_array = np.zeros((len(data["bjets_tight"]), 1))
        empty_array = [None]*len(data["bjets_tight"])
        #Since this is being calculated after the at least one lepton cut is put, there is no need to ensure whether the lepton exists or not

        data_bjet_pt = ak.where(ak.is_none(dr_mask), default_array, data["bjets_tight"].matched_gen.pt)
        #data_bjet_mass = ak.where(ak.is_none(dr_mask), default_array, data["bjets_tight"].mass) #What exactly is this?
        data_bjet_eta = ak.where(ak.is_none(dr_mask), default_array, data["bjets_tight"].matched_gen.eta)
        data_bjet_phi = ak.where(ak.is_none(dr_mask), default_array, data["bjets_tight"].matched_gen.phi)

        data_bpart_eta = ak.where(ak.is_none(dr_part_mask), default_part_array, b_bbar_object.eta)
        data_bpart_phi = ak.where(ak.is_none(dr_part_mask), default_part_array, b_bbar_object.phi)

        dr_dummy_old = np.sqrt(pow(data_bjet_eta[:,0]-data_bpart_eta,2) + pow(data_bjet_phi[:,0]-data_bpart_phi,2))

        pi = 4*math.atan(1)
        #print("value of pi",pi)

        #This is added because of the error found at the time of the mother correction script - this is not working directly, so flatten, compute and apply the mask, and then unflatten
        delta_phi = abs(data_bjet_phi[:,0]-data_bpart_phi)
        delta_phi_flat = ak.flatten(delta_phi)
        #print("Number of b quarks in all events (after flattening)",len(delta_phi_flat))
        pi_array = [2*pi]*len(delta_phi_flat)

        large_phi_mask = ak.mask(delta_phi_flat,delta_phi_flat>pi)
        #large_phi_mask = ak.mask(delta_phi_flat,delta_phi_flat>2) #Different lower limit for debugging
        delta_phi_new = ak.where(ak.is_none(large_phi_mask), delta_phi_flat, pi_array-delta_phi_flat)

        '''
        print("length of delta_phi_new",ak.num(delta_phi_new,axis=0))
        print("Length of data_bpart_phi",ak.num(data_bpart_phi,axis=0))
        print("Length of delta_phi",ak.num(delta_phi,axis=0))
        print("length of delta_phi_flat",ak.num(delta_phi_flat,axis=0))
        '''

        len_data_bpart_phi = 0
        len_delta_phi = 0

        '''
        for i in range(len(data["Jet"])):
            print()
            print("Event number in dR_bjet_chain_match",i)
            ent_data_bpart_phi = ak.num(data_bpart_phi,axis=1)[i]
            print("Number of entries in data_bpart_phi",ent_data_bpart_phi)
            print("data_bjet_phi[:,0]",data_bjet_phi[:,0][i]) # this is turning out to be none, which it really should not have been
            print("dr_mask",dr_mask[i])
            print("Bjet attributes")
            print("pt",data["bjets_tight"][i].pt)
            print("eta",data["bjets_tight"][i].eta)
            print("phi",data["bjets_tight"][i].phi)

            print("Bjet matched_gen attributes")
            print("pt",data["bjets_tight"].matched_gen[i].pt)
            print("eta",data["bjets_tight"].matched_gen[i].eta)
            print("phi",data["bjets_tight"].matched_gen[i].phi)
            print("How many elements are not none", ak.count(data["bjets_tight"].matched_gen.eta,axis=1)[i]) #This should skip the number that is None

            print("Final bjet attributes")
            print("pt",data_bjet_pt[i])
            print("eta",data_bjet_eta[i])
            print("phi",data_bjet_phi[i])

            print("data_bpart_phi",data_bpart_phi[i])
            print("delta_phi",delta_phi[i])
            ent_delta_phi = ak.num(delta_phi,axis=1)[i]
            print("Number of entries in delta_phi",ent_delta_phi)
            #print("Number of entries in data_bjet_phi[:,0]",len(data_bjet_phi[:,0][i]))
            len_data_bpart_phi+=ent_data_bpart_phi
            #len_delta_phi+=ent_delta_phi

        #Both should be equal and amount to ak.num(delta_phi_new,axis=0) [Fully flattened number of entries]
        print("len_data_bpart_phi",len_data_bpart_phi)
        #print("len_delta_phi",len_delta_phi)
        '''

        delta_phi_final = ak.unflatten(delta_phi_new,ak.num(data_bpart_phi,axis=1))
        #hahaha
        #Only for debugging
        count_pi = ak.count(ak.mask(delta_phi,delta_phi>pi),axis=1)

        dr_dummy = np.sqrt(pow(data_bjet_eta[:,0]-data_bpart_eta,2) + pow(delta_phi_final,2))
        min_arr = ak.min(dr_dummy,axis=1)
        #min_delta_r = b_mother[dr_dummy == min_arr]
        b_first_gen_object = data["GenPart"][b_first_object] 
        non_b_mother = data["GenPart"][b_first_gen_object.genPartIdxMother]
        min_delta_r = non_b_mother[dr_dummy == min_arr]

        '''
        b_mother = data["GenPart"][b_bbar_object.genPartIdxMother]
        min_arr = ak.min(dr_dummy,axis=1)
        min_delta_r = b_mother[dr_dummy == min_arr]

        match_pdgId = b_bbar_object[dr_dummy == min_arr].pdgId[:,0]
        b_charge_good = data["GenPart"][data["GenPart"][(data["GenPart"].pdgId)==(match_pdgId)].genPartIdxMother]
        non_b_mother = b_charge_good[b_charge_good.pdgId!=match_pdgId].pdgId

        b_mother_final = ak.where(ak.is_none(dr_mask), empty_array, match_pdgId)
        non_b_mother_final = ak.where(ak.is_none(dr_mask), empty_array, non_b_mother)

        two_mothers = ak.num(non_b_mother_final,axis=1)
        #two_mom_num = ak.num(two_mothers[two_mothers>1],axis=0)
        two_mom_num = ak.num(two_mothers[two_mothers==2],axis=0) #for debugging

        #print("Number of leading jet matched events (b quarks) having more than one mother",two_mom_num)
        #print("Number of leading jet matched events (b quarks) having exactly two mothers",two_mom_num)
        '''
        '''
        for i in range(len(data["Jet"])):
            print()
            print("Event number in dR_bjet_chain_match",i)
            print("b chains for event (first to last)",b_last_object[i])
            print("pdg id from gen part object",b_last_pdgid[i])
            print("Leading B jet pt",data["bjets_tight"][i].pt)
            print("Leading B jet matched gen pt",data["bjets_tight"][i].matched_gen.pt)
            print("dR dummy not adjusted",dr_dummy_old[i])

            print("Wrong phi (not adjusted for greater than pi values)",data_bjet_phi[:,0][i]-data_bpart_phi[i])
            print("Absolute value of unadjusted delta phi",abs(data_bjet_phi[:,0][i]-data_bpart_phi[i]))
            #print("Adjusted value of delta phi",delta_phi[i])
            print("Number of b quarks in event (counts for unflattening delta_phi_new)",ak.num(data_bpart_phi,axis=1)[i])
            print("Adjusted value of delta phi after flattening, applying mask and unflattening",delta_phi_final[i])
            #Only for debugging
            print("Values of delta phi greater than pi",count_pi[i])

            #Comparing with delta_r from the vector.py
            if (len(data["bjets_tight"].matched_gen[i])>0) & (len(b_bbar_object[i])>0):
                dR_function = data["bjets_tight"].matched_gen[i][0].delta_r(b_bbar_object[i])
                print("dR dummy adjusted",dr_dummy[i])
                print("dR function",dR_function)

            print("Last b having minimum dR",b_last_object[dr_dummy == min_arr][i])
            print("First b having minimum dR",b_first_object[dr_dummy == min_arr][i])
            print("Pdg id of all non b mothers",non_b_mother.pdgId[i])
            print("Pdg id of non b mothers",min_delta_r.pdgId[i])

        '''

        #return b_mother_final, non_b_mother_final, dr_dummy
        return data, min_delta_r, dr_dummy


    def dR_bjet_part_match(self, data):

        b_bbar_object = data["GenPart"][(abs(data["GenPart"].pdgId)==5)]

        dr_mask = ak.mask(data,(ak.num(data["bjets_tight"].eta)>0)&(ak.num(data["bjets_tight"].matched_gen.eta)>0)) #So it should have reco pt and also matched gen pt
        #dr_gen_mask = ak.mask(data,ak.num(data["bjets_tight"].matched_gen.eta)>0)
        dr_part_mask = ak.mask(data,ak.num(b_bbar_object.eta)>0)
        #print("Size of events satisfying the mask (having bjets)",len(dr_mask))

        #of course lengths of l_col and "bjets_tight" are supposed to be the same, that is why they are used interchangeably
        default_array = np.zeros((len(data["bjets_tight"]), 2))
        default_part_array = np.zeros((len(data["bjets_tight"]), 1))
        empty_array = [None]*len(data["bjets_tight"])
        #Since this is being calculated after the at least one lepton cut is put, there is no need to ensure whether the lepton exists or not

        data_bjet_pt = ak.where(ak.is_none(dr_mask), default_array, data["bjets_tight"].matched_gen.pt)
        #data_bjet_mass = ak.where(ak.is_none(dr_mask), default_array, data["bjets_tight"].mass) #What exactly is this?
        data_bjet_eta = ak.where(ak.is_none(dr_mask), default_array, data["bjets_tight"].matched_gen.eta)
        data_bjet_phi = ak.where(ak.is_none(dr_mask), default_array, data["bjets_tight"].matched_gen.phi)

        data_bpart_eta = ak.where(ak.is_none(dr_part_mask), default_part_array, b_bbar_object.eta)
        data_bpart_phi = ak.where(ak.is_none(dr_part_mask), default_part_array, b_bbar_object.phi)

        dr_dummy_old = np.sqrt(pow(data_bjet_eta[:,0]-data_bpart_eta,2) + pow(data_bjet_phi[:,0]-data_bpart_phi,2))

        pi = 4*math.atan(1)
        #print("value of pi",pi)

        #This is added because of the error found at the time of the mother correction script - this is not working directly, so flatten, compute and apply the mask, and then unflatten
        delta_phi = abs(data_bjet_phi[:,0]-data_bpart_phi)
        delta_phi_flat = ak.flatten(delta_phi)
        #print("Number of b quarks in all events (after flattening)",len(delta_phi_flat))
        pi_array = [2*pi]*len(delta_phi_flat)

        large_phi_mask = ak.mask(delta_phi_flat,delta_phi_flat>pi)
        #large_phi_mask = ak.mask(delta_phi_flat,delta_phi_flat>2) #Different lower limit for debugging
        delta_phi_new = ak.where(ak.is_none(large_phi_mask), delta_phi_flat, pi_array-delta_phi_flat)

        delta_phi_final = ak.unflatten(delta_phi_new,ak.num(data_bpart_phi,axis=1))

        #Only for debugging
        count_pi = ak.count(ak.mask(delta_phi,delta_phi>pi),axis=1)

        dr_dummy = np.sqrt(pow(data_bjet_eta[:,0]-data_bpart_eta,2) + pow(delta_phi_final,2))
        
        #'''
        '''
        mother_1 = data["GenPart"][b_bbar_object.genPartIdxMother]
        min_arr = ak.min(dr_dummy,axis=1)
        min_delta_r = mother_1[dr_dummy == min_arr]
        '''

        b_mother = data["GenPart"][b_bbar_object.genPartIdxMother]
        min_arr = ak.min(dr_dummy,axis=1)
        min_delta_r = b_mother[dr_dummy == min_arr]

        match_pdgId = b_bbar_object[dr_dummy == min_arr].pdgId[:,0]
        b_charge_good = data["GenPart"][data["GenPart"][(data["GenPart"].pdgId)==(match_pdgId)].genPartIdxMother]
        non_b_mother = b_charge_good[b_charge_good.pdgId!=match_pdgId].pdgId

        b_mother_final = ak.where(ak.is_none(dr_mask), empty_array, match_pdgId)
        non_b_mother_final = ak.where(ak.is_none(dr_mask), empty_array, non_b_mother)

        two_mothers = ak.num(non_b_mother_final,axis=1)
        #two_mom_num = ak.num(two_mothers[two_mothers>1],axis=0)
        two_mom_num = ak.num(two_mothers[two_mothers==2],axis=0) #for debugging

        #print("Number of leading jet matched events (b quarks) having more than one mother",two_mom_num)
        #print("Number of leading jet matched events (b quarks) having exactly two mothers",two_mom_num)

        '''
        for i in range(len(data["Jet"])):
            print()
            print("Event number in dR_bjet_part_match",i)
            print("Leading B jet pt",data["bjets_tight"][i].pt)
            print("Leading B jet matched gen pt",data["bjets_tight"][i].matched_gen.pt)
            print("dR dummy not adjusted",dr_dummy_old[i])

            print("Wrong phi (not adjusted for greater than pi values)",data_bjet_phi[:,0][i]-data_bpart_phi[i])
            print("Absolute value of unadjusted delta phi",abs(data_bjet_phi[:,0][i]-data_bpart_phi[i]))
            #print("Adjusted value of delta phi",delta_phi[i])
            print("Number of b quarks in event (counts for unflattening delta_phi_new)",ak.num(data_bpart_phi,axis=1)[i])
            print("Adjusted value of delta phi after flattening, applying mask and unflattening",delta_phi_final[i])
            #Only for debugging
            print("Values of delta phi greater than pi",count_pi[i])

            #Comparing with delta_r from the vector.py
            if (len(data["bjets_tight"].matched_gen[i])>0) & (len(b_bbar_object[i])>0):
                dR_function = data["bjets_tight"].matched_gen[i][0].delta_r(b_bbar_object[i])
                print("dR dummy adjusted",dr_dummy[i])
                print("dR function",dR_function)

            #if count_pi[i] > 0:
            #    hahaha
            print("first level of b mothers",b_bbar_object[i].pdgId)
            print("second level of b mothers",b_mother[i].pdgId)
            print("Matched pdg id",match_pdgId[i])
            print("Bcharge good",non_b_mother[i])
            print("Bcharge good final",non_b_mother_final[i])
            print("Number of non b mothers",two_mothers[i])
        '''

        #return b_mother_final, non_b_mother_final, dr_dummy
        return data, min_delta_r, dr_dummy

    def b_recursion(self,b_recursion_level,b_mother,data):
        #print("Reached level ", str(b_recursion_level)," of mothers")
        no_b_id = (abs(b_mother.pdgId) != 5)

        repeat_only_pdg = ak.num(b_mother[no_b_id],axis=1)
        repeat_num = ak.num(repeat_only_pdg[repeat_only_pdg==0],axis=0)

        end_index = -1
        #end_index = 10 #Only for debugging and testing since no event at this level actually seems to have the index -1

        #These are the number of events that have only repeat mothers and do not any mothers at the next level (i.e. something like mother)
        repeat_end = ak.num(b_mother[b_mother.genPartIdxMother == end_index],axis=1)
        repeat_end_num = ak.num(repeat_end[(repeat_end==ak.num(b_mother,axis=1)) & (repeat_only_pdg==0)],axis=0)
        #print("Number of events having only repeat b mothers",repeat_num)
        #print("Number of events having only repeat b mothers that are also at the end (beginning)",repeat_end_num)

        repeat_mask = ak.mask(data,repeat_only_pdg==0)
        '''
        for i in range(len(data["Jet"])):
            print()
            print("Event number in part 1 of b_recursion",i)
            #print("Lepton pdg id",data["Lepton"].pdgId[:,0][i])
            print("Non repeating non b quark mothers",repeat_only_pdg[i])
            print("b quark mothers at level",str(b_recursion_level),b_mother.pdgId[i])
            print("Dr dummy array",data["bjet_part_dr"][i])
            print("Repeat mask",repeat_mask[i])
            print("Number of b quark mothers for each event at first level",ak.num(b_mother,axis=1)[i])
            print("Number of particles having mother index as ", str(end_index),"for each event",repeat_end[i])
            print("b quark mother indices at second level",b_mother.genPartIdxMother[i])
        '''    

        if (repeat_num > 0) & (repeat_end_num < repeat_num): #Go the next level of mothers
            b_recursion_level +=1
            b_mother_2 = data["GenPart"][b_mother.genPartIdxMother]

            b_mother = ak.where(ak.is_none(repeat_mask), b_mother[no_b_id], b_mother_2)
            b_mother = self.b_recursion(b_recursion_level,b_mother,data)
            
            '''
            for i in range(len(data["Jet"])):
                print()
                print("Event number in part 2 of b_recursion",i)
                print("b quark mothers at level",str(b_recursion_level),b_mother_2.pdgId[i])
                print("Final b quark mothers at level",str(b_recursion_level),b_mother.pdgId[i])
            '''

        return b_mother

    def top_daughters(self,data):
        broadcast_value = -1
        local_index = ak.local_index(abs(data["GenPart"].pdgId)==6)
        broadcast = ak.broadcast_arrays(abs(data["GenPart"].pdgId)==6, broadcast_value)[1]
        top_index = ak.where(abs(data["GenPart"].pdgId)==6, local_index, broadcast)
        top_index_final = top_index[top_index != broadcast_value]          

        #Using a caveat, just take the last element of all the top indices, obviously we are assuming that they are all part of the same chain (since single top)
        #and obviously if it produces a W and b before that then it won't give another top in the chain
        last_top = top_index_final[:,-1]
        mom_mask = data["GenPart"].genPartIdxMother == last_top

        top_pdgid = data["GenPart"][abs(data["GenPart"].pdgId)==6].pdgId
        top_last_pdgid = top_pdgid[:,-1]
        top_last_sign = top_last_pdgid/abs(top_last_pdgid)

        daughter_top = data["GenPart"][mom_mask]
        daughter_index = ak.where(mom_mask, ak.local_index(mom_mask), ak.broadcast_arrays(mom_mask, broadcast_value)[1])
        daughter_index_b_1_final = daughter_index[daughter_index != broadcast_value]        

        return daughter_index_b_1_final, daughter_top, top_last_sign, data


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


    def b_daughters_recursion(self, n_daughters, recursion_level, daughter_index_b_1_final, daughter_previous, top_last_sign, data):
        broadcast_value = -1
        empty_array = [[-1]]*len(data["Jet"])
        null_array = [[]]*len(data["Jet"])
        default_array = [None]*len(data["Jet"])
        zero_daughters = [0]*len(data["Jet"])

        #First step of finding daughters for b quark from the resonant step
        #===================================================================================================================================================

        b_index_1 = daughter_index_b_1_final[daughter_previous.pdgId == top_last_sign*5]

        b_mask_1 = ak.mask(data,ak.num(b_index_1)>0)
        b_index_1_fill = ak.where(ak.is_none(b_mask_1), empty_array, b_index_1)

        mom_mask_1 = data["GenPart"].genPartIdxMother == b_index_1_fill[:,0]
        daughter_b_1 = data["GenPart"][mom_mask_1]
        daughter_index_b_1 = ak.where(mom_mask_1, ak.local_index(mom_mask_1), ak.broadcast_arrays(mom_mask_1, broadcast_value)[1])
        daughter_index_b_2_final = daughter_index_b_1[daughter_index_b_1 != broadcast_value]        


        #Obviously b_daughters_1 and neutral_fsr_daughters_1 will be set to 0 when we have to replace it with -1
        daughter_b_1 = ak.where(ak.is_none(b_mask_1), null_array, daughter_b_1)
        daughter_index_b_2_final = ak.where(ak.is_none(b_mask_1), null_array, daughter_index_b_2_final)

        b_daughters_1 = ak.num(daughter_b_1[daughter_b_1.pdgId == top_last_sign*5],axis = 1)
        neutral_fsr_daughters_1 = ak.num(daughter_b_1[(abs(daughter_b_1.pdgId) == 21) | (abs(daughter_b_1.pdgId) == 22) | (abs(daughter_b_1.pdgId) == 23)],axis = 1)

        daughter_mask_2 = ak.mask(data,(b_daughters_1==1) & (ak.num(daughter_b_1) == b_daughters_1+neutral_fsr_daughters_1))
        n_daughter_fill = ak.where(ak.is_none(daughter_mask_2), ak.num(daughter_b_1) - (b_daughters_1+neutral_fsr_daughters_1), default_array)

        if recursion_level == 1:
            n_daughters = n_daughter_fill
        elif recursion_level > 1:
            n_daughters = ak.where(ak.is_none(n_daughters), n_daughter_fill, n_daughters)

        #ending_events = ak.num(n_daughters[n_daughters==0],axis = 0)
        ending_events = ak.num(b_daughters_1[b_daughters_1==0],axis = 0)
        #print("Number of events that have b at the end of the chain",ending_events)
        '''
        for i in range(len(data["Jet"])):
            print()
            print("Event number",i)
            print("Recursion level",recursion_level)
            print("Last top sign",top_last_sign[i])
            print("Daughter pdg ids",daughter_previous[i].pdgId)
            print("Daughter particle indices",daughter_index_b_1_final[i])

            print("B index at step (filled with -1s)",b_index_1_fill[i])
            print("Mother matching mask",ak.to_numpy(mom_mask_1[i]))

            print("Daughter pdg ids of b",daughter_b_1[i].pdgId)
            print("Daughter particle indices of b",daughter_index_b_2_final[i])

            print("Number of b daughters of b quark",b_daughters_1[i])
            print("Number of neutral fsr daughters of b quark",neutral_fsr_daughters_1[i])

            print("Daughter mask 2",((b_daughters_1==1) & (ak.num(daughter_b_1) > b_daughters_1+neutral_fsr_daughters_1))[i])
            print("B index mask",(b_daughters_1==1)[i])
            print("Other daughters mask",(ak.num(daughter_b_1) > b_daughters_1+neutral_fsr_daughters_1)[i])

            print("Number of daughters (after first b where it does not give another b)",n_daughter_fill[i])
            print("Final number of daughters at that level",n_daughters[i])
        '''
        #if recursion_level < 3:
        if ending_events < len(data["Jet"]):
            recursion_level+=1
            n_daughters = self.b_daughters_recursion(n_daughters, recursion_level, daughter_index_b_2_final, daughter_b_1, top_last_sign, data) 

        return n_daughters        

    def event_display(self, i, folder, data):
        G = nx.DiGraph()
        pdgIds = {}
        #print()
        #print("Event number",i)
        ngenpart = len(data["GenPart"][i])
        #print("Number of gen particles",ngenpart)
        for gpIdx in range(ngenpart):
            motherIdx = data["GenPart"][i]["genPartIdxMother"][gpIdx]
            pdgId = data["GenPart"][i]["pdgId"][gpIdx]
            pdgIds[gpIdx] = pdgId
            '''
            print()
            print("motherIdx",motherIdx)
            if motherIdx>-1:
                mother_pdgId = data["GenPart"][i]["pdgId"][motherIdx]
                print("pdg id of mother particle?", mother_pdgId)
            
            else:
                print("No mother")
            print("pdgId",pdgId)
            '''
            G.add_node(gpIdx,label=pdgId)
            if motherIdx>-1:
                G.add_edge(motherIdx,gpIdx)


        nx.set_node_attributes(G, pdgIds,"pdgIds")
        pos = graphviz_layout(G,prog="twopi")
        
        #hahaha
        pos_new = {}
        for key in pos.keys():
            #print()
            #print("Int key",int(key))
            #pos[new_int_key] = pos.pop[key]
            pos_new[int(key)] = pos[key]
            str_keys = 0
            #print("New keys",pos_new.keys())
            #print(pos_new)
            #print("Length of keys",len(pos_new))

        pos = pos_new
        #print(pos.keys())
        #print("Graph object",G)
        #print(pos)

        #nx.draw(G,pos,node_size=10,font_size=8,with_labels=True,labels=pdgIds)
        #nx.draw_networkx(G,pos,node_size=10,font_size=8,with_labels=True,labels=pdgIds)    
        nx.draw_networkx(G,pos,node_size=20,font_size=8,with_labels=True,labels=pdgIds)    
        #print("Event string name",str(e.event))
        plt.savefig(folder + "Event_" + str(i) + ".png")
        plt.clf()

    def event_display_new(self, i, mother_new, folder, data):
        G = nx.DiGraph()
        pdgIds = {}
        #print()
        #print("Event number",i)
        ngenpart = len(data["GenPart"][i])
        #print("Number of gen particles",ngenpart)
        for gpIdx in range(ngenpart):
            motherIdx_old = data["GenPart"][i]["genPartIdxMother"][gpIdx]
            motherIdx = mother_new[gpIdx]
            pdgId = data["GenPart"][i]["pdgId"][gpIdx]
            pdgIds[gpIdx] = pdgId
            #print()
            #print("motherIdx_old",motherIdx_old)
            #print("motherIdx",motherIdx)

            '''
            print()
            print("motherIdx",motherIdx)
            if motherIdx>-1:
                mother_pdgId = data["GenPart"][i]["pdgId"][motherIdx]
                print("pdg id of mother particle?", mother_pdgId)
            
            else:
                print("No mother")
            print("pdgId",pdgId)
            '''
            G.add_node(gpIdx,label=pdgId)
            if motherIdx>-1:
                G.add_edge(motherIdx,gpIdx)


        nx.set_node_attributes(G, pdgIds,"pdgIds")
        pos = graphviz_layout(G,prog="twopi")
        
        #hahaha
        pos_new = {}
        for key in pos.keys():
            #print()
            #print("Int key",int(key))
            #pos[new_int_key] = pos.pop[key]
            pos_new[int(key)] = pos[key]
            str_keys = 0
            #print("New keys",pos_new.keys())
            #print(pos_new)
            #print("Length of keys",len(pos_new))

        pos = pos_new
        #print(pos.keys())
        #print("Graph object",G)
        #print(pos)

        #nx.draw(G,pos,node_size=10,font_size=8,with_labels=True,labels=pdgIds)
        #nx.draw_networkx(G,pos,node_size=10,font_size=8,with_labels=True,labels=pdgIds)    
        nx.draw_networkx(G,pos,node_size=20,font_size=8,with_labels=True,labels=pdgIds)    
        #print("Event string name",str(e.event))
        plt.savefig(folder + "Event_new_" + str(i) + ".png")
        plt.clf()

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
