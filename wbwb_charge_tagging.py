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
            selector.add_cut(
                "CrossSection", partial(self.crosssection_scale, dsname))

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

        # lepton columns
        selector.set_multiple_columns(self.pick_electrons)
        selector.set_multiple_columns(self.pick_muons)
        selector.set_column("Muon",partial(self.apply_rochester_corr,selector.rng,is_mc))
        #selector.set_column("Lepton", partial(self.build_lepton_column, is_mc, selector.rng))
        selector.set_column("Lepton", partial(self.build_lepton_column, is_mc, selector.rng,""))
        selector.add_cut("HasLepton", partial(self.require_lepton_nums, is_mc),
                    no_callback=True)

        # Require 1 muon, no veto muons
        selector.add_cut("OneGoodLep", self.one_good_lepton)
        selector.add_cut("NoVetoLeps", self.no_veto_muon)

        selector.set_cat("channel", {"mu", "el" })
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

        idx = selector.data["jetorigin"]["jetIdx"]
        jetorigin = selector.data["jetorigin"]
        #jetorigin["eta"] = selector.data["Jet"][idx]["eta"]
        #jetorigin["pt"] = selector.data["Jet"][idx]["pt"]
        #print("Fields in reco jets",selector.data["Jet"].fields)
        #print("Fields in gen jets",selector.data["GenJet"].fields)

        #This is the info that will be used for get the scores/probability values from the frozen function (tensorflow)

        #===================================================================================================================================================
        selector.set_column("Jet", partial(self.build_jet_column, is_mc, 0))
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

        #print("Number of entries in the gen and reco id arays",len(gen_had_flavor),len(gen_bhad_id),len(reco_btag))
        rec_gen_bhad = ak.from_numpy(np.array(list(zip(gen_had_flavor,gen_bhad_id,reco_btag))).flatten())
        flag_jets = ak.unflatten(rec_gen_bhad, 3)

        #Probably have to include it in selector.data 

        #===================================================================================================================================================
        tfdata = {}
        #'''
        #with uproot.recreate("tensor_list.root") as file:

        for name,featureGroup in featureDict.items():
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
            
            if len(type_array) == len(attr_array):
                for i in range(len(type_array)):
                    arr = ak.to_numpy(ak.flatten(selector.data[type_array[i]][attr_array[i]]))
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
                
                tfdata[name] = array_padded_jet_zeros
                #print("Original Depth (?)",array_padded_jet_zeros.layout.minmax_depth)

                jet_unfl = ak.unflatten(array_padded_jet_zeros, feat_num, axis=1)
                #print("Unflattened array along axis one",jet_unfl)
                #print("Length along axis zero",ak.num(jet_unfl,axis=0))
                #print("Length along axis one",ak.num(jet_unfl,axis=1))
                #print("Depth (?)",jet_unfl.layout.minmax_depth)

                jet_unfl_np = ak.to_numpy(jet_unfl)
                #print(jet_unfl_np)
                #print("Shape of numpy array", jet_unfl_np.shape)
                
                #jet_unfl_tf = tf.convert_to_tensor(jet_unfl)
                #print(jet_unfl_tf)
                #print("Shape of tf tensor", jet_unfl_tf.shape)
                
                
                #for i in range(ak.num(jet_unfl,axis=0)):
                #    print(jet_unfl[i])
                #    print(ak.num(jet_unfl[i],axis=0))
                #    print(ak.num(jet_unfl[i],axis=1))
                
            else:
                tfdata[name] = array_constituent

            #print(tfdata[name])
            name_branch = name + "_branch"
            #file[name] = {name_branch: ak.to_numpy(tfdata[name])}

            #file[name] = {name_branch: tfdata[name]} #This doesn't seem to be working properly, it completely flattens the array if it is 2d along axis 1
            #file[name] = {name: ak.to_numpy(tfdata[name])}
        #'''
        #=================================================================================================================================

        #Potentially very bad method of using one tree for each feature group and assigning only one branch to each
        '''
            for name,featureGroup in featureDict.items():
                print("Feature group tree (and branch) in root file",name)
                #file[name] = uproot.newtree({name : "float32"})
                print("Data Type of feature group",type(name))
                #file[name] = uproot.newtree({name})
                #file[name].extend({name: tfdata[name]})
                name_branch = name + "_branch"
                file[name].extend({name_branch: tfdata[name]})

            #print("Shape of data for feature group",tfdata.shape)
        '''        
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
        
        #with uproot.open("tensor_list.root") as file:
        '''
        print("Contents of global tree in tensor list file")
        #print(file["global"].array("global"))
        print(file["global"].arrays()["global_branch"])
        print("Type of arrays in global branch")
        print(type(file["global"].arrays()["global_branch"]))
        print("Sizes of arrays in global branch")
        print(ak.num(file["global"].arrays()["global_branch"],axis=1))
        print(ak.num(file["cpf"].arrays()["cpf_branch"],axis=1))
        print(file["cpf"].arrays()["cpf_branch"])
        '''
        #jet_total = ak.num(file["global"].arrays()["global_branch"],axis=0)
        jet_total = ak.num(tfdata["global"],axis=0)
        #print("Total number of jets from all the events",jet_total)
        flag_final = []
        logit_final = []
        prob_final = []

        for jet_num in range(jet_total):
            #print("Jet number",jet_num)
            #print(flag_jets[jet_num])
            #print(file["global"].arrays()["global_branch"][jet_num])
            
            #if jet_num >= 5:
            #    break
            batch = {}
            #testModel_mk
            for featureGroup in featureGroups:
                #print("Feature name for frozen function (different order)",featureGroup)
                
                #float_feature = ak.values_astype(file[featureGroup].arrays()[featureGroup + "_branch"][jet_num], 'float32')
                float_feature = ak.values_astype(tfdata[featureGroup][jet_num], 'float32')
                #for k in range(len(file[featureGroup].arrays()[featureGroup + "_branch"][jet_num])):
                #    print(type(file[featureGroup].arrays()[featureGroup + "_branch"][jet_num][k]))
                
                feature_tensor = tf.convert_to_tensor(float_feature)
                #feature_tensor = feature_tensor.astype('float32')
                #feature_tensor = tf.cast(feature_tensor,fl)
                #print("Original shape",feature_tensor.shape)
                #print(type(featureGroup["max"]))
                if 'max' in featureDict[featureGroup].keys():
                    #print(type(featureDict[featureGroup]["max"]))
                    #Had to reshape again instead of directly getting it from the shape because it was not properly stored in the root file
                    nmax = featureDict[featureGroup]["max"]
                    nfeature = feature_tensor.shape[0]/nmax
                    '''
                    print(type(nfeature))
                    print(nfeature)
                    print(type(nmax))
                    print(nmax)
                    '''
                    #print("Type of feature tensor",type(feature_tensor))
                    
                    batch[featureGroup] = tf.reshape(feature_tensor,[1,nmax,int(nfeature)])
                    #feature_tensor = tf.reshape(feature_tensor,[1,nmax,int(nfeature)])
                else:
                    batch[featureGroup] = tf.reshape(feature_tensor,[1,feature_tensor.shape[0]])

                #print("Shape to inputted to function",batch[featureGroup].shape)
                #print(batch[featureGroup])

            logit_jet = frozen_func(**batch)
            prob = softmax(logit_jet)

            #'''
            #print("Original b flavor and hadron id values")
            #print(flag_jets[jet_num])
            flag_final = np.append(flag_final,flag_jets[jet_num]) #Ok if the dimension is not preserved

            #print("Flags for B mesons evaluated")
            #print("Logit values")
            #print(logit_jet)
            logit_final = np.append(logit_final,logit_jet) #Ok if the dimension is not preserved

            #print("Probability values")
            #print(prob.shape) #Very weird shape, what was the need to add another extra dimension
            #print(prob[0][0])
            prob_final = np.append(prob_final,prob[0][0]) #Ok if the dimension is not preserved

        #print("Shape of prob array", np.shape(prob_final))
        #print("Shape of prob array", len(ak.from_numpy(prob_final)))
        
        #selector.set_column("NN_prob",ak.unflatten((ak.from_numpy(prob_final)),4))
        #Assigning the meson categories here - they will only be assigned to jets having the reco btag flag as 1 (since that will be the case for data as well)
        meson_cat = ["bneg","b0bar","b0","bpos"]
        prob_jet = ak.unflatten((ak.from_numpy(prob_final)),4)

        selector.data["prob_neg"] = ak.unflatten(prob_jet[:,0],ak.num(selector.data["Jet"],axis=1))
        selector.data["prob_zerobar"] = ak.unflatten(prob_jet[:,1],ak.num(selector.data["Jet"],axis=1))
        selector.data["prob_zero"] = ak.unflatten(prob_jet[:,2],ak.num(selector.data["Jet"],axis=1))
        selector.data["prob_pos"] = ak.unflatten(prob_jet[:,3],ak.num(selector.data["Jet"],axis=1))

        rec_bflag_jet = ak.unflatten(ak.flatten(selector.data["Jet"]["btagged"]),1)
        empty_array = [-1]*len(rec_bflag_jet) #To be filled for non b jets
        #rec_meson_cat = ak.where(rec_bflag_jet == False, empty_array, ak.argmax(prob_jet,axis=1))
        
        #Since this is not working, the flag will be applied later (while making categories) - right now all the jets have a flavor/index irrespective of whether they are bjets or not
        #rec_meson_cat = ak.where(rec_bflag_jet == False, None, True)
        #rec_meson_cat = ak.where(ak.is_none(rec_bflag_jet), empty_array, ak.argmax(prob_jet,axis=1))
        rec_meson_cat = ak.argmax(prob_jet,axis=1)
        #print("Length of category assigned array",len(rec_meson_cat))

        #selector.data["b_meson_tag"] = ak.unflatten(rec_meson_cat,ak.num(selector.data["Jet"],axis=1))
        selector.set_column("b_meson_tag", ak.unflatten(rec_meson_cat,ak.num(selector.data["Jet"],axis=1)))
        #print(ak.unflatten(rec_meson_cat,ak.num(selector.data["Jet"],axis=1)))
        #print(selector.data["b_meson_tag"])
        
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

        selector.data["Jet"], selector.data["b_meson_tag"],selector.data["b_gen_meson"], selector.data["prob_neg"], selector.data["prob_zerobar"], selector.data["prob_zero"], selector.data["prob_pos"] = self.build_jet_column(is_mc, 1, selector.data)
        #At least to check whether cuts are consistent

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
        selector.add_cut("JetPtReq", partial(self.jet_pt_requirement,"Jet"))
        if is_mc and self.config["compute_systematics"]:
            self.scale_systematics_for_btag(selector, variation, dsname)
        
        #selector.set_column("nbtag", self.num_btags)
        selector.set_column("nbtag", partial(self.num_btags,""))
        #selector.set_column("bjets", partial(self.build_btag_column,is_mc))  
        #selector.set_multiple_columns({"bjets","meson_tag_real"}, partial(self.build_btag_column,is_mc))  #added meson tag

        selector.data["bjets"], selector.data["meson_tag_real"],selector.data["gen_meson_real"], selector.data["prob_neg_real"], selector.data["prob_zerobar_real"], selector.data["prob_zero_real"], selector.data["prob_pos_real"] = self.build_btag_column(is_mc,selector.data)

        #==================================Fake and missing, not quite sure what to do with them at the moment==============================
        if is_mc:
            selector.data["bjets_miss"], selector.data["gen_meson_miss"], selector.data["prob_neg_miss"], selector.data["prob_zerobar_miss"], selector.data["prob_zero_miss"], selector.data["prob_pos_miss"] = self.build_missing_btag_column(is_mc,selector.data)
            selector.data["bjets_fake"], selector.data["meson_tag_fake"],selector.data["gen_meson_fake"], selector.data["prob_neg_fake"], selector.data["prob_zerobar_fake"], selector.data["prob_zero_fake"], selector.data["prob_pos_fake"] = self.build_fake_btag_column(is_mc,selector.data)

        #selector.set_column("mlb",partial(self.mass_lepton_bjet,"Lepton","bjets"))    
        #selector.set_column("dR_lb",partial(self.dR_lepton_bjet,"Lepton","bjets"))    

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
        selector.set_cat("b_meson_cat",set(cat_dict))

        #selector.set_cat("b_meson_cat",{"id_bneg_rec_bneg"}) #Small number of categories for debugging

        #selector.set_cat("b_meson_cat",{"no_b_jet", "id_bneg_rec_bneg", "id_bneg_rec_b0bar"}) #Small number of categories for debugging
        #, 'id_bneg_rec_b0', 'id_bneg_rec_bpos', 'id_b0bar_rec_bneg', 'id_b0bar_rec_b0bar', 'id_b0bar_rec_b0', 'id_b0bar_rec_bpos', 'id_b0_rec_bneg', 'id_b0_rec_b0bar', 'id_b0_rec_b0', 'id_b0_rec_bpos', 'id_bpos_rec_bneg', 'id_bpos_rec_b0bar', 'id_bpos_rec_b0', 'id_bpos_rec_bpos'})
        #selector.set_cat("b_meson_cat",{"no_b_jet"}) #Just for debugging
        selector.set_multiple_columns(partial(self.charge_tag_cat,is_mc)) #Again, only for mc, because data has no chance of having a gen hadron id

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

        for i in range(len(selector.data["Jet"])):
            print("Event number",i)
            print("New order of reco pt",selector.data["Jet"][i]["pt"])
            print("New Matching gen pt",selector.data["Jet"][i].matched_gen.pt) #This is kept as a control because it is unchanged by any scale/correction factor
            print("Assigned meson tag reshaped",selector.data["b_meson_tag"][i])
            print("Assigned gen meson tag",selector.data["b_gen_meson"][i])
            print("Number of jets in each event",ak.num(selector.data["Jet"],axis=1)[i])
            print("Probability values")
            print("B-",selector.data["prob_neg"][i])
            print("B0bar",selector.data["prob_zerobar"][i])
            print("B0",selector.data["prob_zero"][i])
            print("B+",selector.data["prob_pos"][i])

            print("B jet pt",selector.data["bjets"][i]["pt"])
            print("Bjet Matching gen pt",selector.data["bjets"][i].matched_gen.pt) #This is kept as a control because it is unchanged by any scale/correction factor
            print("Bjet meson tag reshaped",selector.data["meson_tag_real"][i])
            print("Bjet gen meson tag",selector.data["gen_meson_real"][i])
            print("Real Probability values")
            print("B-",selector.data["prob_neg_real"][i])
            print("B0bar",selector.data["prob_zerobar_real"][i])
            print("B0",selector.data["prob_zero_real"][i])
            print("B+",selector.data["prob_pos_real"][i])

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

                print("Values of b meson categorization masks")
                for key in cat_dict:
                    print(key,selector.data[key][i])
        '''

        if "btag_sf" in self.config and len(self.config["btag_sf"]) != 0:
            selector.add_cut("btagSF", partial(self.btag_sf, is_mc))

        #'''

        '''
        feat_brnch = []
        for k in featureDict.keys():
            feat_brnch.append(featureDict[k]['branches'])
            print("Feature group",k)
            
            if 'max' in featureDict[k].keys():
                feat_brnch+=[featureDict[k]['length'],featureDict[k]['offset']]

        print(feat_brnch)
        '''
        
        #'''

        #file_tree = self.create_tensor_file(*args_branch)

        #with uproot.open("tensor_list.root") as file:
        #    print("Branch names")
        #    print(file["jet_attr"])

            #file["jet_attr"] = {"branch": tfdata[name]}
            #file["jet_attr"] = {name: tfdata[name]}

        #Right now this is used for mixing and populating the root file with a tree for each feature group
        #=================================================================================================================================
        #'''

        #'''

        #This is for verifying whether the masks have been applied correctly
        '''
        for jet in range(len(selector.data["NN_prob"])):
            print("Jet number", jet)
            print("Gen id",selector.data["Gen_Hadron_Id"][jet])
            print("Rec id",selector.data["Reco_Jet_Flavor"][jet])
            print("Mask value for gen and reco b id match")
            print(selector.data["no_id_no_rec"])
            print(selector.data["id_no_rec"])
            print(selector.data["no_id_rec"])
            print(selector.data["id_rec"])
        '''
        '''
        with uproot.recreate("flavor_charge.root") as file:            
            #Here it doesn't really matter what the dimensions of the values stored is, we can always unflatten/reshape them 

            file["flav"]={"flav": flag_final}
            file["logit"]={"logit": logit_final}
            file["probability"]={"probability": prob_final}
        '''

        '''
        with uproot.open("flavor_charge.root") as file:
            print("Final flavor and id and logit and probability values")
            print(file["flav"].arrays()["flav"])
            print(len(file["flav"].arrays()["flav"]))
            print(file["logit"].arrays()["logit"])
            print(len(file["logit"].arrays()["logit"]))
            print(file["probability"].arrays()["probability"])
            print(len(file["probability"].arrays()["probability"]))
        '''

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
    
    def no_veto_muon(self, data):
        return ak.num(data["VetoMuon"]) + ak.num(data["VetoElectron"]) == 0
    
    def channel_masks(self, data):
        channels = {}
        channels["mu"] = (ak.num(data["Muon"]) == 1)
        channels["el"] = (ak.num(data["Electron"]) == 1)
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
                abs_eta_l = abs(data["bjets"][i][0].eta)
                abs_eta_sl = abs(data["bjets"][i][1].eta)

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
                abs_eta_l = abs(data["bjets"][i][0].eta)

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
        cats["no_b_jet"] = (ak.num(data["bjets"].eta) == 0) & (ak.num(data["Jet"]) >= 2)

        leading_cut = ak.mask(data,(ak.num(data["bjets"].eta)>=1) & (ak.num(data["Jet"]) >= 2))
        subleading_cut = ak.mask(data,(ak.num(data["bjets"].eta)>=2) & (ak.num(data["Jet"]) >= 2))

        #b_match = (id_flag) & (reco_b_flag) #not needed since bjets column already contains all that
        meson_cat = ["bneg","b0bar","b0","bpos"]
        hadron_id = [-521, -511, 511, 521]
        
        #rec_meson = ak.argmax(data["NN_prob"],axis=1)
        #This is the meson type chosen from the probabilities from the nn evaluation, and is a 1 entry per jet awkward array
        cut_array = [False] * len(data["bjets"]) 
        #cut_array = False * len(data["bjets"]) 

        hadron_id_net = ak.to_numpy(ak.unflatten(hadron_id* len(data["bjets"]),4))

        #print("Length of hadron id mask",len(hadron_id_net))

        #================All this is copied from the gen code proc_wbwb_debug================================
        #no_b_mask = ak.mask(data,(ak.num(data["bjets"].eta) > 0) & (ak.num(data["Jet"]) >= 2))

        #default_array = np.zeros((len(data["bjets"]), 2)) #2 since we could have subleading jets too

        #data_gen_meson = ak.where(ak.is_none(no_b_mask), default_array, data["gen_meson_real"])
        #data_meson_tag = ak.where(ak.is_none(no_b_mask), default_array, data["meson_tag_real"])

        #Using this instead of the above (this was also used for reshaping the arrays for the max number of constituents of the jets)
        data_gen_meson = ak.pad_none(data["gen_meson_real"], 2, clip=True)
        data_gen_meson = ak.fill_none(data_gen_meson, 0)
        data_meson_tag = ak.pad_none(data["meson_tag_real"], 2, clip=True)
        data_meson_tag = ak.fill_none(data_meson_tag, 0)

        #=====================================================================================================

        for gen_i in range(len(meson_cat)):
            string_mask = "id_" + meson_cat[gen_i]
            mask_value = (data_gen_meson[:,0] == hadron_id[gen_i])
            cats[string_mask] = ak.where(ak.is_none(leading_cut), cut_array, mask_value)

        for gen_i in range(len(meson_cat)):
            for rec_i in range(len(meson_cat)):
                string_mask = "id_" + meson_cat[gen_i] + "_rec_" + meson_cat[rec_i]
                mask_value = (data_gen_meson[:,0] == hadron_id[gen_i]) & (data_meson_tag[:,0] == rec_i)
                #if string_mask == "id_bneg_rec_bneg":
                    #print(data_gen_meson[:,0])
                    #print(data_meson_tag[:,0])
                    #print(mask_value)
                    #print(len(mask_value))
                cats[string_mask] = ak.where(ak.is_none(leading_cut), cut_array, mask_value)
                #print("Name of mask", string_mask)
        '''
        print(hadron_id_net)
        print(ak.to_numpy(data_meson_tag[:,0]))
        print(ak.to_numpy(data_meson_tag[:,1]))
        #Leading jet meson tag matches gen meson id, also includes sl proper tags
        print(hadron_id_net.shape[0])
        '''

        rec_meson_id_l = hadron_id_net[range(len(ak.to_numpy(data_meson_tag[:,0]))),ak.to_numpy(data_meson_tag[:,0])]
        #rec_meson_id_l = ak.from_numpy(np.choose(ak.to_numpy(data_meson_tag[:,0]), hadron_id_net))
        rec_meson_id_sl = hadron_id_net[range(len(ak.to_numpy(data_meson_tag[:,1]))),ak.to_numpy(data_meson_tag[:,1])]
        '''
        print("Ravelled mask value for correct tag of leading b jet")
        print(data_gen_meson[:,0])
        print(rec_meson_id_l)

        print("Ravelled mask value for correct tag of subleading b jet")
        print(data_gen_meson[:,1])
        print(rec_meson_id_sl)
        '''

        l_correct_tag = (data_gen_meson[:,0] == rec_meson_id_l)
        sl_correct_tag = (data_gen_meson[:,1] == rec_meson_id_sl)

        cats["l_proper_tag"] = ak.where(ak.is_none(leading_cut), cut_array, l_correct_tag)
        cats["l_sl_proper_tag"] = ak.where(ak.is_none(subleading_cut), cut_array, (l_correct_tag) & (sl_correct_tag))
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
        mlb = [0]*len(data[l_col])
        have_bjets = ak.where(ak.num(data[b_col].eta)>0)
        #have_lepton = ak.where(ak.num(data[b_col].eta)>0) #This is required especially for data, because the exact one lepton cut is applied for gen only
        #print("Events having bjets",have_bjets)
        for i in range(len(have_bjets)):
            for j in range(len(have_bjets[i])): #Weird syntax, i is also an array somehow, so have to do this
                #print("Entry number",have_bjets[i][j])
                #print("Lepton object",data[l_col][have_bjets[i][j], 0])
                mlb[have_bjets[i][j]] = (data[l_col][have_bjets[i][j], 0] + data[b_col][have_bjets[i][j], 0]).mass
                #print("Invariant mass excluding MET",mlb[have_bjets[i][j]])
        return mlb

    def dR_lepton_bjet(self, l_col, b_col, data):
        """Return dR between lepton and leading b jet."""
        dr = [0]*len(data[l_col])
        have_bjets = ak.where(ak.num(data[b_col].eta)>0)
        #have_lepton = ak.where(ak.num(data[b_col].eta)>0) #This is required especially for data, because the exact one lepton cut is applied for gen only
        #print("Events having bjets",have_bjets)
        for i in range(len(have_bjets)):
            for j in range(len(have_bjets[i])): #Weird syntax, i is also an array somehow, so have to do this
                #print("Entry number",have_bjets[i][j])
                #print("Lepton object",data[l_col][have_bjets[i][j], 0])
                dr[have_bjets[i][j]] = (data[l_col][have_bjets[i][j], 0] + data[b_col][have_bjets[i][j], 0]).delta_r
                dr[have_bjets[i][j]] = data[b_col][have_bjets[i][j], 0].delta_r(data[l_col][have_bjets[i][j], 0])
                #print("Invariant mass excluding MET",dr[have_bjets[i][j]])
        return dr

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
