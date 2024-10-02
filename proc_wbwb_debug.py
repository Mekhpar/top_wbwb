import pdb
import numpy as np
import pepper
import awkward as ak
from functools import partial
from copy import copy
import logging

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
        #print("Start of selection")
        #print(selector.data)
        #print(type(selector.data))
        #print(selector.data.columns)
        era = self.get_era(selector.data, is_mc)
        # lumi mask
        if not is_mc:
            selector.add_cut("Lumi", partial(
                self.good_lumimask, is_mc, dsname))
        if is_mc:
            selector.add_cut(
                "CrossSection", partial(self.crosssection_scale, dsname))

        if is_mc and "pileup_reweighting" in self.config:
            selector.add_cut("Reweighting", partial(self.do_pileup_reweighting, dsname))

        # Only allow events that pass triggers specified in config
        # This also takes into account a trigger order to avoid triggering
        # the same event if it's in two different data datasets.
        pos_triggers, neg_triggers = pepper.misc.get_trigger_paths_for(
            dsname, is_mc, self.config["dataset_trigger_map"],
            self.config["dataset_trigger_order"])

        #Not needed for gen
        if not is_mc:
            selector.add_cut("Trigger", partial(self.passing_trigger, pos_triggers, neg_triggers))

        if is_mc:
            selector.data["GenElectron"] = selector.data["GenPart"][abs(selector.data["GenPart"].pdgId) == 11]
            selector.data["GenMuon"] = selector.data["GenPart"][abs(selector.data["GenPart"].pdgId) == 13]

        # lepton columns
        #Technically these are not event cuts (that is probably why they are not in the add_cut function) but lepton cuts
        selector.set_multiple_columns(self.pick_electrons)
        selector.set_multiple_columns(self.pick_muons)

        if is_mc:
            selector.set_multiple_columns(self.pick_gen_electrons)
            selector.set_multiple_columns(self.pick_gen_muons)

        #No reason the correction cannot be applied to gen and rec muons at the same time (columns are different, not done eventwise)
        #Since there is a need to look at the muon wise scale factor value as well, the function apply_rochester_corr and apply_rochester_corr_gen is also going to
        #return that, so set_column is not being used

        #selector.set_column("Muon",partial(self.apply_rochester_corr,selector.rng,is_mc))
        #selector.set_column("GenMuon",partial(self.apply_rochester_corr_gen,selector.rng,is_mc)) #Very likely has to be applied for gen muons too because it is MC

        #print("Event Muon size after muon iso,id,pt cuts",len(selector.data["Muon"]))
        #mcSF1,mcSF2,mcSF,smear_gen = self.apply_rochester_corr(selector.rng, is_mc, selector.data) #Only for debugging, was trying to figure out how the mcsf is populated from smearing and spread sf
        #mcSF2,mcSF = self.apply_rochester_corr_gen(selector.rng, is_mc, selector.data)

        #selector.data["Muon"],mcSF_rec = self.apply_rochester_corr(selector.rng, is_mc, selector.data)

        #THIS SHOULD NEVER BE USED - REC AND DETECTOR LEVEL CORRECTION NOT USED FOR GEN MUONS!!
        #selector.data["GenMuon"],mcSF_gen = self.apply_rochester_corr_gen(selector.rng, is_mc, selector.data)

        '''
        for i in range(len(selector.data["Muon"])):
            print()
            print("Muon rec pt",selector.data["Muon"][i].pt)
            print("Matching gen pt",selector.data["Muon"][i].matched_gen.pt)
            print("Muon gen pt",selector.data["GenMuon"][i].pt)

            print("Muon rec eta",selector.data["Muon"][i].eta)
            print("Matching gen eta",selector.data["Muon"][i].matched_gen.eta)
            print("Muon gen eta",selector.data["GenMuon"][i].eta)

            print("Rec muon pdg id",selector.data["Muon"][i]["pdgId"])
            #print("Actual b flavor or not",selector.data["GenMuon"][i]["hadronFlavour"])
            print("Veto rec muon pt",selector.data["VetoMuon"][i].pt)
            print("Veto rec muon eta",selector.data["VetoMuon"][i].eta)

            print("Veto gen muon pt",selector.data["VetoGenMuon"][i].pt)
            print("Veto gen muon eta",selector.data["VetoGenMuon"][i].eta)

            print("Rec scale factor",mcSF_rec[i])

            #print("Rec Muon charge",selector.data["Muon"][i]["charge"])
            #print("Rec Muon charge from pdg id",-selector.data["Muon"][i]["pdgId"]/abs(selector.data["Muon"][i]["pdgId"]))
        '''

        selector.set_column("Lepton", partial(self.build_lepton_column, is_mc, selector.rng,""))
        
        if is_mc:
            selector.set_column("GenLepton", partial(self.build_lepton_column, is_mc, selector.rng,"Gen"))
            #selector.add_cut("HasGenLepton", partial(self.require_lepton_nums_gen, is_mc),no_callback=True)
            selector.add_cut("HasLepton", partial(self.require_lepton_nums_gen, is_mc),no_callback=True)

        #Do NOT use both simultaneously unless explicitly required
        #Not sure whether this is required for data, the argument is_mc was already there so probably not
        elif not is_mc:
            selector.add_cut("HasLepton", partial(self.require_lepton_nums, is_mc),no_callback=True)
        

        #has_lep_rec, rec_sys = self.require_lepton_nums(is_mc,selector.data)
        #has_lep_gen = self.require_lepton_nums_gen(is_mc,selector.data) #If this is after the cut, of course it will always be 1

        #print()
        # Require 1 muon, no veto muons
        #No scale factor computation here (not as long as this is not combined with HasGenLepton)
        #selector.add_cut("OneGoodLep", partial(self.one_good_lepton,""))
        if is_mc:
            selector.add_cut("OneGoodLep", partial(self.one_good_lepton,"Gen"))

        elif not is_mc: #Now this one might actually be for both MC and data - selecting exactly one lepton
            selector.add_cut("OneGoodLep", partial(self.one_good_lepton,""))

        one_lep_rec = self.one_good_lepton("",selector.data)
        if is_mc:
            one_lep_gen = self.one_good_lepton("Gen",selector.data) #If this is after the cut, of course it will always be 1
            selector.add_cut("NoVetoLeps", partial(self.no_veto_muon,"Gen"))

        elif not is_mc:
            selector.add_cut("NoVetoLeps", partial(self.no_veto_muon,""))    

        no_veto_rec = self.no_veto_muon("",selector.data)
        if is_mc:
            no_veto_gen = self.no_veto_muon("Gen",selector.data) #If this is after the cut, of course it will always be 1

        '''
        selector.set_cat("channel", {"mu", "el" }) #This does not seem specific to rec or gen
        print("Rec categories")
        print(selector.cats["channel"])
        selector.set_multiple_columns(partial(self.channel_masks,"")) #continuation of the above line
        '''

        #Again another one for gen
    
        selector.set_cat("channel", {"mu", "el" }) #This does not seem specific to rec or gen
        #print("Gen categories")
        #print(selector.cats["channel"])
        #selector.set_multiple_columns(partial(self.channel_masks,"Gen")) #continuation of the above line
        selector.set_multiple_columns(partial(self.channel_masks,is_mc)) #continuation of the above line

        selector.set_column("mTW", partial(self.w_mT,""))
        if is_mc:
            selector.set_column("GenmTW", partial(self.w_mT,"Gen"))

        '''
        for i in range(len(selector.data["Lepton"])):
            print()
            print("Lepton rec pt",selector.data["Lepton"][i].pt)
            print("Lepton gen pt",selector.data["GenLepton"][i].pt)

            print("Lepton rec eta",selector.data["Lepton"][i].eta)
            print("Lepton gen eta",selector.data["GenLepton"][i].eta)

            print("Rec Lepton pdg id",selector.data["Lepton"][i]["pdgId"])
            print("Gen Lepton pdg id",selector.data["GenLepton"][i]["pdgId"])

            print("Veto rec muon pt",selector.data["VetoMuon"][i].pt)
            print("Veto rec muon eta",selector.data["VetoMuon"][i].eta)

            print("Veto gen muon pt",selector.data["VetoGenMuon"][i].pt)
            print("Veto gen muon eta",selector.data["VetoGenMuon"][i].eta)

            print("Rec MET pt",selector.data["MET"][i].pt)
            print("Gen MET pt",selector.data["GenMET"][i].pt) #This is directly available in the event content

            #Not available (not for GenMET at least)
            #print("Rec MET eta",selector.data["MET"][i].eta)
            #print("Gen MET eta",selector.data["GenMET"][i].eta) #This is directly available in the event content

            #Obviously this will not be "correct" since the first of the datasets (usually, for which the spew is compared) is for ZZ process
            print("Rec transverse mass of W",selector.data["mTW"][i])
            print("Gen transverse mass of W",selector.data["GenmTW"][i])
        '''

        #Removing this since it is not required for gen
        #if "trigger_sf_sl" in self.config and is_mc:
        #    selector.add_cut("trigSF", self.do_trigger_sf_sl)

        # # gen jet info
        # if is_mc: 
        #     selector.set_column("jetorigin", self.build_jet_geninfo)
        # selector.set_column("OrigJet", selector.data["Jet"])

        #JEC variations 
        '''
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
        '''    

        #nominal JEC
        #'''
        #self.process_selection_jet_part(selector, is_mc, self.get_jetmet_nominal_arg(),dsname, filler, era)
        #'''
        #Gen jet selection cuts
        self.process_selection_jet_part(selector, is_mc, dsname, filler, era)
        logger.debug("Selection done")


        # apply btag sf, define categories



        # Current end of selection steps. After this, define relevant methods

    def process_selection_jet_part(self, selector, is_mc, 
                                   #variation,
                                   dsname,
                                   filler, era):
        """Part of the selection that needs to be repeated for
        every systematic variation done for the jet energy correction,
        resultion and for MET"""

        #Again, not needed for gen objects
        #reapply_jec = ("reapply_jec" in self.config and self.config["reapply_jec"])
        #selector.set_multiple_columns(partial(self.compute_jet_factors, is_mc, reapply_jec, variation.junc,variation.jer, selector.rng))

        selector.set_column("OrigJet", selector.data["Jet"])
        selector.set_column("Jet", partial(self.build_jet_column, is_mc))

        if is_mc:
            selector.set_column("OrigGenJet", selector.data["GenJet"])
            selector.set_column("GenJet", partial(self.build_gen_jet_column, is_mc))

        #At the moment this is put before the pt cut since it is confusing as what jets (from the jet id) actually pass the cut and which ones are just missing
        '''
        print("Event jet size?",len(selector.data["Jet"]))
        print("Event muon size?",len(selector.data["Muon"]))
        print("Event gen particles size?",len(selector.data["GenPart"]))
        '''
        #for i in range(len(selector.data["GenPart"])):
        #    print("Pdg ids for each event",selector.data["GenPart"][i].pdgId)

        #Not required since gen            
        '''
        if "jet_puid_sf" in self.config and is_mc:
            selector.add_cut("JetPUIdSFs", self.jet_puid_sfs)
        '''

        selector.set_column("Jet", self.jets_with_puid) #Probably ok to set this column

        #Again ok to set this column as it is not a part of gen, but variation is not calculated so skip it
        #smear_met = "smear_met" in self.config and self.config["smear_met"]
        #selector.set_column("MET", partial(self.build_met_column, is_mc, variation.junc, variation.jer if smear_met else None, selector.rng, era, variation=variation.met))

        
        #print("Event jet size after cut",len(selector.data["Jet"]))

        #Do not need this, was used for debugging only
        '''
        #selector.data["GenJet"]["btagged"] = selector.data["GenJet"][selector.data["GenJet"].hadronFlavour == 5]
        gen_jet_flat = ak.flatten(selector.data["GenJet"])
        count_events = ak.num(selector.data["GenJet"]) #Think this is the number of events
        print("Flattened array (might help for gen hadron flavor 'tagging')",gen_jet_flat.hadronFlavour)

        flat_btag = gen_jet_flat.hadronFlavour == 5 #only add the flag, not the whole cut on the jets
        print("Flattened hadron flavor array (gen btag)",flat_btag)
        print("UnFlattened array (might help for gen hadron flavor 'tagging')",ak.unflatten(gen_jet_flat,count_events).hadronFlavour)
        #print("Unflattened btag array",ak.unflatten(flat_btag,count_events))
        selector.data["GenJet"]["btagged"] = ak.unflatten(flat_btag,count_events)
        '''
        

        #Again this is something that is scaled wrt btag eff so this is not required for gen (bjets are known)
        #if is_mc and self.config["compute_systematics"]:
        #    self.scale_systematics_for_btag(selector, variation, dsname)

        selector.set_column("nbtag", partial(self.num_btags,""))
        selector.set_column("bjets", partial(self.build_btag_column,is_mc))
        #selector.set_cat("jet_btag", {"j2_b0", "j2_b1","j2_b2","j2+_b0", "j2+_b1","j2+_b2"})
        #selector.set_multiple_columns(self.btag_categories)


        if not is_mc:
            selector.set_column("mlb",partial(self.mass_lepton_bjet,"Lepton","bjets"))    
            selector.set_column("dR_lb",partial(self.dR_lepton_bjet,"Lepton","bjets"))    

        if is_mc:
            selector.set_column("nbtag_gen", partial(self.num_btags,"Gen"))
            selector.set_column("bjets_gen", partial(self.build_gen_btag_column,is_mc))

            #print("Calling invariant mass function")
            #mlb_debug = self.mass_lepton_bjet("GenLepton","bjets_gen",selector.data)

            selector.set_column("mlb_gen",partial(self.mass_lepton_bjet,"GenLepton","bjets_gen"))    
            selector.set_column("dR_lb_gen",partial(self.dR_lepton_bjet,"GenLepton","bjets_gen"))    

        selector.set_cat("gen_jet_btag", {"j2_b0", "j2_b1","j2_b2","j2+_b0", "j2+_b1","j2+_b2"})
        selector.set_multiple_columns(partial(self.btag_gen_categories,is_mc))



        #selector.set_cat("gen_bjet_eta", {"l_be","l_fwd","l_be_sl_be", "l_be_sl_fwd", "l_fwd_sl_be", "l_fwd_sl_fwd"})
        selector.set_cat("gen_bjet_eta", {"no_bjet","l_be","l_fwd","sl_be","sl_fwd","l_be_sl_be","l_fwd_sl_be","l_be_sl_fwd","l_fwd_sl_fwd"})

        #Removed for debugging purposes
        print("Calling eta categories function")
        #eta_categories = self.eta_gen_cat("bjets_gen","GenJet",selector.data)
        if is_mc:
            selector.set_multiple_columns(partial(self.eta_gen_cat,"bjets_gen","GenJet"))
        if not is_mc:
            selector.set_multiple_columns(partial(self.eta_gen_cat,"bjets","Jet"))

        #print("Type of rec masks", type(selector.data["j2_b0"]))
        #print("Type of eta masks", type(selector.data["l_be"]))

        #This was originally not put here, but the btag sf was removed so this is probably needed to get the num jet and num bjets= categories for plotting
        #'''
        if is_mc:
            selector.add_cut("JetPtReq", partial(self.jet_pt_requirement,"GenJet")) #Reminder that this is not a pt cut, this is an event cut selecting the number of jets with a pt cut
        
        #Removal for debugging
        elif not is_mc:
            selector.add_cut("JetPtReq", partial(self.jet_pt_requirement,"Jet"))
        #'''

        rec_jet_mask = self.jet_pt_requirement("Jet",selector.data)
        if is_mc:
            gen_jet_mask = self.jet_pt_requirement("GenJet",selector.data) #This is applied for gen jets independent of rec jets, not the matched ones

        if not is_mc:
            selector.set_multiple_columns(self.build_data_gen_column)
            
        #'''
        for i in range(len(selector.data["Jet"])):
            if is_mc:
                print()
                print("MC Gen set")
                

                #This is only to compare with previous spew files (of course the event number could also have been printed)
                print("Lepton rec pt",selector.data["Lepton"][i].pt)
                print("Rec transverse mass of W",selector.data["mTW"][i])

                print("Jet rec pt",selector.data["Jet"][i].pt)
                print("Jet rec eta",selector.data["Jet"][i].eta)

                print("Btag flag",selector.data["Jet"][i]["btagged"])
                print("Number of btagged rec jets",selector.data["nbtag"][i])
                #If rec_jet_mask and gen_jet_mask are calculated before applying the actual cut (which they will be) this indexing will be wrong so it is better to get rid of it
                
                print("B gen jet pt",selector.data["bjets_gen"][i].pt) 
                print("B gen jet eta",selector.data["bjets_gen"][i].eta) 

                print("Lepton gen pt",selector.data["GenLepton"][i].pt)
                print("Gen transverse mass of W",selector.data["GenmTW"][i])
                print("Matching gen pt",selector.data["Jet"][i].matched_gen.pt)
                #print("Muon gen pt?",selector.data["GenPart"][abs(selector.data["GenPart"].pdgId) == 13][i].pt)
                print("Jet gen pt",selector.data["GenJet"][i].pt)

                print("Matching gen eta",selector.data["Jet"][i].matched_gen.eta)
                print("Jet gen eta",selector.data["GenJet"][i].eta)

                print("Actual b flavor or not",selector.data["GenJet"][i]["hadronFlavour"]) #Of course this is obtained from hadron flavor, not a discriminator
                print("Actual b flag for gen jets",selector.data["GenJet"][i]["btagged"]) #Of course this is obtained from hadron flavor, not a discriminator

                print("Number of btagged gen jets",selector.data["nbtag_gen"][i])
                print("Invariant gen lepton and b jet mass (excluding MET)", selector.data["mlb_gen"][i])
                
                print("dr between lepton and b jet",selector.data["dR_lb_gen"][i])
                #'''
                print("Category masks for gen")
                print(selector.data["j2_b0"][i])
                print(selector.data["j2_b1"][i])
                print(selector.data["j2_b2"][i])
                print(selector.data["j2+_b0"][i])
                print(selector.data["j2+_b1"][i])
                print(selector.data["j2+_b2"][i])

                print("Eta masks for bjets")
                print("No b jets")
                print(selector.data["no_bjet"][i])
                print("Leading")
                print(selector.data["l_be"][i])
                print(selector.data["l_fwd"][i])
                print("Subleading")
                print(selector.data["sl_be"][i])
                print(selector.data["sl_fwd"][i])
                print("Combo masks")
                print(selector.data["l_be_sl_be"][i])
                print(selector.data["l_fwd_sl_be"][i])
                print(selector.data["l_be_sl_fwd"][i])
                print(selector.data["l_fwd_sl_fwd"][i])

                #print("Rec jet mask",rec_jet_mask[i])
                #print("Gen jet mask",gen_jet_mask[i])
                #'''
            
            '''
            elif not is_mc:
                print()
                print("Data 'Gen' set")

                #This is only to compare with previous spew files (of course the event number could also have been printed)
                print("Lepton rec pt",selector.data["Lepton"][i].pt)
                print("Rec transverse mass of W",selector.data["mTW"][i])

                print("Jet rec pt",selector.data["Jet"][i].pt)
                print("Jet rec eta",selector.data["Jet"][i].eta)

                print("Btag flag",selector.data["Jet"][i]["btagged"])
                print("Number of btagged rec jets",selector.data["nbtag"][i])
                #If rec_jet_mask and gen_jet_mask are calculated before applying the actual cut (which they will be) this indexing will be wrong so it is better to get rid of it
                
                print("B gen jet pt",selector.data["bjets_gen"][i].pt) 
                print("B gen jet eta",selector.data["bjets_gen"][i].eta) 

                print("Lepton gen pt",selector.data["GenLepton"][i].pt)
                print("Gen transverse mass of W",selector.data["GenmTW"][i])
                #print("Matching gen pt",selector.data["Jet"][i].matched_gen.pt)
                #print("Muon gen pt?",selector.data["GenPart"][abs(selector.data["GenPart"].pdgId) == 13][i].pt)
                print("Jet gen pt",selector.data["GenJet"][i].pt)

                #print("Matching gen eta",selector.data["Jet"][i].matched_gen.eta)
                print("Jet gen eta",selector.data["GenJet"][i].eta)

                #print("Actual b flavor or not",selector.data["GenJet"][i]["hadronFlavour"]) #Of course this is obtained from hadron flavor, not a discriminator
                print("Actual b flag for gen jets",selector.data["GenJet"][i]["btagged"]) #Of course this is obtained from hadron flavor, not a discriminator

                print("Number of btagged gen jets",selector.data["nbtag_gen"][i])

                print("Category masks for gen")
                print(selector.data["j2_b0"][i])
                print(selector.data["j2_b1"][i])
                print(selector.data["j2_b2"][i])
                print(selector.data["j2+_b0"][i])
                print(selector.data["j2+_b1"][i])
                print(selector.data["j2+_b2"][i])

                print("Eta masks for bjets")
                print("No b jets")
                print(selector.data["no_bjet"][i])

                print("Leading")
                print(selector.data["l_be"][i])
                print(selector.data["l_fwd"][i])
                print("Subleading")
                print(selector.data["sl_be"][i])
                print(selector.data["sl_fwd"][i])
                print("Combo masks")
                print(selector.data["l_be_sl_be"][i])
                print(selector.data["l_fwd_sl_be"][i])
                print(selector.data["l_be_sl_fwd"][i])
                print(selector.data["l_fwd_sl_fwd"][i])
                '''
        #'''
        #if "btag_sf" in self.config and len(self.config["btag_sf"]) != 0:
        #    selector.add_cut("btagSF", partial(self.btag_sf, is_mc))

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

    def require_lepton_nums_gen(self, is_mc, data):
        """Select events that contain at least one lepton."""
        accept = np.asarray(ak.num(data["GenMuon"]) >= 1) #At least one lepton or at least one muon?
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

    def pick_gen_electrons(self, data):
        good_id, good_pt, good_iso = self.config[["good_ele_id", "good_ele_pt_min", "good_ele_iso"]]
        veto_id, veto_pt, veto_iso = self.config[["veto_ele_id", "veto_ele_pt_min", "veto_ele_iso"]]
        eta_min, eta_max = self.config[["ele_eta_min", "ele_eta_max"]]

        gen_ele = data["GenElectron"]

        # remove barrel-endcap transreg
        sc_eta_abs = abs(gen_ele.eta) #gen_ele.deltaEtaSC is removed (correction?) but only because it doesnt exist for gen electrons and I don't know how to put it in
        is_in_transreg = (1.444 < sc_eta_abs) & (sc_eta_abs < 1.566)
        etacuts = (eta_min < gen_ele.eta) & (eta_max > gen_ele.eta)

        # Electron ID, as an example we use the MVA one here
        #has_id = self.electron_id(self.config["good_ele_id"], gen_ele) #why is this even there if the good_id is used below?

        # Finally combine all the requirements
        # to-do add iso - IMPORTANT!! (not for this dataset)
        is_good = (
            #self.electron_id(good_id, gen_ele)
            #&
            (~is_in_transreg)
            & etacuts
            & (good_pt < gen_ele.pt))

        is_veto = (
            ~is_good
            #& self.electron_id(veto_id, gen_ele)
            & (~is_in_transreg)
            & etacuts
            & (veto_pt < gen_ele.pt))

        # Return all electrons with are deemed to be good
        return {"GenElectron": gen_ele[is_good], "VetoGenElectron": gen_ele[is_veto]}

    def pick_muons(self, data):

        good_id, good_pt, good_iso = self.config[["good_muon_id", "good_muon_pt_min", "good_muon_iso"]]
        veto_id, veto_pt, veto_iso = self.config[["veto_muon_id", "veto_muon_pt_min", "veto_muon_iso"]]
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

    def pick_gen_muons(self, data):

        good_id, good_pt, good_iso = self.config[["good_muon_id", "good_muon_pt_min", "good_muon_iso"]]
        veto_id, veto_pt, veto_iso = self.config[["veto_muon_id", "veto_muon_pt_min", "veto_muon_iso"]]
        eta_min, eta_max = self.config[["muon_eta_min", "muon_eta_max"]]

        gen_muon = data["GenMuon"]

        etacuts = (eta_min < gen_muon.eta) & (eta_max > gen_muon.eta)

        is_good = (
            #self.muon_id(good_id, gen_muon)
            #& self.muon_iso(good_iso, gen_muon)
            #& 
            etacuts
            & (good_pt < gen_muon.pt))
        
        is_veto = (
            ~is_good
            #& self.muon_id(veto_id, gen_muon)
            #& self.muon_iso(veto_iso, gen_muon)
            & etacuts
            & (veto_pt < gen_muon.pt))

        return {"GenMuon": gen_muon[is_good], "VetoGenMuon": gen_muon[is_veto]}

    def one_good_lepton(self, prefix, data):
        #print()
        #This will probably leave out missing reconstructed muons completely, although fakes might be a bit more obvious
        #print("Event size?",len(data["Muon"]))
        '''
        for i in range(len(data["Muon"])):
            print("Muon rec pt",data["Muon"][i].pt)
            print("Matching gen pt",data["Muon"][i].matched_gen.pt)

            #print("Matching gen muon id",data["Muon"].Jet_muonIdx1)
            #print("Matching gen muon id",data["jetorigin"][i]["jetIdx"])
            #print("Muon pt directly from nano aod files",data["Muon_pt"]) #Checking whether this is the same as any of the gen or rec, Muon_pt is the name of the branch
            print("Muon rec eta",data["Muon"][i].eta)
            print("Matching gen eta",data["Muon"][i].matched_gen.eta)
        '''
        #print("Muon eta directly from nano aod files",data["Muon_eta"]) #Checking whether this is the same as any of the gen or rec, Muon_eta is the name of the branch

        return ak.num(data[prefix+"Electron"]) + ak.num(data[prefix+"Muon"]) == 1 #Why not just use the data["Lepton"] column? Maybe because it is after the selections from pick_electrons and pick_muons
    
    #probably no veto lepton column made - should it be made?
    def no_veto_muon(self, prefix, data):
        return ak.num(data["Veto"+prefix+"Muon"]) + ak.num(data["Veto"+prefix+"Electron"]) == 0
    
    '''
    def channel_masks(self, prefix, data):
        channels = {}
        channels[prefix+"mu"] = (ak.num(data[prefix+"Muon"]) == 1)
        channels[prefix+"el"] = (ak.num(data[prefix+"Electron"]) == 1)
        # channels["fake_mu"] 


        return channels
    '''

    def channel_masks(self, is_mc, data):
        channels = {}
        if is_mc:
            channels["mu"] = (ak.num(data["GenMuon"]) == 1)
            channels["el"] = (ak.num(data["GenElectron"]) == 1)

        elif not is_mc:
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

    def btag_gen_categories(self,is_mc,data):
        
        cats = {}
        
        btagged = data["nbtag"]
        njet = ak.num(data["Jet"])

        if is_mc:
            btagged_gen = data["nbtag_gen"]
            njet_gen = ak.num(data["GenJet"])

            cats["j2_b0"] = (btagged_gen == 0) & (njet_gen == 2)
            cats["j2_b1"] = (btagged_gen == 1) & (njet_gen == 2)
            cats["j2_b2"] = (btagged_gen == 2) & (njet_gen == 2)
            cats["j2+_b0"] = (btagged_gen == 0) & (njet_gen >= 2)
            cats["j2+_b1"] = (btagged_gen == 1) & (njet_gen >= 2)
            cats["j2+_b2"] = (btagged_gen == 2) & (njet_gen >= 2)

            #print("Length of jet masks after populating for mc",len(cats["j2_b2"]),len(cats["j2+_b0"]))

        elif not is_mc:
            cats["j2_b0"] = (btagged == 0) & (njet == 2)
            cats["j2_b1"] = (btagged == 1) & (njet == 2)
            cats["j2_b2"] = (btagged == 2) & (njet == 2)
            cats["j2+_b0"] = (btagged == 0) & (njet >= 2)
            cats["j2+_b1"] = (btagged == 1) & (njet >= 2)
            cats["j2+_b2"] = (btagged == 2) & (njet >= 2)

        return cats

    #'''
    #Adding all 8 categories (possible combos of eta for the leading and subleading jets)
    def eta_gen_cat(self,b_col,jet_col,data):
        
        cats = {}

        default_l = np.zeros((len(data[b_col]), 1))
        default_sl = np.zeros((len(data[b_col]), 2))

        cut_array = [False] * len(data[b_col]) 
        '''
        cats["no_bjet"] = [False] * len(data[b_col]) 
        #Technically this was not required, but the j2_b0 and j2+_b0 events need somewhere to go and appear in the final sum histogram without being cut
        cats["l_be"] = [False] * len(data[b_col])
        cats["l_fwd"] = [False] * len(data[b_col])
        cats["sl_be"] = [False] * len(data[b_col])
        cats["sl_fwd"] = [False] * len(data[b_col])

        cats["l_be_sl_be"] = [False] * len(data[b_col])
        cats["l_fwd_sl_be"] = [False] * len(data[b_col])
        cats["l_be_sl_fwd"] = [False] * len(data[b_col])
        cats["l_fwd_sl_fwd"] = [False] * len(data[b_col])
        '''
        '''
        if is_mc:
            btagged = data["nbtag_gen"]
            njet = ak.num(data["GenJet"])

        elif not is_mc:
            btagged = data["nbtag"]
            njet = ak.num(data[b_col])          
        '''

        #btagged = ak.num(data[jet_col])
        #njet = ak.num(data[jet_col])


        #Not sure whether there is a need of ak.mask here or whether it means the same
        '''
        no_b_cut = (btagged == 0) & (njet >= 2)
        leading_cut = (btagged>=1) & (njet >= 2)
        subleading_cut = (btagged>=2) & (njet >= 2)
        '''
        #ak.mask(data,ak.num(data[b_col].eta)>0)
        #This seems to be able to return a none value as opposed to false
        #no_b_cut = ak.mask(data,(ak.num(data[b_col].eta) == 0) & (ak.num(data[jet_col]) >= 2))
        leading_cut = ak.mask(data,(ak.num(data[b_col].eta)>=1) & (ak.num(data[jet_col]) >= 2))
        subleading_cut = ak.mask(data,(ak.num(data[b_col].eta)>=2) & (ak.num(data[jet_col]) >= 2))

        fwd_cut = self.config["btag_jet_eta_max"]

        lead_bjet_eta = ak.where(ak.is_none(leading_cut), default_sl, data[b_col].eta)
        sublead_bjet_eta = ak.where(ak.is_none(subleading_cut), default_sl, data[b_col].eta)

        l_be = abs(lead_bjet_eta[:,0]) < fwd_cut
        l_fwd = abs(lead_bjet_eta[:,0]) >= fwd_cut

        sl_be = abs(sublead_bjet_eta[:,1]) < fwd_cut
        sl_fwd = abs(sublead_bjet_eta[:,1]) >= fwd_cut

        l_be_sl_be = (abs(sublead_bjet_eta[:,0]) < fwd_cut) & (abs(sublead_bjet_eta[:,1]) < fwd_cut)
        l_fwd_sl_be = (abs(sublead_bjet_eta[:,0]) >= fwd_cut) & (abs(sublead_bjet_eta[:,1]) < fwd_cut)
        l_be_sl_fwd = (abs(sublead_bjet_eta[:,0]) < fwd_cut) & (abs(sublead_bjet_eta[:,1]) >= fwd_cut)
        l_fwd_sl_fwd = (abs(sublead_bjet_eta[:,0]) >= fwd_cut) & (abs(sublead_bjet_eta[:,1]) >= fwd_cut)

        cats["no_bjet"] = (ak.num(data[b_col].eta) == 0) & (ak.num(data[jet_col]) >= 2)

        cats["l_be"] = ak.where(ak.is_none(leading_cut), cut_array, l_be)
        cats["l_fwd"] = ak.where(ak.is_none(leading_cut), cut_array, l_fwd)
        cats["sl_be"] = ak.where(ak.is_none(subleading_cut), cut_array, sl_be)
        cats["sl_fwd"] = ak.where(ak.is_none(subleading_cut), cut_array, sl_fwd)

        cats["l_be_sl_be"] = ak.where(ak.is_none(subleading_cut), cut_array, l_be_sl_be)
        cats["l_fwd_sl_be"] = ak.where(ak.is_none(subleading_cut), cut_array, l_fwd_sl_be)
        cats["l_be_sl_fwd"] = ak.where(ak.is_none(subleading_cut), cut_array, l_be_sl_fwd)
        cats["l_fwd_sl_fwd"] = ak.where(ak.is_none(subleading_cut), cut_array, l_fwd_sl_fwd)

        '''
        for i in range(len(data[b_col])):
            print()
            print("MC Gen set")
            print("Event number",i)            

            #This is only to compare with previous spew files (of course the event number could also have been printed)
            print("B gen jet pt",data[b_col][i].pt) 
            #print("Dummy B gen jet pt",data_bjet_pt[i]) 

            print("B gen jet eta",data[b_col][i].eta) 
            #print("Dummy B gen jet eta",data_bjet_eta[i]) 
            print("No b jet cut value",cats["no_bjet"][i])
            print("Leading b jet cut value", leading_cut[i])
            print("Subleading b jet cut value",subleading_cut[i])

            print("Dummy values of masks")
            print("l_be",l_be[i])
            print("l_fwd",l_fwd[i])
            print("sl_be",sl_be[i])
            print("sl_fwd",sl_fwd[i])
            print("l_be_sl_be",l_be_sl_be[i])
            print("l_be_sl_fwd",l_be_sl_fwd[i])
            print("l_fwd_sl_be",l_fwd_sl_be[i])
            print("l_fwd_sl_fwd",l_fwd_sl_fwd[i])

            print("Actual values of masks")
            print("No b jets",cats["no_bjet"][i])
            print("l_be",cats["l_be"][i])
            print("l_fwd",cats["l_fwd"][i])
            print("sl_be",cats["sl_be"][i])
            print("sl_fwd",cats["sl_fwd"][i])
            print("l_be_sl_be",cats["l_be_sl_be"][i])
            print("l_be_sl_fwd",cats["l_be_sl_fwd"][i])
            print("l_fwd_sl_be",cats["l_fwd_sl_be"][i])
            print("l_fwd_sl_fwd",cats["l_fwd_sl_fwd"][i])
        '''

        '''
        for i in range(len(data[b_col])):
            #print("Entry number",i,"Leading and subleading Cut values",leading_cut[i],subleading_cut[i])
            cats["no_bjet"][i] = no_b_cut[i]

            if subleading_cut[i]:
                if is_mc:
                    abs_eta_l = abs(data[b_col][i][0].eta)
                    abs_eta_sl = abs(data[b_col][i][1].eta)

                elif not is_mc:
                    abs_eta_l = abs(data["bjets"][i][0].eta)
                    abs_eta_sl = abs(data["bjets"][i][1].eta)

                #print("Jet etas",abs_eta_l,abs_eta_sl)
                #print("Entry number", i, ": two or more b jets",len(data[b_col][i].eta) > 1)
                #eta_leading = data[b_col][i][2].eta
                cats["sl_be"][i] = abs_eta_sl < fwd_cut
                cats["sl_fwd"][i] = abs_eta_sl >= fwd_cut

                cats["l_be_sl_be"][i] = (abs_eta_l < fwd_cut) & (abs_eta_sl < fwd_cut)
                cats["l_fwd_sl_be"][i] = (abs_eta_l >= fwd_cut) & (abs_eta_sl < fwd_cut)
                cats["l_be_sl_fwd"][i] = (abs_eta_l < fwd_cut) & (abs_eta_sl >= fwd_cut)
                cats["l_fwd_sl_fwd"][i] = (abs_eta_l >= fwd_cut) & (abs_eta_sl >= fwd_cut)

            if leading_cut[i]:
                if is_mc:
                    abs_eta_l = abs(data[b_col][i][0].eta)

                elif not is_mc:
                    abs_eta_l = abs(data["bjets"][i][0].eta)

                cats["l_be"][i] = abs_eta_l < fwd_cut
                cats["l_fwd"][i] = abs_eta_l >= fwd_cut
            
        '''
        #cats["l_fwd"] = ak.Array(cats["l_fwd"])
    
        return cats
    #'''

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
        #print("Size of events satisfying the mask (having bjets)",len(dr_mask))

        #of course lengths of l_col and b_col are supposed to be the same, that is why they are used interchangeably
        default_array = np.zeros((len(data[b_col]), 2))
        empty_array = [None]*len(data[l_col])
        #Since this is being calculated after the at least one lepton cut is put, there is no need to ensure whether the lepton exists or not

        data_bjet_pt = ak.where(ak.is_none(dr_mask), default_array, data[b_col].pt)
        data_bjet_mass = ak.where(ak.is_none(dr_mask), default_array, data[b_col].mass) #What exactly is this?
        data_bjet_eta = ak.where(ak.is_none(dr_mask), default_array, data[b_col].eta)
        data_bjet_phi = ak.where(ak.is_none(dr_mask), default_array, data[b_col].phi)

        dr_dummy = np.sqrt(pow(data[l_col].eta[:,0]-data_bjet_eta[:,0],2) + pow(data[l_col].phi[:,0]-data_bjet_phi[:,0],2))
        dr_final = ak.where(ak.is_none(dr_mask), empty_array, dr_dummy)
        '''
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

        '''        
        return dr_final

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

    '''
    def invar_mass(self,cols,data)
        cosh = np.cosh(leps.eta[:,0]-leps.eta[:,1])
        cos = np.cos(leps.phi[:,0]-leps.phi[:,1])
        #
        mll = np.sqrt(2*leps.pt[:,0]*leps.pt[:,1]*(cosh - cos))
    '''

    def w_mT(self, prefix, data):
        """Get the pt of the four vector difference of the MET and the
        neutrinos"""
        met = data[prefix+"MET"]
        lep = data[prefix+"Lepton"]
        mt = np.sqrt(2*(met.pt*lep.pt-met.x*lep.x-met.y*lep.y))
        return mt
