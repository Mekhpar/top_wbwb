#import tensorflow as tf
import math

featureDict = {
    "global": {
        "branches": [
            "global_pt",
            "global_eta",
            "global_phi",
            "global_mass",
            "global_energy",

            "global_area",

            "global_beta",
            "global_dR2Mean",
            "global_frac01",
            "global_frac02",
            "global_frac03",
            "global_frac04",

            "global_jetR",
            "global_jetRchg",

            "global_n60",
            "global_n90",

            "global_chargedEmEnergyFraction",
            "global_chargedHadronEnergyFraction",
            "global_chargedMuEnergyFraction",
            "global_electronEnergyFraction",

            "global_tau1",
            "global_tau2",
            "global_tau3",

            "global_relMassDropMassAK",
            "global_relMassDropMassCA",
            "global_relSoftDropMassAK",
            "global_relSoftDropMassCA",

            "global_thrust",
            "global_sphericity",
            "global_circularity",
            "global_isotropy",
            "global_eventShapeC",
            "global_eventShapeD",

            "global_numberCpf",
            "global_numberNpf",
            "global_numberSv",
            "global_numberMuon",
            "global_numberElectron",

            "csv_trackSumJetEtRatio",
            "csv_trackSumJetDeltaR",
            "csv_vertexCategory",
            "csv_trackSip2dValAboveCharm",
            "csv_trackSip2dSigAboveCharm",
            "csv_trackSip3dValAboveCharm",
            "csv_trackSip3dSigAboveCharm",
            "csv_jetNTracksEtaRel",
            "csv_jetNSelectedTracks",
        ],
    },

    
    "cpf_charge": {
        "branches": [
            "cpf_charge",
        ],
        "length":"length_cpf",
        "offset":"length_cpf_offset",
        "max": 25
    },
    
    "cpf": {
        "branches": [

            "cpf_ptrel",
            "cpf_deta",
            "cpf_dphi",
            "cpf_deltaR",
            
            "cpf_trackEtaRel",
            "cpf_trackPtRel",
            "cpf_trackPPar",
            "cpf_trackDeltaR",
            "cpf_trackPParRatio",
            "cpf_trackPtRatio",
            "cpf_trackSip2dVal",
            "cpf_trackSip2dSig",
            "cpf_trackSip3dVal",
            "cpf_trackSip3dSig",
            "cpf_trackJetDistVal",
            "cpf_trackJetDistSig",
            "cpf_drminsv",
            "cpf_vertex_association",
            "cpf_fromPV",
            "cpf_puppi_weight",
            "cpf_track_chi2",
            "cpf_track_quality",
            "cpf_track_numberOfValidPixelHits",
            "cpf_track_pixelLayersWithMeasurement",
            "cpf_track_numberOfValidStripHits",
            "cpf_track_stripLayersWithMeasurement",
            "cpf_relmassdrop",

            "cpf_trackSip2dValSV",
            "cpf_trackSip2dSigSV",
            "cpf_trackSip3dValSV",
            "cpf_trackSip3dSigSV",

            "cpf_matchedMuon",
            "cpf_matchedElectron",
            "cpf_matchedSV",
            "cpf_track_ndof",

            "cpf_dZmin"
        ],
        "length":"length_cpf",
        "offset":"length_cpf_offset",
        "max": 25
    },

    "npf": {
        "branches": [
            "npf_ptrel",
            "npf_deta",
            "npf_dphi",
            "npf_deltaR",
            "npf_isGamma",
            "npf_hcal_fraction",
            "npf_drminsv",
            "npf_puppi_weight",
            "npf_relmassdrop",
        ],
        "length":"length_npf",
        "offset":"length_npf_offset",
        "max": 25
    },

    "sv": {
        "branches": [
            "sv_ptrel",
            "sv_deta",
            "sv_dphi",
            "sv_deltaR",
            "sv_mass",
            "sv_ntracks",
            "sv_chi2",
            "sv_ndof",
            "sv_dxy",
            "sv_dxysig",
            "sv_d3d",
            "sv_d3dsig",
            "sv_costhetasvpv",
            "sv_enratio"
        ],
        "length":"length_sv",
        "offset":"length_sv_offset",
        "max": 4
    },
    
    "muon_charge": {
        "branches": [
            "muon_charge",
        ],
        "length":"length_mu",
        "offset":"length_mu_offset",
        "max": 2
    },
    
    "muon": {
        "branches": [
            "muon_ptrel",
            "muon_deta",
            "muon_dphi",
            
            "muon_energy",
            "muon_et",
            "muon_deltaR",
            "muon_numberOfMatchedStations",

            "muon_IP2d",
            "muon_IP2dSig",
            "muon_IP3d",
            "muon_IP3dSig",

            "muon_EtaRel",
            "muon_dxy",
            "muon_dxyError",
            "muon_dxySig",
            "muon_dz",
            "muon_dzError",
            "muon_dzSig",
            "muon_numberOfValidPixelHits",
            "muon_numberOfpixelLayersWithMeasurement",
            "muon_numberOfstripLayersWithMeasurement",

            "muon_chi2",
            "muon_ndof",

            "muon_caloIso",
            "muon_ecalIso",
            "muon_hcalIso",

            "muon_sumPfChHadronPt",
            "muon_sumPfNeuHadronEt",
            "muon_Pfpileup",
            "muon_sumPfPhotonEt",

            "muon_sumPfChHadronPt03",
            "muon_sumPfNeuHadronEt03",
            "muon_Pfpileup03",
            "muon_sumPfPhotonEt03",


            "muon_timeAtIpInOut",
            "muon_timeAtIpInOutErr",
            "muon_timeAtIpOutIn"
        ],
        "length":"length_mu",
        "offset":"length_mu_offset",
        "max": 2
    },
    
    'electron_charge': {
        "branches": [
            "electron_charge",
        ],
        "length":"length_ele",
        "offset":"length_ele_offset",
        "max": 2
    },

    "electron": {
        "branches": [
            "electron_ptrel",
            "electron_deltaR",
            "electron_deta",
            "electron_dphi",
            "electron_energy",
            "electron_EtFromCaloEn",
            "electron_isEB",
            "electron_isEE",
            "electron_ecalEnergy",
            "electron_isPassConversionVeto",
            "electron_convDist",
            "electron_convFlags",
            "electron_convRadius",
            "electron_hadronicOverEm",
            "electron_ecalDrivenSeed",
            "electron_IP2d",
            "electron_IP2dSig",
            "electron_IP3d",
            "electron_IP3dSig",

            "electron_elecSC_energy",
            "electron_elecSC_deta",
            "electron_elecSC_dphi",
            "electron_elecSC_et",
            "electron_elecSC_eSuperClusterOverP",
            #"electron_scPixCharge",
            "electron_superClusterFbrem",

            "electron_eSeedClusterOverP",
            "electron_eSeedClusterOverPout",
            "electron_eSuperClusterOverP",

            "electron_sigmaEtaEta",
            "electron_sigmaIetaIeta",
            "electron_sigmaIphiIphi",
            "electron_e5x5",
            "electron_e5x5Rel",
            "electron_e1x5Overe5x5",
            "electron_e2x5MaxOvere5x5",
            "electron_r9",
            "electron_hcalOverEcal",
            "electron_hcalDepth1OverEcal",
            "electron_hcalDepth2OverEcal",

            "electron_deltaEtaEleClusterTrackAtCalo",
            "electron_deltaEtaSeedClusterTrackAtCalo",
            "electron_deltaPhiSeedClusterTrackAtCalo", 
            "electron_deltaEtaSeedClusterTrackAtVtx",
            "electron_deltaEtaSuperClusterTrackAtVtx",
            "electron_deltaPhiEleClusterTrackAtCalo",
            "electron_deltaPhiSuperClusterTrackAtVtx",

            "electron_sCseedEta",

            "electron_EtaRel",
            "electron_dxy",
            "electron_dxyError",
            "electron_dxySig",
            "electron_dz",
            "electron_dzError",
            "electron_dzSig",
            "electron_nbOfMissingHits",
            #"electron_gsfCharge",
            "electron_ndof",
            "electron_chi2",
            "electron_numberOfBrems",
            "electron_fbrem",

            "electron_neutralHadronIso",
            "electron_particleIso",
            "electron_photonIso",
            "electron_puChargedHadronIso",
            "electron_trackIso",
            "electron_ecalPFClusterIso",
            "electron_hcalPFClusterIso",

            "electron_pfSumPhotonEt",
            "electron_pfSumChargedHadronPt", 
            "electron_pfSumNeutralHadronEt",
            "electron_pfSumPUPt",

            "electron_dr04TkSumPt",
            "electron_dr04EcalRecHitSumEt",
            "electron_dr04HcalDepth1TowerSumEt",
            "electron_dr04HcalDepth1TowerSumEtBc",
            "electron_dr04HcalDepth2TowerSumEt",
            "electron_dr04HcalDepth2TowerSumEtBc",
            "electron_dr04HcalTowerSumEt",
            "electron_dr04HcalTowerSumEtBc"
        ],
        "length":"length_ele",
        "offset":"length_ele_offset",
        "max": 2,
        
    }
}
