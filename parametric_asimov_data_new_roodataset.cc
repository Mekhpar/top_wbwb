#include "TStyle.h"
#include "TGraph.h"
#include "TH1.h"
#include "TCanvas.h"
#include "TPad.h"
#include "TAxis.h"
#include "TGaxis.h"
#include "TLegend.h"
#include <TMath.h>
#include <TString.h>
#include <vector>
#include <iostream>
#include <sstream>
#include <cmath>
#include <utility>
#include <fstream>
#include <algorithm>
#include <TMultiGraph.h>
#include <TFormula.h>
#include <RooFit.h>
#include </afs/desy.de/user/p/paranjpe/top_wbwb/pepper/scripts/src/parametric_signal_bkg.h>

void parametric_asimov_data_new_roodataset()
{
    RooRealVar mass("mass", "mass", 0.0, 1000.0);
    //RooRealVar mass("mass", "mass", 0.0, 500.0); //dummy range only

    //Trying to read the RooFormulaVar from the workspace has not been successful
    //There is no provision for reading RooFormulaVar - casting by getting pdf (from RooAbsPdf) did not help either
    //Casting directly to a RooGenericPdf gave the same error - that the object is a null pointer

    /*
    std::string workspace_qcd_path = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/workspace_datacard_0621/workspace_bkg_qcd_os_ss.root";
    TFile *f_qcd = new TFile(workspace_qcd_path.c_str(), "READ");    
    RooWorkspace *w_qcd = (RooWorkspace *)f_qcd->Get("workspace_bkg_qcd_os_ss");
    std::cout << "Contents of qcd workspace" << std::endl;
    w_qcd->Print();
    std::cout << "Value of alpha "<< w_qcd->var("alpha_os_qcd")->getVal() << std::endl;
    std::cout << "Value of os_ss_norm_ratio_qcd "<< w_qcd->var("os_ss_norm_ratio_qcd")->getVal() << std::endl;
    RooGenericPdf * shape_qcd_ss = (RooGenericPdf*) (w_qcd->pdf("crystalball_gauss_ss"));

    std::string ss_path = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_ttbar_tw_wjets_0508/bkg_ss_parametric_cc_qcd_from_rooworkspace.png";
    //plot_final_mass(plot_vs_mass, *shape_qcd_ss, ss_path);
    string canvas_title = "c_mass";
    TCanvas *c1 = new TCanvas(canvas_title.c_str(),canvas_title.c_str(), 700,500);

    //variable_intermediate[i]->plotOn(plot_vs_mass);
    shape_qcd_ss->plotOn(plot_vs_mass);
    
    plot_vs_mass->Draw();
    c1->Draw();
    c1->SaveAs(ss_path.c_str());
    c1->Close();

    */
    //std::cout << "Value of reverse crystalball gauss function "<< w_qcd->var("crystalball_gauss_ss")->getVal() << std::endl;

    std::string odir_main_bkg = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_ttbar_tw_wjets_0508/";
    std::string odir_main_signal = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/width_19/";

    ///*
    std::string two_gauss_ss_norm = "@1*exp(-0.5*pow((@0-@2)/@6,2))*(((@0-@2)/@3)<=0) + @1*exp(-0.5*pow((@0-@2)/@3,2))*(((@0-@2)/@3)>0)";
    //std::string two_gauss_os_norm = "@1*@2*exp(-0.5*pow((@0-@3)/@7,2))*(((@0-@3)/@4)<=0) + @1*@2*exp(-0.5*pow((@0-@3)/@4,2))*(((@0-@3)/@4)>0)";
    std::string two_gauss_os_norm = "@1*exp(-0.5*pow((@0-@2)/@6,2))*(((@0-@2)/@3)<=0) + @1*exp(-0.5*pow((@0-@2)/@3,2))*(((@0-@2)/@3)>0)";

    std::string gauss_only_ss_norm = "@1*exp(-0.5*pow((@0-@2)/@3,2))";
    //std::string gauss_only_os_norm = "@1*@2*exp(-0.5*pow((@0-@3)/@4,2))";
    std::string gauss_only_os_norm = "@1*exp(-0.5*pow((@0-@2)/@3,2))";

    //Normalization not added here, will be added separately (or here itself later but by a factor of true_norm(i.e. number of events)/(current integral value))
    std::string two_gauss_ss = "exp(-0.5*pow((@0-@1)/@5,2))*(((@0-@1)/@2)<=0) + exp(-0.5*pow((@0-@1)/@2,2))*(((@0-@1)/@2)>0)";
    std::string two_gauss_os = "exp(-0.5*pow((@0-@1)/@5,2))*(((@0-@1)/@2)<=0) + exp(-0.5*pow((@0-@1)/@2,2))*(((@0-@1)/@2)>0)";

    std::string gauss_only_ss = "exp(-0.5*pow((@0-@1)/@2,2))";
    std::string gauss_only_os = "exp(-0.5*pow((@0-@1)/@2,2))";    
    //*/

    /*
    std::string two_gauss_ss_norm = "crystalball_gauss(x[0], x[1], x[2], x[3], x[4], x[5], x[6])";
    //std::string two_gauss_os_norm = "@1*@2*exp(-0.5*pow((@0-@3)/@7,2))*(((@0-@3)/@4)<=0) + @1*@2*exp(-0.5*pow((@0-@3)/@4,2))*(((@0-@3)/@4)>0)";
    std::string two_gauss_os_norm = "crystalball_gauss(x[0], x[1], x[2], x[3], x[4], x[5], x[6])";

    std::string gauss_only_ss_norm = "crystalball(x[0], x[1], x[2], x[3], x[4], x[5])";
    //std::string gauss_only_os_norm = "@1*@2*exp(-0.5*pow((@0-@3)/@4,2))";
    std::string gauss_only_os_norm = "crystalball(x[0], x[1], x[2], x[3], x[4], x[5])";

    //Normalization not added here, will be added separately (or here itself later but by a factor of true_norm(i.e. number of events)/(current integral value))
    std::string two_gauss_ss = "crystalball_gauss_no_norm(x[0], x[1], x[2], x[3], x[4], x[5])";
    std::string two_gauss_os = "crystalball_gauss_no_norm(x[0], x[1], x[2], x[3], x[4], x[5])";

    std::string gauss_only_ss = "crystalball_no_norm(x[0], x[1], x[2], x[3], x[4])";
    std::string gauss_only_os = "crystalball_no_norm(x[0], x[1], x[2], x[3], x[4])";    
    */

    std::string all_proc_json_file = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/net_procs_0508/cutflows.json";

    //===========================================================================QCD===========================================================================================

    //process_shapes qcd_shapes("crystalball_gauss", "crystalball_gauss", "(x[0], x[1]*x[2], x[3], x[4], x[5], x[6], x[7])", "(x[0], x[1], x[2], x[3], x[4], x[5], x[6])", "qcd");
    //process_shapes qcd_shapes("crystalball_gauss", "crystalball_gauss", two_gauss_os.c_str(), two_gauss_ss.c_str(), "qcd");
    process_shapes qcd_shapes("crystalball_gauss", "crystalball_gauss", two_gauss_os.c_str(), two_gauss_ss.c_str(), two_gauss_os_norm.c_str(), two_gauss_ss_norm.c_str(), "qcd");

    //All of these processes were run at the same time so they belong to the same json file
    std::vector<std::string> qcd_process_names = {"QCD_Pt-1000_MuEnrichedPt5_TuneCP5_13TeV-pythia8", "QCD_Pt-120To170_MuEnrichedPt5_TuneCP5_13TeV-pythia8",
    "QCD_Pt-120to170_EMEnriched_TuneCP5_13TeV-pythia8", "QCD_Pt-15To20_MuEnrichedPt5_TuneCP5_13TeV-pythia8", "QCD_Pt-15to20_EMEnriched_TuneCP5_13TeV-pythia8",
    "QCD_Pt-170To300_MuEnrichedPt5_TuneCP5_13TeV-pythia8", "QCD_Pt-170to300_EMEnriched_TuneCP5_13TeV-pythia8", "QCD_Pt-20To30_MuEnrichedPt5_TuneCP5_13TeV-pythia8",
    "QCD_Pt-20to30_EMEnriched_TuneCP5_13TeV-pythia8", "QCD_Pt-300To470_MuEnrichedPt5_TuneCP5_13TeV-pythia8", "QCD_Pt-300toInf_EMEnriched_TuneCP5_13TeV-pythia8",
    "QCD_Pt-30To50_MuEnrichedPt5_TuneCP5_13TeV-pythia8", "QCD_Pt-30to50_EMEnriched_TuneCP5_13TeV-pythia8", "QCD_Pt-470To600_MuEnrichedPt5_TuneCP5_13TeV-pythia8",
    "QCD_Pt-50to80_EMEnriched_TuneCP5_13TeV-pythia8", "QCD_Pt-600To800_MuEnrichedPt5_TuneCP5_13TeV-pythia8", "QCD_Pt-800To1000_MuEnrichedPt5_TuneCP5_13TeV-pythia8",
    "QCD_Pt-80To120_MuEnrichedPt5_TuneCP5_13TeV-pythia8", "QCD_Pt-80to120_EMEnriched_TuneCP5_13TeV-pythia8"};

    std::vector<float> norm_true_qcd = norm_calc(qcd_process_names, all_proc_json_file);
    std::cout << "Contents of norm_true_qcd" << std::endl;
    print_vector_ft(norm_true_qcd);

    //std::vector<RooFormulaVar> shape_both_qcd = shape_calc<RooFormulaVar>(&qcd_shapes, &mass, odir_main_bkg, 0);
    std::vector<RooFormulaVar> shape_both_qcd = shape_calc<RooFormulaVar>(&qcd_shapes, norm_true_qcd, &mass, odir_main_bkg, 0);
    //std::vector<RooGenericPdf> shape_both_qcd_pdf = shape_calc<RooGenericPdf>(&qcd_shapes, &mass, odir_main_bkg, 0);

    //========================================================================================================================================================================

    //=======================================================================ttbar+tW=========================================================================================

    //process_shapes ttbar_tw_shapes("crystalball", "crystalball_gauss", "(x[0], x[1]*x[2], x[3], x[4], x[5], x[6])", "(x[0], x[1], x[2], x[3], x[4], x[5], x[6])", "ttbar_tw");
    //process_shapes ttbar_tw_shapes("crystalball", "crystalball_gauss", gauss_only_os.c_str(), two_gauss_ss.c_str(), "ttbar_tw");
    process_shapes ttbar_tw_shapes("crystalball", "crystalball_gauss", gauss_only_os.c_str(), two_gauss_ss.c_str(), gauss_only_os_norm.c_str(), two_gauss_ss_norm.c_str(), "ttbar_tw");

    //All of these processes were run at the same time so they belong to the same json file
    std::vector<std::string> ttbar_tw_process_names = {"ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8", "TTToHadronic_TuneCP5_13TeV-powheg-pythia8", "ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
    "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8", "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8"};

    std::vector<float> norm_true_ttbar_tw = norm_calc(ttbar_tw_process_names, all_proc_json_file);
    std::cout << "Contents of norm_true_ttbar_tw" << std::endl;
    print_vector_ft(norm_true_ttbar_tw);

    //std::vector<RooFormulaVar> shape_both_ttbar_tw = shape_calc<RooFormulaVar>(&ttbar_tw_shapes, &mass, odir_main_bkg, 0);
    std::vector<RooFormulaVar> shape_both_ttbar_tw = shape_calc<RooFormulaVar>(&ttbar_tw_shapes, norm_true_ttbar_tw, &mass, odir_main_bkg, 0);

    //========================================================================================================================================================================

    //=====================================================================wjets+dy+diboson===================================================================================

    //process_shapes wjets_dy_diboson_shapes("crystalball", "crystalball_gauss", "(x[0], x[1]*x[2], x[3], x[4], x[5], x[6])", "(x[0], x[1], x[2], x[3], x[4], x[5], x[6])", "wjets_dy_diboson");
    //process_shapes wjets_dy_diboson_shapes("crystalball", "crystalball_gauss", gauss_only_os.c_str(), two_gauss_ss.c_str(), "wjets_dy_diboson");
    process_shapes wjets_dy_diboson_shapes("crystalball", "crystalball_gauss", gauss_only_os.c_str(), two_gauss_ss.c_str(), gauss_only_os_norm.c_str(), two_gauss_ss_norm.c_str(), "wjets_dy_diboson");

    std::vector<std::string> wjets_dy_diboson_process_names = {"WJetsToLNu_2J_TuneCP5_13TeV-amcatnloFXFX-pythia8", "WJetsToLNu_1J_TuneCP5_13TeV-amcatnloFXFX-pythia8", "WJetsToLNu_0J_TuneCP5_13TeV-amcatnloFXFX-pythia8",
    "WW_TuneCP5_13TeV-pythia8", "WZ_TuneCP5_13TeV-pythia8", "ZZ_TuneCP5_13TeV-pythia8", "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8"};

    std::vector<float> norm_true_wjets_dy_diboson = norm_calc(wjets_dy_diboson_process_names, all_proc_json_file);
    std::cout << "Contents of norm_true_wjets_dy_diboson" << std::endl;
    print_vector_ft(norm_true_wjets_dy_diboson);

    //std::vector<RooFormulaVar> shape_both_wjets_dy_diboson = shape_calc<RooFormulaVar>(&wjets_dy_diboson_shapes, &mass, odir_main_bkg, 0);
    std::vector<RooFormulaVar> shape_both_wjets_dy_diboson = shape_calc<RooFormulaVar>(&wjets_dy_diboson_shapes, norm_true_wjets_dy_diboson, &mass, odir_main_bkg, 0);

    //========================================================================================================================================================================    
    
    //=======================================================wbj signal - SM predicted width sample only======================================================================

    //process_shapes signal_shapes("crystalball", "crystalball", "(x[0], x[1]*x[2], x[3], x[4], x[5], x[6])", "(x[0], x[1], x[2], x[3], x[4], x[5])", "signal");
    //process_shapes signal_shapes("crystalball", "crystalball", gauss_only_os.c_str(), gauss_only_ss.c_str(), "signal");
    process_shapes signal_shapes("crystalball", "crystalball", gauss_only_os.c_str(), gauss_only_ss.c_str(), gauss_only_os_norm.c_str(), gauss_only_ss_norm.c_str(), "signal");
    
    std::vector<std::string> signal_process_names = {"WbjToLNu_4f_TuneCP5_13TeV-madgraph-pythia8"};

    std::vector<float> norm_true_signal = norm_calc(signal_process_names, all_proc_json_file);
    std::cout << "Contents of norm_true_signal" << std::endl;
    print_vector_ft(norm_true_signal);
    
    //std::vector<RooFormulaVar> shape_both_signal = shape_calc<RooFormulaVar>(&signal_shapes, &mass, odir_main_signal, 1);
    std::vector<RooFormulaVar> shape_both_signal = shape_calc<RooFormulaVar>(&signal_shapes, norm_true_signal, &mass, odir_main_signal, 1);

    //========================================================================================================================================================================    

    ///*
    RooFormulaVar shape_os_total("shape_os_total", "shape_os_total", "@0+@1+@2+@3", RooArgList(shape_both_qcd.at(0), shape_both_ttbar_tw.at(0), shape_both_wjets_dy_diboson.at(0), shape_both_signal.at(0)));
    RooFormulaVar shape_ss_total("shape_ss_total", "shape_ss_total", "@0+@1+@2+@3", RooArgList(shape_both_qcd.at(1), shape_both_ttbar_tw.at(1), shape_both_wjets_dy_diboson.at(1), shape_both_signal.at(1)));

    RooGenericPdf shape_os_total_pdf("shape_os_total_pdf", "shape_os_total_pdf", "@0+@1+@2+@3", RooArgList(shape_both_qcd.at(0), shape_both_ttbar_tw.at(0), shape_both_wjets_dy_diboson.at(0), shape_both_signal.at(0)));
    RooGenericPdf shape_ss_total_pdf("shape_ss_total_pdf", "shape_ss_total_pdf", "@0+@1+@2+@3", RooArgList(shape_both_qcd.at(1), shape_both_ttbar_tw.at(1), shape_both_wjets_dy_diboson.at(1), shape_both_signal.at(1)));

    RooPlot *plot_vs_mass = mass.frame();

    std::vector<Color_t> colors_net = {kBlack, kRed, kGreen, kBlue, kOrange};
    std::vector<Color_t> colors_net_pdf = {kBlack};
    //std::vector<Color_t> colors_net = {kOrange};
    //*/
    /*
    RooPlot *plot_vs_mass = mass.frame();

    std::vector<Color_t> colors_net = {kRed, kGreen, kBlue, kOrange};
    */

    //========================================================All the vectors of the RooFormulaVars and RooGenericPdfs for plotting===================================

    ///*
    std::vector<RooFormulaVar> final_shape_net_os = {shape_os_total, shape_both_qcd.at(0), shape_both_ttbar_tw.at(0), shape_both_wjets_dy_diboson.at(0), shape_both_signal.at(0)};
    std::vector<RooGenericPdf> final_pdf_os = {shape_os_total_pdf}; //Since the total shape is going to be plotted as a pdf, it does not make much sense to look at the individual shapes

    std::vector<string> legend_titles_os = {"shape_os_total", "shape_qcd_os", "shape_ttbar_tw_os", "shape_wjets_dy_diboson_os", "shape_signal_os"};
    std::vector<string> legend_keys_os = {"Total asimov shape opp sign", "QCD opp sign", "ttbar + tW opp sign", "W+Jets, DY and diboson opp sign", "Signal opp sign"};

    std::vector<string> legend_titles_os_pdf = {"shape_os_total_pdf"};
    std::vector<string> legend_keys_os_pdf = {"Total asimov shape opp sign area normalized pdf"};

    std::vector<RooFormulaVar> final_shape_net_ss = {shape_ss_total, shape_both_qcd.at(1), shape_both_ttbar_tw.at(1), shape_both_wjets_dy_diboson.at(1), shape_both_signal.at(1)};
    std::vector<RooGenericPdf> final_pdf_ss = {shape_ss_total_pdf}; //Since the total shape is going to be plotted as a pdf, it does not make much sense to look at the individual shapes

    std::vector<string> legend_titles_ss = {"shape_ss_total", "shape_qcd_ss", "shape_ttbar_tw_ss", "shape_wjets_dy_diboson_ss", "shape_signal_ss"};
    std::vector<string> legend_keys_ss = {"Total asimov shape same sign", "QCD same sign", "ttbar + tW same sign", "W+Jets, DY and diboson same sign", "Signal same sign"};

    std::vector<string> legend_titles_ss_pdf = {"shape_ss_total_pdf"};
    std::vector<string> legend_keys_ss_pdf = {"Total asimov shape same sign area normalized pdf"};
    //*/

    /*
    std::vector<RooFormulaVar> final_shape_net_os = {shape_both_qcd.at(0), shape_both_ttbar_tw.at(0), shape_both_wjets_dy_diboson.at(0), shape_both_signal.at(0)};
    std::vector<string> legend_titles_os = {"shape_qcd_os", "shape_ttbar_tw_os", "shape_wjets_dy_diboson_os", "shape_signal_os"};

    std::vector<RooFormulaVar> final_shape_net_ss = {shape_both_qcd.at(1), shape_both_ttbar_tw.at(1), shape_both_wjets_dy_diboson.at(1), shape_both_signal.at(1)};
    std::vector<string> legend_titles_ss = {"shape_qcd_ss", "shape_ttbar_tw_ss", "shape_wjets_dy_diboson_ss", "shape_signal_ss"};
    */

    //==================================================================================================================================================================

    //==============================================================================Actual plotting=====================================================================

    plot_vs_mass = mass.frame();
    //Plotting all the shapes (not normalized) together
    //std::string net_path_os = odir_main_bkg + "0701_os_all_asimov_data.png";
    std::string net_path_os = odir_main_bkg + "0705_os_all_asimov_data.png";
    plot_final_mass_multiple(plot_vs_mass, final_shape_net_os, colors_net, legend_titles_os, legend_keys_os, net_path_os);

    //So that this will be plotted along with the actual total shape when the plot_vs_mass is not redefined
    //Trying the command in the link https://root-forum.cern.ch/t/building-a-pdf-using-roogenericpdf-and-extract-parameters-from-fitting-an-asimov-data/34417

    //There seems to be an inherent binsize of 10 GeV while sampling, so dividing the number of events to be sampled by 10 in order to get same normalization as MC
    float inherent_binsize_os = 10.0; //in GeV in the x axis (mass)
    float asimov_os_events = (norm_true_qcd.at(0) + norm_true_ttbar_tw.at(0) + norm_true_wjets_dy_diboson.at(0) + norm_true_signal.at(0))/inherent_binsize_os;
    //RooDataSet *data_os_asimov = shape_os_total_pdf.generate(mass, 20000);
    RooDataSet *data_os_asimov = shape_os_total_pdf.generate(mass, asimov_os_events);
    std::string dataset_os_path = odir_main_bkg + "0705_bkg_os_all_asimov_dataset.png";
    plot_final_mass_dataset(plot_vs_mass, *data_os_asimov, dataset_os_path);

    /*
    plot_vs_mass = mass.frame();
    //std::string net_path_os_pdf = odir_main_bkg + "0701_os_asimov_data_pdf.png";
    std::string net_path_os_pdf = odir_main_bkg + "0705_os_asimov_data_pdf.png";
    plot_final_mass_multiple(plot_vs_mass, final_pdf_os, colors_net_pdf, legend_titles_os_pdf, legend_keys_os_pdf, net_path_os_pdf);
    */

    plot_vs_mass = mass.frame();
    //std::string net_path_ss = odir_main_bkg + "0701_ss_all_asimov_data.png";
    std::string net_path_ss = odir_main_bkg + "0705_ss_all_asimov_data.png";
    plot_final_mass_multiple(plot_vs_mass, final_shape_net_ss, colors_net, legend_titles_ss, legend_keys_ss, net_path_ss);

    //So that this will be plotted along with the actual total shape when the plot_vs_mass is not redefined
    //Trying the command in the link https://root-forum.cern.ch/t/building-a-pdf-using-roogenericpdf-and-extract-parameters-from-fitting-an-asimov-data/34417

    //There seems to be an inherent binsize of 10 GeV while sampling, so dividing the number of events to be sampled by 10 in order to get same normalization as MC
    float inherent_binsize_ss = 10.0; //in GeV in the x axis (mass)
    float asimov_ss_events = (norm_true_qcd.at(1) + norm_true_ttbar_tw.at(1) + norm_true_wjets_dy_diboson.at(1) + norm_true_signal.at(1))/inherent_binsize_ss;
    
    //RooDataSet *data_ss_asimov = shape_ss_total_pdf.generate(mass, 20000);
    //RooDataSet *data_ss_asimov = shape_ss_total_pdf.generate(mass, 15000);
    RooDataSet *data_ss_asimov = shape_ss_total_pdf.generate(mass, asimov_ss_events);
    std::string dataset_ss_path = odir_main_bkg + "0705_bkg_ss_all_asimov_dataset.png";
    plot_final_mass_dataset(plot_vs_mass, *data_ss_asimov, dataset_ss_path);

    /*
    plot_vs_mass = mass.frame();
    //std::string net_path_ss_pdf = odir_main_bkg + "0701_ss_asimov_data_pdf.png";
    std::string net_path_ss_pdf = odir_main_bkg + "0705_ss_asimov_data_pdf.png";
    plot_final_mass_multiple(plot_vs_mass, final_pdf_ss, colors_net_pdf, legend_titles_ss_pdf, legend_keys_ss_pdf, net_path_ss_pdf);
    */

    //==================================================================================================================================================================

    //Saving both in the same workspace since the names of the other variables are different anyway
    ///*
    //std::string workspace_dir = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/workspace_datacard_0621/";
    //std::string workspace_dir = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/workspace_datacard_0628_gauss_only/";
    std::string workspace_dir = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/workspace_datacard_0701_gauss_pdf/";
    std::string workspace_file_os_ss = workspace_dir + "workspace_asimov_os_ss.root";

    dataset_os_path = odir_main_bkg + "0705_bkg_os_all_asimov_dataset_only.png";
    dataset_ss_path = odir_main_bkg + "0705_bkg_ss_all_asimov_dataset_only.png";
    
    plot_vs_mass = mass.frame();
    plot_final_mass_dataset(plot_vs_mass, *data_os_asimov, dataset_os_path);

    plot_vs_mass = mass.frame();
    plot_final_mass_dataset(plot_vs_mass, *data_ss_asimov, dataset_ss_path);


    ///*
    std::vector<RooDataSet> final_shape_net_dataset = {*data_os_asimov, *data_ss_asimov};
    save_workspace_roodataset<RooDataSet>(final_shape_net_dataset, workspace_file_os_ss, "workspace_asimov_os_ss");
    //*/
    //*/

    /*
    for(int i = 0; i < num_bins; i++)
    {
        std::cout << "Number in opp sign " << i << std::endl;
        std::cout << "mass value in opp sign " << (2*i+1)*binsize/2.0 << std::endl;
        std::cout << "bin height in opp sign in TH1F " << os_vs_mass_bins->GetBinContent(i+1) << std::endl;
        data_os_hist->get(i);
        std::cout << "bin height in opp sign roodatahist " << data_os_hist->weight() << std::endl;
        data_os_hist->get(i);
        std::cout << "bin error in opp sign roodatahist " << data_os_hist->weightError() << std::endl;
        std::cout << "bin error/height ratio in opp sign roodatahist " << data_os_hist->weightError()/data_os_hist->weight() << std::endl;
    }
    */
    /*
    std::vector<RooDataHist> final_shape_net;
    final_shape_net.push_back(*data_os_hist);
    final_shape_net.push_back(*data_ss_hist);
    //save_workspace_datahist(final_shape_net, workspace_file_os_ss, "workspace_asimov_os_ss");
    //save_workspace<RooDataHist>(final_shape_net, workspace_file_os_ss, "workspace_asimov_os_ss");
    */
    //std::cout << "Integral for shape_both_qcd_pdf " << shape_both_qcd_pdf.at(0).createIntegral(RooArgSet(mass))->getVal() << " " << shape_both_qcd_pdf.at(1).createIntegral(RooArgSet(mass))->getVal() << std::endl;

    gApplication->Terminate();
}
