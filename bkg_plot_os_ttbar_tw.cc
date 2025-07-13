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
#include </afs/desy.de/user/p/paranjpe/top_wbwb/pepper/scripts/src/parametric_signal_bkg.h>
#include <json/json.h>

void bkg_plot_os_ttbar_tw()
{
    //RooRealVar width("width","width", 0.5, 10); //Obviously there is no width here
    RooRealVar mass("mass", "mass", 0.0, 1000.0);
    RooArgList args_ss(mass);
    RooArgList args_os(mass);
    RooPlot *plot_vs_mass = mass.frame();

    std::string odir_main = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_ttbar_tw_wjets_0508/";

    std::string textfile_path_os = odir_main + "0623_crystalball_j2_b1_net_mass_bkg_os_ttbar_tw.txt";
    std::string textfile_path_ss = odir_main + "0623_crystalball_gauss_j2_b1_net_mass_bkg_ss_ttbar_tw.txt";

    /*
    std::string crystalballFormula_os = "crystalball(x[0], x[1]*x[2], x[3], x[4], x[5], x[6])";
    std::string crystalball_gaussFormula_ss = "crystalball_gauss(x[0], x[1], x[2], x[3], x[4], x[5], x[6])";
    */

    /*
    std::string crystalballFormula_os = "@1*@2*exp(-0.5*pow((@0-@3)/@4,2))";
    std::string crystalball_gaussFormula_ss = "@1*exp(-0.5*pow((@0-@2)/@6,2))*(((@0-@2)/@3)<=0) + @1*exp(-0.5*pow((@0-@2)/@3,2))*(((@0-@2)/@3)>0)";
    */

    //Not including the norm here since there will be an area normalization anyway
    ///*
    std::string crystalballFormula_os = "exp(-0.5*pow((@0-@1)/@2,2))";
    std::string crystalball_gaussFormula_ss = "exp(-0.5*pow((@0-@1)/@5,2))*(((@0-@1)/@2)<=0) + exp(-0.5*pow((@0-@1)/@2,2))*(((@0-@1)/@2)>0)";
    //*/

    //Not including the norm here since there will be an area normalization anyway
    /*
    std::string crystalballFormula_os = "crystalball_no_norm(x[0], x[1], x[2], x[3], x[4])";
    std::string crystalball_gaussFormula_ss = "crystalball_gauss_no_norm(x[0], x[1], x[2], x[3], x[4], x[5])";
    */

    bin_bkg_types ttbar_tw_ss("ss","","ttbar_tw",0);
    std::cout << "Args before calling shape_ttbar_tw_ss " << args_ss.size() << std::endl;
    //RooGenericPdf shape_ttbar_tw_ss = shape_calc_bkg<RooGenericPdf>(&args_ss, & args_ss, textfile_path_ss, "crystalball_gauss_ss_ttbar_tw", crystalball_gaussFormula_ss, ttbar_tw_ss);
    RooGenericPdf shape_ttbar_tw_ss = shape_calc_bkg<RooGenericPdf>(&args_ss, textfile_path_ss, "crystalball_gauss_ss_ttbar_tw", crystalball_gaussFormula_ss, ttbar_tw_ss);
    std::cout << "Args after calling shape_ttbar_tw_ss " << args_ss.size() << std::endl;
        
    bin_bkg_types ttbar_tw_os("os","","ttbar_tw",1);
    std::cout << "Args before calling shape_ttbar_tw_os " << args_os.size() << std::endl;
    //RooGenericPdf shape_ttbar_tw_os = shape_calc_bkg<RooGenericPdf>(&args_os, & args_ss, textfile_path_os, "crystalball_os_ttbar_tw", crystalballFormula_os, ttbar_tw_os);
    RooGenericPdf shape_ttbar_tw_os = shape_calc_bkg<RooGenericPdf>(&args_os, textfile_path_os, "crystalball_os_ttbar_tw", crystalballFormula_os, ttbar_tw_os);
    std::cout << "Args after calling shape_ttbar_tw_os " << args_os.size() << std::endl;

    //==============================================================Plotting to canvas (final mass)================================================================

    //std::string total_path = odir_main + "0701_bkg_os_ss_ttbar_tw.png";
    std::string total_path = odir_main + "0705_bkg_os_ss_ttbar_tw.png"; //Actually made after 0705 but same date to be consistent with the signal date
    //std::string total_path = odir_main + "0705_bkg_os_ss_ttbar_tw_crystalball.png"; //Actually made after 0705 but same date to be consistent with the signal date

    std::vector<RooGenericPdf> final_shape_net = {shape_ttbar_tw_os, shape_ttbar_tw_ss};
    std::vector<Color_t> colors_net = {kRed,kBlue};
    std::vector<string> legend_titles = {"shape_ttbar_tw_os", "shape_ttbar_tw_ss"};
    std::vector<string> legend_keys = {"ttbar_tw opp sign", "ttbar_tw same sign"};

    plot_final_mass_multiple(plot_vs_mass, final_shape_net, colors_net, legend_titles, legend_keys, total_path);

    //==============================================================================================================================================================

    
    //======================================Trying to get the separate normalization from the cutflow files (json format)========================================

    //area_norm_shape_vs_width("WbjToLNu_4f_TuneCP5_13TeV-madgraph-pythia8", "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/");
    std::string all_proc_json_file = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_ttbar_tw_wjets_0508/cutflows.json";

    //All of these processes were run at the same time so they belong to the same json file
    std::vector<std::string> all_process_names = {"ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8", "TTToHadronic_TuneCP5_13TeV-powheg-pythia8", "ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
    "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8", "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8"};

    std::vector<float> norm_true = norm_calc(all_process_names, all_proc_json_file);
    std::cout << "Contents of norm_true" << std::endl;
    print_vector_ft(norm_true);

    RooRealVar norm_ss_ttbar_tw("norm_ss_ttbar_tw","norm_ss_ttbar_tw", norm_true.at(1));
    RooRealVar os_ss_norm_ratio_ttbar_tw("os_ss_norm_ratio_ttbar_tw", "os_ss_norm_ratio_ttbar_tw", norm_true.at(0)/norm_true.at(1));

    norm_ss_ttbar_tw.setConstant(false);
    os_ss_norm_ratio_ttbar_tw.setConstant(true);

    //There is probably no use defining both as RooRealVar since one has to depend on the other since it has same xsec uncertainty
    RooFormulaVar *area_norm_ss_ttbar_tw = new RooFormulaVar("crystalball_gauss_ss_ttbar_tw_norm","crystalball_gauss_ss_ttbar_tw_norm", "@0", RooArgList(norm_ss_ttbar_tw));
    RooFormulaVar *area_norm_os_ttbar_tw = new RooFormulaVar("crystalball_os_ttbar_tw_norm","crystalball_os_ttbar_tw_norm", "@0*@1", RooArgList(norm_ss_ttbar_tw,os_ss_norm_ratio_ttbar_tw));

    std::cout << "Just before making the norm vector" << std::endl;
    std::vector<RooFormulaVar> norm_net = {*area_norm_os_ttbar_tw, *area_norm_ss_ttbar_tw};

    //============================================================================================================================================================

    //Saving both in the same workspace since the names of the other variables are different anyway
    ///*
    //std::string workspace_dir = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/workspace_datacard_0621/";
    //std::string workspace_dir = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/workspace_datacard_0628_gauss_only/";
    std::string workspace_dir = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/workspace_datacard_0701_gauss_pdf/";
    std::string workspace_file_os_ss = workspace_dir + "workspace_bkg_ttbar_tw_os_ss.root";

    //std::cout << mass.getVal() << std::endl;
    save_workspace<RooGenericPdf, RooFormulaVar>(final_shape_net, norm_net, workspace_file_os_ss, "workspace_bkg_ttbar_tw_os_ss");
    //*/

    //*/
    gApplication->Terminate();
}
