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

void bkg_plot_os_wjets_dy_diboson()
{
    //RooRealVar width("width","width", 0.5, 10); //Obviously there is no width here
    RooRealVar mass("mass", "mass", 0.0, 1000.0);
    RooArgList args_ss(mass);
    RooArgList args_os(mass);
    RooPlot *plot_vs_mass = mass.frame();

    std::string odir_main = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_ttbar_tw_wjets_0508/";

    std::string textfile_path_os = odir_main + "0623_crystalball_j2_b1_net_mass_bkg_os_wjets_dy_diboson.txt";
    std::string textfile_path_ss = odir_main + "0623_crystalball_gauss_j2_b1_net_mass_bkg_ss_wjets_dy_diboson.txt";

    /*
    std::string crystalball_gaussFormula_ss = "crystalball_gauss(x[0], x[1], x[2], x[3], x[4], x[5], x[6])";
    std::string crystalballFormula_os = "crystalball(x[0], x[1]*x[2], x[3], x[4], x[5], x[6])";
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

    bin_bkg_types wjets_dy_diboson_ss("ss","","wjets_dy_diboson",0);
    std::cout << "Args before calling shape_wjets_dy_diboson_ss " << args_ss.size() << std::endl;
    //RooGenericPdf shape_wjets_dy_diboson_ss = shape_calc_bkg<RooGenericPdf>(&args_ss, &args_ss, textfile_path_ss, "crystalball_gauss_ss_wjets_dy_diboson", crystalball_gaussFormula_ss, wjets_dy_diboson_ss);
    RooGenericPdf shape_wjets_dy_diboson_ss = shape_calc_bkg<RooGenericPdf>(&args_ss, textfile_path_ss, "crystalball_gauss_ss_wjets_dy_diboson", crystalball_gaussFormula_ss, wjets_dy_diboson_ss);
    std::cout << "Args after calling shape_wjets_dy_diboson_ss " << args_ss.size() << std::endl;
    
    bin_bkg_types wjets_dy_diboson_os("os","","wjets_dy_diboson",1);
    std::cout << "Args before calling shape_wjets_dy_diboson_os " << args_os.size() << std::endl;
    //RooGenericPdf shape_wjets_dy_diboson_os = shape_calc_bkg<RooGenericPdf>(&args_os, &args_ss, textfile_path_os, "crystalball_os_wjets_dy_diboson", crystalballFormula_os, wjets_dy_diboson_os);
    RooGenericPdf shape_wjets_dy_diboson_os = shape_calc_bkg<RooGenericPdf>(&args_os, textfile_path_os, "crystalball_os_wjets_dy_diboson", crystalballFormula_os, wjets_dy_diboson_os);
    std::cout << "Args after calling shape_wjets_dy_diboson_os " << args_os.size() << std::endl;

    //==============================================================Plotting to canvas (final mass)================================================================

    //std::string total_path = odir_main + "0701_bkg_os_ss_wjets_dy_diboson.png";
    std::string total_path = odir_main + "0705_bkg_os_ss_wjets_dy_diboson.png";
    //std::string total_path = odir_main + "0705_bkg_os_ss_wjets_dy_diboson_crystalball.png";

    std::vector<RooGenericPdf> final_shape_net = {shape_wjets_dy_diboson_os, shape_wjets_dy_diboson_ss};
    std::vector<Color_t> colors_net = {kRed,kBlue};
    std::vector<string> legend_titles = {"shape_wjets_dy_diboson_os", "shape_wjets_dy_diboson_ss"};
    std::vector<string> legend_keys = {"wjets_dy_diboson opp sign", "wjets_dy_diboson same sign"};

    plot_final_mass_multiple(plot_vs_mass, final_shape_net, colors_net, legend_titles, legend_keys, total_path);

    //==============================================================================================================================================================


    //======================================Trying to get the separate normalization from the cutflow files (json format)========================================

    //area_norm_shape_vs_width("WbjToLNu_4f_TuneCP5_13TeV-madgraph-pythia8", "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/");
    std::string all_proc_json_file = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/net_procs_0508/cutflows.json";

    std::vector<std::string> all_process_names = {"WJetsToLNu_2J_TuneCP5_13TeV-amcatnloFXFX-pythia8", "WJetsToLNu_1J_TuneCP5_13TeV-amcatnloFXFX-pythia8", "WJetsToLNu_0J_TuneCP5_13TeV-amcatnloFXFX-pythia8",
    "WW_TuneCP5_13TeV-pythia8", "WZ_TuneCP5_13TeV-pythia8", "ZZ_TuneCP5_13TeV-pythia8", "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8"};

    std::vector<float> norm_true = norm_calc(all_process_names, all_proc_json_file);
    std::cout << "Contents of norm_true" << std::endl;
    print_vector_ft(norm_true);

    RooRealVar norm_ss_wjets_dy_diboson("norm_ss_wjets_dy_diboson","norm_ss_wjets_dy_diboson", norm_true.at(1));
    RooRealVar os_ss_norm_ratio_wjets_dy_diboson("os_ss_norm_ratio_wjets_dy_diboson", "os_ss_norm_ratio_wjets_dy_diboson", norm_true.at(0)/norm_true.at(1));

    norm_ss_wjets_dy_diboson.setConstant(false);
    os_ss_norm_ratio_wjets_dy_diboson.setConstant(true);

    //There is probably no use defining both as RooRealVar since one has to depend on the other since it has same xsec uncertainty
    RooFormulaVar *area_norm_ss_wjets_dy_diboson = new RooFormulaVar("crystalball_gauss_ss_wjets_dy_diboson_norm","crystalball_gauss_ss_wjets_dy_diboson_norm", "@0", RooArgList(norm_ss_wjets_dy_diboson));
    RooFormulaVar *area_norm_os_wjets_dy_diboson = new RooFormulaVar("crystalball_os_wjets_dy_diboson_norm","crystalball_os_wjets_dy_diboson_norm", "@0*@1", RooArgList(norm_ss_wjets_dy_diboson,os_ss_norm_ratio_wjets_dy_diboson));

    std::cout << "Just before making the norm vector" << std::endl;
    std::vector<RooFormulaVar> norm_net = {*area_norm_os_wjets_dy_diboson, *area_norm_ss_wjets_dy_diboson};


    //============================================================================================================================================================


    //Saving both in the same workspace since the names of the other variables are different anyway
    ///*
    //std::string workspace_dir = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/workspace_datacard_0621/";
    //std::string workspace_dir = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/workspace_datacard_0628_gauss_only/";
    std::string workspace_dir = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/workspace_datacard_0701_gauss_pdf/";
    std::string workspace_file_os_ss = workspace_dir + "workspace_bkg_wjets_dy_diboson_os_ss.root";

    //std::cout << mass.getVal() << std::endl;
    save_workspace<RooGenericPdf, RooFormulaVar>(final_shape_net, norm_net, workspace_file_os_ss, "workspace_bkg_wjets_dy_diboson_os_ss");
    //*/

    //*/
    gApplication->Terminate();
}
