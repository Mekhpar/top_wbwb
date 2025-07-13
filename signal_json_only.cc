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
#include </afs/desy.de/user/p/paranjpe/top_wbwb/pepper/scripts/src/parametric_signal_bkg.h>
#include <json/json.h>
#include <yaml-cpp/yaml.h>
//#include <json.hpp>

//Using the nlohmann namespace - not working so not being used
/*
void read_json_cc(std::string json_file)
{
    nlohmann::json signal = nlohmann::json::parse(json_file);
    //Just looking at a specific set of keys for now
    std::cout << " Contents of signal region for wbj" << std::endl;
    std::cout << signal["WbjToLNu_4f_TuneCP5_13TeV-madgraph-pythia8"]["mu"]["j2_b1"] << std::endl;

    //Looking at what exception is produced if the contents are read from a non existent key
    std::cout << signal["WbjToLNu_4f_TuneCP5_13TeV-madgraph-pythia8"]["mu"]["j2_b3"] << std::endl;

}
*/

void fit_area_norm(std::vector<float> y_vals, std::vector<float> x_vals, std::string cutflow_file_path, std::string graph_object_name, std::string graph_title, std::string save_path)
{
    TGraph* area_norm_vs_width = new TGraph(x_vals.size(), &x_vals[0],&y_vals[0]);    

    TCanvas *c1 = new TCanvas("c1", "c1",0,0,700,500);
    gStyle->SetOptStat(0);
    c1->SetHighLightColor(2);
    c1->Range(0,0,1,1);
    c1->SetFillColor(0);
    c1->SetBorderMode(0);
    c1->SetBorderSize(2);
    c1->SetFrameBorderMode(0);    

    area_norm_vs_width->SetName(graph_object_name.c_str());
    area_norm_vs_width->SetTitle(graph_title.c_str());
    area_norm_vs_width->GetHistogram()->GetXaxis()->SetTitle("Width set in MC (in GeV)");
    area_norm_vs_width->GetHistogram()->GetYaxis()->SetTitle("Cutflow area normalization values");
    area_norm_vs_width->SetFillStyle(1000);
    area_norm_vs_width->SetMarkerStyle(20);
    area_norm_vs_width->Draw("ap");

    c1->Modified();
    c1->cd();
    c1->SetSelected(c1);
    std::string png_path = save_path + ".png";
    std::string macro_path = save_path + ".C";
    c1->Print(png_path.c_str());
    c1->Print(macro_path.c_str());
    c1->Close();

}

//std::vector<float> area_norm_shape_vs_width(std::string process_name, std::string cutflow_file_path) //Only for signal
void area_norm_shape_vs_width(std::string process_name, std::string cutflow_file_path) //Only for signal
{
    std::vector<float> opp_sign_norm, same_sign_norm, width_vals;
    //for(int width_num=1; width_num<22; width_num++)
    std::string width_yaml_file = "/afs/desy.de/user/p/paranjpe/top_wbwb/pepper/inputs/common/width_weights.yaml";
    YAML::Node width_dict = YAML::LoadFile(width_yaml_file);
    
    for(int width_num=1; width_num<22; width_num++) //Only for debugging
    {
        std::string json_file = cutflow_file_path + "/width_" + std::to_string(width_num) + "/cutflows.json";
        std::cout << "Chronological width num " << width_num << std::endl;
        std::vector<float> net_counts = area_norm(process_name, json_file);
        opp_sign_norm.push_back(net_counts.at(0));
        same_sign_norm.push_back(net_counts.at(1));
        float width_current = width_dict[process_name]["Width"][std::to_string(width_num)]["Value"].as<float>();
        width_vals.push_back(width_current);
    }

    std::cout << std::endl;
    std::cout << "opp_sign_norm" << std::endl;
    print_vector_ft(opp_sign_norm);
    std::cout << "same_sign_norm" << std::endl;
    print_vector_ft(same_sign_norm);
    std::cout << "width_vals" << std::endl;
    print_vector_ft(width_vals);

    fit_area_norm(opp_sign_norm, width_vals, cutflow_file_path, "opp_area_norm_vs_width", "Area norm opp sign with fit vs width", cutflow_file_path + "/area_cutflow_opp_sign_norm_vs_width_signal");
    fit_area_norm(same_sign_norm, width_vals, cutflow_file_path, "same_area_norm_vs_width", "Area norm same sign with fit vs width", cutflow_file_path + "/area_cutflow_same_sign_norm_vs_width_signal");

}

void signal_json_only()
{
    //======================================Trying to get the separate normalization from the cutflow files (json format)========================================

    //std::vector<string> opp_sign_keys = {"WbjToLNu_4f_TuneCP5_13TeV-madgraph-pythia8","mu","j2_b1", "opp_sign", "lep_neg", "onshell", "b_loose_1+_leading"};
    
    area_norm_shape_vs_width("WbjToLNu_4f_TuneCP5_13TeV-madgraph-pythia8", "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/");

    //============================================================================================================================================================
    gApplication->Terminate();
}
