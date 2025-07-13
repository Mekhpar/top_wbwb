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

/*
RooFormulaVar shape_calc_signal(RooArgList *args, RooArgList *args_previous, std::string textfile_path, std::string fit_string, std::string formula_name, std::string formula_string, bin_bkg_types process_sign)
{
    int len_array = sizeof(fit_string) / sizeof(fit_string[0]);    

}
*/

void parametric_plot_os()
{
    RooRealVar width("width","width", 0.5, 10);
    //RooArgList args(width);
    //RooRealVar mass{"mass", "mass", 0.0, 1000.0};
    RooRealVar mass("mass", "mass", 0.0, 1000.0);
    /*
    RooArgList args_os_final(mass);
    RooArgList args_ss_final(mass);
    */

    RooFormulaVar *variable_intermediate_os[4];
    RooFormulaVar *variable_intermediate_ss[4];

    //RooRealVar mean_mass_os_signal{"mean_mass_os_signal", "mean_mass_os_signal", 172.5};
    RooRealVar mean_mass_os_signal("mean_mass_os_signal", "mean_mass_os_signal", 172.5);
    RooRealVar mean_mass_ss_signal("mean_mass_ss_signal", "mean_mass_ss_signal", 172.5);

    //Newly added instead of in the norm loop
    RooArgList args_ss_final(mass, mean_mass_ss_signal);
    RooArgList args_os_final(mass, mean_mass_os_signal);

    /*
    std::string crystalballFormula_os = "crystalball(x[0], x[1], x[2], x[3], x[4], x[5])";
    std::string crystalballFormula_ss = "crystalball(x[0], x[1], x[2], x[3], x[4], x[5])";
    */
    /*
    std::string crystalballFormula_os = "@1*exp(-0.5*pow((@0-@2)/@3,2))";
    std::string crystalballFormula_ss = "@1*exp(-0.5*pow((@0-@2)/@3,2))";
    */

    //Not including the norm here since there will be an area normalization anyway
    ///*
    std::string crystalballFormula_os = "exp(-0.5*pow((@0-@1)/@2,2))";
    std::string crystalballFormula_ss = "exp(-0.5*pow((@0-@1)/@2,2))";
    //*/

    //Not including the norm here since there will be an area normalization anyway
    /*
    std::string crystalballFormula_os = "crystalball_no_norm(x[0], x[1], x[2], x[3], x[4])";
    std::string crystalballFormula_ss = "crystalball_no_norm(x[0], x[1], x[2], x[3], x[4])";
    */

    std::string odir_main = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/";

    //RooPlot *plot_vs_width = width.frame();

    //textfile_path_os, "crystalball_gauss_os", crystalball_gaussFormula_os, qcd_os
    //shape_calc_signal(&args_ss, & args_ss, std::string textfile_path_ss, std::string fit_string_ss, std::string formula_name, std::string formula_string, bin_bkg_types process_sign)    

    /*
    string fit_string_ss[4] = {"norm","sigma","alpha","power"};
    string fit_string_os[4] = {"norm","sigma","alpha","power"};
    */
    string fit_string_ss[3] = {"sigma","alpha","power"};
    string fit_string_os[3] = {"sigma","alpha","power"};

    bin_signal_types signal_ss("ss","",0);
    bin_signal_types signal_os("os","",1);
    
    //RooArgList *args_ss[4], *args_os[4];
    RooArgList *args_ss[3], *args_os[3];

    //for(int i=0;i<4;i++)
    for(int i=0;i<3;i++)
    {    
        string fit_par_ss = fit_string_ss[i];
        string fit_par_os = fit_string_os[i];

        string canvas_title = "c_norm";
        TCanvas *c1 = new TCanvas(canvas_title.c_str(),canvas_title.c_str(), 700,500);

        //string title_intermediate_ss = fit_par_ss + "_same_sign";
        string title_intermediate_ss = fit_par_ss + "_ss_signal";
        formula_par_attr_signal formula_fit_params_ss = read_text_file_signal(fit_par_ss,odir_main + "0612_" + fit_par_ss + "_vs_width_with_errors_same_sign_root_original.txt");

        //RooArgList args = get_intermediate_fit_pars_signal(formula_fit_params, fit_par);
        //args_ss[i] = new RooArgList args_ss[i](width);
        args_ss[i] = new RooArgList (width);
        //args_os[i] = new RooArgList args_os[i](width);
        args_os[i] = new RooArgList (width);
        
        std::cout << "Args ss size before calling get_intermediate_fit_pars_signal in the loop " << args_ss[i]->size() << std::endl;
        //get_intermediate_fit_pars_signal(formula_fit_params_ss, fit_par_ss, &args_ss, &args_ss, signal_ss);
        get_intermediate_fit_pars_signal(&formula_fit_params_ss, fit_par_ss, args_ss[i], signal_ss);
        std::cout << "Args ss size after calling get_intermediate_fit_pars_signal in the loop " << args_ss[i]->size() << std::endl;
        std::cout << "Formula string " << formula_fit_params_ss.formula_string << std::endl;
        variable_intermediate_ss[i] = new RooFormulaVar(title_intermediate_ss.c_str(), title_intermediate_ss.c_str(), formula_fit_params_ss.formula_string.c_str(), RooArgList(*args_ss[i]));

        std::cout << " Just after ss RooFormulaVar" << std::endl;
        //norm_os.plotOn(plot_vs_width);
        RooPlot *plot_vs_width = width.frame();
        variable_intermediate_ss[i]->plotOn(plot_vs_width);
        plot_vs_width->Draw();

        string title_intermediate_os = fit_par_os + "_os_signal";
        formula_par_attr_signal formula_fit_params_os = read_text_file_signal(fit_par_os,odir_main + "0611_" + fit_par_os + "_vs_width_with_errors_opp_sign_root.txt");

        //RooArgList args = get_intermediate_fit_pars_signal(formula_fit_params, fit_par);
        
        std::cout << "Args os size before calling get_intermediate_fit_pars_signal in the loop " << args_os[i]->size() << std::endl;
        //get_intermediate_fit_pars_signal(formula_fit_params_os, fit_par_os, &args_os[i], &args_ss[i], signal_os);
        get_intermediate_fit_pars_signal(&formula_fit_params_os, fit_par_os, args_os[i], signal_os);
        std::cout << "Args os size after calling get_intermediate_fit_pars_signal in the loop " << args_os[i]->size() << std::endl;
        
        variable_intermediate_os[i] = new RooFormulaVar(title_intermediate_os.c_str(), title_intermediate_os.c_str(), formula_fit_params_os.formula_string.c_str(), RooArgList(*args_os[i]));

        std::cout << " Just after os RooFormulaVar" << std::endl;
        //norm_os.plotOn(plot_vs_width);
        //RooPlot *plot_vs_width = width.frame();
        //plot_vs_width = width.frame();
        variable_intermediate_os[i]->plotOn(plot_vs_width);
        
        plot_vs_width->Draw();
        c1->Draw();
        //c1->SaveAs("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/" + "norm" + "_vs_width_os_parametric_cc_header_file.png");
        string os_path = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/" + fit_par_os + "_vs_width_os_parametric_cc_header_file.png";
        c1->SaveAs(os_path.c_str());
        c1->Close();
            

        //args_intermediate.add(*variable_intermediate[i]); //Remove this out of the loop most probably
    }
    //*/

    /*
    RooFormulaVar *norm_os_signal, *sigma_os_signal, *alpha_os_signal, *power_os_signal;
    RooFormulaVar *norm_ss_signal, *sigma_ss_signal, *alpha_ss_signal, *power_ss_signal;
    */
    RooFormulaVar *sigma_os_signal, *alpha_os_signal, *power_os_signal;
    RooFormulaVar *sigma_ss_signal, *alpha_ss_signal, *power_ss_signal;

    //for(int i=0;i<4;i++)
    for(int i=0;i<3;i++)
    {
        string fit_par_ss = fit_string_ss[i];
        //string title_intermediate_ss = fit_par_ss + "_same_sign";
        string title_intermediate_ss = fit_par_ss + "_ss_signal";
        formula_par_attr_signal formula_fit_params_ss = read_text_file_signal(fit_par_ss,"/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/0612_" + fit_par_ss + "_vs_width_with_errors_same_sign_root_original.txt");
        get_intermediate_fit_pars_signal(&formula_fit_params_ss, fit_par_ss, args_ss[i], signal_ss);

        /*
        if(i==0)
        {
            std::cout << "title_intermediate_ss " << title_intermediate_ss << std::endl;
            norm_ss_signal = new RooFormulaVar(title_intermediate_ss.c_str(), title_intermediate_ss.c_str(), formula_fit_params_ss.formula_string.c_str(), RooArgList(*args_ss[i]));
            args_ss_final.add(*norm_ss_signal);
            args_ss_final.add(mean_mass_ss_signal);
        }
        */
        if(i==0)
        {
            sigma_ss_signal = new RooFormulaVar(title_intermediate_ss.c_str(), title_intermediate_ss.c_str(), formula_fit_params_ss.formula_string.c_str(), RooArgList(*args_ss[i]));
            args_ss_final.add(*sigma_ss_signal);

        }
        else if(i==1)
        {
            alpha_ss_signal = new RooFormulaVar(title_intermediate_ss.c_str(), title_intermediate_ss.c_str(), formula_fit_params_ss.formula_string.c_str(), RooArgList(*args_ss[i]));
            args_ss_final.add(*alpha_ss_signal);
        }
        else if(i==2)
        {
            power_ss_signal = new RooFormulaVar(title_intermediate_ss.c_str(), title_intermediate_ss.c_str(), formula_fit_params_ss.formula_string.c_str(), RooArgList(*args_ss[i]));
            args_ss_final.add(*power_ss_signal);
        }

    }

    ///*
    //for(int i=0;i<4;i++)
    for(int i=0;i<3;i++)
    {
        string fit_par_os = fit_string_os[i];
        string title_intermediate_os = fit_par_os + "_os_signal";
        formula_par_attr_signal formula_fit_params_os = read_text_file_signal(fit_par_os,"/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/0611_" + fit_par_os + "_vs_width_with_errors_opp_sign_root.txt");
        get_intermediate_fit_pars_signal(&formula_fit_params_os, fit_par_os, args_os[i], signal_os);

        /*
        if(i==0)
        {
            std::cout << "title_intermediate_os " << title_intermediate_os << std::endl;
            norm_os_signal = new RooFormulaVar(title_intermediate_os.c_str(), title_intermediate_os.c_str(), formula_fit_params_os.formula_string.c_str(), RooArgList(*args_os[i]));
            args_os_final.add(*norm_os_signal);
            args_os_final.add(mean_mass_os_signal);
        }
        */
        if(i==0)
        {
            sigma_os_signal = new RooFormulaVar(title_intermediate_os.c_str(), title_intermediate_os.c_str(), formula_fit_params_os.formula_string.c_str(), RooArgList(*args_os[i]));
            args_os_final.add(*sigma_os_signal);
        }
        else if(i==1)
        {
            alpha_os_signal = new RooFormulaVar(title_intermediate_os.c_str(), title_intermediate_os.c_str(), formula_fit_params_os.formula_string.c_str(), RooArgList(*args_os[i]));
            args_os_final.add(*alpha_os_signal);
        }
        else if(i==2)
        {
            power_os_signal = new RooFormulaVar(title_intermediate_os.c_str(), title_intermediate_os.c_str(), formula_fit_params_os.formula_string.c_str(), RooArgList(*args_os[i]));
            args_os_final.add(*power_os_signal);
        }

    }

    //RooFormulaVar crystalball{"crystalball", crystalballFormula, {args_intermediate}};
    std::cout << " Just before crystalball RooFormulaVar os" << std::endl;
    //RooFormulaVar crystalball_os{"crystalball_os_signal", crystalballFormula_os.c_str(), {mass, *norm_os_signal, mean_mass_os_signal, *sigma_os_signal, *alpha_os_signal, *power_os_signal}};
    
    //RooFormulaVar crystalball_os{"crystalball_os_signal", crystalballFormula_os.c_str(), {RooArgList(args_os_final)}};
    RooGenericPdf crystalball_os{"crystalball_os_signal", crystalballFormula_os.c_str(), {RooArgList(args_os_final)}};

    //RooFormulaVar crystalball{"crystalball", crystalballFormula, {mass, mean_mass_os_signal, norm_os_signal, sigma_os_signal, alpha_os_signal, power_os_signal}};
    std::cout << " Just after crystalball RooFormulaVar os" << std::endl;
    //*/

    std::cout << " Just before crystalball RooFormulaVar ss" << std::endl;
    std::cout << "args_ss_final size " << args_ss_final.size() << std::endl;
    //RooFormulaVar crystalball_os{"crystalball_os_signal", crystalballFormula_os.c_str(), {mass, *norm_os_signal, mean_mass_os_signal, *sigma_os_signal, *alpha_os_signal, *power_os_signal}};
    //RooFormulaVar crystalball_ss{"crystalball_ss_signal", crystalballFormula_ss.c_str(), {RooArgList(args_ss_final)}};
    RooGenericPdf crystalball_ss{"crystalball_ss_signal", crystalballFormula_ss.c_str(), {RooArgList(args_ss_final)}};
    
    //RooFormulaVar crystalball{"crystalball", crystalballFormula, {mass, mean_mass_os_signal, norm_os_signal, sigma_os_signal, alpha_os_signal, power_os_signal}};
    std::cout << " Just after crystalball RooFormulaVar ss" << std::endl;

    RooPlot *plot_vs_mass = mass.frame();
    std::vector<float> width_vals = {0.5,2.5,5.5,7.5,10.0};
    std::vector<Color_t> colors_net = {kRed,kBlue,kGreen,kOrange,kBlack};
    
    ///*
    string os_mass_path = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/0705_mass_vs_width_os_signal_parametric_abspdf.png";
    string ss_mass_path = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/0705_mass_vs_width_ss_signal_parametric_abspdf.png";
    //*/

    /*
    string os_mass_path = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/0705_mass_vs_width_os_signal_parametric_abspdf_crystalball.png";
    string ss_mass_path = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/0705_mass_vs_width_ss_signal_parametric_abspdf_crystalball.png";
    */

    /*
    TCanvas *c2 = new TCanvas("Parametric mass","Parametric mass", 700,500);

    TLegend *leg = new TLegend(0.3,0.7,0.45,0.85,NULL,"brNDC");
    leg->SetBorderSize(0);
    leg->SetLineColor(1);
    leg->SetLineStyle(1);
    leg->SetLineWidth(1);
    leg->SetFillColor(0);
    leg->SetFillStyle(1001);
    TLegendEntry *entry;

    for(int width_num=0;width_num<width_vals.size();width_num++)
    {
        float current_width = width_vals.at(width_num);
        width.setVal(current_width);
        //std::cout << "Norm for " << current_width << " GeV " << norm_os_signal->getVal() << " Sigma for " << current_width << " GeV " << sigma_os_signal->getVal() << " Alpha for " << current_width << " GeV " << alpha_os_signal->getVal() << " Power for " << current_width << " GeV " << power_os_signal->getVal() << std::endl;
        //std::cout << " Sigma for " << current_width << " GeV " << sigma_os_signal->getVal() << " Alpha for " << current_width << " GeV " << alpha_os_signal->getVal() << " Power for " << current_width << " GeV " << power_os_signal->getVal() << std::endl;
        crystalball_os.plotOn(plot_vs_mass);
        plot_vs_mass->getAttLine()->SetLineColor(colors_net.at(width_num));
        plot_vs_mass->Draw();

        std::stringstream stream_key;
        stream_key << std::fixed << std::setprecision(1) << current_width;
        std::string width_key = "Width - " + stream_key.str() + " GeV";

        std::string width_title = "width_" + std::to_string(width_num);
        //std::string width_key = "Width - " + std::to_string(current_width) + " GeV";

        entry=leg->AddEntry(width_title.c_str(),width_key.c_str(),"L");

        entry->SetLineColor(colors_net.at(width_num));
        entry->SetLineStyle(1);
        entry->SetLineWidth(1);
        entry->SetMarkerColor(1);
        entry->SetMarkerStyle(20);
        entry->SetMarkerSize(1.15);
        entry->SetTextFont(42);
        entry->SetTextSize(0.02);

    }

    leg->Draw();
    c2->Draw();

    c2->SaveAs(os_mass_path.c_str());
    c2->Close();


    TCanvas *c4 = new TCanvas("Parametric mass","Parametric mass", 700,500);
    plot_vs_mass = mass.frame();
    leg = new TLegend(0.3,0.7,0.45,0.85,NULL,"brNDC");
    leg->SetBorderSize(0);
    leg->SetLineColor(1);
    leg->SetLineStyle(1);
    leg->SetLineWidth(1);
    leg->SetFillColor(0);
    leg->SetFillStyle(1001);    
    for(int width_num=0;width_num<width_vals.size();width_num++)
    {
        float current_width = width_vals.at(width_num);
        width.setVal(current_width);
        //std::cout << "Norm for " << current_width << " GeV " << norm_ss_signal->getVal() << " Sigma for " << current_width << " GeV " << sigma_ss_signal->getVal() << " Alpha for " << current_width << " GeV " << alpha_ss_signal->getVal() << " Power for " << current_width << " GeV " << power_ss_signal->getVal() << std::endl;
        crystalball_ss.plotOn(plot_vs_mass);
        plot_vs_mass->getAttLine()->SetLineColor(colors_net.at(width_num));
        plot_vs_mass->Draw();

        std::stringstream stream_key;
        stream_key << std::fixed << std::setprecision(1) << current_width;
        std::string width_key = "Width - " + stream_key.str() + " GeV";

        std::string width_title = "width_" + std::to_string(width_num);
        //std::string width_key = "Width - " + std::to_string(current_width) + " GeV";

        entry=leg->AddEntry(width_title.c_str(),width_key.c_str(),"L");
        entry->SetLineColor(colors_net.at(width_num));
        entry->SetLineStyle(1);
        entry->SetLineWidth(1);
        entry->SetMarkerColor(1);
        entry->SetMarkerStyle(20);
        entry->SetMarkerSize(1.15);
        entry->SetTextFont(42);
        entry->SetTextSize(0.02);

    }

    leg->Draw();
    c4->Draw();

    c4->SaveAs(ss_mass_path.c_str());
    c4->Close();
    */

    ///*

    //Only for debugging
    //mass.setVal(172);

    //Adding final norm
    RooArgList *args_ss_norm = new RooArgList (width);
    RooArgList *args_os_norm = new RooArgList (width);

    formula_par_attr_signal formula_final_norm_ss = read_text_file_signal("norm","/afs/desy.de/user/p/paranjpe/top_wbwb/pepper/scripts/0705_area_cutflow_same_sign_norm_vs_width_signal.txt");
    get_intermediate_fit_pars_signal_norm(&formula_final_norm_ss, args_ss_norm, args_ss_norm, signal_ss);
    RooFormulaVar *area_norm_ss_signal = new RooFormulaVar("crystalball_ss_signal_norm", "crystalball_ss_signal_norm", formula_final_norm_ss.formula_string.c_str(), RooArgList(*args_ss_norm));

    formula_par_attr_signal formula_final_norm_os = read_text_file_signal("norm","/afs/desy.de/user/p/paranjpe/top_wbwb/pepper/scripts/0705_area_cutflow_opp_sign_norm_vs_width_signal.txt");
    get_intermediate_fit_pars_signal_norm(&formula_final_norm_os, args_os_norm, args_ss_norm, signal_os);
    RooFormulaVar *area_norm_os_signal = new RooFormulaVar("crystalball_os_signal_norm", "crystalball_os_signal_norm", formula_final_norm_os.formula_string.c_str(), RooArgList(*args_os_norm));

    std::cout << "Just before making the vector and plotting area norm" << std::endl;
    std::vector<RooFormulaVar> norm_net = {*area_norm_os_signal, *area_norm_ss_signal};

    
    string canvas_norm = "c_area_norm";
    TCanvas *c5 = new TCanvas(canvas_norm.c_str(),canvas_norm.c_str(), 700,500);
    RooPlot *plot_norm_vs_width = width.frame();
    area_norm_ss_signal->plotOn(plot_norm_vs_width);
    
    plot_norm_vs_width->Draw();
    c5->Draw();
    //c5->SaveAs("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/" + "norm" + "_vs_width_os_parametric_cc_header_file.png");
    string ss_path = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/area_norm_vs_width_ss_parametric_cc_header_file.png";
    c5->SaveAs(ss_path.c_str());
    c5->Close();

    TCanvas *c6 = new TCanvas(canvas_norm.c_str(),canvas_norm.c_str(), 700,500);
    plot_norm_vs_width = width.frame();
    area_norm_os_signal->plotOn(plot_norm_vs_width);
    
    plot_norm_vs_width->Draw();
    c6->Draw();
    string os_path = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/area_norm_vs_width_os_parametric_cc_header_file.png";
    c6->SaveAs(os_path.c_str());
    c6->Close();

    //Not at all sure about these settings
    width.setConstant(false); //This is the POI - ideally we want it to remain floating?
    //norm_os_signal->setConstant(false); //Presumably because we are including signal xsec uncertainty

    //Not sure how to constrain the other parameters since (no shape uncertainty at the moment) but width is still varying
    //Saving both in the same workspace since the names of the other variables are different anyway
    ///*
    //std::string workspace_dir = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/workspace_datacard_0621/";
    //std::string workspace_dir = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/workspace_datacard_0628_gauss_only/";
    std::string workspace_dir = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/workspace_datacard_0701_gauss_pdf/";
    std::string workspace_file_os_ss = workspace_dir + "workspace_signal_os_ss.root";

    //std::cout << mass.getVal() << std::endl;
    //std::vector<RooFormulaVar> final_shape_net;
    std::vector<RooGenericPdf> final_shape_net;
    final_shape_net.push_back(crystalball_os);
    final_shape_net.push_back(crystalball_ss);
    //save_workspace<RooGenericPdf>(final_shape_net, workspace_file_os_ss, "workspace_signal_os_ss");
    //save_workspace<RooGenericPdf>(final_shape_net, norm_net, workspace_file_os_ss, "workspace_signal_os_ss");
    save_workspace<RooGenericPdf, RooFormulaVar>(final_shape_net, norm_net, workspace_file_os_ss, "workspace_signal_os_ss");

    /*
    RooAbsReal* integral_os = crystalball_os.createIntegral(RooArgSet(mass));
    std::cout << "integral_os " << *integral_os << std::endl;
    */

    /*=======================================================Histogram for calculating integral=========================================================*/
    /*
    width.setVal(10);
    int num_bins = 100000;
    float upper_limit = 1000;
    float lower_limit = 0;
    float binsize = (upper_limit-lower_limit)/(num_bins); //Assuming equal binsize at the moment

    TH1F *os_vs_mass_bins = new TH1F("os_vs_mass_bins","bin heights for opp sign signal",num_bins,lower_limit,upper_limit);
    TH1F *ss_vs_mass_bins = new TH1F("ss_vs_mass_bins","bin heights for same sign signal",num_bins,lower_limit,upper_limit);
    for (int i = 0; i < num_bins; i++) 
    {
        mass.setVal((2*i+1)*binsize/2.0);
        float current_shape_os_val = crystalball_os.getVal();
        os_vs_mass_bins->SetBinContent(i+1,current_shape_os_val);
        std::cout << "Number in opp sign " << i << std::endl;
        std::cout << "mass value in opp sign " << mass.getVal() << std::endl;
        std::cout << "bin height in opp sign " << os_vs_mass_bins->GetBinContent(i+1) << std::endl;

        float current_shape_ss_val = crystalball_ss.getVal();
        ss_vs_mass_bins->SetBinContent(i+1,current_shape_ss_val);
        std::cout << "Number in same sign " << i << std::endl;
        std::cout << "mass value in same sign " << mass.getVal() << std::endl;
        std::cout << "bin height in same sign " << ss_vs_mass_bins->GetBinContent(i+1) << std::endl;

    }

    //Calculate integral to check whether it is matching in magnitude at least with the RooRombergIntegrator (latter gives a negative rate)
    float integral_ss = ss_vs_mass_bins->Integral();
    float integral_os = os_vs_mass_bins->Integral();

    std::cout << "Integral of os with binsize " << binsize << " GeV " << integral_os << std::endl;
    std::cout << "Integral of ss with binsize " << binsize << " GeV " << integral_ss << std::endl;
    */
    //=====================================================================================================================================================

    //*/

    gApplication->Terminate();
}
