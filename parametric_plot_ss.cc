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

void print_vector_ft(vector<float> const &ptrit) //function to print a vector of floats (floating point numbers)
{
 for(unsigned int i=0; i<ptrit.size(); i++)
 {
  std::cout << ptrit.at(i) << ' ';
 }
 std::cout << std::endl;
}

void print_vector_str(vector<string> const &ptrit) //function to print a vector of floats (floating point numbers)
{
 for(unsigned int i=0; i<ptrit.size(); i++)
 {
  std::cout << ptrit.at(i) << ' ';
 }
 std::cout << std::endl;
}


double reverse_crystalball(double x, double mean, double norm, double sigma, double alpha, double n)
{
    //Sign reversed since we want positive tail
    double arg = (x-mean)/sigma;
    double abs_alpha = abs(alpha);
    double reverse_crystalball;
    if (arg <= -alpha)
    {
        reverse_crystalball = norm*exp(-0.5*pow(arg,2));
    }
    else if (arg > -alpha)
    {
        ///*
        double nDivAlpha = n/abs_alpha;
        double AA =  exp(-0.5 * pow(abs_alpha,2));
        double B = nDivAlpha - abs_alpha;
        //arg_exp = nDivAlpha/(B-arg)
        double arg_exp = nDivAlpha/(B+arg);
        reverse_crystalball = norm*AA * (pow(arg_exp,n));
        //*/

        //reverse_crystalball = 0; //Only for debugging
    }
    return reverse_crystalball;
}

double num_string_check(string potential_float)
{
    //std::cout << std::stof("0.45") << std::endl;
    bool float_flag = 0;
    
    try
    {
        //std::cout << std::stof(potential_float) << std::endl;
        std::stof(potential_float);
        float_flag = 1;
    }
    catch(invalid_argument e)
    {
        //std::cout << "Caught exception: " << e.what() << std::endl;
        float_flag = 0;
    }
    
    //std::cout << "Final float_flag value " << float_flag << std::endl;
    return float_flag;
}

string find_replace_pars(string formula_string, std::vector<string> par_names, std::vector<string> par_index, int vals_size)
{
    //Find and replace each instance of each parameter mention by the corresponding @index
    size_t pos_x = formula_string.find("x[0]");
    if (pos_x != string::npos) 
    {
        formula_string.replace(pos_x, 4, "@0");
    }
    if(par_index.size() == par_names.size() && par_names.size() == vals_size)
    {
        std::cout << "Proceeding with the fit " << std::endl;
        for(int j=0;j<par_index.size();j++)
        {
            string find_string = par_names.at(j);
            string replace_string = "@" + par_index.at(j);
            //Copied directly from the standard code of finding all occurrences
            size_t pos = formula_string.find(find_string);

            // Iterate through the string and replace all occurrences
            while (pos != string::npos) 
            {
                std::cout << "Position of " << find_string << " is " << pos << std::endl;
                // Replace the substring with the specified string
                formula_string.replace(pos, find_string.size(), replace_string);

                std::cout << "Formula after replacing one occurrence " << formula_string << std::endl;
                // Find the next occurrence of the substring
                pos = formula_string.find(find_string,pos + replace_string.size()); //So here they are telling to only search again in the 'rest of the string'
            }            
        }
    }
    else
    {
        std::cout << "Parameter values, indices, and names not properly populated " << std::endl;
    }

    //std::cout << "Final formula after all the variable names with indices " << formula_string << std::endl;
    return formula_string;
}

struct formula_par_attr
{
    string formula_string;
    std::vector<string> par_names;
    std::vector<float> par_vals;
    std::vector<string> par_index;        
};


auto read_text_file(string fit_par, string text_file_path)
{
    string myText;
    //Starting simple - trying to read from one of the text files with fit parameters
    ifstream fitfile(text_file_path);
    //ifstream fitfile("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/0612_norm_vs_width_with_errors_same_sign_root_bkp.txt");

    std::vector<std::vector<string>> textfile; //Basically similar to the readlines in python, each line is also a vector of the separated words
    // Use a while loop together with the getline() function to read the file line by line
    while (getline (fitfile, myText)) 
    {
        // Output the text from the file
        std::cout << "Current line" << std::endl;
        std::cout << myText;
        //char* word = strtok(myText, " ");
        //char* word = strtok(myText.c_str(), " ");
        char arr[myText.length()+1];
        char* word = strtok(strcpy(arr, myText.c_str()), " ");
        std::vector<string> current_line;
        while (word != NULL)
        {
            std::cout << std::endl;
            std::cout << "word" << std::endl;
            std::cout << word;
            current_line.push_back(word);
            word = strtok (NULL, " ");
        }    
        std::cout << std::endl;
        textfile.push_back(current_line);
    }

    std::cout << "Printing the contents of the vector after reading with strtok " << std::endl;
    bool formula_flag = 0;
    int i_formula = 0;
    std::vector<string> formula;
    for(int i=0;i<textfile.size();i++)
    {
        if(textfile.at(i).size()>=2)
        {
            if(textfile.at(i).at(0)== "Fitting" && textfile.at(i).at(1)== "formula")
            {
                formula_flag = 1;
                i_formula = i;
                for(int j=0;j<textfile.at(i).size();j++)
                {
                    std::cout << textfile.at(i).at(j) << " ";
                }
                std::cout << std::endl;
            }
            else
            {
                formula_flag = 0;
            }
        }
        if(formula_flag==1)
        {
            copy(textfile.at(i).begin(), textfile.at(i).end(), back_inserter(formula));
            formula.erase(formula.begin(), formula.begin() + 2);            
        }

    }


    formula_par_attr return_objects;
    return_objects.formula_string = "";

    //There is supposed to be only one formula
    std::cout << "Line number with formula " << i_formula << std::endl;
    std::cout << "Final formula with strings" << std::endl;
    //string formula_string = "";
    for(int j=0;j<formula.size();j++)
    {
        //std::cout << formula.at(j) << " ";
        return_objects.formula_string += formula.at(j);
    }
    std::cout << "Formula string as a whole string (not broken into vector of strings) " << return_objects.formula_string << std::endl;

    /*
    std::vector<string> par_names;
    std::vector<float> par_vals;

    //Still not decided whether to keep this as a string or convert it to int
    //std::vector<int> par_index;
    std::vector<string> par_index;
    */

    for(int i=i_formula+1;i<textfile.size();i++)
    {
        std::vector<string> fit_line = textfile.at(i);
        std::cout << "Lines after the formula " << i_formula << std::endl;
        for(int j=0;j<fit_line.size();j++)
        {
            std::cout << fit_line.at(j) << " ";
            //bool float_flag = num_string_check(fit_line.at(j));
        }
        std::cout << std::endl;

        if(fit_line.size()>=3)
        {
            if(num_string_check(fit_line.at(0))==1 && num_string_check(fit_line.at(2))==1 && num_string_check(fit_line.at(1))==0)
            {
                std::cout << "fit_line value " << std::endl;
                std::cout << fit_line.at(2) << std::endl;
                return_objects.par_names.push_back(fit_line.at(1));
                return_objects.par_vals.push_back(std::stof(fit_line.at(2)));

                //Still not decided whether to keep this as a string or convert it to int
                //par_index.push_back(std::stoi(fit_line.at(0)));
                return_objects.par_index.push_back(fit_line.at(0));
            }
        }

    }
    // Close the file
    fitfile.close();

    return_objects.formula_string = find_replace_pars(return_objects.formula_string, return_objects.par_names, return_objects.par_index, return_objects.par_vals.size());
    std::cout << "Final formula after all the variable names with indices outside the function " << return_objects.formula_string << std::endl;

    //std::cout << std::endl;
    return return_objects;
}

//RooArgList get_intermediate_fit_pars(formula_par_attr formula_fit_params, string fit_par, RooArgList *args)
void get_intermediate_fit_pars(formula_par_attr formula_fit_params, string fit_par, RooArgList *args)
{
    //RooRealVar width("width","width", 0.5, 10);
    //RooArgList args(width);

    //====================================================================Lots of spew==============================================================================

    std::cout << "formula_fit_params formula " << formula_fit_params.formula_string << std::endl;
    std::cout << "formula_fit_params par_index " << std::endl;
    print_vector_str(formula_fit_params.par_index);
    std::cout << "formula_fit_params par_names " << std::endl;
    print_vector_str(formula_fit_params.par_names);
    std::cout << "formula_fit_params par_vals " << std::endl;
    print_vector_ft(formula_fit_params.par_vals);

    //===============================================================================================================================================================

    
    RooRealVar *variable[formula_fit_params.par_names.size()];
    std::cout << " Just before ss RooFormulaVar" << std::endl;
    if(formula_fit_params.par_index.size() == formula_fit_params.par_names.size() && formula_fit_params.par_names.size() == formula_fit_params.par_vals.size())
    {
        std::cout << "Proceeding with the fit and adding to the Rooarglist (only this part in a loop)" << std::endl;
        for(int j=0;j<formula_fit_params.par_index.size();j++)
        {
            string par_name_var = formula_fit_params.par_names.at(j) + "_" + fit_par;
            //According to the explanation in https://root-forum.cern.ch/t/array-of-roorealvars/29350 RooRealVar object is not meant to be fully copyable, so that is why the pointer array
            variable[j] = new RooRealVar(par_name_var.c_str(), par_name_var.c_str(), formula_fit_params.par_vals.at(j));
            //args.add(*variable[j]); //Dereferencing the pointer
            args->add(*variable[j]); //Dereferencing the pointer
            std::cout << " Variable added " << par_name_var << std::endl;
        }
    }
    else
    {
        std::cout << "Parameter values, indices, and names not properly populated " << std::endl;
    }

    //string title_intermediate = "norm_same_sign";
    //RooFormulaVar *variable_intermediate = new RooFormulaVar(title_intermediate.c_str(), title_intermediate.c_str(), formula_fit_params.formula_string.c_str(), RooArgList(*args));

    //return args;
}


auto reverse_crystalballFormula = "reverse_crystalball(x[0], x[1], x[2], x[3], x[4], x[5])";

void parametric_plot_ss()
{
    RooRealVar width("width","width", 0.5, 10);
    //RooArgList args(width);
    //RooRealVar mass{"mass", "mass", 0.0, 1000.0};
    RooRealVar mass("mass", "mass", 0.0, 1000.0);
    //RooRealVar mean_mass{"mean_mass", "mean_mass", 172.5};
    RooRealVar mean_mass("mean_mass", "mean_mass", 172.5);

    RooArgList args_intermediate(mass,mean_mass);

    //RooPlot *plot_vs_width = width.frame();

    string fit_string[4] = {"norm","sigma","alpha","power"};

    RooFormulaVar *variable_intermediate[4];

    //args = get_intermediate_fit_pars(formula_fit_params, "norm", args);
    /*
    std::cout << args.size() << std::endl;
    get_intermediate_fit_pars(formula_fit_params, "norm", &args);
    std::cout << args.size() << std::endl;
    string title_intermediate = "norm_same_sign";
    //variable_intermediate[0] = new RooFormulaVar(title_intermediate.c_str(), title_intermediate.c_str(), formula_fit_params.formula_string.c_str(), RooArgList(args));    
    //variable_intermediate[0] = new RooFormulaVar(title_intermediate.c_str(), title_intermediate.c_str(), formula_fit_params.formula_string.c_str(), RooArgList(args));    
    */
    ///*
    for(int i=0;i<4;i++)
    {    
        string fit_par = fit_string[i];

        string canvas_title = "c_norm";
        TCanvas *c1 = new TCanvas(canvas_title.c_str(),canvas_title.c_str(), 700,500);

        string title_intermediate = fit_par + "_same_sign";
        formula_par_attr formula_fit_params = read_text_file(fit_par,"/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/0612_" + fit_par + "_vs_width_with_errors_same_sign_root_original.txt");

        //RooArgList args = get_intermediate_fit_pars(formula_fit_params, fit_par);
        RooArgList args(width);
        std::cout << "Args before calling get_intermediate_fit_pars in the loop " << args.size() << std::endl;
        get_intermediate_fit_pars(formula_fit_params, fit_par, &args);
        std::cout << "Args before calling get_intermediate_fit_pars in the loop " << args.size() << std::endl;

        variable_intermediate[i] = new RooFormulaVar(title_intermediate.c_str(), title_intermediate.c_str(), formula_fit_params.formula_string.c_str(), RooArgList(args));


        std::cout << " Just after ss RooFormulaVar" << std::endl;
        //norm_ss.plotOn(plot_vs_width);
        RooPlot *plot_vs_width = width.frame();
        variable_intermediate[i]->plotOn(plot_vs_width);
        
        
        plot_vs_width->Draw();
        c1->Draw();
        //c1->SaveAs("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/" + "norm" + "_vs_width_ss_parametric_cc.png");
        string ss_path = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/" + fit_par + "_vs_width_ss_parametric_cc.png";
        c1->SaveAs(ss_path.c_str());
        c1->Close();
            

        //args_intermediate.add(*variable_intermediate[i]); //Remove this out of the loop most probably
    }
    //*/

    RooFormulaVar *norm_ss, *sigma_ss, *alpha_ss, *power_ss;
    for(int i=0;i<4;i++)
    {
        RooArgList args_inter(width);     
        string fit_par = fit_string[i];
        string title_intermediate = fit_par + "_same_sign";
        formula_par_attr formula_fit_params = read_text_file(fit_par,"/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/0612_" + fit_par + "_vs_width_with_errors_same_sign_root_original.txt");
        get_intermediate_fit_pars(formula_fit_params, fit_par, &args_inter);
        if(i==0)
        {
            norm_ss = new RooFormulaVar(title_intermediate.c_str(), title_intermediate.c_str(), formula_fit_params.formula_string.c_str(), RooArgList(args_inter));
        }
        else if(i==1)
        {
            sigma_ss = new RooFormulaVar(title_intermediate.c_str(), title_intermediate.c_str(), formula_fit_params.formula_string.c_str(), RooArgList(args_inter));
        }
        else if(i==2)
        {
            alpha_ss = new RooFormulaVar(title_intermediate.c_str(), title_intermediate.c_str(), formula_fit_params.formula_string.c_str(), RooArgList(args_inter));
        }
        else if(i==3)
        {
            power_ss = new RooFormulaVar(title_intermediate.c_str(), title_intermediate.c_str(), formula_fit_params.formula_string.c_str(), RooArgList(args_inter));
        }

    }

    //RooFormulaVar reverse_crystalball{"reverse_crystalball", reverse_crystalballFormula, {args_intermediate}};
    std::cout << " Just before reverse_crystalball RooFormulaVar" << std::endl;
    RooFormulaVar reverse_crystalball{"reverse_crystalball", reverse_crystalballFormula, {mass, mean_mass, *norm_ss, *sigma_ss, *alpha_ss, *power_ss}};
    //RooFormulaVar reverse_crystalball{"reverse_crystalball", reverse_crystalballFormula, {mass, mean_mass, norm_ss, sigma_ss, alpha_ss, power_ss}};
    std::cout << " Just after reverse_crystalball RooFormulaVar" << std::endl;

    TCanvas *c2 = new TCanvas("Parametric mass","Parametric mass", 700,500);
    RooPlot *plot_vs_mass = mass.frame();
    width.setVal(0.5);
    reverse_crystalball.plotOn(plot_vs_mass);
    plot_vs_mass->Draw();

    width.setVal(10);
    reverse_crystalball.plotOn(plot_vs_mass);
    plot_vs_mass->Draw();

    c2->Draw();

    string ss_mass_path = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/mass_vs_width_ss_parametric_cc.png";
    c2->SaveAs(ss_mass_path.c_str());
    c2->Close();
    //*/
    /*
    //Values of constants taken from the spew when the width is set to 0.5 GeV
    RooRealVar mass{"mass", "mass", 0.0, 1000.0};
    RooRealVar mean_mass{"mean_mass", "mean_mass", 172.5};
    RooRealVar norm_ss{"norm_ss", "norm_ss", 16162.615598692373};
    RooRealVar sigma_ss{"sigma_ss", "sigma_ss", 25.813359533637403};
    RooRealVar alpha_ss{"alpha_ss", "alpha_ss", -1.1465359605047318};
    RooRealVar power_ss{"power_ss", "power_ss", 4.404586573118257};

    //model = ROOT.RooGenericPdf("reverse_crystalball", reverse_crystalballFormula, (mass, mean_mass, norm_ss, sigma_ss, alpha_ss, power_ss))
    //RooGenericPdf reverse_crystalball{"reverse_crystalball", reverse_crystalballFormula, {mass, mean_mass, norm_ss, sigma_ss, alpha_ss, power_ss}};
    //RooAbsPdf reverse_crystalball{"reverse_crystalball", reverse_crystalballFormula, {mass, mean_mass, norm_ss, sigma_ss, alpha_ss, power_ss}};
    RooFormulaVar reverse_crystalball{"reverse_crystalball", reverse_crystalballFormula, {mass, mean_mass, norm_ss, sigma_ss, alpha_ss, power_ss}};
    //reverse_crystalball->Print();

    //mass.setVal(172);
    //std::cout << reverse_crystalball.getVal() << std::endl;
    RooPlot *plot_vs_mass = mass.frame();
    //reverse_crystalball.plotOn(plot_vs_mass, RooFit.LineColor(3));
    reverse_crystalball.plotOn(plot_vs_mass);
    plot_vs_mass->Draw();
    */

    gApplication->Terminate();
}
