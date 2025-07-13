#ifndef parametric_signal_bkg
#define parametric_signal_bkg

#include <json/json.h>

void print_vector_ft(vector<float> const &ptrit) //function to print a vector of floats (floating point numbers)
{
 for(unsigned int i=0; i<ptrit.size(); i++)
 {
  std::cout << ptrit.at(i) << ' ';
 }
 std::cout << std::endl;
}


//Removing one function to see whether it complains
///*
void print_vector_str(vector<string> const &ptrit) //function to print a vector of floats (floating point numbers)
{
 for(unsigned int i=0; i<ptrit.size(); i++)
 {
  std::cout << ptrit.at(i) << ' ';
 }
 std::cout << std::endl;
}
//*/

double crystalball(double x, double norm, double mean, double sigma, double alpha, double n)
{
    //Sign reversed since we want positive tail
    double arg = (x-mean)/sigma;
    double abs_alpha = abs(alpha);
    double crystalball;
    if (arg <= -alpha)
    {
        crystalball = norm*exp(-0.5*pow(arg,2));
    }
    else if (arg > -alpha)
    {
        ///*
        double nDivAlpha = n/abs_alpha;
        double AA =  exp(-0.5 * pow(abs_alpha,2));
        double B = nDivAlpha - abs_alpha;
        //arg_exp = nDivAlpha/(B-arg)
        double arg_exp = nDivAlpha/(B+arg);
        crystalball = norm*AA * (pow(arg_exp,n));
        //*/

        //crystalball = 0; //Only for debugging
    }
    return crystalball;
}

//It is a major assumption at this point that the order of the other parameters (other than x) is the same as what was in the fitting script get_bin_hgts_ss_bkg.py
double crystalball_gauss(double x,double norm, double mean, double sigma, double alpha, double n, double r)
{
    //Sign reversed since we want positive tail
    double arg = (x-mean)/sigma;
    double arg_narrow = (x-mean)/r;
    double abs_alpha = abs(alpha);
    double crystal_ball_gauss;
    if (arg <= 0)
    {
        crystal_ball_gauss = norm*exp(- 0.5 * arg_narrow * arg_narrow);
    }
    else if (arg <= -alpha && arg > 0)
    {
        crystal_ball_gauss = norm*exp(- 0.5 * arg * arg);
    }           
    else if (arg > -alpha)
    {
        double nDivAlpha = n/abs_alpha;
        double AA =  exp(-0.5 * abs_alpha * abs_alpha);
        double B = nDivAlpha - abs_alpha;
        double arg_exp = nDivAlpha/(B+arg);
        crystal_ball_gauss = norm*AA * (pow(arg_exp,n));
    }
    return crystal_ball_gauss;
}    


double crystalball_no_norm(double x, double mean, double sigma, double alpha, double n)
{
    //Sign reversed since we want positive tail
    double arg = (x-mean)/sigma;
    double abs_alpha = abs(alpha);
    double crystalball_no_norm;
    if (arg <= -alpha)
    {
        crystalball_no_norm = exp(-0.5*pow(arg,2));
    }
    else if (arg > -alpha)
    {
        ///*
        double nDivAlpha = n/abs_alpha;
        double AA =  exp(-0.5 * pow(abs_alpha,2));
        double B = nDivAlpha - abs_alpha;
        //arg_exp = nDivAlpha/(B-arg)
        double arg_exp = nDivAlpha/(B+arg);
        crystalball_no_norm = AA * (pow(arg_exp,n));
        //*/

        //crystalball_no_norm = 0; //Only for debugging
    }
    return crystalball_no_norm;
}

//It is a major assumption at this point that the order of the other parameters (other than x) is the same as what was in the fitting script get_bin_hgts_ss_bkg.py
double crystalball_gauss_no_norm(double x, double mean, double sigma, double alpha, double n, double r)
{
    //Sign reversed since we want positive tail
    double arg = (x-mean)/sigma;
    double arg_narrow = (x-mean)/r;
    double abs_alpha = abs(alpha);
    double crystalball_gauss_no_norm;
    if (arg <= 0)
    {
        crystalball_gauss_no_norm = exp(- 0.5 * arg_narrow * arg_narrow);
    }
    else if (arg <= -alpha && arg > 0)
    {
        crystalball_gauss_no_norm = exp(- 0.5 * arg * arg);
    }           
    else if (arg > -alpha)
    {
        double nDivAlpha = n/abs_alpha;
        double AA =  exp(-0.5 * abs_alpha * abs_alpha);
        double B = nDivAlpha - abs_alpha;
        double arg_exp = nDivAlpha/(B+arg);
        crystalball_gauss_no_norm = AA * (pow(arg_exp,n));
    }
    return crystalball_gauss_no_norm;
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
    /*
    if (pos_x != string::npos) 
    {
        formula_string.replace(pos_x, 4, "@0");
    }
    */

    while (pos_x != string::npos) 
    {
        formula_string.replace(pos_x, 4, "@0");
        pos_x = formula_string.find("x[0]",pos_x + 2); //So here they are telling to only search again in the 'rest of the string'
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

string find_replace_pars_signal(string formula_string, int vals_size)
{
    //Find and replace each instance of each parameter mention by the corresponding @index
    std::cout << "Proceeding with the fit (just replacing the norm_os with norm_ss*(norm_os/norm_ss))" << std::endl;
    string replace_string;
    size_t pos;
    string find_string;

    for(int k=0;k<vals_size;k++)
    {
        if(k==1)
        {
            find_string = "@" + std::to_string(k);
            replace_string = "@" + std::to_string(k+vals_size); //Dummy crappy value so that it does not get replaced in the next loop
            pos = formula_string.find(find_string);

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

    //for(int k=0;k<vals_size;k++)
    for(int k=vals_size-1;k>=0;k--) //Start from the highest index instead of the lowest so that the replaced strings will not get updated again
    {
        if(k>1)
        {
            find_string = "@" + std::to_string(k);
            replace_string = "@" + std::to_string(k+1);
            //Copied directly from the standard code of finding all occurrences
            pos = formula_string.find(find_string);

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

    int k_dummy = 1;
    find_string = "@" + std::to_string(k_dummy+vals_size);
    replace_string = "@" + std::to_string(k_dummy) + "*@" + std::to_string(k_dummy+1);
    pos = formula_string.find(find_string);

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


    //std::cout << "Final formula after all the variable names with indices " << formula_string << std::endl;
    return formula_string;
}


//This is for the bkg but I am probably not going to rename it promptly
struct formula_par_attr
{
    //string formula_string;
    std::vector<string> par_names;
    std::vector<float> par_vals;
    std::vector<string> par_index;        
};

struct formula_par_attr_signal
{
    string formula_string;
    std::vector<string> par_names;
    std::vector<float> par_vals;
    std::vector<string> par_index;        
};

struct bin_bkg_types
{
    std::string sign; //"ss" or "os"
    std::string mass_window; //"onshell" or "offshell"
    std::string process_type; //signal or one of the three bkg groups
    bool os_shape_calc;
    bin_bkg_types(std::string sign_input, std::string mass_window_input, std::string process_type_input, bool os_shape_calc_input)
    {
        std::cout << "Initializing struct bin_bkg_types" << std::endl;
        sign = sign_input;
        mass_window = mass_window_input;
        process_type = process_type_input;
        os_shape_calc = os_shape_calc_input;

        std::cout << "sign " << sign << " mass_window " <<  mass_window << " process_type " << process_type << " whether to use previous args or not " << (os_shape_calc ? "true" : "false") << std::endl;
    }
};

struct bin_signal_types
{
    std::string sign; //"ss" or "os"
    std::string mass_window; //"onshell" or "offshell"
    bool os_shape_calc;
    bin_signal_types(std::string sign_input, std::string mass_window_input, bool os_shape_calc_input)
    {
        std::cout << "Initializing struct bin_signal_types" << std::endl;
        sign = sign_input;
        mass_window = mass_window_input;
        os_shape_calc = os_shape_calc_input;

        std::cout << "sign " << sign << " mass_window " <<  mass_window << " whether to use previous args or not " << (os_shape_calc ? "true" : "false") << std::endl;
    }
};

struct process_shapes
{
    //RooFormulaVar *shape_ss, *shape_os;
    std::string formula_name_os, formula_name_ss; //This will be crystalball, crystalball gauss, or whatever, no reverse added unfortunately
    std::string formula_args_os, formula_args_ss; //This is the list of the args in the formula

    //Adding extra formulae for norm addition in the asimov dataset
    std::string formula_args_os_asimov, formula_args_ss_asimov; //This is the list of the args in the formula

    std::string process_type;
    //process_shapes(std::string formula_name_os_input, std::string formula_name_ss_input, std::string formula_args_os_input, std::string formula_args_ss_input, std::string process_type_input)
    process_shapes(std::string formula_name_os_input, std::string formula_name_ss_input, std::string formula_args_os_input, std::string formula_args_ss_input, std::string formula_args_os_asimov_input, std::string formula_args_ss_asimov_input, std::string process_type_input)
    {
        std::cout << "Initializing struct process_shapes" << std::endl;
        formula_name_os = formula_name_os_input;
        formula_name_ss = formula_name_ss_input;
        formula_args_os = formula_args_os_input;
        formula_args_ss = formula_args_ss_input;

        formula_args_os_asimov = formula_args_os_asimov_input;
        formula_args_ss_asimov = formula_args_ss_asimov_input;

        process_type = process_type_input;

        std::cout << "formula_name_os " << formula_name_os << " formula_name_ss " <<  formula_name_ss << " formula_args_os " << formula_args_os 
        << " formula_args_ss " << formula_args_ss << " formula_args_os_asimov " << formula_args_os_asimov 
        << " formula_args_ss_asimov " << formula_args_ss_asimov << " process_type " << process_type  << std::endl;
    }    
};

std::vector<string> split_line(string full_line)
{
    char arr[full_line.length()+1];
    char* word = strtok(strcpy(arr, full_line.c_str()), " ");
    std::vector<string> current_line;
    while (word != NULL)
    {
        /*
        std::cout << std::endl;
        std::cout << "word" << std::endl;
        std::cout << word;
        */
        current_line.push_back(word);
        word = strtok (NULL, " ");
    }    
    //std::cout << std::endl;

    return current_line;
}

//This function is specifically for reading the bkg fit text file because the format is different from the signal text file
formula_par_attr read_text_file_bkg(string text_file_path)
{
    string myText;
    formula_par_attr return_objects;
    //Starting simple - trying to read from one of the text files with fit parameters
    ifstream fitfile(text_file_path);
    //ifstream fitfile("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_ttbar_tw_wjets_0508/0612_norm_vs_width_with_errors_same_sign_root_bkp.txt");

    std::vector<std::vector<string>> textfile; //Basically similar to the readlines in python, each line is also a vector of the separated words
    // Use a while loop together with the getline() function to read the file line by line
    int line_number = 0;
    while (getline (fitfile, myText)) 
    {
        line_number++;
        // Output the text from the file
        std::cout << "Current line" << std::endl;
        std::cout << myText;
        std::cout << std::endl;
        //char* word = strtok(myText, " ");
        //char* word = strtok(myText.c_str(), " ");

        std::vector<string> current_line = split_line(myText);
        textfile.push_back(current_line);
        if(current_line.size()>=1)
        {
            std::cout << "Current line first element (supposed to be FCN) " << current_line.at(0) << std::endl;
            //This assumes the first instance of FCN only, which is good for bkg,
            //and also suits us for the signal same sign looping over power but since we 
            //only want to pick the first one at this moment even though it overestimates the tail a little
            if (current_line.at(0).find("FCN=")!= string::npos)
            {
                int line_number_fcn = line_number;
                std::cout << "Current line number " << line_number_fcn << std::endl;
                print_vector_str(current_line);
                int line_next = 1;
                int line_skip = 3;

                //Not working the way I want it to (does not 'skip' the required number of lines and read from the next line after that)
                /*
                int line_next = 3;
                //3 lines the number of which will probably always remain constant irrespective of the number of fitting parameters
                //                     EDM=1.03053e-06    STRATEGY= 1      ERROR MATRIX ACCURATE
                //EXT PARAMETER                                   STEP         FIRST
                //NO.   NAME      VALUE            ERROR          SIZE      DERIVATIVE
                */
                /*
                line_number = line_number_fcn + line_skip;
                int line_number_fit = line_number;
                std::cout << "Current line number after skipping " << line_number_fit << std::endl;
                */
                
                while(true)
                {
                    std::cout << std::endl;
                    //line_next++;
                    int line_next_absolute = line_number + line_next;
                    //The only one equal sign is probably deliberate
                    //Extra set of parentheses to silence the 'assignment, not comparison' warning
                    bool fit_flag = 0;
                    std::vector<string> current_line_fit;
                    if((line_number = line_next_absolute))
                    {
                        getline (fitfile, myText);
                        std::cout << "Next line after FCN starting from the first fitting parameter " << line_number << std::endl;
                        std::cout << myText << std::endl;
                        current_line_fit = split_line(myText);
                        std::cout << "First element (supposed to be a fitting parameter index) " << current_line_fit.at(0) << std::endl;
                        
                        if(current_line_fit.size()>=3)
                        {
                            fit_flag = num_string_check(current_line_fit.at(0))==1 && num_string_check(current_line_fit.at(2))==1 && num_string_check(current_line_fit.at(1))==0;
                        }
                        //break;
                    }
                    if(line_next_absolute>line_number_fcn+3 && fit_flag == 0)
                    //if(line_next_absolute==line_number_fcn+15)
                    {
                        break;
                    }
                    else if(fit_flag==1)
                    {
                        std::cout << myText << std::endl;
                        std::cout << "First element (supposed to be a fitting parameter index) " << current_line_fit.at(0) << std::endl;
                        std::cout << "fit_flag " << fit_flag << std::endl;
                        return_objects.par_names.push_back(current_line_fit.at(1));
                        return_objects.par_vals.push_back(std::stof(current_line_fit.at(2)));
                        return_objects.par_index.push_back(current_line_fit.at(0));
                    }

                }
                break;
            }
        }
    }

    return return_objects;
    //return textfile;
}

formula_par_attr_signal read_text_file_signal(string fit_par, string text_file_path)
{
    string myText;
    //Starting simple - trying to read from one of the text files with fit parameters
    ifstream fitfile(text_file_path);
    //ifstream fitfile("/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/0611_norm_vs_width_with_errors_opp_sign_root_bkp.txt");

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


    formula_par_attr_signal return_objects;
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

//void get_intermediate_fit_pars_bkg(formula_par_attr formula_fit_params, RooArgList *args, RooArgList *args_previous, bin_bkg_types sign_process)
//void get_intermediate_fit_pars_bkg(formula_par_attr formula_fit_params, RooArgList *args, bin_bkg_types sign_process)
void get_intermediate_fit_pars_bkg(formula_par_attr formula_fit_params, RooArgList *args, bin_bkg_types sign_process)
{
    //Do absolutely nothing to args_previous
    //This is only going to be used for correlating the norm of ss and os (and therefore not going to be used if ss)

    //RooRealVar width("width","width", 0.5, 10);
    //RooArgList args(width);

    //====================================================================Lots of spew==============================================================================

    std::cout << "formula_fit_params par_index " << std::endl;
    print_vector_str(formula_fit_params.par_index);
    std::cout << "formula_fit_params par_names " << std::endl;
    print_vector_str(formula_fit_params.par_names);
    std::cout << "formula_fit_params par_vals " << std::endl;
    print_vector_ft(formula_fit_params.par_vals);

    //===============================================================================================================================================================

    
    RooRealVar *variable[formula_fit_params.par_names.size()];
    RooRealVar *os_ss_norm_ratio;
    std::cout << " Just before RooFormulaVar" << std::endl;
    if(formula_fit_params.par_index.size() == formula_fit_params.par_names.size() && formula_fit_params.par_names.size() == formula_fit_params.par_vals.size())
    {
        std::cout << "Proceeding with the fit and adding to the Rooarglist (only this part in a loop)" << std::endl;
        for(int j=0;j<formula_fit_params.par_index.size();j++)
        {
            //string par_name_var = formula_fit_params.par_names.at(j) + "_" + fit_par;
            //string par_name_var = formula_fit_params.par_names.at(j);
            string par_name_var = formula_fit_params.par_names.at(j) + "_" + sign_process.sign + "_" + sign_process.process_type;
            //According to the explanation in https://root-forum.cern.ch/t/array-of-roorealvars/29350 RooRealVar object is not meant to be fully copyable, so that is why the pointer array
            variable[j] = new RooRealVar(par_name_var.c_str(), par_name_var.c_str(), formula_fit_params.par_vals.at(j));
            //args.add(*variable[j]); //Dereferencing the pointer

            if(formula_fit_params.par_names.at(j) == "norm")
            {
                std::cout << "Not adding normalization (area normalization will be added separately)" << std::endl;
            }

            else
            {
                std::cout << "Parameter to be set constant " << par_name_var << std::endl;
                variable[j]->setConstant(true);

                args->add(*variable[j]); //Dereferencing the pointer
                std::cout << " Variable added " << par_name_var << std::endl;
            }
            //args->add(*variable[j]); //Dereferencing the pointer
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

void get_intermediate_fit_pars_bkg_asimov_norm(formula_par_attr formula_fit_params, float norm_asimov, RooArgList *args, bin_bkg_types sign_process)
{
    //Do absolutely nothing to args_previous
    //This is only going to be used for correlating the norm of ss and os (and therefore not going to be used if ss)

    //RooRealVar width("width","width", 0.5, 10);
    //RooArgList args(width);

    //====================================================================Lots of spew==============================================================================

    std::cout << "formula_fit_params par_index " << std::endl;
    print_vector_str(formula_fit_params.par_index);
    std::cout << "formula_fit_params par_names " << std::endl;
    print_vector_str(formula_fit_params.par_names);
    std::cout << "formula_fit_params par_vals " << std::endl;
    print_vector_ft(formula_fit_params.par_vals);

    //===============================================================================================================================================================

    
    RooRealVar *variable[formula_fit_params.par_names.size()];
    RooRealVar *norm_asimov_net;
    std::cout << " Just before RooFormulaVar" << std::endl;
    if(formula_fit_params.par_index.size() == formula_fit_params.par_names.size() && formula_fit_params.par_names.size() == formula_fit_params.par_vals.size())
    {
        std::cout << "Proceeding with the fit and adding to the Rooarglist (only this part in a loop)" << std::endl;
        for(int j=0;j<formula_fit_params.par_index.size();j++)
        {
            //string par_name_var = formula_fit_params.par_names.at(j) + "_" + fit_par;
            //string par_name_var = formula_fit_params.par_names.at(j);
            string par_name_var = formula_fit_params.par_names.at(j) + "_" + sign_process.sign + "_" + sign_process.process_type;
            //According to the explanation in https://root-forum.cern.ch/t/array-of-roorealvars/29350 RooRealVar object is not meant to be fully copyable, so that is why the pointer array
            variable[j] = new RooRealVar(par_name_var.c_str(), par_name_var.c_str(), formula_fit_params.par_vals.at(j));
            //args.add(*variable[j]); //Dereferencing the pointer

            if(formula_fit_params.par_names.at(j) == "norm")
            {
                std::cout << "Adding a different normalization for components of asimov dataset" << std::endl;
                std::string norm_asimov_name = "norm_asimov_" + sign_process.sign + "_" + sign_process.process_type;
                norm_asimov_net = new RooRealVar(norm_asimov_name.c_str(),norm_asimov_name.c_str(),norm_asimov);
                norm_asimov_net->setConstant(true);
                args->add(*norm_asimov_net);
            }

            else
            {
                std::cout << "Parameter to be set constant " << par_name_var << std::endl;
                variable[j]->setConstant(true);

                args->add(*variable[j]); //Dereferencing the pointer
                std::cout << " Variable added " << par_name_var << std::endl;
            }
            //args->add(*variable[j]); //Dereferencing the pointer
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



void get_intermediate_fit_pars_bkg_norm(formula_par_attr formula_fit_params, RooArgList *args, RooArgList *args_previous, bin_bkg_types sign_process)
{
    //Do absolutely nothing to args_previous
    //This is only going to be used for correlating the norm of ss and os (and therefore not going to be used if ss)

    //RooRealVar width("width","width", 0.5, 10);
    //RooArgList args(width);

    //====================================================================Lots of spew==============================================================================

    std::cout << "formula_fit_params par_index " << std::endl;
    print_vector_str(formula_fit_params.par_index);
    std::cout << "formula_fit_params par_names " << std::endl;
    print_vector_str(formula_fit_params.par_names);
    std::cout << "formula_fit_params par_vals " << std::endl;
    print_vector_ft(formula_fit_params.par_vals);

    //===============================================================================================================================================================

    
    RooRealVar *variable[formula_fit_params.par_names.size()];
    RooRealVar *os_ss_norm_ratio;
    std::cout << " Just before RooFormulaVar" << std::endl;
    if(formula_fit_params.par_index.size() == formula_fit_params.par_names.size() && formula_fit_params.par_names.size() == formula_fit_params.par_vals.size())
    {
        std::cout << "Proceeding with the fit and adding to the Rooarglist (only this part in a loop)" << std::endl;
        for(int j=0;j<formula_fit_params.par_index.size();j++)
        {
            //string par_name_var = formula_fit_params.par_names.at(j) + "_" + fit_par;
            //string par_name_var = formula_fit_params.par_names.at(j);
            string par_name_var = formula_fit_params.par_names.at(j) + "_" + sign_process.sign + "_" + sign_process.process_type;
            //According to the explanation in https://root-forum.cern.ch/t/array-of-roorealvars/29350 RooRealVar object is not meant to be fully copyable, so that is why the pointer array
            variable[j] = new RooRealVar(par_name_var.c_str(), par_name_var.c_str(), formula_fit_params.par_vals.at(j));
            //args.add(*variable[j]); //Dereferencing the pointer

            if(formula_fit_params.par_names.at(j) == "norm")
            {
                if(sign_process.os_shape_calc == 0)
                {
                    std::cout << "Parameter to be changed (not set constant) " << par_name_var << std::endl;
                    variable[j]->setConstant(false);
                    std::cout << "Value of current norm " << variable[j]->getVal() << std::endl;

                    args->add(*variable[j]); //Dereferencing the pointer
                }
                ///*
                else if (sign_process.os_shape_calc == 1)
                {
                    if (args_previous->size() > 1)
                    {
                        //This is some sort of casting since the args_previous->at(1) from the RooArgList is a RooAbsArg* or something
                        RooRealVar * args_ss_previous_norm = (RooRealVar*) (args_previous->at(1)); //1 is the index of the norm supposedly, have to rely on a specific order for this one
                        
                        float ss_norm = args_ss_previous_norm->getVal();
                        std::cout << "Supposed value from same sign norm " << ss_norm << std::endl;
                        float os_norm = variable[j]->getVal();
                        //variable[j]->setVal(ss_norm); //This variable will not be added separately, because we already have args_ss_previous_norm
                        std::string os_ss_ratio = "os_ss_norm_ratio_" + sign_process.process_type;
                        os_ss_norm_ratio = new RooRealVar(os_ss_ratio.c_str(), os_ss_ratio.c_str(), os_norm/ss_norm);

                        os_ss_norm_ratio->setConstant(true);
                        //args->add(*variable[j]); //Dereferencing the pointer
                        args->add(*args_ss_previous_norm); //Dereferencing the pointer
                        std::cout << " Variable added " << "args_ss_previous_norm" << std::endl;
                        args->add(*os_ss_norm_ratio);
                        std::cout << " Variable added " << "os_ss_norm_ratio" << std::endl;
                    }
                    
                }

                //*/
            }

            else
            {
                std::cout << "Parameter to be set constant " << par_name_var << std::endl;
                variable[j]->setConstant(true);

                args->add(*variable[j]); //Dereferencing the pointer
                std::cout << " Variable added " << par_name_var << std::endl;
            }
            //args->add(*variable[j]); //Dereferencing the pointer
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


void get_intermediate_fit_pars_signal(formula_par_attr_signal *formula_fit_params, string fit_par, RooArgList *args, bin_signal_types sign_process)
{
    //RooRealVar width("width","width", 0.5, 10);
    //RooArgList args(width);

    //====================================================================Lots of spew==============================================================================

    std::cout << "formula_fit_params formula " << formula_fit_params->formula_string << std::endl;
    std::cout << "formula_fit_params par_index " << std::endl;
    print_vector_str(formula_fit_params->par_index);
    std::cout << "formula_fit_params par_names " << std::endl;
    print_vector_str(formula_fit_params->par_names);
    std::cout << "formula_fit_params par_vals " << std::endl;
    print_vector_ft(formula_fit_params->par_vals);

    //===============================================================================================================================================================

    
    RooRealVar *variable[formula_fit_params->par_names.size()];
    RooRealVar *os_ss_norm_ratio;
    std::cout << " Just before os RooFormulaVar" << std::endl;
    if(formula_fit_params->par_index.size() == formula_fit_params->par_names.size() && formula_fit_params->par_names.size() == formula_fit_params->par_vals.size())
    {
        std::cout << "Proceeding with the fit and adding to the Rooarglist (only this part in a loop)" << std::endl;
        for(int j=0;j<formula_fit_params->par_index.size();j++)
        {

            string par_name_var = formula_fit_params->par_names.at(j) + "_" + fit_par + "_" + sign_process.sign;
            //According to the explanation in https://root-forum.cern.ch/t/array-of-roorealvars/29350 RooRealVar object is not meant to be fully copyable, so that is why the pointer array
            variable[j] = new RooRealVar(par_name_var.c_str(), par_name_var.c_str(), formula_fit_params->par_vals.at(j));
            std::cout << "Parameter to be set constant " << par_name_var << std::endl;
            variable[j]->setConstant(true);
            args->add(*variable[j]); //Dereferencing the pointer

        }

    }
    else
    {
        std::cout << "Parameter values, indices, and names not properly populated " << std::endl;
    }
}

void get_intermediate_fit_pars_signal_norm(formula_par_attr_signal *formula_fit_params, RooArgList *args, RooArgList *args_previous, bin_signal_types sign_process)
{
    //RooRealVar width("width","width", 0.5, 10);
    //RooArgList args(width);

    //====================================================================Lots of spew==============================================================================

    std::cout << "formula_fit_params formula " << formula_fit_params->formula_string << std::endl;
    std::cout << "formula_fit_params par_index " << std::endl;
    print_vector_str(formula_fit_params->par_index);
    std::cout << "formula_fit_params par_names " << std::endl;
    print_vector_str(formula_fit_params->par_names);
    std::cout << "formula_fit_params par_vals " << std::endl;
    print_vector_ft(formula_fit_params->par_vals);

    //===============================================================================================================================================================

    
    RooRealVar *variable[formula_fit_params->par_names.size()];
    RooRealVar *os_ss_norm_ratio;
    std::cout << " Just before os RooFormulaVar" << std::endl;
    if(formula_fit_params->par_index.size() == formula_fit_params->par_names.size() && formula_fit_params->par_names.size() == formula_fit_params->par_vals.size())
    {
        std::cout << "Proceeding with the fit and adding to the Rooarglist (only this part in a loop)" << std::endl;
        for(int j=0;j<formula_fit_params->par_index.size();j++)
        {
            
            string par_name_var = formula_fit_params->par_names.at(j) + "_area_norm_" + sign_process.sign;
            //According to the explanation in https://root-forum.cern.ch/t/array-of-roorealvars/29350 RooRealVar object is not meant to be fully copyable, so that is why the pointer array
            variable[j] = new RooRealVar(par_name_var.c_str(), par_name_var.c_str(), formula_fit_params->par_vals.at(j));

            //No need for this - fit_par == "norm"
            if(formula_fit_params->par_names.at(j) == "y_shift")
            {
                std::cout << "Par_names" << std::endl;
                print_vector_str(formula_fit_params->par_names);
                std::cout << "Formula_string " << formula_fit_params->formula_string << std::endl;
                
                if(sign_process.os_shape_calc == 0)
                {
                    std::cout << "Parameter to be changed (not set constant) " << par_name_var << std::endl;
                    //variable[j]->setConstant(false);
                    //Norm is to be provided separately so probably will get rid of this if condition anyway (or not because this is a part of the underlying formula, not the norm as a whole)
                    variable[j]->setConstant(false);
                    std::cout << "Value of current norm " << variable[j]->getVal() << std::endl;

                    args->add(*variable[j]); //Dereferencing the pointer
                }
                ///*
                else if (sign_process.os_shape_calc == 1)
                {
                    if (args_previous->size() > 1)
                    {
                        //This is some sort of casting since the args_previous->at(1) from the RooArgList is a RooAbsArg* or something
                        RooRealVar * args_ss_previous_norm = (RooRealVar*) (args_previous->at(1)); //1 is the index of the norm supposedly, have to rely on a specific order for this one
                        
                        float ss_norm = args_ss_previous_norm->getVal();
                        std::cout << "Supposed value from same sign norm " << ss_norm << std::endl;
                        float os_norm = variable[j]->getVal();
                        //variable[j]->setVal(ss_norm); //This variable will not be added separately, because we already have args_ss_previous_norm
                        std::string os_ss_ratio = "os_ss_norm_ratio";
                        os_ss_norm_ratio = new RooRealVar(os_ss_ratio.c_str(), os_ss_ratio.c_str(), os_norm/ss_norm);

                        os_ss_norm_ratio->setConstant(true);
                        //args->add(*variable[j]); //Dereferencing the pointer
                        args->add(*args_ss_previous_norm); //Dereferencing the pointer
                        std::cout << " Variable added " << "args_ss_previous_norm" << std::endl;
                        args->add(*os_ss_norm_ratio);
                        std::cout << " Variable added " << "os_ss_norm_ratio" << std::endl;
                    }
                }
            }
            else
            {
                std::cout << "Parameter to be set constant " << par_name_var << std::endl;
                variable[j]->setConstant(true);
                args->add(*variable[j]); //Dereferencing the pointer

            }

            std::cout << " Variable added " << par_name_var << std::endl;
            std::cout << "Value of current variable " << variable[j]->getVal() << std::endl;

        }

    }
    else
    {
        std::cout << "Parameter values, indices, and names not properly populated " << std::endl;
    }

    for(int j=0;j<formula_fit_params->par_index.size();j++)
    {
        //if ((fit_par == "norm") & (sign_process.os_shape_calc == 1) & (formula_fit_params->par_names.at(j) == "y_shift"))        
        if ((sign_process.os_shape_calc == 1) & (formula_fit_params->par_names.at(j) == "y_shift"))        
        {
            std::cout << "Formula_string before replacing with os_ss_norm_ratio " << formula_fit_params->formula_string << std::endl;
            std::cout << "Args size " << args->size() << std::endl;
            formula_fit_params->formula_string = find_replace_pars_signal(formula_fit_params->formula_string, args->size());
            std::cout << "Formula_string after  replacing with os_ss_norm_ratio " << formula_fit_params->formula_string << std::endl;

        }

    }

    //string title_intermediate = "norm_opp_sign";
    //RooFormulaVar *variable_intermediate = new RooFormulaVar(title_intermediate.c_str(), title_intermediate.c_str(), formula_fit_params.formula_string.c_str(), RooArgList(*args));

    //return args;
}




//RooFormulaVar shape_calc_bkg(RooArgList *args, RooArgList *args_previous, std::string textfile_path, std::string formula_name, std::string formula_string, bin_bkg_types process_sign)
//RooGenericPdf shape_calc_bkg(RooArgList *args, RooArgList *args_previous, std::string textfile_path, std::string formula_name, std::string formula_string, bin_bkg_types process_sign)
//auto shape_calc_bkg(RooArgList *args, RooArgList *args_previous, std::string textfile_path, std::string formula_name, std::string formula_string, bin_bkg_types process_sign)
//template <typename T> T shape_calc_bkg(RooArgList *args, RooArgList *args_previous, std::string textfile_path, std::string formula_name, std::string formula_string, bin_bkg_types process_sign)
template <typename T> T shape_calc_bkg(RooArgList *args, std::string textfile_path, std::string formula_name, std::string formula_string, bin_bkg_types process_sign)
{
    formula_par_attr formula_fit_params = read_text_file_bkg(textfile_path);
    std::cout << "formula_fit_params par_index " << std::endl;
    print_vector_str(formula_fit_params.par_index);
    std::cout << "formula_fit_params par_names " << std::endl;
    print_vector_str(formula_fit_params.par_names);
    std::cout << "formula_fit_params par_vals " << std::endl;
    print_vector_ft(formula_fit_params.par_vals);

    //get_intermediate_fit_pars_bkg(formula_fit_params, fit_par, &args);
    //get_intermediate_fit_pars_bkg(formula_fit_params, &args);

    //Since we are already passing a pointer, have to remove the &
    //get_intermediate_fit_pars_bkg(formula_fit_params, args);
    //get_intermediate_fit_pars_bkg(formula_fit_params, args, args_previous, process_sign);
    get_intermediate_fit_pars_bkg(formula_fit_params, args, process_sign);

    //Converting datatype of formula from auto to string (since we might require it to be passed in a function)
    //RooFormulaVar final_shape{formula_name.c_str(), formula_string.c_str(), {RooArgList(args)}};
    //RooFormulaVar final_shape{formula_name.c_str(), formula_string.c_str(), {RooArgList(*args)}};
    //RooFormulaVar final_shape{formula_name.c_str(), formula_string.c_str(), {RooArgList(*args)}};
    T final_shape{formula_name.c_str(), formula_string.c_str(), {RooArgList(*args)}};

    std::cout << " Just after os RooGenericPdf" << std::endl;

    return final_shape;
}

//This actually multiplies the shape with the 
template <typename T> T shape_calc_bkg_asimov_norm(RooArgList *args, float norm_true, float integral_calc, std::string textfile_path, std::string formula_name, std::string formula_string, bin_bkg_types process_sign)
{
    formula_par_attr formula_fit_params = read_text_file_bkg(textfile_path);
    std::cout << "formula_fit_params par_index " << std::endl;
    print_vector_str(formula_fit_params.par_index);
    std::cout << "formula_fit_params par_names " << std::endl;
    print_vector_str(formula_fit_params.par_names);
    std::cout << "formula_fit_params par_vals " << std::endl;
    print_vector_ft(formula_fit_params.par_vals);

    //get_intermediate_fit_pars_bkg(formula_fit_params, fit_par, &args);
    //get_intermediate_fit_pars_bkg(formula_fit_params, &args);

    //Since we are already passing a pointer, have to remove the &
    //get_intermediate_fit_pars_bkg(formula_fit_params, args);
    //get_intermediate_fit_pars_bkg(formula_fit_params, args, args_previous, process_sign);
    float norm_asimov = norm_true/integral_calc;
    get_intermediate_fit_pars_bkg_asimov_norm(formula_fit_params, norm_asimov, args, process_sign);

    //Converting datatype of formula from auto to string (since we might require it to be passed in a function)
    //RooFormulaVar final_shape{formula_name.c_str(), formula_string.c_str(), {RooArgList(args)}};
    //RooFormulaVar final_shape{formula_name.c_str(), formula_string.c_str(), {RooArgList(*args)}};
    //RooFormulaVar final_shape{formula_name.c_str(), formula_string.c_str(), {RooArgList(*args)}};
    T final_shape{formula_name.c_str(), formula_string.c_str(), {RooArgList(*args)}};

    std::cout << " Just after os RooGenericPdf" << std::endl;

    return final_shape;
}



/*
RooGenericPdf shape_calc_bkg(RooArgList *args, RooArgList *args_previous, std::string textfile_path, std::string formula_name, std::string formula_string, bin_bkg_types process_sign)
//RooGenericPdf shape_calc_bkg(RooArgList *args, RooArgList *args_previous, std::string textfile_path, std::string formula_name, std::string formula_string, bin_bkg_types process_sign)
//auto shape_calc_bkg(RooArgList *args, RooArgList *args_previous, std::string textfile_path, std::string formula_name, std::string formula_string, bin_bkg_types process_sign)
{
    formula_par_attr formula_fit_params = read_text_file_bkg(textfile_path);
    std::cout << "formula_fit_params par_index " << std::endl;
    print_vector_str(formula_fit_params.par_index);
    std::cout << "formula_fit_params par_names " << std::endl;
    print_vector_str(formula_fit_params.par_names);
    std::cout << "formula_fit_params par_vals " << std::endl;
    print_vector_ft(formula_fit_params.par_vals);

    //get_intermediate_fit_pars_bkg(formula_fit_params, fit_par, &args);
    //get_intermediate_fit_pars_bkg(formula_fit_params, &args);

    //Since we are already passing a pointer, have to remove the &
    //get_intermediate_fit_pars_bkg(formula_fit_params, args);
    get_intermediate_fit_pars_bkg(formula_fit_params, args, args_previous, process_sign);

    //Converting datatype of formula from auto to string (since we might require it to be passed in a function)
    //RooFormulaVar final_shape{formula_name.c_str(), formula_string.c_str(), {RooArgList(args)}};
    //RooFormulaVar final_shape{formula_name.c_str(), formula_string.c_str(), {RooArgList(*args)}};
    RooGenericPdf final_shape{formula_name.c_str(), formula_string.c_str(), {RooArgList(*args)}};

    std::cout << " Just after os RooGenericPdf" << std::endl;

    return final_shape;
}
*/

//std::vector<RooFormulaVar> shape_calc(process_shapes *process_details, RooRealVar *mass, std::string odir_main, bool signal_flag)
//std::vector<RooGenericPdf> shape_calc(process_shapes *process_details, RooRealVar *mass, std::string odir_main, bool signal_flag)
//template <typename T> std::vector<T> shape_calc(process_shapes *process_details, RooRealVar *mass, std::string odir_main, bool signal_flag)
template <typename T> std::vector<T> shape_calc(process_shapes *process_details, std::vector<float> norm_true, RooRealVar *mass, std::string odir_main, bool signal_flag)

//auto shape_calc(process_shapes *process_details, RooRealVar *mass, std::string odir_main, bool signal_flag)
{
    RooArgList args_ss(*mass);
    RooArgList args_os(*mass);
    //RooPlot *plot_vs_mass = mass.frame();
    std::string textfile_path_os, textfile_path_ss;

    if(signal_flag == 1)
    {
        //20GeV_binsize_0605_signal_crystalball_new_j2_b1_net_mass_ss_width_19.txt
        //20GeV_binsize_0605_signal_crystalball_new_j2_b1_net_mass_os_width_19.txt
        textfile_path_os = odir_main + "20GeV_binsize_0605_" + process_details->process_type + "_" + process_details->formula_name_os + "_new_j2_b1_net_mass_os_width_19.txt";
        textfile_path_ss = odir_main + "20GeV_binsize_0605_" + process_details->process_type + "_" + process_details->formula_name_ss + "_new_j2_b1_net_mass_ss_width_19.txt";
    }
    else if(signal_flag == 0)
    {
        textfile_path_os = odir_main + "0623_" + process_details->formula_name_os + "_j2_b1_net_mass_bkg_os_" + process_details->process_type + ".txt";
        textfile_path_ss = odir_main + "0623_" + process_details->formula_name_ss + "_j2_b1_net_mass_bkg_ss_" + process_details->process_type + ".txt";
    }

    std::cout << "textfile_path_os in shape_calc " << textfile_path_os << std::endl;
    std::cout << "textfile_path_ss in shape_calc " << textfile_path_ss << std::endl;

    /*
    std::string final_Formula_os = process_details->formula_name_os + process_details->formula_args_os;
    std::string final_Formula_ss = process_details->formula_name_ss + process_details->formula_args_ss;
    */
    //This is when the formula args is in the form of a string rather than a custom made function
    std::string final_Formula_os = process_details->formula_args_os;
    std::string final_Formula_ss = process_details->formula_args_ss;


    bin_bkg_types ss("ss","",process_details->process_type,0);
    std::cout << "Args before calling shape_ss " << args_ss.size() << std::endl;

    //RooFormulaVar shape_ss = shape_calc_bkg(&args_ss, & args_ss, textfile_path_ss, process_details->formula_name_ss + "_ss_" + process_details->process_type + "_asimov", final_Formula_ss, ss);
    //RooGenericPdf shape_ss = shape_calc_bkg(&args_ss, & args_ss, textfile_path_ss, process_details->formula_name_ss + "_ss_" + process_details->process_type + "_asimov", final_Formula_ss, ss);
    //T shape_ss = shape_calc_bkg<T>(&args_ss, & args_ss, textfile_path_ss, process_details->formula_name_ss + "_ss_" + process_details->process_type + "_asimov", final_Formula_ss, ss);
    T shape_ss = shape_calc_bkg<T>(&args_ss, textfile_path_ss, process_details->formula_name_ss + "_ss_" + process_details->process_type + "_asimov", final_Formula_ss, ss);

    //process_details->shape_ss = (RooFormulaVar*) (shape_calc_bkg(&args_ss, & args_ss, textfile_path_ss, process_details->formula_name_ss + "_ss_" + process_details->process_type, final_Formula_ss, ss));

    std::cout << "Args after calling shape_ss " << args_ss.size() << std::endl;
    
    bin_bkg_types os("os","",process_details->process_type,1);
    std::cout << "Args before calling shape_os " << args_os.size() << std::endl;

    //RooFormulaVar shape_os = shape_calc_bkg(&args_os, & args_ss, textfile_path_os, process_details->formula_name_os + "_os_" + process_details->process_type + "_asimov", final_Formula_os, os);
    //RooGenericPdf shape_os = shape_calc_bkg(&args_os, & args_ss, textfile_path_os, process_details->formula_name_os + "_os_" + process_details->process_type + "_asimov", final_Formula_os, os);
    //T shape_os = shape_calc_bkg<T>(&args_os, & args_ss, textfile_path_os, process_details->formula_name_os + "_os_" + process_details->process_type + "_asimov", final_Formula_os, os);
    T shape_os = shape_calc_bkg<T>(&args_os, textfile_path_os, process_details->formula_name_os + "_os_" + process_details->process_type + "_asimov", final_Formula_os, os);

    float int_ss = shape_ss.createIntegral(RooArgSet(*mass))->getVal();
    float int_os = shape_os.createIntegral(RooArgSet(*mass))->getVal();
    //Looking at values of createintegral for both shape_both_qcd and shape_both_qcd_pdf - actually probably only the RooFormulaVar one will suffice
    std::cout << "Integral for shape_ss " << int_ss << std::endl;
    std::cout << "Integral for shape_os " << int_os << std::endl;
    
    std::cout << "Args after calling shape_os " << args_os.size() << std::endl;   

    std::string formula_name_ss_final = "shape_ss_final_" + process_details->process_type + "_asimov";
    std::string formula_name_os_final = "shape_os_final_" + process_details->process_type + "_asimov";

    std::string formula_integral_ss = "hght_norm_integral_ss_" + process_details->process_type + "_asimov";
    std::string formula_integral_os = "hght_norm_integral_os_" + process_details->process_type + "_asimov";

    std::string formula_norm_net_ss = "norm_net_ss_" + process_details->process_type + "_asimov";
    std::string formula_norm_net_os = "norm_net_os_" + process_details->process_type + "_asimov";
    
    /*
    RooRealVar hght_norm_integral_ss("hght_norm_integral_ss", "hght_norm_integral_ss", int_ss);
    RooRealVar hght_norm_integral_os("hght_norm_integral_os", "hght_norm_integral_os", int_os);

    RooFormulaVar *norm_net_ss = new RooFormulaVar("norm_net_ss","norm_net_ss", "@0/@1", RooArgList(norm_true.at(1),hght_norm_integral_ss));
    RooFormulaVar *norm_net_os = new RooFormulaVar("norm_net_os","norm_net_os", "@0/@1", RooArgList(norm_true.at(0),hght_norm_integral_os));
    */

    RooRealVar hght_norm_integral_ss(formula_integral_ss.c_str(), formula_integral_ss.c_str(), int_ss);
    RooRealVar hght_norm_integral_os(formula_integral_os.c_str(), formula_integral_os.c_str(), int_os);

    /*
    RooFormulaVar *norm_net_ss = new RooFormulaVar(formula_norm_net_ss.c_str(),formula_norm_net_ss.c_str(), "@0/@1", RooArgList(norm_true.at(1),hght_norm_integral_ss));
    RooFormulaVar *norm_net_os = new RooFormulaVar(formula_norm_net_os.c_str(),formula_norm_net_os.c_str(), "@0/@1", RooArgList(norm_true.at(0),hght_norm_integral_os));
    */

    RooFormulaVar norm_net_ss(formula_norm_net_ss.c_str(),formula_norm_net_ss.c_str(), "1/@0", RooArgList(hght_norm_integral_ss));
    RooFormulaVar norm_net_os(formula_norm_net_os.c_str(),formula_norm_net_os.c_str(), "1/@0", RooArgList(hght_norm_integral_os));

    /*
    T shape_ss_final{formula_name_ss_final.c_str(), "@0*@1", {RooArgList(shape_ss,norm_net_ss)}};
    T shape_os_final{formula_name_ss_final.c_str(), "@0*@1", {RooArgList(shape_os,norm_net_os)}};
    */

    /*
    RooFormulaVar *shape_os_final = new RooFormulaVar(formula_name_os_final.c_str(), formula_name_os_final.c_str(), "@0", RooArgList(shape_os));
    RooFormulaVar *shape_ss_final = new RooFormulaVar(formula_name_ss_final.c_str(), formula_name_ss_final.c_str(), "@0", RooArgList(shape_ss));
    */

    /*
    T shape_os_final = shape_calc_bkg<T>(&args_os, textfile_path_os, process_details->formula_name_os + "_os_" + process_details->process_type + "_asimov", final_Formula_os, os);
    T shape_ss_final = shape_calc_bkg<T>(&args_ss, textfile_path_ss, process_details->formula_name_ss + "_ss_" + process_details->process_type + "_asimov", final_Formula_ss, ss);
    */

    RooArgList args_ss_new(*mass);
    RooArgList args_os_new(*mass);

    std::string final_Formula_os_asimov = process_details->formula_args_os_asimov;
    std::string final_Formula_ss_asimov = process_details->formula_args_ss_asimov;


    T shape_os_final = shape_calc_bkg_asimov_norm<T>(&args_os_new, norm_true.at(0), int_os, textfile_path_os, process_details->formula_name_os + "_os_" + process_details->process_type + "_asimov", final_Formula_os_asimov, os);
    T shape_ss_final = shape_calc_bkg_asimov_norm<T>(&args_ss_new, norm_true.at(1), int_ss, textfile_path_ss, process_details->formula_name_ss + "_ss_" + process_details->process_type + "_asimov", final_Formula_ss_asimov, ss);

    //std::vector<RooFormulaVar> shapes_both;
    //std::vector<RooGenericPdf> shapes_both;
    std::vector<T> shapes_both = {shape_os_final, shape_ss_final};
    //std::vector<T> shapes_both = {*shape_os_final};
    //auto shapes_both;
    /*
    shapes_both.push_back(shape_os);
    shapes_both.push_back(shape_ss);
    */

    /*
    shapes_both.push_back(*shape_os_final);
    shapes_both.push_back(*shape_ss_final);
    */

    std::cout << "Just before exiting the shape_calc function" << std::endl;
    return shapes_both;
}


//void plot_final_mass(RooPlot *plot_vs_mass, RooFormulaVar final_shape, std::string png_path)
void plot_final_mass(RooPlot *plot_vs_mass, RooGenericPdf final_shape, std::string png_path)
{
    string canvas_title = "c_mass";
    TCanvas *c1 = new TCanvas(canvas_title.c_str(),canvas_title.c_str(), 700,500);

    //variable_intermediate[i]->plotOn(plot_vs_mass);
    final_shape.plotOn(plot_vs_mass);
    
    plot_vs_mass->Draw();
    c1->Draw();
    c1->SaveAs(png_path.c_str());
    c1->Close();
    
}

void plot_final_mass_dataset(RooPlot *plot_vs_mass, RooDataSet final_shape, std::string png_path)
{
    string canvas_title = "c_mass";
    TCanvas *c1 = new TCanvas(canvas_title.c_str(),canvas_title.c_str(), 700,500);

    //variable_intermediate[i]->plotOn(plot_vs_mass);
    final_shape.plotOn(plot_vs_mass);
    
    plot_vs_mass->Draw();
    c1->Draw();
    c1->SaveAs(png_path.c_str());
    c1->Close();
    
}


//void plot_final_mass_multiple(RooPlot *plot_vs_mass, std::vector<RooFormulaVar> final_shape_net, std::vector<Color_t> colors_net, std::string png_path)
//void plot_final_mass_multiple(RooPlot *plot_vs_mass, std::vector<RooGenericPdf> final_shape_net, std::vector<Color_t> colors_net, std::vector<string> legend_titles, std::vector<string> legend_keys, std::string png_path)

template <typename T> void plot_final_mass_multiple(RooPlot *plot_vs_mass, std::vector<T> final_shape_net, std::vector<Color_t> colors_net, std::vector<string> legend_titles, std::vector<string> legend_keys, std::string png_path)
{
    string canvas_title = "c_mass";
    TCanvas *c1 = new TCanvas(canvas_title.c_str(),canvas_title.c_str(), 700,500);

    TLegend *leg = new TLegend(0.3,0.7,0.45,0.85,NULL,"brNDC");
    leg->SetBorderSize(0);
    leg->SetLineColor(1);
    leg->SetLineStyle(1);
    leg->SetLineWidth(1);
    leg->SetFillColor(0);
    leg->SetFillStyle(1001);
    TLegendEntry *entry;

    //variable_intermediate[i]->plotOn(plot_vs_mass);
    for(int i=0; i<final_shape_net.size(); i++)
    {        
        final_shape_net.at(i).plotOn(plot_vs_mass);
        //final_shape_net.at(i).plotOn(plot_vs_mass, plot_vs_mass->SetLineColor(kRed));
        //plot_vs_mass->SetLineColor(colors_net.at(i));
        //plot_vs_mass->SetLineColor(kRed);
        //final_shape_net.at(i).SetLineColor(kRed);
        plot_vs_mass->getAttLine()->SetLineColor(colors_net.at(i));
        plot_vs_mass->Draw();

        entry=leg->AddEntry(legend_titles.at(i).c_str(),legend_keys.at(i).c_str(),"L");

        entry->SetLineColor(colors_net.at(i));
        entry->SetLineStyle(1);
        entry->SetLineWidth(1);
        entry->SetMarkerColor(1);
        entry->SetMarkerStyle(20);
        entry->SetMarkerSize(1.15);
        entry->SetTextFont(42);
        entry->SetTextSize(0.02);
        
    }
    leg->Draw();

    c1->Draw();
    c1->SaveAs(png_path.c_str());
    c1->Close();
    
}

/*
//Unfortunately have to create another function since probably cannot assign auto as the data type
void plot_final_mass_multiple_formula(RooPlot *plot_vs_mass, std::vector<RooFormulaVar> final_shape_net, std::vector<Color_t> colors_net, std::vector<string> legend_titles, std::vector<string> legend_keys, std::string png_path)
{
    string canvas_title = "c_mass";
    TCanvas *c1 = new TCanvas(canvas_title.c_str(),canvas_title.c_str(), 700,500);

    TLegend *leg = new TLegend(0.3,0.7,0.45,0.85,NULL,"brNDC");
    leg->SetBorderSize(0);
    leg->SetLineColor(1);
    leg->SetLineStyle(1);
    leg->SetLineWidth(1);
    leg->SetFillColor(0);
    leg->SetFillStyle(1001);
    TLegendEntry *entry;

    //variable_intermediate[i]->plotOn(plot_vs_mass);
    for(int i=0; i<final_shape_net.size(); i++)
    {        
        final_shape_net.at(i).plotOn(plot_vs_mass);
        //final_shape_net.at(i).plotOn(plot_vs_mass, plot_vs_mass->SetLineColor(kRed));
        //plot_vs_mass->SetLineColor(colors_net.at(i));
        //plot_vs_mass->SetLineColor(kRed);
        //final_shape_net.at(i).SetLineColor(kRed);
        plot_vs_mass->getAttLine()->SetLineColor(colors_net.at(i));
        plot_vs_mass->Draw();

        entry=leg->AddEntry(legend_titles.at(i).c_str(),legend_keys.at(i).c_str(),"L");

        entry->SetLineColor(colors_net.at(i));
        entry->SetLineStyle(1);
        entry->SetLineWidth(1);
        entry->SetMarkerColor(1);
        entry->SetMarkerStyle(20);
        entry->SetMarkerSize(1.15);
        entry->SetTextFont(42);
        entry->SetTextSize(0.02);
        
    }
    leg->Draw();

    c1->Draw();
    c1->SaveAs(png_path.c_str());
    c1->Close();
    
}
*/

//void save_workspace(std::vector<RooFormulaVar> final_shape_net, std::string workspace_file_path, std::string workspace_name)
//template <typename T> void save_workspace(std::vector<T> final_shape_net, std::string workspace_file_path, std::string workspace_name)
template <typename T1, typename T2> void save_workspace(std::vector<T1> final_shape_net, std::vector<T2> norm_net, std::string workspace_file_path, std::string workspace_name)
{
    TFile *f_out = new TFile(workspace_file_path.c_str(), "RECREATE");
    RooWorkspace *w = new RooWorkspace(workspace_name.c_str(),workspace_name.c_str());
    for(int i=0; i<final_shape_net.size(); i++)
    {
        w->import(final_shape_net.at(i));
        w->import(norm_net.at(i));
        w->Print();
        w->Write();
    }
    f_out->Close();    

}

//This is for the dataset where you do not have to save an additional normalization
//WARNING - Number of events are divided by 10 so as to match the inherent 10 GeV binsize when the dataset is sampled from the RooGenericPdf
template <typename T1> void save_workspace_roodataset(std::vector<T1> final_shape_net, std::string workspace_file_path, std::string workspace_name)
{
    TFile *f_out = new TFile(workspace_file_path.c_str(), "RECREATE");
    RooWorkspace *w = new RooWorkspace(workspace_name.c_str(),workspace_name.c_str());
    for(int i=0; i<final_shape_net.size(); i++)
    {
        w->import(final_shape_net.at(i));
        w->Print();
        w->Write();
    }
    f_out->Close();    

}


struct json_object
{
    Json::Value dict_portion;
    bool read_flag;
    json_object(Json::Value dict_portion_input, bool read_flag_input)
    {
        std::cout << "Initializing struct json_object" << std::endl;
        dict_portion = dict_portion_input;
        read_flag = read_flag_input;

        //std::cout << "dict_portion " << dict_portion << " read_flag " <<  read_flag << std::endl;
    }    
};

///*
json_object read_json_cc(Json::Value chunk_init, std::vector<string> keys)
{
    /*
    //Just looking at a specific set of keys for now
    std::cout << " Contents of signal region for wbj" << std::endl;
    std::cout << chunk_init["WbjToLNu_4f_TuneCP5_13TeV-madgraph-pythia8"]["mu"]["j2_b1"] << std::endl;
    */

    bool read_flag;
    Json::Value chunk_init_nest = chunk_init;
    read_flag = 1;

    for(int key_num=0;key_num<keys.size();key_num++)
    {
        if(chunk_init_nest[keys.at(key_num)].empty() == false)
        {
            std::cout << "Current key " << keys.at(key_num) << std::endl;
            chunk_init_nest = chunk_init_nest[keys.at(key_num)];
        }

        else
        {
            std::cout << "Key does not exist" << std::endl;
            read_flag = 0;
            break;
        }
    }

    json_object chunk_json(chunk_init_nest, read_flag);
    return chunk_json;
}
//*/

std::vector<float> norm_counts(json_object chunk_inter, std::vector<string> lep_charge, std::vector<string> btag_veto)
//void norm_counts(json_object chunk_inter, std::vector<string> lep_charge, std::vector<string> btag_veto)
{
    float onshell_counts = 0, offshell_counts = 0;

    if(chunk_inter.read_flag == 1) //i.e. main chunk read successfully
    {
        std::vector<string> subset_key_list;
        std::string onshell_string = "onshell";
        std::string offshell_string = "offshell";
        std::string final_cut_string = "btagSF";
        for(int i=0;i<lep_charge.size();i++)
        {
            subset_key_list.push_back(lep_charge.at(i));
            subset_key_list.push_back(onshell_string);
            std::cout << "subset_key_list outside the inner loop " << std::endl;
            print_vector_str(subset_key_list);

            int outside_loop_size = subset_key_list.size();
            for(int j=0;j<btag_veto.size();j++)
            {
                std::cout << std::endl;
                subset_key_list.push_back(btag_veto.at(j));
                subset_key_list.push_back(final_cut_string);
                std::cout << "subset_key_list" << std::endl;
                print_vector_str(subset_key_list);
                json_object individual_counts_onshell = read_json_cc(chunk_inter.dict_portion, subset_key_list);
                std::cout << "individual_counts_onshell dict " << individual_counts_onshell.dict_portion << std::endl;

                onshell_counts += individual_counts_onshell.dict_portion.asFloat();

                std::replace(subset_key_list.begin(), subset_key_list.end(), onshell_string, offshell_string);
                std::cout << "subset_key_list" << std::endl;
                print_vector_str(subset_key_list);
                json_object individual_counts_offshell = read_json_cc(chunk_inter.dict_portion, subset_key_list);
                std::cout << "individual_counts_offshell dict " << individual_counts_offshell.dict_portion << std::endl;

                offshell_counts += individual_counts_offshell.dict_portion.asFloat();
                
                int inside_loop_size = subset_key_list.size();
                //===============================Reverse all the operations performed before moving on to the next loop==================================

                std::replace(subset_key_list.begin(), subset_key_list.end(), offshell_string, onshell_string);

                for(int k=0; k < (inside_loop_size-outside_loop_size); k++)
                {
                    subset_key_list.pop_back(); //Removed last element again before going to the next iteration
                }
                //=======================================================================================================================================
            }
            subset_key_list.clear();
        }   
    }

    std::vector<float> onshell_offshell = {onshell_counts,offshell_counts};
    return onshell_offshell;
}

//Creating a different function just for inside the width loop since it can be used for bkg processes as well (no width dependence there)
std::vector<float> area_norm(std::string process_name, std::string json_file)
{
    //Return a small chunk of the dict so do not have to read the whole file again
    Json::Value signal;
    std::ifstream signal_file(json_file, std::ifstream::binary);
    signal_file >> signal;
    
    //This is just to verify the counts against the existing datacard width_1_noqcd_xsec_syst.json (from validatedatacards.py) where the jet multiplicity was used
    //std::vector<string> opp_sign_keys = {process_name, "mu", "j2+_b1", "opp_sign"}; 

    std::vector<string> opp_sign_keys = {process_name, "mu", "j2_b1", "opp_sign"};
    //For summing over everything, combinations of these two will be made and the corresponding values extracted
    std::vector<string> lep_charge = {"lep_neg", "lep_pos"};
    std::vector<string> btag_veto = {"b_loose_0", "b_loose_1+_leading", "b_loose_1+_no_effect"};

    //std::vector<string> opp_sign_keys = {process_name,"mu","j2_b1", "opp_"};
    json_object opp_sign_json = read_json_cc(signal,opp_sign_keys);
    /*
    std::cout << "opp_sign_json dict " << opp_sign_json.dict_portion << std::endl;
    std::cout << "opp_sign_json flag " << opp_sign_json.read_flag << std::endl;
    */
    std::vector<float> opp_sign_counts = norm_counts(opp_sign_json, lep_charge, btag_veto); //At the moment the size of the nesting is not variable, only 4 - charge + mass window + btag veto + final key

    //This is just to verify the counts against the existing datacard width_1_noqcd_xsec_syst.json (from validatedatacards.py) where the jet multiplicity was used
    //std::vector<string> same_sign_keys = {process_name,"mu","j2+_b1", "same_sign"}; 

    std::vector<string> same_sign_keys = {process_name, "mu", "j2_b1", "same_sign"}; 
    json_object same_sign_json = read_json_cc(signal,same_sign_keys);
    /*
    std::cout << "same_sign_json dict " << same_sign_json.dict_portion << std::endl;
    std::cout << "same_sign_json flag " << same_sign_json.read_flag << std::endl;
    */

    std::vector<float> same_sign_counts = norm_counts(same_sign_json, lep_charge, btag_veto); //At the moment the size of the nesting is not variable, only 4 - charge + mass window + btag veto + final key

    std::cout << "opp sign offshell_counts " << opp_sign_counts.at(1) << std::endl;
    std::cout << "opp sign onshell_counts " << opp_sign_counts.at(0) << std::endl;

    std::cout << "same sign offshell_counts " << same_sign_counts.at(1) << std::endl;
    std::cout << "same sign onshell_counts " << same_sign_counts.at(0) << std::endl;

    std::vector<float> net_counts = {opp_sign_counts.at(0) + opp_sign_counts.at(1), same_sign_counts.at(0) + same_sign_counts.at(1)};

    return net_counts;
}

//Calculate real norm (real number of events) reading from the cutflows.json file
std::vector<float> norm_calc(std::vector<std::string> all_process_names, std::string all_proc_json_file)
{
    std::vector<std::vector<float>> all_process_counts;
    float net_counts_os = 0, net_counts_ss = 0;
    for(int i = 0; i < all_process_names.size(); i++)
    {
        std::vector<float> net_counts = area_norm(all_process_names.at(i), all_proc_json_file);
        all_process_counts.push_back(net_counts);
        std::cout << "Current process name " << all_process_names.at(i) << std::endl;
        print_vector_ft(all_process_counts.at(i));
        std::cout << std::endl;

        net_counts_os += all_process_counts.at(i).at(0);
        net_counts_ss += all_process_counts.at(i).at(1);

        std::cout << "net_counts_os " << net_counts_os << std::endl;
        std::cout << "net_counts_ss " << net_counts_ss << std::endl;
    }

    std::vector<float> norm_net = {net_counts_os, net_counts_ss}; //Same order as the shapes_os and shapes_ss in shape_calc

    return norm_net;
}

#endif // parametric_signal_bkg
