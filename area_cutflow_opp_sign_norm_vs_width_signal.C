#ifdef __CLING__
#pragma cling optimize(0)
#endif

double linear(double * x, double *p) 
{
   double shift = p[0];
   double lin_coeff = p[1];
   /*
   double neg_power_coeff = p[2];
   double mean = p[3];
   double neg_power = p[4];
   */
   //double linear = shift - lin_coeff*(x[0]-mean) - neg_power_coeff/pow((x[0]-mean),neg_power);
   //double linear = shift*(1 - lin_coeff*(x[0]-mean) - neg_power_coeff/pow((x[0]-mean),neg_power));
   double linear = shift*(1 - lin_coeff*x[0]); //Again the shift is pulled out so that that can be set to varying with xsec (the only parameter)
   return linear;
}

void area_cutflow_opp_sign_norm_vs_width_signal()
{
//=========Macro generated from canvas: c1/c1
//=========  (Thu Jul  3 21:14:04 2025) by ROOT version 6.28/04
   TCanvas *c1 = new TCanvas("c1", "c1",0,0,700,500);
   gStyle->SetOptStat(0);
   c1->SetHighLightColor(2);
   c1->Range(-1.36875,50500.87,12.31875,59134.51);
   c1->SetFillColor(0);
   c1->SetBorderMode(0);
   c1->SetBorderSize(2);
   c1->SetFrameBorderMode(0);
   c1->SetFrameBorderMode(0);
   
   Double_t opp_area_norm_vs_width_fx1[21] = { 10, 9.5, 9, 8.5, 8, 7.5, 7, 6.5, 6, 5.5, 5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1.322, 1, 0.5 };
   Double_t opp_area_norm_vs_width_fy1[21] = { 51939.81, 52210.25, 52484.41, 52762.41, 53044.26, 53330, 53619.66, 53913.28, 54210.86, 54512.42, 54817.93, 55127.32, 55440.44, 55756.91, 56075.93, 56395.82, 56713.43,
   57024.29, 57133.1, 57331.57, 57695.57 };
   TGraph *graph = new TGraph(21,opp_area_norm_vs_width_fx1,opp_area_norm_vs_width_fy1);

   graph->SetName("opp_area_norm_vs_width");
   graph->SetTitle("Area norm opp sign with fit vs width");
   graph->GetHistogram()->GetXaxis()->SetTitle("Width set in MC (in GeV)");
   graph->GetHistogram()->GetYaxis()->SetTitle("Cutflow area normalization values");
   graph->SetFillStyle(1000);
   graph->SetMarkerStyle(20);
   graph->Draw("ap");
      
   TF1 *f = new TF1("graph_linear", linear, 0.3, 10.0, 2); //For width_1 (10 GeV)
   //TF1 *f = new TF1("graph_linear", linear, 160.0, 180.0, 3);
   f->SetTitle(";x;Density");
   f->SetParNames("y_shift","linear_coefficient");

   //string formula = "y_shift - linear_coefficient*(x[0]-x_shift) - negative_power_coefficent/pow((x[0]-x_shift),negative_power)";
   string formula = "y_shift*(1 - linear_coefficient*x[0])";
   std::cout << "Fitting formula " << formula << std::endl;

   f->SetLineColor(kGreen);
   f->SetParameters(4800.0, 55.0);
   //f->SetParLimits(0,0,5000);
   //f->SetParLimits(1,0,100);
   

   //ROOT::Math::MinimizerOptions::SetDefaultMaxFunctionCalls(10000); //Trying to change the status of the fit from call limit to hopefully something better
   graph->Fit("graph_linear", "R");
   TF1 *g = (TF1*)graph->GetListOfFunctions()->FindObject("graph_linear");
   f->SetParameters(g->GetParameter(0), g->GetParameter(1));

   graph->Draw("ap");
   f->Draw("SAME");


   TPaveText *pt = new TPaveText(0.194023,0.9339831,0.805977,0.995,"blNDC");
   pt->SetName("title");
   pt->SetBorderSize(0);
   pt->SetFillColor(0);
   pt->SetFillStyle(0);
   pt->SetTextFont(42);
   TText *pt_LaTex = pt->AddText("Area norm opp sign with fit vs width");
   pt->Draw();
   c1->Modified();
   c1->cd();
   c1->SetSelected(c1);

   std::string image_path = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/area_cutflow_opp_sign_norm_fit_vs_width_signal.png";
   c1->Print(image_path.c_str());

   gApplication->Terminate();
}
