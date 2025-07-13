#ifdef __CLING__
#pragma cling optimize(0)
#endif

double lin(double * x, double *p) 
{
   double shift = p[0];
   double lin_coeff = p[1];
   double lin_power = p[2];
   double mean = p[3];
   //double lin = shift + pos_power_coeff*pow((x[0]-mean),pos_power) + neg_power_coeff/pow((x[0]-mean),neg_power);
   double lin = shift + lin_coeff*pow((x[0]-mean),lin_power);
   return lin;
}

void sigma_vs_width_with_errors_opp_sign_root()
{
//=========Macro generated from canvas: c1/c1
//=========  (Thu Jun 12 04:58:56 2025) by ROOT version 6.28/04
   TCanvas *c1 = new TCanvas("c1", "c1",0,0,700,500);
   gStyle->SetOptStat(0);
   c1->SetHighLightColor(2);
   c1->Range(0,0,1,1);
   c1->SetFillColor(0);
   c1->SetBorderMode(0);
   c1->SetBorderSize(2);
   c1->SetFrameBorderMode(0);
   
   Double_t Graph0_fx1003[21] = { 0.5, 1, 1.322, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8,
   8.5, 9, 9.5, 10 };
   Double_t Graph0_fy1003[21] = { 25.809, 25.9122, 25.9675, 25.9968, 26.0753, 26.1486, 26.2182, 26.3012, 26.3824, 26.4615, 26.539, 26.6151, 26.6901, 26.7641, 26.8372, 26.9094, 26.9808,
   27.0513, 27.1211, 27.1902, 27.2585 };
   Double_t Graph0_fex1003[21] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0 };
   Double_t Graph0_fey1003[21] = { 0.0952728, 0.116793, 0.117025, 0.117198, 0.117922, 0.118785, 0.123676, 0.0993695, 0.100239, 0.101124, 0.10202, 0.102515, 0.10384, 0.104762, 0.10569, 0.106624, 0.107563,
   0.108978, 0.108598, 0.110403, 0.112241 };
   TGraphErrors *gre = new TGraphErrors(21,Graph0_fx1003,Graph0_fy1003,Graph0_fex1003,Graph0_fey1003);
   gre->SetName("Graph0");

   gre->SetTitle("Sigma opp sign with fit vs width");
   gre->GetHistogram()->GetXaxis()->SetTitle("Width set in MC (in GeV)");
   gre->GetHistogram()->GetYaxis()->SetTitle("Gaussian sigma values");
   gre->SetFillStyle(1000);
   gre->SetMarkerStyle(20);
   gre->Draw("ap");

   float upper_limit = 10.0;
   float lower_limit = 0.4;

   TF1 *f = new TF1("gre_lin", lin, lower_limit, upper_limit, 4); //For width_1 (10 GeV)
   //TF1 *f = new TF1("gre_lin", lin, 160.0, 180.0, 3);
   f->SetTitle(";x;Density");
   //f->SetParNames("shift","exponential_coefficient","linear_coefficient","decay_constant");
   f->SetParNames("y_shift","linear_coefficient","linear_like_power","mean");

   string formula = "y_shift + linear_coefficient*pow((x[0]-mean),linear_like_power)";
   std::cout << "Fitting formula " << formula << std::endl;   

   f->SetLineColor(kGreen);

   //New function definition
   //f->SetParameters(0, 1, 0.05, 0.3);
   f->SetParameters(25.8, 0.14, 1, 0);

   f->SetParLimits(0,0,100);
   f->SetParLimits(1,0,10);
   f->SetParLimits(2,0,2);
   f->SetParLimits(3,-1000,lower_limit);
   f->Draw("SAME");

   gre->Fit("gre_lin", "R");
   TF1 *g = (TF1*)gre->GetListOfFunctions()->FindObject("gre_lin");
   f->SetParameters(g->GetParameter(0), g->GetParameter(1),g->GetParameter(2),g->GetParameter(3));
   
   gre->Draw("ap");

   f->Draw("SAME");
   /*
   f->SetLineColor(kRed);
   //f->SetParameters(30.4349, 0.0266376,30.9553,-0.612850,1.0,0.00323300);

   //f->SetParameters(4, 2.5, 0.05, 1.0, 1.5);
   //f->SetParameters(4, 2.5, 0.05, 0.7, 1.5);
   //f->SetParameters(4, 2.5, 0.05, 0.7, 1.5);
   f->SetParameters(4, 3.0, 0.05, 0.5, 1.5);
   
   f->Draw("SAME");
   */
   c1->Modified();
   c1->cd();
   c1->SetSelected(c1);
   string image_path = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/sigma_vs_width_with_errors_opp_sign_fit.pdf";
   c1->Print(image_path.c_str());
   c1->Close();
   gApplication->Terminate();         
}
