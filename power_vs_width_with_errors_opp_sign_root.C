#ifdef __CLING__
#pragma cling optimize(0)
#endif

double exp(double * x, double *p) 
{
   double shift = p[0];
   double exp_coeff = p[1];
   double decay_const = p[2];
   //double mean = p[4];
   //double exp = shift + pos_power_coeff*pow((x[0]-mean),pos_power) + neg_power_coeff/pow((x[0]-mean),neg_power);
   double exp = shift + exp_coeff*TMath::Exp(-decay_const*(x[0]));
   return exp;
}


void power_vs_width_with_errors_opp_sign_root()
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
   
   Double_t Graph0_fx1005[21] = { 0.5, 1, 1.322, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8,
   8.5, 9, 9.5, 10 };
   Double_t Graph0_fy1005[21] = { 4.55041, 4.14086, 4.01763, 3.9654, 3.84996, 3.7619, 3.69094, 3.63378, 3.58583, 3.54504, 3.51007, 3.47993, 3.45381, 3.4312, 3.41159, 3.39465, 3.38002,
   3.36751, 3.35689, 3.34796, 3.34057 };
   Double_t Graph0_fex1005[21] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0 };
   Double_t Graph0_fey1005[21] = { 0.134697, 0.117686, 0.112492, 0.110138, 0.104481, 0.0997448, 0.0957688, 0.092141, 0.0892961, 0.0868322, 0.0846911, 0.0811227, 0.0811897, 0.0797585, 0.0785036, 0.0774039, 0.0764389,
   0.0756214, 0.0742752, 0.0742262, 0.0737096 };
   TGraphErrors *gre = new TGraphErrors(21,Graph0_fx1005,Graph0_fy1005,Graph0_fex1005,Graph0_fey1005);
   gre->SetName("Graph0");
   gre->SetTitle("power opp sign with fit vs width");
   gre->GetHistogram()->GetXaxis()->SetTitle("Width set in MC (in GeV)");
   gre->GetHistogram()->GetYaxis()->SetTitle("Power values");
   gre->SetFillStyle(1000);
   gre->SetMarkerStyle(20);
   gre->Draw("ap");
   
   ///*
   TF1 *f = new TF1("gre_exp", exp, 0.4, 10.0, 3); //For width_1 (10 GeV)

   f->SetTitle(";x;Density");
   //f->SetParNames("shift","exponential_coefficient","linear_coefficient","decay_constant");
   f->SetParNames("y_shift","exponential_coefficient","decay_constant");

   string formula = "y_shift + exponential_coefficient*TMath::Exp(-decay_constant*(x[0]))";
   std::cout << "Fitting formula " << formula << std::endl;   

   f->SetLineColor(kGreen);

   //New function definition
   //f->SetParameters(0, 1, 0.05, 0.3);
   f->SetParameters(3, 4, 1);
   /*
   f->SetParLimits(0,0,1000);
   f->SetParLimits(1,0.9,10);
   f->SetParLimits(2,0,1000);
   */

   f->Draw("SAME");

   gre->Fit("gre_exp", "R");
   TF1 *g = (TF1*)gre->GetListOfFunctions()->FindObject("gre_exp");
   f->SetParameters(g->GetParameter(0), g->GetParameter(1),g->GetParameter(2));
   
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
   string image_path = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/power_vs_width_with_errors_opp_sign_fit.pdf";
   c1->Print(image_path.c_str());
   c1->Close();
   gApplication->Terminate();         
}
