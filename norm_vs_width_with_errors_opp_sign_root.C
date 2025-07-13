#ifdef __CLING__
#pragma cling optimize(0)
#endif

double lin(double * x, double *p) 
{
   double shift = p[0];
   double lin_coeff = p[1];
   double lin_power = p[2];
   double mean = p[3];
   //double normalization_norm = p[4];
   //double lin = shift + pos_power_coeff*pow((x[0]-mean),pos_power) + neg_power_coeff/pow((x[0]-mean),neg_power);
   //double lin = shift - lin_coeff*pow((x[0]-mean),lin_power);

   //double lin = normalization_norm*(shift - lin_coeff*pow((x[0]-mean),lin_power));
   double lin = shift*(1 - lin_coeff*pow((x[0]-mean),lin_power)); //Obviously the lin_coeff will be different from the previous parameter
   return lin;
}

void norm_vs_width_with_errors_opp_sign_root()
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
   
   Double_t xsec_signal_os_unc = 1.3;

   Double_t Graph0_fx1001[21] = { 0.5, 1, 1.322, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8,
   8.5, 9, 9.5, 10 };
   Double_t Graph0_fy1001[21] = { 16089.4, 15842, 15704.3, 15630.7, 15429.5, 15234.3, 15044.3, 14838.8, 14638, 14442.7, 14252.3, 14066.4, 13884.9, 13707.4, 13533.7, 13363.8, 13197.4,
   13034.6, 12875, 12718.7, 12565.6 };

   //Trying to fit the normalization values that will be obtained by the ln norm uncertainty 'implementation'
   Double_t Graph0_fy1001_up[21];
   Double_t Graph0_fy1001_down[21];

   for(int i=0;i<21;i++)
   {
      Graph0_fy1001_up[i] = Graph0_fy1001[i]*xsec_signal_os_unc;
      Graph0_fy1001_down[i] = Graph0_fy1001[i]/xsec_signal_os_unc;
   }

   /*
   for(int i=0;i<21;i++)
   {
      std::cout << std::endl;
      std::cout << "Chronological Width number " << i << std::endl;
      std::cout << "up " << Graph0_fy1001_up[i] << std::endl;
      std::cout << "nominal " << Graph0_fy1001[i] << std::endl;
      std::cout << "down " << Graph0_fy1001_down[i] << std::endl;
   }
   */
   ///*
   Double_t Graph0_fex1001[21] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0 };
   Double_t Graph0_fey1001[21] = { 179.976, 75.855, 77.0665, 76.726, 73.6198, 105.082, 132.082, 165.617, 138.146, 127.034, 42.1513, 33.6775, 29.5817, 25.8491, 22.9888, 20.7269, 18.8969,
   18.1471, 15.7923, 15.0057, 14.0679 };
   TGraphErrors *gre = new TGraphErrors(21,Graph0_fx1001,Graph0_fy1001,Graph0_fex1001,Graph0_fey1001);
   gre->SetName("Graph0");

   gre->SetTitle("Norm opp sign with fit vs width");
   gre->GetHistogram()->GetXaxis()->SetTitle("Width set in MC (in GeV)");
   gre->GetHistogram()->GetYaxis()->SetTitle("Gaussian Norm values");
   gre->SetFillStyle(1000);
   gre->SetMarkerStyle(20);
   gre->SetMarkerColor(kBlack);
   gre->Draw("ap");

   //Hard coded y limits
   gre->GetHistogram()->SetMaximum(25000.);   // along          
   gre->GetHistogram()->SetMinimum(8000.);  //   Y     


   TGraphErrors *gre_up = new TGraphErrors(21,Graph0_fx1001,Graph0_fy1001_up,Graph0_fex1001,Graph0_fey1001);
   gre_up->SetName("Graph0");

   gre_up->SetTitle("Norm opp sign up variation with fit vs width");
   gre_up->GetHistogram()->GetXaxis()->SetTitle("Width set in MC (in GeV)");
   gre_up->GetHistogram()->GetYaxis()->SetTitle("Gaussian Norm values");
   gre_up->SetFillStyle(1000);
   gre_up->SetMarkerStyle(20);
   gre_up->SetMarkerColor(kRed);
   gre_up->Draw("p same");

   TGraphErrors *gre_down = new TGraphErrors(21,Graph0_fx1001,Graph0_fy1001_down,Graph0_fex1001,Graph0_fey1001);
   gre_down->SetName("Graph0");

   gre_down->SetTitle("Norm opp sign down variation  with fit vs width");
   gre_down->GetHistogram()->GetXaxis()->SetTitle("Width set in MC (in GeV)");
   gre_down->GetHistogram()->GetYaxis()->SetTitle("Gaussian Norm values");
   gre_down->SetFillStyle(1000);
   gre_down->SetMarkerStyle(20);
   gre_down->SetMarkerColor(kBlue);
   gre_down->Draw("p same");

   ///*
   float upper_limit = 10.0;
   float lower_limit = 0.4;

   TF1 *f = new TF1("gre_lin", lin, lower_limit, upper_limit, 4); //For width_1 (10 GeV)
   //TF1 *f = new TF1("gre_lin", lin, lower_limit, upper_limit, 5); //For width_1 (10 GeV)
   //TF1 *f = new TF1("gre_lin", lin, 160.0, 180.0, 3);
   f->SetTitle(";x;Density");
   //f->SetParNames("shift","exponential_coefficient","linear_coefficient","decay_constant");
   //f->SetParNames("y_shift","linear_coefficient","linear_like_power","mean", "normalization_norm");
   f->SetParNames("y_shift","linear_coefficient","linear_like_power","mean");

   //string formula = "normalization_norm*(y_shift - linear_coefficient*pow((x[0]-mean),linear_like_power))";
   string formula = "y_shift*(1 - linear_coefficient*pow((x[0]-mean),linear_like_power))";
   std::cout << "Fitting formula " << formula << std::endl;   


   //New function definition
   //f->SetParameters(0, 1, 0.05, 0.3);
   f->SetParameters(1, 350, 1, 0, 16000);

   //f->SetParLimits(0,0,100000);
   //f->SetParLimits(0,13000,18000);
   //f->SetParLimits(0,13000,20000);
   //f->SetParLimits(0,13000,50000);
   //f->SetParLimits(0,13000,30000);
   //f->SetParLimits(0,10000,30000);
   f->SetParLimits(0,Graph0_fy1001[0]-2000,Graph0_fy1001[0]+2000);
   //f->SetParLimits(0,0,10);
   f->SetParLimits(1,0,10);
   //f->SetParLimits(2,0,2);
   f->SetParLimits(2,0.7,1.3);
   f->SetParLimits(3,-1000,lower_limit);
   //f->SetParLimits(3,-50,lower_limit);
   //f->SetParLimits(4,10000,25000);
   //f->SetParLimits(4,13000,18000);
   //f->FixParameter(4,13000);
   //f->Draw("SAME");

   ROOT::Math::MinimizerOptions::SetDefaultMaxFunctionCalls(10000); //Trying to change the status of the fit from call limit to hopefully something better
   gre->Fit("gre_lin", "R");
   TF1 *g = (TF1*)gre->GetListOfFunctions()->FindObject("gre_lin");
   f->SetParameters(g->GetParameter(0), g->GetParameter(1),g->GetParameter(2),g->GetParameter(3));
   
   gre->Draw("ap");
   f->SetLineColor(kBlack);
   f->Draw("SAME");

   //f->SetParLimits(0,0,10);
   f->SetParLimits(0,0,100000);
   f->SetParLimits(1,0,10);
   f->SetParLimits(2,0,2);
   f->SetParLimits(3,-1000,lower_limit);
   //f->SetParLimits(4,18000,25000);

   /*
   gre_up->Fit("gre_lin", "R");
   TF1 *g_up = (TF1*)gre_up->GetListOfFunctions()->FindObject("gre_lin");
   f->SetParameters(g_up->GetParameter(0), g_up->GetParameter(1),g_up->GetParameter(2),g_up->GetParameter(3));
   
   
   gre_up->Draw("p same");
   f->SetLineColor(kRed);
   f->Draw("SAME");
   

   //f->SetParLimits(0,0,10);
   f->SetParLimits(0,0,100000);
   f->SetParLimits(1,0,10);
   f->SetParLimits(2,0,2);
   f->SetParLimits(3,-1000,lower_limit);
   //f->SetParLimits(4,10000,13000);
   //f->FixParameter(4,10700);

   gre_down->Fit("gre_lin", "R");
   TF1 *g_down = (TF1*)gre_down->GetListOfFunctions()->FindObject("gre_lin");
   f->SetParameters(g_down->GetParameter(0), g_down->GetParameter(1),g_down->GetParameter(2),g_down->GetParameter(3));
   
   gre_down->Draw("p same");
   f->SetLineColor(kBlue);
   f->Draw("SAME");
   */

   //*/
   c1->Modified();
   c1->cd();
   c1->SetSelected(c1);
   string image_path = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/up_down_nominal_norm_vs_width_with_errors_opp_sign_fit.pdf";
   c1->Print(image_path.c_str());
   c1->Close();
   //*/  
   gApplication->Terminate();       
   
}
