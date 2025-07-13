#ifdef __CLING__
#pragma cling optimize(0)
#endif

double power_sum(double * x, double *p) 
{
   double shift = p[0];
   double pos_power_coeff = p[1];
   double neg_power_coeff = p[2];
   double mean = p[3];
   double pos_power = p[4];
   double neg_power = p[5];
   //double power_sum = shift + pos_power_coeff*pow((x[0]-mean),pos_power) + neg_power_coeff/pow((x[0]-mean),neg_power);
   double power_sum = shift + pos_power_coeff*pow((x[0]-mean),pos_power) - neg_power_coeff*pow((x[0]-mean),neg_power);
   return power_sum;
}

void alpha_vs_width_with_errors_opp_sign_root()
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
   
   Double_t Graph0_fx1004[21] = { 0.5, 1, 1.322, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8,
   8.5, 9, 9.5, 10 };
   Double_t Graph0_fy1004[21] = { -1.13733, -1.13975, -1.13595, -1.13339, -1.12546, -1.11695, -1.10805, -1.09874, -1.08923, -1.07959, -1.06989, -1.06017, -1.05045, -1.04076, -1.03111, -1.02151, -1.01198,
   -1.00252, -0.993132, -0.98383, -0.974616 };
   Double_t Graph0_fex1004[21] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0 };
   Double_t Graph0_fey1004[21] = { 0.0136501, 0.0137348, 0.0137107, 0.0136824, 0.0135672, 0.0134227, 0.0132693, 0.0130966, 0.0129409, 0.012788, 0.0126392, 0.0124206, 0.0123555, 0.0122208, 0.0120909, 0.0119653, 0.0118438,
   0.0117255, 0.0115215, 0.0115016, 0.0114587 };
   TGraphErrors *gre = new TGraphErrors(21,Graph0_fx1004,Graph0_fy1004,Graph0_fex1004,Graph0_fey1004);
   gre->SetName("Graph0");
   gre->SetTitle("alpha opp sign with fit vs width");
   gre->GetHistogram()->GetXaxis()->SetTitle("Width set in MC (in GeV)");
   gre->GetHistogram()->GetYaxis()->SetTitle("Gaussian alpha values");
   gre->SetFillStyle(1000);
   gre->SetMarkerStyle(20);
   gre->Draw("ap");

   //TF1 *f = new TF1("gre_power_sum", power_sum, 0.3, 10.0, 6); //For width_1 (10 GeV)
   TF1 *f = new TF1("gre_power_sum", power_sum, 0.4, 10.0, 6); //For width_1 (10 GeV)
   //TF1 *f = new TF1("gre_power_sum", power_sum, 160.0, 180.0, 3);
   f->SetTitle(";x;Density");
   f->SetParNames("y_shift","positive_power_coefficient","negative_power_coefficient","x_shift","positive_power","negative_power");

   string formula = "y_shift + positive_power_coefficient*pow((x[0]-x_shift),positive_power) - negative_power_coefficient*pow((x[0]-x_shift),negative_power)";
   std::cout << "Fitting formula " << formula << std::endl;   

   f->SetLineColor(kGreen);
   //f->SetParameters(20.0, 40.0, 4.0);
   //f->SetParameters(5000.0, 200.0, 5.0);
   //f->SetParameters(4800.0, 55.0, 4500.0, 0.0, 2.0);
   f->SetParameters(26.0, 0.5, 1.0, -1.0, 0.8, 2.0);
   //f->SetParameters(10000.0, 200.0, 0.3);
   //f->SetParameters(0.0, 50.0, 4.0, 50.0);
   /*
   f->SetParLimits(0,20,35);
   f->SetParLimits(1,0,100);
   f->SetParLimits(2,0,1000);
   f->SetParLimits(3,-10,0);
   f->SetParLimits(4,0,20);
   f->SetParLimits(5,0,20);
   */
   //f->Draw(i == 0 ? "" : "SAME");
   //f->Draw("SAME");
   //f->Draw();
   ///*
   gre->Fit("gre_power_sum", "R");
   TF1 *g = (TF1*)gre->GetListOfFunctions()->FindObject("gre_power_sum");
   f->SetParameters(g->GetParameter(0), g->GetParameter(1),g->GetParameter(2),g->GetParameter(3),g->GetParameter(4),g->GetParameter(5));

   gre->Draw("ap");
   f->Draw("SAME");
   //*/


   c1->Modified();
   c1->cd();
   c1->SetSelected(c1);
   string image_path = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/alpha_vs_width_with_errors_opp_sign_fit.pdf";
   c1->Print(image_path.c_str());
   c1->Close();
   gApplication->Terminate();      
}
