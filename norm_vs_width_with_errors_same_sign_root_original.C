#ifdef __CLING__
#pragma cling optimize(0)
#endif

double power_sum(double * x, double *p) 
{
   double shift = p[0];
   double lin_coeff = p[1];
   double neg_power_coeff = p[2];
   double mean = p[3];
   double neg_power = p[4];
   //double power_sum = shift - lin_coeff*(x[0]-mean) - neg_power_coeff/pow((x[0]-mean),neg_power);
   double power_sum = shift*(1 - lin_coeff*(x[0]-mean) - neg_power_coeff/pow((x[0]-mean),neg_power));
   return power_sum;
}


void norm_vs_width_with_errors_same_sign_root_original()
{
//=========Macro generated from canvas: c1/c1
//=========  (Fri Jun 13 06:49:09 2025) by ROOT version 6.28/04
   TCanvas *c1 = new TCanvas("c1", "c1",0,0,700,500);
   gStyle->SetOptStat(0);
   c1->SetHighLightColor(2);
   c1->Range(0,0,1,1);
   c1->SetFillColor(0);
   c1->SetBorderMode(0);
   c1->SetBorderSize(2);
   c1->SetFrameBorderMode(0);
   
   Double_t Graph0_fx1001[21] = { 0.5, 1, 1.322, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8,
   8.5, 9, 9.5, 10 };
   Double_t Graph0_fy1001[21] = { 4375.27, 4405.85, 4404.23, 4400.16, 4381.6, 4356.91, 4328.99, 4299.42, 4269.13, 4238.67, 4208.39, 4178.48, 4149.09, 4120.29, 4092.12, 4064.59, 4037.72,
   4011.48, 3985.88, 3960.89, 3936.49 };
   Double_t Graph0_fex1001[21] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0 };
   Double_t Graph0_fey1001[21] = { 15.1709, 16.9589, 17.6728, 17.9544, 18.2977, 18.3892, 18.1212, 17.6975, 17.2143, 16.7343, 16.2861, 15.8894, 15.5232, 15.185, 14.9041, 14.6792, 14.4611,
   14.2644, 14.086, 13.9248, 13.78 };
   TGraphErrors *gre = new TGraphErrors(21,Graph0_fx1001,Graph0_fy1001,Graph0_fex1001,Graph0_fey1001);
   gre->SetName("Graph0");
   gre->SetTitle("Norm same sign with fit vs width");
   gre->GetHistogram()->GetXaxis()->SetTitle("Width set in MC (in GeV)");
   gre->GetHistogram()->GetYaxis()->SetTitle("Gaussian normalization values");
   gre->SetFillStyle(1000);
   gre->SetMarkerStyle(20);
   gre->Draw("ap");

   //g_graph->GetHistogram()

   TF1 *f = new TF1("gre_power_sum", power_sum, 0.3, 10.0, 5); //For width_1 (10 GeV)
   //TF1 *f = new TF1("gre_power_sum", power_sum, 160.0, 180.0, 3);
   f->SetTitle(";x;Density");
   f->SetParNames("y_shift","linear_coefficient","negative_power_coefficent","x_shift","negative_power");

   //string formula = "y_shift - linear_coefficient*(x[0]-x_shift) - negative_power_coefficent/pow((x[0]-x_shift),negative_power)";
   string formula = "y_shift*(1 - linear_coefficient*(x[0]-x_shift) - negative_power_coefficent/pow((x[0]-x_shift),negative_power))";
   std::cout << "Fitting formula " << formula << std::endl;

   f->SetLineColor(kGreen);
   //f->SetParameters(20.0, 40.0, 4.0);
   //f->SetParameters(5000.0, 200.0, 5.0);
   //f->SetParameters(4800.0, 55.0, 4500.0, 0.0, 2.0);
   f->SetParameters(4800.0, 55.0, 4500.0, -4.0, 2.0);
   //f->SetParameters(10000.0, 200.0, 0.3);
   //f->SetParameters(0.0, 50.0, 4.0, 50.0);
   f->SetParLimits(0,0,5000);
   f->SetParLimits(1,0,100);
   //f->SetParLimits(2,0,10000);
   f->SetParLimits(2,0,100);
   f->SetParLimits(3,-100,0);
   f->SetParLimits(4,0,20);
   
   //f->Draw(i == 0 ? "" : "SAME");
   f->Draw("SAME");
   //f->Draw();
   ///*

   ROOT::Math::MinimizerOptions::SetDefaultMaxFunctionCalls(10000); //Trying to change the status of the fit from call limit to hopefully something better
   gre->Fit("gre_power_sum", "R");
   TF1 *g = (TF1*)gre->GetListOfFunctions()->FindObject("gre_power_sum");
   f->SetParameters(g->GetParameter(0), g->GetParameter(1),g->GetParameter(2),g->GetParameter(3),g->GetParameter(4));

   gre->Draw("ap");
   f->Draw("SAME");
   //*/
   c1->Modified();
   c1->cd();
   c1->SetSelected(c1);
   string image_path = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/up_down_original_norm_vs_width_with_errors_same_sign_fit.pdf";
   c1->Print(image_path.c_str());
   c1->Close();
   gApplication->Terminate();   
}
