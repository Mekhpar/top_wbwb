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
   double power_sum = shift + pos_power_coeff*pow((x[0]-mean),pos_power) + neg_power_coeff/pow((x[0]-mean),neg_power);
   return power_sum;
}

void sigma_vs_width_with_errors_same_sign_root_original()
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
   
   Double_t Graph0_fx1003[21] = { 0.5, 1, 1.322, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8,
   8.5, 9, 9.5, 10 };
   Double_t Graph0_fy1003[21] = { 27.6525, 27.4565, 27.4668, 27.4883, 27.5875, 27.675, 27.7753, 27.8718, 27.9634, 28.0501, 28.1322, 28.2099, 28.2836, 28.3535, 28.4202, 28.4837, 28.5444,
   28.6026, 28.6584, 28.7119, 28.7633 };
   Double_t Graph0_fex1003[21] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0 };
   Double_t Graph0_fey1003[21] = { 0.145843, 0.14723, 0.148506, 0.149253, 0.147674, 0.153088, 0.154704, 0.156136, 0.157416, 0.158575, 0.159634, 0.16061, 0.16152, 0.162179, 0.16308, 0.16393, 0.16465,
   0.165335, 0.165991, 0.166621, 0.167228 };
   TGraphErrors *gre = new TGraphErrors(21,Graph0_fx1003,Graph0_fy1003,Graph0_fex1003,Graph0_fey1003);
   gre->SetName("Graph0");
   gre->SetTitle("Sigma same sign with fit vs width");
   gre->GetHistogram()->GetXaxis()->SetTitle("Width set in MC (in GeV)");
   gre->GetHistogram()->GetYaxis()->SetTitle("Gaussian sigma values");
   gre->SetFillStyle(1000);
   gre->SetMarkerStyle(20);
   gre->Draw("ap");

   //TF1 *f = new TF1("gre_power_sum", power_sum, 0.3, 10.0, 6); //For width_1 (10 GeV)
   TF1 *f = new TF1("gre_power_sum", power_sum, 0.4, 10.0, 6); //For width_1 (10 GeV)
   //TF1 *f = new TF1("gre_power_sum", power_sum, 160.0, 180.0, 3);
   f->SetTitle(";x;Density");
   f->SetParNames("y_shift","positive_power_coefficient","negative_power_coefficient","x_shift","positive_power","negative_power");

   string formula = "y_shift + positive_power_coefficient*pow((x[0]-x_shift),positive_power) + negative_power_coefficient/pow((x[0]-x_shift),negative_power)";
   std::cout << "Fitting formula " << formula << std::endl;

   f->SetLineColor(kGreen);
   //f->SetParameters(20.0, 40.0, 4.0);
   //f->SetParameters(5000.0, 200.0, 5.0);
   //f->SetParameters(4800.0, 55.0, 4500.0, 0.0, 2.0);
   f->SetParameters(26.0, 0.5, 1.0, -1.0, 0.8, 2.0);
   //f->SetParameters(10000.0, 200.0, 0.3);
   //f->SetParameters(0.0, 50.0, 4.0, 50.0);
   ///*
   f->SetParLimits(0,20,35);
   f->SetParLimits(1,0,100);
   f->SetParLimits(2,0,1000);
   f->SetParLimits(3,-10,0);
   f->SetParLimits(4,0,20);
   f->SetParLimits(5,0,20);
   //*/
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
   string image_path = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0430_wbj_widths_new/original_sigma_vs_width_with_errors_same_sign_fit.pdf";
   c1->Print(image_path.c_str());
   c1->Close();
   gApplication->Terminate();      
}
