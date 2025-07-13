#ifdef __CLING__
#pragma cling optimize(0)
#endif
void area_cutflow_same_sign_norm_vs_width_signal()
{
//=========Macro generated from canvas: c1/c1
//=========  (Thu Jul  3 21:14:04 2025) by ROOT version 6.28/04
   TCanvas *c1 = new TCanvas("c1", "c1",0,0,700,500);
   gStyle->SetOptStat(0);
   c1->SetHighLightColor(2);
   c1->Range(-1.36875,22122.86,12.31875,23974.46);
   c1->SetFillColor(0);
   c1->SetBorderMode(0);
   c1->SetBorderSize(2);
   c1->SetFrameBorderMode(0);
   c1->SetFrameBorderMode(0);
   
   Double_t same_area_norm_vs_width_fx2[21] = { 10, 9.5, 9, 8.5, 8, 7.5, 7, 6.5, 6, 5.5, 5, 4.5, 4, 3.5, 3, 2.5, 2,
   1.5, 1.322, 1, 0.5 };
   Double_t same_area_norm_vs_width_fy2[21] = { 23665.86, 23605.87, 23544.89, 23482.9, 23419.87, 23355.78, 23290.59, 23224.24, 23156.69, 23087.85, 23017.64, 22945.95, 22872.67, 22797.71, 22721.08, 22643.02, 22564.61,
   22489.55, 22465.55, 22431.46, 22439.4 };
   TGraph *graph = new TGraph(21,same_area_norm_vs_width_fx2,same_area_norm_vs_width_fy2);
   graph->SetName("same_area_norm_vs_width");
   graph->SetTitle("Area norm same sign with fit vs width");
   graph->SetFillStyle(1000);
   graph->SetMarkerStyle(20);
   
   TH1F *Graph_same_area_norm_vs_width2 = new TH1F("Graph_same_area_norm_vs_width2","Area norm same sign with fit vs width",100,0,10.95);
   Graph_same_area_norm_vs_width2->SetMinimum(22308.02);
   Graph_same_area_norm_vs_width2->SetMaximum(23789.3);
   Graph_same_area_norm_vs_width2->SetDirectory(nullptr);
   Graph_same_area_norm_vs_width2->SetStats(0);

   Int_t ci;      // for color index setting
   TColor *color; // for color definition with alpha
   ci = TColor::GetColor("#000099");
   Graph_same_area_norm_vs_width2->SetLineColor(ci);
   Graph_same_area_norm_vs_width2->GetXaxis()->SetTitle("Width set in MC (in GeV)");
   Graph_same_area_norm_vs_width2->GetXaxis()->SetLabelFont(42);
   Graph_same_area_norm_vs_width2->GetXaxis()->SetTitleOffset(1);
   Graph_same_area_norm_vs_width2->GetXaxis()->SetTitleFont(42);
   Graph_same_area_norm_vs_width2->GetYaxis()->SetTitle("Cutflow area normalization values");
   Graph_same_area_norm_vs_width2->GetYaxis()->SetLabelFont(42);
   Graph_same_area_norm_vs_width2->GetYaxis()->SetTitleFont(42);
   Graph_same_area_norm_vs_width2->GetZaxis()->SetLabelFont(42);
   Graph_same_area_norm_vs_width2->GetZaxis()->SetTitleOffset(1);
   Graph_same_area_norm_vs_width2->GetZaxis()->SetTitleFont(42);
   graph->SetHistogram(Graph_same_area_norm_vs_width2);
   
   graph->Draw("ap");
   
   TPaveText *pt = new TPaveText(0.181092,0.9339831,0.818908,0.995,"blNDC");
   pt->SetName("title");
   pt->SetBorderSize(0);
   pt->SetFillColor(0);
   pt->SetFillStyle(0);
   pt->SetTextFont(42);
   TText *pt_LaTex = pt->AddText("Area norm same sign with fit vs width");
   pt->Draw();
   c1->Modified();
   c1->cd();
   c1->SetSelected(c1);
}
