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
#include <string>
#include <TMultiGraph.h>

void hist_combo()
{
   int cat_num = 4, gen_cat_num = 4;
   TH1F* h_cat[cat_num];
   const char* cat[4] = {"bneg","b0bar","b0","bpos"};
   const char* leg_ent[4] = {"B-","B0bar","B0","B+"};
   Color_t col[4] = {kRed,kBlue,kGreen,kOrange};
   string str_proc = "TTToHadronic_TuneCP5_13TeV-powheg-pythia8/nominal/mu/j2+_b1/id_";
   string str_2 = "/hist";
   string file_path = "/afs/desy.de/user/p/paranjpe/outputs_wbwb/chunk_1115_ttbar_had/";
   string cut_req = "Cut_009_btagSF_";

   const char* gen_cat[4] = {"prob_pos","prob_neg","prob_zero","prob_zerobar"};
   for (int j=0;j<gen_cat_num;j++)
   {
      TCanvas *c = new TCanvas("c", "c", 800,800);
      c->Draw();
      string file_str = file_path + cut_req + gen_cat[j] + ".root";
      TFile *f_1 = new TFile(file_str.c_str());

      std::cout << str_proc.c_str() << " "<< str_2.c_str() <<std::endl;

      //================================================Creating legend==========================================

      TLegend *leg = new TLegend(0.7,0.7,0.85,0.85,NULL,"brNDC");
      leg->SetBorderSize(0);
      leg->SetLineColor(1);
      leg->SetLineStyle(1);
      leg->SetLineWidth(1);

      //=========================================================================================================

      for(int i=0;i<cat_num;i++)
      {
         string str_hist = str_proc + cat[i] + str_2;
         h_cat[i] =(TH1F*)f_1->Get(str_hist.c_str());
      }

      TLegendEntry *entry;

      //============================Normalizing and replotting (overlapping) the histograms having the same gen category=======================

      for(int i=0;i<cat_num;i++)
      { 

         //Double_t scale = h_cat[i]->GetXaxis()->GetBinWidth(1)/(h_cat[i]->Integral());
         Double_t scale = 1.0/(h_cat[i]->GetMaximum());
         std::cout << cat[i] << std::endl;
         std::cout << h_cat[i]->GetMaximum() << std::endl;
         std::cout << "Normalization factor " << scale << std::endl;
         h_cat[i]->Scale(scale);
         h_cat[i]->SetLineColor(col[i]);

         if(i==0)
         {
            h_cat[i]->SetStats(0);
            h_cat[i]->Draw("hist");
         }

         else
         {  
            h_cat[i]->SetStats(0);
            h_cat[i]->Draw("hist same");
         }
      entry=leg->AddEntry(h_cat[i],leg_ent[i],"L");
      entry->SetLineColor(1);
      entry->SetLineStyle(1);
      entry->SetLineWidth(1);
      entry->SetMarkerColor(1);
      entry->SetMarkerStyle(20);
      entry->SetMarkerSize(1.15);
      entry->SetTextFont(42);
      entry->SetTextSize(0.02);

      }

      //======================================================================================================================================

      leg->Draw("same");

      //==============================Saving the histograms as .C, .png and .pdf files==========================================

      string macro_path = file_path + cut_req + gen_cat[j] + ".C";
      string png_path = file_path + cut_req + gen_cat[j] + ".png";
      string pdf_path = file_path + cut_req + gen_cat[j] + ".pdf";

      c->Print(macro_path.c_str());
      c->Print(png_path.c_str());
      c->Print(pdf_path.c_str());
      c->Close();

      //===================================================================================================================
   }

   gApplication->Terminate();
}
