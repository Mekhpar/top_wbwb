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

void sf_plot()
{
   string file_path = "/afs/desy.de/user/p/paranjpe/top_wbwb/pepper/inputs/user/ul17/muon/";
   string file_str = file_path + "NUM_IsoMu27_DEN_CutBasedIdTight_and_PFIsoVeryTight_abseta_pt.root";
   TFile *f = new TFile(file_str.c_str());
   string str_hist = "NUM_IsoMu27_DEN_CutBasedIdTight_and_PFIsoVeryTight_abseta_pt";
   TH2F* h_sf = (TH2F*)f->Get(str_hist.c_str());
   int bin_eta = h_sf->GetNbinsX();
   int bin_pt = h_sf->GetNbinsY();
   std::cout << "Number of pt bins " << bin_pt << " Number of bins in eta " << bin_eta << std::endl;
   for(int pt=1;pt<=bin_pt;pt++)
   {
      for(int eta=1;eta<=bin_eta;eta++)
      {
         float current_pt = h_sf->GetYaxis()->GetBinLowEdge(pt);
         float current_eta = h_sf->GetXaxis()->GetBinLowEdge(eta);
         float sf_val = h_sf->GetBinContent(eta,pt);
         std::cout << std::endl;
         std::cout << "current_pt " << current_pt << " current_eta " << current_eta << std::endl;
         std::cout << "Sf value " << sf_val << std::endl;
      }
   }
   //double bincontent = h_sf->GetBinContent(bin);   

   TCanvas *c = new TCanvas("c", "c", 400,400);

   h_sf->Draw("hist");   
   string muon_sf = "muon_sf";
   string macro_path = file_path + muon_sf + ".C";
   string png_path = file_path + muon_sf + ".png";
   string pdf_path = file_path + muon_sf + ".pdf";
   c->Print(macro_path.c_str());
   c->Print(png_path.c_str());
   c->Print(pdf_path.c_str());
   c->Close();

   gApplication->Terminate();
}
