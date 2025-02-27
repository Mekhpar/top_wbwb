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

void pz_comp_wbj()
{
   int cat_num = 2, b_id_mask = 2;
   //int mom_num = 2;
   int mom_num = 1;
   //int mom_num = 5;
   //int mom_num = 6;
   //const char* mom_file[2] = {"non_b_mother_final","b_mother_final"};
   //const char* mom_file[2] = {"mlb","dR_lb"};
   //const char* mom_file[3] = {"mlb","dR_lb","top_max_pt"};
   //const char* mom_file[3] = {"bW_mass_final","W_mass_final","b_lnu_mass_final"};
   //const char* mom_file[2] = {"bW_mass","b_lnu_mass"};
   //const char* mom_file[2] = {"nu_pt","GenMET_pt"};

   //const char* mom_file[4] = {"b_lMET_min_mass","b_lMET_max_mass","b_lnu_mass_final","bW_mass_final"};
   //const char* mom_file[4] = {"bW_mass_final","b_lnu_mass_final","b_lMET_min_mass","b_lMET_max_mass"};
   //const char* mom_file[6] = {"bW_mass_final","b_lnu_mass_final","b_lMET_min_mass","b_lMET_max_mass","b_l_recMET_min_mass_final","b_l_recMET_max_mass_final"};
   //const char* mom_file[2] = {"b_l_recMET_min_mass_final","b_l_recMET_max_mass_final"};
   //const char* mom_file[2] = {"b_l_recMET_min_mass","b_l_recMET_max_mass"};
   const char* mom_file[1] = {"1st_lep_pt"};
   //const char* mom_file[1] = {"b_l_recMET_min_mass"};
   //const char* mom_file[3] = {"min_pz_met","max_pz_met","Gen_nu_pz"};
   //const char* mom_file[5] = {"min_pz_met","max_pz_met","min_pz_recmet", "max_pz_recmet", "Gen_nu_pz"};

   //const char* mom_file[3] = {"1st_lep_iso_4","METphi","METpt"};
   //const char* cat[4] = {"bneg","b0bar","b0","bpos"};
   const char* cat[2] = {"bpos","bneg"};
   //const char* cat[1] = {"bneg"};
   
   string str_proc = "WbjToLNu_4f_TuneCP5_13TeV-madgraph-pythia8/mu/j2+_b1/";
   //string str_proc = "WbjToLNu_4f_TuneCP5_13TeV-madgraph-pythia8/mu/j2_b1/";
   string str_2 = "/hist";
   //string file_path = "/afs/desy.de/user/p/paranjpe/outputs_wbwb/chunk_0206_wbj/";
   //string file_path = "/afs/desy.de/user/p/paranjpe/outputs_wbwb/chunk_0206_wbj_1/";
   //string file_path = "/afs/desy.de/user/p/paranjpe/outputs_wbwb/chunk_0213_wbj/";
   //string file_path = "/afs/desy.de/user/p/paranjpe/outputs_wbwb/debug_0214_wbj_cuts/";

   //string file_path = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0218_wbj_1/";
   //string file_path = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/debug_0224_wbj/";
   //string file_path = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0224_wbj_30/";
   //string file_path = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0224_wbj/";

   //string file_path = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/debug_0225_wbj/";
   string file_path = "/data/dust/user/paranjpe/af-cms.merged/outputs_wbwb/chunk_0225_wbj/";

   string cut_req = "Cut_009_btagSF_";
   //string cut_req = "Cut_010_btagSF_"; //After adding cut hasfreshgenlepton
   //string cut_req = "Cut_009_HasOneGenLepton_"; //After adding cut hasfreshgenlepton

   //const char* gen_cat[4] = {"prob_pos","prob_neg","prob_zero","prob_zerobar"};

      //Color_t col[6] = {kRed,kBlue,kGreen,kOrange,kMagenta,kCyan};
      //Color_t col[2] = {kRed,kBlue};
      Color_t col[4] = {kGreen,kOrange,kRed,kBlue}; //Overlap so 2*2
      //Color_t col[5] = {kRed,kBlue,kGreen,kOrange,kMagenta};
      //Color_t col[6] = {kRed,kBlue,kGreen,kOrange,kMagenta,kCyan};

      //TCanvas *c = new TCanvas("c", "c", 800,1200);
      TCanvas *c = new TCanvas("c", "c", 800,400);
      //TCanvas *c = new TCanvas("c", "c", 800,800);
      //c->Divide(b_id_mask,mom_num);
      c->Divide(b_id_mask,1);
      //string discr_sign_str = "real";

      //btag veto categories
      string discr_all[5] = {"b_loose_0","b_loose_1+_no_effect","b_loose_1+_leading_inconclusive_50","b_loose_1+_leading_conclusive_50_right","b_loose_1+_leading_conclusive_50_wrong"};
      string discr_sign_str = "b_loose_0";

   for (int j=0;j<cat_num;j++)
   {

      const char* b_mask[2];
      //const char* leg_entry[2];
      const char* leg_entry[4];
      if(strcmp(cat[j],"bpos")==0)
      {
         ///*
         //b_mask[0] = "top_b_w_l_r/";
         //b_mask[1] = "no_top_l_r_lpos/";

         b_mask[0] = "top_same_sign/";
         b_mask[1] = "no_top_same_sign/";
         
         //leg_entry[0] = "Wrongly chosen b jets from onshell top";
         //leg_entry[1] = "Offshell top (positive W charge) supposed t channel";

         leg_entry[0] = "Wrongly chosen b jets from onshell top (zero loose jets)";
         leg_entry[1] = "Offshell top (positive W charge) supposed t channel (zero loose jets)";
         leg_entry[2] = "Wrongly chosen b jets from onshell top";
         leg_entry[3] = "Offshell top (positive W charge) supposed t channel";

         //*/

         /*
         b_mask[0] = "antitop_b_r_l_r/";
         b_mask[1] = "top_b_w_l_r/";

         leg_entry[0] = "Wrongly chosen b jets from onshell top";
         leg_entry[1] = "Correctly chosen b jets from onshell antitop";

         */

      }
      else if(strcmp(cat[j],"bneg")==0)
      {
         ///*
         //b_mask[0] = "antitop_b_w_l_r/";
         //b_mask[1] = "no_top_l_r_lneg/";

         b_mask[0] = "antitop_same_sign/";
         b_mask[1] = "no_top_same_sign/";

         //leg_entry[0] = "Wrongly chosen b jets from onshell antitop";
         //leg_entry[1] = "Offshell antitop (negative W charge) supposed t channel";

         leg_entry[0] = "Wrongly chosen b jets from onshell antitop (zero loose jets)";
         leg_entry[1] = "Offshell antitop (negative W charge) supposed t channel (zero loose jets)";
         leg_entry[2] = "Wrongly chosen b jets from onshell antitop";
         leg_entry[3] = "Offshell antitop (negative W charge) supposed t channel";

         //*/

         /*
         b_mask[0] = "top_b_r_l_r/";
         b_mask[1] = "antitop_b_w_l_r/";

         leg_entry[0] = "Wrongly chosen b jets from onshell antitop";
         leg_entry[1] = "Correctly chosen b jets from onshell top";

         */


      }

      //TCanvas *c = new TCanvas("c", "c", 400,400);
      //c->Divide(b_id_mask,1);
      TH1F* h_cat[mom_num][b_id_mask];
      TH1F* h_sum[mom_num][b_id_mask];
      //const char* b_mask[2];

      for(int i=0;i<b_id_mask;i++)
      {
         string id = "id_", rec = "rec_";

         string id_cat = id + cat[j] + "_" + rec + cat[j];
         std::cout << "i " << i << "cat[j] " << cat[j] << std::endl;
         string str_hist = str_proc + id_cat + "/conclusive_50/" + b_mask[i] + discr_sign_str + str_2;
         //string str_hist = str_proc + id_cat + "/conclusive_50/" + b_mask[i] + str_2;
         std::cout << "Histogram full path " << str_hist << std::endl;

         for(int k=0;k<mom_num;k++)
         {
            //string file_str = file_path + cut_req + mom_file[k] + "_final.root";
            string file_str = file_path + cut_req + mom_file[k] + ".root";
            TFile *f = new TFile(file_str.c_str());
            std::cout << "File path " << file_str << std::endl;

            std::cout << str_proc.c_str() << " "<< str_2.c_str() <<std::endl;
            
            h_cat[k][i] = (TH1F*)f->Get(str_hist.c_str());
            std::cout <<  "Got individual hist " <<std::endl;
            TList *list = new TList;
            TH1F* h_sum_ind[5];
            std::cout <<  "Defined list and hists for addition " <<std::endl;
            for(int hist_cat=0;hist_cat<5;hist_cat++)
            {  
               string str_hist_sum = str_proc + id_cat + "/conclusive_50/" + b_mask[i] + discr_all[hist_cat] + str_2;
               h_sum_ind[hist_cat] = (TH1F*)f->Get(str_hist_sum.c_str());               
            }
            for(int hist_cat=0;hist_cat<5;hist_cat++)
            {
               list->Add(h_sum_ind[hist_cat]);  
            }
            std::cout <<  "Added everything to list " <<std::endl;
            h_sum[k][i] = (TH1F*)h_sum_ind[0]->Clone("h_sum[k][i]");
            h_sum[k][i]->Reset();            
            h_sum[k][i]->Merge(list);
         }
      }

      std::cout << "b_id_mask " << b_id_mask << std::endl;
      //================================================Creating legend==========================================

      //TLegend *leg = new TLegend(0.7,0.7,0.85,0.85,NULL,"brNDC");
      TLegend *leg = new TLegend(0.5,0.7,0.65,0.85,NULL,"brNDC");
      //TLegend *leg = new TLegend(0.15,0.7,0.25,0.85,NULL,"brNDC");
      leg->SetBorderSize(0);
      leg->SetLineColor(1);
      leg->SetLineStyle(1);
      leg->SetLineWidth(1);
      TLegendEntry *entry;

      std::cout << "j inside plotting loop " << j << std::endl;
      int canvas_div = j+1;
      std::cout << "Canvas division " << canvas_div << std::endl;
      c->cd(canvas_div);

      //=========================================================================================================

      for(int i=0;i<b_id_mask;i++)
      {

         float mass_min = 100., mass_max = 250.;
         for(int k=0;k<mom_num;k++)
         {
            //int index = i*b_id_mask+k;
            int index = i*mom_num+k;
            h_cat[k][i]->SetStats(0);
            h_cat[k][i]->SetLineColor(col[index]);

            h_sum[k][i]->SetStats(0);
            h_sum[k][i]->SetLineColor(col[b_id_mask+index]);

            //float scale = 1/h_cat[k][i]->Integral();
            float scale = 1/h_cat[k][i]->GetMaximum();
            std::cout << "Maximum height " << scale << std::endl;
            std::cout << "Integral (total number of events) " << h_cat[k][i]->Integral() << std::endl;

            std::cout << "Integral in h_sum (total number of events) " << h_sum[k][i]->Integral() << std::endl;
            std::cout << "Integral in h_sum (events in 20-30 GeV) " << h_sum[k][i]->Integral(2,3) << std::endl;
            //h_cat[k][i]->Scale(scale);

            /*
            if(i==0 && k==0)
            {
             h_cat[k][i]->Draw("hist");
            }
            else
            {
             h_cat[k][i]->Draw("hist same");
            }
             */

             if(i==0 && k==0)
             {
              h_sum[k][i]->Draw("hist");
             }
             else
             {
               h_sum[k][i]->Draw("hist same");
             }
 
             h_cat[k][i]->Draw("hist same");
             h_cat[k][i]->GetXaxis()->SetTitle(mom_file[k]);
            //h_cat[k][i]->GetXaxis()->SetRangeUser(100.,250.);
            //h_cat[k][i]->GetXaxis()->SetRangeUser(-300.,300.); //For actual (with sign) value of pz
            //h_cat[k][i]->GetXaxis()->SetRangeUser(0.,200.); //For pt comparison
            std::string buf(mom_file[k]);
            buf.append("_");
            buf.append(cat[j]);
            std::cout << "Buf " << buf << std::endl;
            //entry=leg->AddEntry(h_cat[k][i], mom_file[k], "L");
            //entry=leg->AddEntry(h_cat[k][i], buf.c_str(), "L");
            //entry=leg->AddEntry(h_cat[k][i], b_mask[i], "L");
            entry=leg->AddEntry(h_cat[k][i], leg_entry[i], "L");
            std::cout << "Index " << index << std::endl;
            std::cout << "Index " << b_id_mask+index << std::endl;
            std::cout << "Legend color " << col[index] << std::endl;

            entry->SetLineColor(col[index]);
            entry->SetLineStyle(1);
            entry->SetLineWidth(1);
            entry->SetMarkerColor(1);
            entry->SetMarkerStyle(20);
            entry->SetMarkerSize(1.15);
            entry->SetTextFont(42);
            entry->SetTextSize(0.02);

            entry=leg->AddEntry(h_sum[k][i], leg_entry[b_id_mask+i], "L");
            entry->SetLineColor(col[b_id_mask+index]);

            entry->SetLineStyle(1);
            entry->SetLineWidth(1);
            entry->SetMarkerColor(1);
            entry->SetMarkerStyle(20);
            entry->SetMarkerSize(1.15);
            entry->SetTextFont(42);
            entry->SetTextSize(0.02);

         }

         //c->Draw();
      }
      
      leg->Draw("same");
      c->Draw();
   }
      //==============================Saving the histograms as .C, .png and .pdf files==========================================
      ///*
      /*
      string macro_path = file_path + cut_req +"nu_MET_pt_"+ discr_sign_str + ".C";
      string png_path = file_path + cut_req + "nu_MET_pt_"+ discr_sign_str + ".png";
      string pdf_path = file_path + cut_req + "nu_MET_pt_"+ discr_sign_str + ".pdf";
      */
      /*
      string macro_path = file_path + cut_req + "b_lMET_min_mass_reco_overlap_"+ discr_sign_str + ".C";
      string png_path = file_path + cut_req + "b_lMET_min_mass_reco_overlap_"+ discr_sign_str + ".png";
      string pdf_path = file_path + cut_req + "b_lMET_min_mass_reco_overlap_"+ discr_sign_str + ".pdf";
      */

      ///*
      string macro_path = file_path + cut_req + "1st_lep_pt_"+ discr_sign_str + ".C";
      string png_path = file_path + cut_req + "1st_lep_pt_"+ discr_sign_str + ".png";
      string pdf_path = file_path + cut_req + "1st_lep_pt_"+ discr_sign_str + ".pdf";
      //*/
      
      /*
      string macro_path = file_path + cut_req + "pz_comp_"+ discr_sign_str + ".C";
      string png_path = file_path + cut_req + "pz_comp_"+ discr_sign_str + ".png";
      string pdf_path = file_path + cut_req + "pz_comp_"+ discr_sign_str + ".pdf";
      */
      ///*
      c->Print(macro_path.c_str());
      c->Print(png_path.c_str());
      c->Print(pdf_path.c_str());
      c->Close();
      //*/
      //*/
      //===================================================================================================================
 
   /*
   for (int j=0;j<cat_num;j++)
   {
      //TCanvas *c = new TCanvas("c", "c", 800,1200);
      TCanvas *c = new TCanvas("c", "c", 800,400);
      //TCanvas *c = new TCanvas("c", "c", 800,800);
      //c->Divide(b_id_mask,mom_num);
      c->Divide(b_id_mask,1);

      const char* b_mask[2];
      if(strcmp(cat[j],"bpos")==0)
      {
       b_mask[0] = "top/b_w_l_r/non_zero_daughters";
       b_mask[1] = "antitop/b_r_l_r/non_zero_daughters";

      }
      else if(strcmp(cat[j],"bneg")==0)
      {
       //b_mask[0] = "antitop/b_w_l_r";
       //b_mask[1] = "top/b_r_l_r";

       b_mask[0] = "antitop/b_w_l_r/non_zero_daughters";
       b_mask[1] = "top/b_r_l_r/non_zero_daughters";

      }

  }
   */

   gApplication->Terminate();
}
