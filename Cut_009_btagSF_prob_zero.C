#ifdef __CLING__
#pragma cling optimize(0)
#endif
void Cut_009_btagSF_prob_zero()
{
//=========Macro generated from canvas: c/c
//=========  (Mon Nov 18 10:20:54 2024) by ROOT version 6.28/04
   TCanvas *c = new TCanvas("c", "c",0,0,800,800);
   c->Range(0,0,1,1);
   c->SetFillColor(0);
   c->SetBorderMode(0);
   c->SetBorderSize(2);
   c->SetFrameBorderMode(0);
   
   TH1D *hist__9 = new TH1D("hist__9","",100,0,1);
   hist__9->SetBinContent(1,0.1266925);
   hist__9->SetBinContent(2,0.6286864);
   hist__9->SetBinContent(3,0.7289763);
   hist__9->SetBinContent(4,0.7497472);
   hist__9->SetBinContent(5,0.8705286);
   hist__9->SetBinContent(6,0.8275658);
   hist__9->SetBinContent(7,0.8318696);
   hist__9->SetBinContent(8,0.8189638);
   hist__9->SetBinContent(9,0.8220377);
   hist__9->SetBinContent(10,0.9373125);
   hist__9->SetBinContent(11,0.9775023);
   hist__9->SetBinContent(12,1);
   hist__9->SetBinContent(13,0.9488532);
   hist__9->SetBinContent(14,0.878943);
   hist__9->SetBinContent(15,0.903408);
   hist__9->SetBinContent(16,0.9281133);
   hist__9->SetBinContent(17,0.8072202);
   hist__9->SetBinContent(18,0.7908519);
   hist__9->SetBinContent(19,0.8579255);
   hist__9->SetBinContent(20,0.9526801);
   hist__9->SetBinContent(21,0.958494);
   hist__9->SetBinContent(22,0.8481279);
   hist__9->SetBinContent(23,0.7927045);
   hist__9->SetBinContent(24,0.6532011);
   hist__9->SetBinContent(25,0.6944971);
   hist__9->SetBinContent(26,0.9528216);
   hist__9->SetBinContent(27,0.5631497);
   hist__9->SetBinContent(28,0.5431543);
   hist__9->SetBinContent(29,0.5268245);
   hist__9->SetBinContent(30,0.4142767);
   hist__9->SetBinContent(31,0.4230012);
   hist__9->SetBinContent(32,0.308127);
   hist__9->SetBinContent(33,0.2508915);
   hist__9->SetBinContent(34,0.3103318);
   hist__9->SetBinContent(35,0.2353811);
   hist__9->SetBinContent(36,0.2582363);
   hist__9->SetBinContent(37,0.1687572);
   hist__9->SetBinContent(38,0.1543411);
   hist__9->SetBinContent(39,0.2424806);
   hist__9->SetBinContent(40,0.1216555);
   hist__9->SetBinContent(41,0.1130282);
   hist__9->SetBinContent(42,0.1025169);
   hist__9->SetBinContent(43,0.09512249);
   hist__9->SetBinContent(44,0.1145637);
   hist__9->SetBinContent(45,0.06738892);
   hist__9->SetBinContent(46,0.0681394);
   hist__9->SetBinContent(47,0.04863111);
   hist__9->SetBinContent(48,0.03993307);
   hist__9->SetBinContent(49,0.05201445);
   hist__9->SetBinContent(50,0.04261764);
   hist__9->SetBinContent(51,0.01931431);
   hist__9->SetBinContent(52,0.03682948);
   hist__9->SetBinContent(53,0.01491981);
   hist__9->SetBinContent(54,0.03177257);
   hist__9->SetBinContent(55,0.02417584);
   hist__9->SetBinContent(56,0.02258288);
   hist__9->SetBinContent(57,0.007476204);
   hist__9->SetBinContent(58,0.01392373);
   hist__9->SetBinContent(59,0.01348009);
   hist__9->SetBinContent(61,0.02076819);
   hist__9->SetBinContent(62,0.005320031);
   hist__9->SetBinContent(63,0.005052467);
   hist__9->SetBinContent(64,0.01404092);
   hist__9->SetBinContent(65,0.005048657);
   hist__9->SetBinContent(68,0.004808959);
   hist__9->SetBinContent(81,0.006449419);
   hist__9->SetBinError(1,0.02555284);
   hist__9->SetBinError(2,0.05595328);
   hist__9->SetBinError(3,0.05981337);
   hist__9->SetBinError(4,0.06121697);
   hist__9->SetBinError(5,0.0665797);
   hist__9->SetBinError(6,0.06419589);
   hist__9->SetBinError(7,0.06401661);
   hist__9->SetBinError(8,0.06403004);
   hist__9->SetBinError(9,0.0640437);
   hist__9->SetBinError(10,0.06902947);
   hist__9->SetBinError(11,0.07052555);
   hist__9->SetBinError(12,0.07080739);
   hist__9->SetBinError(13,0.06867067);
   hist__9->SetBinError(14,0.06661451);
   hist__9->SetBinError(15,0.06788341);
   hist__9->SetBinError(16,0.0677909);
   hist__9->SetBinError(17,0.0645884);
   hist__9->SetBinError(18,0.0627185);
   hist__9->SetBinError(19,0.06561809);
   hist__9->SetBinError(20,0.06960294);
   hist__9->SetBinError(21,0.07016557);
   hist__9->SetBinError(22,0.06478272);
   hist__9->SetBinError(23,0.06347568);
   hist__9->SetBinError(24,0.05717215);
   hist__9->SetBinError(25,0.05906439);
   hist__9->SetBinError(26,0.0695399);
   hist__9->SetBinError(27,0.0523973);
   hist__9->SetBinError(28,0.05201854);
   hist__9->SetBinError(29,0.05144827);
   hist__9->SetBinError(30,0.04583437);
   hist__9->SetBinError(31,0.04626597);
   hist__9->SetBinError(32,0.03913418);
   hist__9->SetBinError(33,0.03621844);
   hist__9->SetBinError(34,0.04029571);
   hist__9->SetBinError(35,0.03404533);
   hist__9->SetBinError(36,0.03618186);
   hist__9->SetBinError(37,0.02835009);
   hist__9->SetBinError(38,0.02882162);
   hist__9->SetBinError(39,0.03521214);
   hist__9->SetBinError(40,0.02411348);
   hist__9->SetBinError(41,0.02331024);
   hist__9->SetBinError(42,0.02206955);
   hist__9->SetBinError(43,0.02152803);
   hist__9->SetBinError(44,0.02381705);
   hist__9->SetBinError(45,0.01836615);
   hist__9->SetBinError(46,0.01841499);
   hist__9->SetBinError(47,0.01727297);
   hist__9->SetBinError(48,0.01335376);
   hist__9->SetBinError(49,0.01577511);
   hist__9->SetBinError(50,0.01426493);
   hist__9->SetBinError(51,0.009667848);
   hist__9->SetBinError(52,0.01397156);
   hist__9->SetBinError(53,0.008669429);
   hist__9->SetBinError(54,0.01307495);
   hist__9->SetBinError(55,0.01085267);
   hist__9->SetBinError(56,0.01141459);
   hist__9->SetBinError(57,0.00552923);
   hist__9->SetBinError(58,0.008058494);
   hist__9->SetBinError(59,0.007804392);
   hist__9->SetBinError(61,0.01040112);
   hist__9->SetBinError(62,0.005320031);
   hist__9->SetBinError(63,0.005052467);
   hist__9->SetBinError(64,0.008148656);
   hist__9->SetBinError(65,0.005048657);
   hist__9->SetBinError(68,0.004808959);
   hist__9->SetBinError(81,0.006449419);
   hist__9->SetEntries(350.8596);
   hist__9->SetStats(0);

   Int_t ci;      // for color index setting
   TColor *color; // for color definition with alpha
   ci = TColor::GetColor("#ff0000");
   hist__9->SetLineColor(ci);
   hist__9->GetXaxis()->SetTitle("$prob_{B0}$");
   hist__9->GetXaxis()->SetLabelFont(42);
   hist__9->GetXaxis()->SetTitleOffset(1);
   hist__9->GetXaxis()->SetTitleFont(42);
   hist__9->GetYaxis()->SetLabelFont(42);
   hist__9->GetYaxis()->SetTitleOffset(1);
   hist__9->GetYaxis()->SetTitleFont(42);
   hist__9->GetZaxis()->SetLabelFont(42);
   hist__9->GetZaxis()->SetTitleOffset(1);
   hist__9->GetZaxis()->SetTitleFont(42);
   hist__9->Draw("hist");
   
   TH1D *hist__10 = new TH1D("hist__10","",100,0,1);
   hist__10->SetBinContent(1,0.2227837);
   hist__10->SetBinContent(2,0.3582324);
   hist__10->SetBinContent(3,0.4961394);
   hist__10->SetBinContent(4,0.521952);
   hist__10->SetBinContent(5,0.6892379);
   hist__10->SetBinContent(6,0.569884);
   hist__10->SetBinContent(7,0.6706065);
   hist__10->SetBinContent(8,0.8178784);
   hist__10->SetBinContent(9,0.7667809);
   hist__10->SetBinContent(10,0.8422644);
   hist__10->SetBinContent(11,0.8907095);
   hist__10->SetBinContent(12,0.8103974);
   hist__10->SetBinContent(13,0.9497581);
   hist__10->SetBinContent(14,1);
   hist__10->SetBinContent(15,0.9677867);
   hist__10->SetBinContent(16,0.957514);
   hist__10->SetBinContent(17,0.9661686);
   hist__10->SetBinContent(18,0.9203395);
   hist__10->SetBinContent(19,0.9751216);
   hist__10->SetBinContent(20,0.9155255);
   hist__10->SetBinContent(21,0.7901169);
   hist__10->SetBinContent(22,0.8108269);
   hist__10->SetBinContent(23,0.8621656);
   hist__10->SetBinContent(24,0.7100531);
   hist__10->SetBinContent(25,0.5947664);
   hist__10->SetBinContent(26,0.9829938);
   hist__10->SetBinContent(27,0.6860354);
   hist__10->SetBinContent(28,0.440237);
   hist__10->SetBinContent(29,0.4787136);
   hist__10->SetBinContent(30,0.4359084);
   hist__10->SetBinContent(31,0.4561493);
   hist__10->SetBinContent(32,0.3371394);
   hist__10->SetBinContent(33,0.3034498);
   hist__10->SetBinContent(34,0.2921581);
   hist__10->SetBinContent(35,0.251279);
   hist__10->SetBinContent(36,0.1911461);
   hist__10->SetBinContent(37,0.1954411);
   hist__10->SetBinContent(38,0.1631829);
   hist__10->SetBinContent(39,0.1289713);
   hist__10->SetBinContent(40,0.1443541);
   hist__10->SetBinContent(41,0.07872484);
   hist__10->SetBinContent(42,0.06961415);
   hist__10->SetBinContent(43,0.07199774);
   hist__10->SetBinContent(44,0.06620483);
   hist__10->SetBinContent(45,0.04553547);
   hist__10->SetBinContent(46,0.05339327);
   hist__10->SetBinContent(47,0.02017969);
   hist__10->SetBinContent(48,0.02859606);
   hist__10->SetBinContent(49,0.03040111);
   hist__10->SetBinContent(50,0.02505752);
   hist__10->SetBinContent(51,0.04514421);
   hist__10->SetBinContent(52,0.0411019);
   hist__10->SetBinContent(53,0.004208674);
   hist__10->SetBinContent(54,0.05078202);
   hist__10->SetBinContent(55,0.02220073);
   hist__10->SetBinContent(56,0.0107583);
   hist__10->SetBinContent(57,0.03043398);
   hist__10->SetBinContent(58,0.01542163);
   hist__10->SetBinContent(59,0.02033245);
   hist__10->SetBinContent(60,0.02279026);
   hist__10->SetBinContent(61,0.01920621);
   hist__10->SetBinContent(62,0.006778058);
   hist__10->SetBinContent(63,0.004746823);
   hist__10->SetBinContent(64,0.005095966);
   hist__10->SetBinContent(65,0.0105677);
   hist__10->SetBinContent(66,0.005007535);
   hist__10->SetBinContent(67,0.004699797);
   hist__10->SetBinContent(68,0.01017951);
   hist__10->SetBinContent(69,0.005467766);
   hist__10->SetBinContent(70,0.004769813);
   hist__10->SetBinContent(71,0.01116757);
   hist__10->SetBinContent(73,0.009513778);
   hist__10->SetBinError(1,0.03377861);
   hist__10->SetBinError(2,0.04153793);
   hist__10->SetBinError(3,0.04958124);
   hist__10->SetBinError(4,0.05104368);
   hist__10->SetBinError(5,0.05843236);
   hist__10->SetBinError(6,0.05381812);
   hist__10->SetBinError(7,0.05834084);
   hist__10->SetBinError(8,0.06449667);
   hist__10->SetBinError(9,0.0620225);
   hist__10->SetBinError(10,0.06461121);
   hist__10->SetBinError(11,0.06656135);
   hist__10->SetBinError(12,0.06334062);
   hist__10->SetBinError(13,0.06905645);
   hist__10->SetBinError(14,0.07041511);
   hist__10->SetBinError(15,0.06942009);
   hist__10->SetBinError(16,0.06912223);
   hist__10->SetBinError(17,0.0698433);
   hist__10->SetBinError(18,0.06800409);
   hist__10->SetBinError(19,0.07040347);
   hist__10->SetBinError(20,0.06788949);
   hist__10->SetBinError(21,0.06351407);
   hist__10->SetBinError(22,0.06436288);
   hist__10->SetBinError(23,0.0658874);
   hist__10->SetBinError(24,0.05995197);
   hist__10->SetBinError(25,0.05458597);
   hist__10->SetBinError(26,0.07065085);
   hist__10->SetBinError(27,0.05887582);
   hist__10->SetBinError(28,0.04667328);
   hist__10->SetBinError(29,0.04857613);
   hist__10->SetBinError(30,0.04723028);
   hist__10->SetBinError(31,0.04757596);
   hist__10->SetBinError(32,0.04089958);
   hist__10->SetBinError(33,0.03925465);
   hist__10->SetBinError(34,0.03834461);
   hist__10->SetBinError(35,0.0348686);
   hist__10->SetBinError(36,0.0307726);
   hist__10->SetBinError(37,0.0311278);
   hist__10->SetBinError(38,0.02836067);
   hist__10->SetBinError(39,0.02461411);
   hist__10->SetBinError(40,0.02701617);
   hist__10->SetBinError(41,0.01851139);
   hist__10->SetBinError(42,0.01824101);
   hist__10->SetBinError(43,0.01941402);
   hist__10->SetBinError(44,0.01792574);
   hist__10->SetBinError(45,0.014775);
   hist__10->SetBinError(46,0.01615593);
   hist__10->SetBinError(47,0.01010208);
   hist__10->SetBinError(48,0.01291349);
   hist__10->SetBinError(49,0.01175538);
   hist__10->SetBinError(50,0.01127629);
   hist__10->SetBinError(51,0.0160452);
   hist__10->SetBinError(52,0.01461512);
   hist__10->SetBinError(53,0.004208674);
   hist__10->SetBinError(54,0.01615121);
   hist__10->SetBinError(55,0.01128621);
   hist__10->SetBinError(56,0.007722712);
   hist__10->SetBinError(57,0.01169667);
   hist__10->SetBinError(58,0.008925286);
   hist__10->SetBinError(59,0.01033463);
   hist__10->SetBinError(60,0.01051635);
   hist__10->SetBinError(61,0.009625233);
   hist__10->SetBinError(62,0.005080737);
   hist__10->SetBinError(63,0.004746823);
   hist__10->SetBinError(64,0.005095966);
   hist__10->SetBinError(65,0.007505839);
   hist__10->SetBinError(66,0.005007535);
   hist__10->SetBinError(67,0.004699797);
   hist__10->SetBinError(68,0.007217545);
   hist__10->SetBinError(69,0.005467766);
   hist__10->SetBinError(70,0.004769813);
   hist__10->SetBinError(71,0.007905633);
   hist__10->SetBinError(73,0.00674654);
   hist__10->SetEntries(332.1914);
   hist__10->SetStats(0);

   ci = TColor::GetColor("#0000ff");
   hist__10->SetLineColor(ci);
   hist__10->GetXaxis()->SetTitle("$prob_{B0}$");
   hist__10->GetXaxis()->SetLabelFont(42);
   hist__10->GetXaxis()->SetTitleOffset(1);
   hist__10->GetXaxis()->SetTitleFont(42);
   hist__10->GetYaxis()->SetLabelFont(42);
   hist__10->GetYaxis()->SetTitleOffset(1);
   hist__10->GetYaxis()->SetTitleFont(42);
   hist__10->GetZaxis()->SetLabelFont(42);
   hist__10->GetZaxis()->SetTitleOffset(1);
   hist__10->GetZaxis()->SetTitleFont(42);
   hist__10->Draw("hist same");
   
   TH1D *hist__11 = new TH1D("hist__11","",100,0,1);
   hist__11->SetBinContent(1,0.009317409);
   hist__11->SetBinContent(2,0.004987231);
   hist__11->SetBinContent(3,0.03970499);
   hist__11->SetBinContent(4,0.0774617);
   hist__11->SetBinContent(5,0.1456185);
   hist__11->SetBinContent(6,0.1909939);
   hist__11->SetBinContent(7,0.2019747);
   hist__11->SetBinContent(8,0.2056091);
   hist__11->SetBinContent(9,0.2901408);
   hist__11->SetBinContent(10,0.3094317);
   hist__11->SetBinContent(11,0.3252529);
   hist__11->SetBinContent(12,0.473923);
   hist__11->SetBinContent(13,0.5703948);
   hist__11->SetBinContent(14,0.5174586);
   hist__11->SetBinContent(15,0.7052806);
   hist__11->SetBinContent(16,0.6564633);
   hist__11->SetBinContent(17,0.6585434);
   hist__11->SetBinContent(18,0.7214827);
   hist__11->SetBinContent(19,0.700648);
   hist__11->SetBinContent(20,0.5874864);
   hist__11->SetBinContent(21,0.8184967);
   hist__11->SetBinContent(22,0.8605479);
   hist__11->SetBinContent(23,0.8868706);
   hist__11->SetBinContent(24,0.9736515);
   hist__11->SetBinContent(25,0.9133216);
   hist__11->SetBinContent(26,1);
   hist__11->SetBinContent(27,0.8152514);
   hist__11->SetBinContent(28,0.7736638);
   hist__11->SetBinContent(29,0.7034878);
   hist__11->SetBinContent(30,0.7490596);
   hist__11->SetBinContent(31,0.5639527);
   hist__11->SetBinContent(32,0.6118728);
   hist__11->SetBinContent(33,0.5530292);
   hist__11->SetBinContent(34,0.4939382);
   hist__11->SetBinContent(35,0.4226049);
   hist__11->SetBinContent(36,0.4820343);
   hist__11->SetBinContent(37,0.4196996);
   hist__11->SetBinContent(38,0.3688708);
   hist__11->SetBinContent(39,0.3947272);
   hist__11->SetBinContent(40,0.4384535);
   hist__11->SetBinContent(41,0.3593443);
   hist__11->SetBinContent(42,0.3257339);
   hist__11->SetBinContent(43,0.3575171);
   hist__11->SetBinContent(44,0.2841911);
   hist__11->SetBinContent(45,0.2587126);
   hist__11->SetBinContent(46,0.2762032);
   hist__11->SetBinContent(47,0.196286);
   hist__11->SetBinContent(48,0.19193);
   hist__11->SetBinContent(49,0.1507471);
   hist__11->SetBinContent(50,0.1943727);
   hist__11->SetBinContent(51,0.1447034);
   hist__11->SetBinContent(52,0.1810746);
   hist__11->SetBinContent(53,0.1636204);
   hist__11->SetBinContent(54,0.1364279);
   hist__11->SetBinContent(55,0.1516607);
   hist__11->SetBinContent(56,0.1377542);
   hist__11->SetBinContent(57,0.164886);
   hist__11->SetBinContent(58,0.09280839);
   hist__11->SetBinContent(59,0.08911639);
   hist__11->SetBinContent(60,0.09072891);
   hist__11->SetBinContent(61,0.1030888);
   hist__11->SetBinContent(62,0.08992709);
   hist__11->SetBinContent(63,0.04024189);
   hist__11->SetBinContent(64,0.08452445);
   hist__11->SetBinContent(65,0.08012991);
   hist__11->SetBinContent(66,0.05763808);
   hist__11->SetBinContent(67,0.08797259);
   hist__11->SetBinContent(68,0.07449717);
   hist__11->SetBinContent(69,0.04183133);
   hist__11->SetBinContent(70,0.05998693);
   hist__11->SetBinContent(71,0.04697206);
   hist__11->SetBinContent(72,0.01355486);
   hist__11->SetBinContent(73,0.05145198);
   hist__11->SetBinContent(74,0.03016645);
   hist__11->SetBinContent(75,0.01495987);
   hist__11->SetBinContent(76,0.03002257);
   hist__11->SetBinContent(77,0.009414985);
   hist__11->SetBinContent(78,0.02347174);
   hist__11->SetBinContent(79,0.002682157);
   hist__11->SetBinContent(80,0.02572003);
   hist__11->SetBinContent(81,0.03765422);
   hist__11->SetBinContent(82,0.01634698);
   hist__11->SetBinContent(83,0.008779598);
   hist__11->SetBinContent(84,0.009698718);
   hist__11->SetBinContent(85,0.03412412);
   hist__11->SetBinContent(86,0.004277829);
   hist__11->SetBinContent(87,0.004050353);
   hist__11->SetBinContent(88,0.01723826);
   hist__11->SetBinContent(90,0.01056572);
   hist__11->SetBinContent(91,0.004484216);
   hist__11->SetBinError(1,0.006637327);
   hist__11->SetBinError(2,0.004987231);
   hist__11->SetBinError(3,0.0135104);
   hist__11->SetBinError(4,0.02015852);
   hist__11->SetBinError(5,0.02716205);
   hist__11->SetBinError(6,0.0305855);
   hist__11->SetBinError(7,0.03128382);
   hist__11->SetBinError(8,0.03132548);
   hist__11->SetBinError(9,0.03680231);
   hist__11->SetBinError(10,0.0403172);
   hist__11->SetBinError(11,0.04048864);
   hist__11->SetBinError(12,0.04824833);
   hist__11->SetBinError(13,0.05293822);
   hist__11->SetBinError(14,0.05068261);
   hist__11->SetBinError(15,0.05900218);
   hist__11->SetBinError(16,0.0567803);
   hist__11->SetBinError(17,0.0572475);
   hist__11->SetBinError(18,0.0593735);
   hist__11->SetBinError(19,0.05810559);
   hist__11->SetBinError(20,0.05385933);
   hist__11->SetBinError(21,0.06355199);
   hist__11->SetBinError(22,0.06578634);
   hist__11->SetBinError(23,0.06579696);
   hist__11->SetBinError(24,0.06948942);
   hist__11->SetBinError(25,0.0674388);
   hist__11->SetBinError(26,0.07021983);
   hist__11->SetBinError(27,0.06433056);
   hist__11->SetBinError(28,0.06214126);
   hist__11->SetBinError(29,0.05948695);
   hist__11->SetBinError(30,0.06061171);
   hist__11->SetBinError(31,0.05259596);
   hist__11->SetBinError(32,0.0544928);
   hist__11->SetBinError(33,0.05195764);
   hist__11->SetBinError(34,0.05020451);
   hist__11->SetBinError(35,0.04566859);
   hist__11->SetBinError(36,0.04895689);
   hist__11->SetBinError(37,0.04525126);
   hist__11->SetBinError(38,0.04325247);
   hist__11->SetBinError(39,0.04367416);
   hist__11->SetBinError(40,0.0474393);
   hist__11->SetBinError(41,0.04269581);
   hist__11->SetBinError(42,0.03914412);
   hist__11->SetBinError(43,0.04195351);
   hist__11->SetBinError(44,0.03767489);
   hist__11->SetBinError(45,0.0357517);
   hist__11->SetBinError(46,0.03709921);
   hist__11->SetBinError(47,0.03095715);
   hist__11->SetBinError(48,0.02979026);
   hist__11->SetBinError(49,0.02708279);
   hist__11->SetBinError(50,0.03098353);
   hist__11->SetBinError(51,0.0273965);
   hist__11->SetBinError(52,0.02990368);
   hist__11->SetBinError(53,0.02865993);
   hist__11->SetBinError(54,0.02778532);
   hist__11->SetBinError(55,0.02681558);
   hist__11->SetBinError(56,0.02643724);
   hist__11->SetBinError(57,0.02863057);
   hist__11->SetBinError(58,0.02168234);
   hist__11->SetBinError(59,0.02098197);
   hist__11->SetBinError(60,0.02056688);
   hist__11->SetBinError(61,0.02271652);
   hist__11->SetBinError(62,0.02171751);
   hist__11->SetBinError(63,0.01366515);
   hist__11->SetBinError(64,0.02083224);
   hist__11->SetBinError(65,0.01951547);
   hist__11->SetBinError(66,0.01680029);
   hist__11->SetBinError(67,0.02079703);
   hist__11->SetBinError(68,0.01798371);
   hist__11->SetBinError(69,0.01417116);
   hist__11->SetBinError(70,0.01758295);
   hist__11->SetBinError(71,0.0157588);
   hist__11->SetBinError(72,0.00786344);
   hist__11->SetBinError(73,0.0164247);
   hist__11->SetBinError(74,0.01255615);
   hist__11->SetBinError(75,0.008663637);
   hist__11->SetBinError(76,0.01310032);
   hist__11->SetBinError(77,0.006668962);
   hist__11->SetBinError(78,0.01056314);
   hist__11->SetBinError(79,0.002682157);
   hist__11->SetBinError(80,0.01161289);
   hist__11->SetBinError(81,0.01346602);
   hist__11->SetBinError(82,0.009501022);
   hist__11->SetBinError(83,0.006215903);
   hist__11->SetBinError(84,0.006858609);
   hist__11->SetBinError(85,0.01298879);
   hist__11->SetBinError(86,0.004277829);
   hist__11->SetBinError(87,0.004050353);
   hist__11->SetBinError(88,0.008630831);
   hist__11->SetBinError(90,0.007534006);
   hist__11->SetBinError(91,0.004484216);
   hist__11->SetEntries(340.2862);
   hist__11->SetStats(0);

   ci = TColor::GetColor("#00ff00");
   hist__11->SetLineColor(ci);
   hist__11->GetXaxis()->SetTitle("$prob_{B0}$");
   hist__11->GetXaxis()->SetLabelFont(42);
   hist__11->GetXaxis()->SetTitleOffset(1);
   hist__11->GetXaxis()->SetTitleFont(42);
   hist__11->GetYaxis()->SetLabelFont(42);
   hist__11->GetYaxis()->SetTitleOffset(1);
   hist__11->GetYaxis()->SetTitleFont(42);
   hist__11->GetZaxis()->SetLabelFont(42);
   hist__11->GetZaxis()->SetTitleOffset(1);
   hist__11->GetZaxis()->SetTitleFont(42);
   hist__11->Draw("hist same");
   
   TH1D *hist__12 = new TH1D("hist__12","",100,0,1);
   hist__12->SetBinContent(1,0.05195965);
   hist__12->SetBinContent(2,0.3015057);
   hist__12->SetBinContent(3,0.4682573);
   hist__12->SetBinContent(4,0.6951655);
   hist__12->SetBinContent(5,0.6932085);
   hist__12->SetBinContent(6,0.6529802);
   hist__12->SetBinContent(7,0.7785087);
   hist__12->SetBinContent(8,0.7622148);
   hist__12->SetBinContent(9,0.8635491);
   hist__12->SetBinContent(10,0.8587182);
   hist__12->SetBinContent(11,0.8052745);
   hist__12->SetBinContent(12,0.8046121);
   hist__12->SetBinContent(13,0.8383475);
   hist__12->SetBinContent(14,0.761788);
   hist__12->SetBinContent(15,0.7896085);
   hist__12->SetBinContent(16,0.8334649);
   hist__12->SetBinContent(17,0.844095);
   hist__12->SetBinContent(18,0.9772985);
   hist__12->SetBinContent(19,0.8123783);
   hist__12->SetBinContent(20,0.8225879);
   hist__12->SetBinContent(21,0.8385803);
   hist__12->SetBinContent(22,0.7897652);
   hist__12->SetBinContent(23,0.769916);
   hist__12->SetBinContent(24,0.8100934);
   hist__12->SetBinContent(25,0.6600366);
   hist__12->SetBinContent(26,1);
   hist__12->SetBinContent(27,0.7258275);
   hist__12->SetBinContent(28,0.5917189);
   hist__12->SetBinContent(29,0.4435083);
   hist__12->SetBinContent(30,0.4085454);
   hist__12->SetBinContent(31,0.3917299);
   hist__12->SetBinContent(32,0.3611467);
   hist__12->SetBinContent(33,0.3947003);
   hist__12->SetBinContent(34,0.2808506);
   hist__12->SetBinContent(35,0.2585227);
   hist__12->SetBinContent(36,0.2615006);
   hist__12->SetBinContent(37,0.1692789);
   hist__12->SetBinContent(38,0.2488862);
   hist__12->SetBinContent(39,0.2407765);
   hist__12->SetBinContent(40,0.1647);
   hist__12->SetBinContent(41,0.1330633);
   hist__12->SetBinContent(42,0.1048078);
   hist__12->SetBinContent(43,0.08917082);
   hist__12->SetBinContent(44,0.1008887);
   hist__12->SetBinContent(45,0.0998061);
   hist__12->SetBinContent(46,0.07318651);
   hist__12->SetBinContent(47,0.06642518);
   hist__12->SetBinContent(48,0.03606224);
   hist__12->SetBinContent(49,0.04478839);
   hist__12->SetBinContent(50,0.08542741);
   hist__12->SetBinContent(51,0.04595841);
   hist__12->SetBinContent(52,0.0580839);
   hist__12->SetBinContent(53,0.05968076);
   hist__12->SetBinContent(54,0.04406763);
   hist__12->SetBinContent(55,0.01489123);
   hist__12->SetBinContent(56,0.0379722);
   hist__12->SetBinContent(57,0.02472193);
   hist__12->SetBinContent(58,0.01722248);
   hist__12->SetBinContent(59,0.02374305);
   hist__12->SetBinContent(60,0.009008617);
   hist__12->SetBinContent(61,0.01804904);
   hist__12->SetBinContent(62,0.004521042);
   hist__12->SetBinContent(63,0.01741859);
   hist__12->SetBinContent(64,0.007815447);
   hist__12->SetBinContent(65,0.004494426);
   hist__12->SetBinContent(67,0.004487715);
   hist__12->SetBinContent(68,0.01343906);
   hist__12->SetBinContent(69,0.009402186);
   hist__12->SetBinContent(70,0.01444141);
   hist__12->SetBinContent(71,0.004813895);
   hist__12->SetBinContent(72,0.008458684);
   hist__12->SetBinContent(74,0.004338561);
   hist__12->SetBinContent(75,0.002038941);
   hist__12->SetBinContent(76,0.01351402);
   hist__12->SetBinContent(77,0.01006126);
   hist__12->SetBinContent(78,0.008076825);
   hist__12->SetBinContent(81,0.005124153);
   hist__12->SetBinContent(85,0.006150584);
   hist__12->SetBinContent(86,0.004443498);
   hist__12->SetBinContent(90,0.003937494);
   hist__12->SetBinError(1,0.01636256);
   hist__12->SetBinError(2,0.03760471);
   hist__12->SetBinError(3,0.04688847);
   hist__12->SetBinError(4,0.05786525);
   hist__12->SetBinError(5,0.0568995);
   hist__12->SetBinError(6,0.05643206);
   hist__12->SetBinError(7,0.06016302);
   hist__12->SetBinError(8,0.05965179);
   hist__12->SetBinError(9,0.06418285);
   hist__12->SetBinError(10,0.06394726);
   hist__12->SetBinError(11,0.0619853);
   hist__12->SetBinError(12,0.06128232);
   hist__12->SetBinError(13,0.06307902);
   hist__12->SetBinError(14,0.05971287);
   hist__12->SetBinError(15,0.0611397);
   hist__12->SetBinError(16,0.06315267);
   hist__12->SetBinError(17,0.06335851);
   hist__12->SetBinError(18,0.06772625);
   hist__12->SetBinError(19,0.06221986);
   hist__12->SetBinError(20,0.06295693);
   hist__12->SetBinError(21,0.06179998);
   hist__12->SetBinError(22,0.06207943);
   hist__12->SetBinError(23,0.06068445);
   hist__12->SetBinError(24,0.06118599);
   hist__12->SetBinError(25,0.0551465);
   hist__12->SetBinError(26,0.06976918);
   hist__12->SetBinError(27,0.05911031);
   hist__12->SetBinError(28,0.0527386);
   hist__12->SetBinError(29,0.0455717);
   hist__12->SetBinError(30,0.04444618);
   hist__12->SetBinError(31,0.04436528);
   hist__12->SetBinError(32,0.04156072);
   hist__12->SetBinError(33,0.0431799);
   hist__12->SetBinError(34,0.03616503);
   hist__12->SetBinError(35,0.03517715);
   hist__12->SetBinError(36,0.03541852);
   hist__12->SetBinError(37,0.02797257);
   hist__12->SetBinError(38,0.03416166);
   hist__12->SetBinError(39,0.03450993);
   hist__12->SetBinError(40,0.0273534);
   hist__12->SetBinError(41,0.02492163);
   hist__12->SetBinError(42,0.0226639);
   hist__12->SetBinError(43,0.02059357);
   hist__12->SetBinError(44,0.02161172);
   hist__12->SetBinError(45,0.02281717);
   hist__12->SetBinError(46,0.01905934);
   hist__12->SetBinError(47,0.01795239);
   hist__12->SetBinError(48,0.01277676);
   hist__12->SetBinError(49,0.01428601);
   hist__12->SetBinError(50,0.02121194);
   hist__12->SetBinError(51,0.01462037);
   hist__12->SetBinError(52,0.01713004);
   hist__12->SetBinError(53,0.01739132);
   hist__12->SetBinError(54,0.01474247);
   hist__12->SetBinError(55,0.008628067);
   hist__12->SetBinError(56,0.01276646);
   hist__12->SetBinError(57,0.01122118);
   hist__12->SetBinError(58,0.008988096);
   hist__12->SetBinError(59,0.01067118);
   hist__12->SetBinError(60,0.006372417);
   hist__12->SetBinError(61,0.009030259);
   hist__12->SetBinError(62,0.004521042);
   hist__12->SetBinError(63,0.008710718);
   hist__12->SetBinError(64,0.005695845);
   hist__12->SetBinError(65,0.004494426);
   hist__12->SetBinError(67,0.004487715);
   hist__12->SetBinError(68,0.007762222);
   hist__12->SetBinError(69,0.006704205);
   hist__12->SetBinError(70,0.008352827);
   hist__12->SetBinError(71,0.004813895);
   hist__12->SetBinError(72,0.005988889);
   hist__12->SetBinError(74,0.004338561);
   hist__12->SetBinError(75,0.002038941);
   hist__12->SetBinError(76,0.007836051);
   hist__12->SetBinError(77,0.007121033);
   hist__12->SetBinError(78,0.00576266);
   hist__12->SetBinError(81,0.005124153);
   hist__12->SetBinError(85,0.006150584);
   hist__12->SetBinError(86,0.004443498);
   hist__12->SetBinError(90,0.003937494);
   hist__12->SetEntries(353.209);
   hist__12->SetStats(0);

   ci = TColor::GetColor("#ffcc00");
   hist__12->SetLineColor(ci);
   hist__12->GetXaxis()->SetTitle("$prob_{B0}$");
   hist__12->GetXaxis()->SetLabelFont(42);
   hist__12->GetXaxis()->SetTitleOffset(1);
   hist__12->GetXaxis()->SetTitleFont(42);
   hist__12->GetYaxis()->SetLabelFont(42);
   hist__12->GetYaxis()->SetTitleOffset(1);
   hist__12->GetYaxis()->SetTitleFont(42);
   hist__12->GetZaxis()->SetLabelFont(42);
   hist__12->GetZaxis()->SetTitleOffset(1);
   hist__12->GetZaxis()->SetTitleFont(42);
   hist__12->Draw("hist same");
   
   TLegend *leg = new TLegend(0,0,0,0,NULL,"brNDC");
   leg->SetBorderSize(0);
   leg->SetLineColor(1);
   leg->SetLineStyle(1);
   leg->SetLineWidth(1);
   leg->SetFillColor(0);
   leg->SetFillStyle(1001);
   TLegendEntry *entry=leg->AddEntry("hist","B-","L");
   entry->SetLineColor(1);
   entry->SetLineStyle(1);
   entry->SetLineWidth(1);
   entry->SetMarkerColor(1);
   entry->SetMarkerStyle(20);
   entry->SetMarkerSize(1.15);
   entry->SetTextFont(42);
   entry->SetTextSize(0.02);
   entry=leg->AddEntry("hist","B0bar","L");
   entry->SetLineColor(1);
   entry->SetLineStyle(1);
   entry->SetLineWidth(1);
   entry->SetMarkerColor(1);
   entry->SetMarkerStyle(20);
   entry->SetMarkerSize(1.15);
   entry->SetTextFont(42);
   entry->SetTextSize(0.02);
   entry=leg->AddEntry("hist","B0","L");
   entry->SetLineColor(1);
   entry->SetLineStyle(1);
   entry->SetLineWidth(1);
   entry->SetMarkerColor(1);
   entry->SetMarkerStyle(20);
   entry->SetMarkerSize(1.15);
   entry->SetTextFont(42);
   entry->SetTextSize(0.02);
   entry=leg->AddEntry("hist","B+","L");
   entry->SetLineColor(1);
   entry->SetLineStyle(1);
   entry->SetLineWidth(1);
   entry->SetMarkerColor(1);
   entry->SetMarkerStyle(20);
   entry->SetMarkerSize(1.15);
   entry->SetTextFont(42);
   entry->SetTextSize(0.02);
   leg->Draw();
   c->Modified();
   c->cd();
   c->SetSelected(c);
}
