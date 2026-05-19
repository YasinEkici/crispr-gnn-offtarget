![](assets/pictures/_page_0_Picture_2.jpeg)

![](assets/pictures/_page_0_Picture_3.jpeg)

# Nucleosome breathing and remodeling constrain CRISPR-Cas9 function

R Stefan Isaac1,2, Fuguo Jiang3,4, Jennifer A Doudna4,5,6,7,8, Wendell A Lim9,10,11\*, Geeta J Narlikar$^{1}$\*, Ricardo Almeida10,11,12

$^{1}$Department of Biochemistry and Biophysics, University of California, San Francisco, San Francisco, United States; $^{2}$Tetrad Graduate Program, University of California, San Francisco, San Francisco, United States; $^{3}$Department of Molecular and Cell Biology, University of California, Berkeley, Berkeley, United States; $^{4}$California Institute for Quantitative Biosciences, University of California, Berkeley, Berkeley, United States; $^{5}$Department of Molecular and Cell Biology, Howard Hughes Medical Institute, University of California, Berkeley, Berkeley, United States; $^{6}$Department of Chemistry, University of California, Berkeley, Berkeley, United States; $^{7}$Physical Biosciences Division, Lawrence Berkeley National Laboratory, Berkeley, United States; $^{8}$ Innovative Genomics Initiative, University of California, Berkeley, Berkeley, United States; $^{9}$Department of Cellular and Molecular Pharmacology, Howard Hughes Medical Institute, University of California, San Francisco, San Francisco, United States; $^{10}$Center for Systems and Synthetic Biology, University of California, San Francisco, San Francisco, United States; $^{11}$California Institute for Quantitative Biosciences, University of California, San Francisco, San Francisco, United States; $^{12}$Department of Cellular and Molecular Pharmacology, University of California, San Francisco, San Francisco, United States

\*For correspondence: Wendell. Lim@ucsf.edu (WAL); Geeta. Narlikar@ucsf.edu (GJN)

Competing interest: [See](#page-10-0) [page 11](#page-10-0)

Funding: [See page 11](#page-10-0)

Received: 02 December 2015 Accepted: 16 April 2016 Published: 28 April 2016

Reviewing editor: Karen Adelman, National Institute of Environmental Health Sciences, United States

Copyright Isaac et al. This article is distributed under the terms of the [Creative Commons](http://creativecommons.org/licenses/by/4.0/) [Attribution License,](http://creativecommons.org/licenses/by/4.0/) which permits unrestricted use and redistribution provided that the original author and source are credited.

Abstract The CRISPR-Cas9 bacterial surveillance system has become a versatile tool for genome editing and gene regulation in eukaryotic cells, yet how CRISPR-Cas9 contends with the barriers presented by eukaryotic chromatin is poorly understood. Here we investigate how the smallest unit of chromatin, a nucleosome, constrains the activity of the CRISPR-Cas9 system. We find that nucleosomes assembled on native DNA sequences are permissive to Cas9 action. However, the accessibility of nucleosomal DNA to Cas9 is variable over several orders of magnitude depending on dynamic properties of the DNA sequence and the distance of the PAM site from the nucleosome dyad. We further find that chromatin remodeling enzymes stimulate Cas9 activity on nucleosomal templates. Our findings imply that the spontaneous breathing of nucleosomal DNA together with the action of chromatin remodelers allow Cas9 to effectively act on chromatin in vivo. [DOI: 10.7554/eLife.13450.001](http://dx.doi.org/10.7554/eLife.13450.001)

# Introduction

The recent development of CRISPR (clustered regularly interspaced short palindromic repeats) systems, particularly the type II CRISPR-Cas9 mechanism from Streptomyces pyogenes, as an artificial tool for genome engineering, gene regulation, and live imaging is a remarkable achievement with profound impact in a wide variety of research fields and applications ([Makarova et al., 2015](#page-12-0); [Doudna and Charpentier, 2014](#page-11-0); [Cong et al., 2013](#page-11-0); [Jinek et al., 2012](#page-11-0); [2013](#page-11-0); [Mali et al., 2013](#page-12-0)). Despite its successful adoption across numerous eukaryotic organisms, relatively few details are known of the mechanism by which bacterial CRISPR-Cas9 systems operate in eukaryotic cells ([Doudna and Charpentier, 2014](#page-11-0); [Ghorbal et al., 2014](#page-11-0); [Vyas et al., 2015](#page-13-0)).

![](assets/pictures/_page_1_Picture_1.jpeg)

eLife digest CRISPR is a method of editing the genetic material inside living cells and has enabled dramatic advances in a broad variety of research fields in recent years. The method relies on a bacterial enzyme called Cas9 that can be programmed, via short guide molecules made from RNA, to target specific sites in the cell's DNA. Once bound to its target, the Cas9 enzyme cuts the DNA molecule; this often leads to changes in the DNA sequence. In nature, bacteria use the CRISPR-Cas9 system to defend themselves against viruses. However, this system also works in other cell types and can be reprogrammed to target almost any site in the DNA.

To date, the CRISPR-Cas9 system has been used in fungi, worms, flies, plants, mammals and other eukaryotes. Yet, unlike in bacteria, much of the DNA in eukaryotes is wrapped around proteins called histones to form units referred to as nucleosomes. This means eukaryotic DNA is often tightly packaged, which makes it less accessible to other proteins. Nevertheless, eukaryotic DNA will spontaneously detach and reattach to the histones – a phenomenon that is commonly known as DNA "breathing". Also, protein machines known as chromatin remodelers can move, assemble and take apart the nucleosomes in eukaryotic cells. However, because there is much still to learn about how CRISPR-Cas9 works in eukaryotic cells, it is not clear how nucleosomes affect this system's activity.

Isaac et al. have now used a simplified biochemical system to test how nucleosomes and chromatin remodelers affect CRISP-Cas9 activity. The system comprised purified Cas9 enzymes, short guide RNA molecules and nucleosomes. The experiments revealed that the Cas9 enzyme was able to cut DNA on nucleosomes when the DNA sequence allowed more spontaneous breathing or when chromatin remodelers were present to destabilize or move the nucleosome out of the way.

These results suggest that by taking the placement of the nucleosomes into account, researchers can better predict how effective the CRISPR-Cas9 system will be at targeting a specific DNA sequence in a eukaryotic cell. The findings also suggest ways to make genome editing with CRISPR-Cas9 even more efficient.

[DOI: 10.7554/eLife.13450.002](http://dx.doi.org/10.7554/eLife.13450.002)

CRISPR-Cas9 originated in bacteria, where genomic DNA generally consists of supercoiled circular molecules associated with nucleoid-associated proteins ([Travers and Muskhelishvili, 2005](#page-12-0)). In contrast, eukaryotic chromosomes are linear, packaged with histone octamers into nucleosomes, and further organized into higher-order structures ([Luger et al., 1997](#page-12-0); [Olins and Olins, 1974](#page-12-0); [Woodcock et al., 1976](#page-13-0); [Dixon et al., 2012](#page-11-0)). The packaging of DNA into nucleosomes generally inhibits the binding of sequence specific DNA binding factors. In the simplest model, nucleosomes would analogously inhibit Cas9 action. Further, in eukaryotes ATP-dependent chromatin remodelers reposition, remove, or restructure nucleosomes to regulate the access of DNA binding factors ([Clapier and Cairns, 2009](#page-11-0); [Narlikar et al., 2013](#page-12-0)). It can therefore be imagined that the action of remodelers also regulates the action of Cas9 on nucleosomes.

To quantitatively test the above models we performed biochemical studies to measure Cas9 activity on nucleosomes assembled with native and artificial nucleosome positioning sequences. We find that the combination of nucleosome breathing, by which DNA transiently disengages from the histone octamer, and the action of chromatin remodeling enzymes allow Cas9 to act on nucleosomal DNA with rates comparable to naked DNA. The results provide a biochemical explanation for the efficacy of Cas9 in eukaryotic cells.

# Results

#### Nucleosomes assembled on the 601 sequence inhibit Cas9 binding and cleavage of target DNA

To determine if a nucleosome inhibits the ability of Cas9 to scan, recognize, and cleave sgRNAdirected DNA targets, we performed in vitro Cas9 cleavage assays using mononucleosomes (single nucleosomes on short dsDNA molecules) reconstituted using the Widom 601 positioning sequence with 80 base pairs of flanking DNA on both sides (referred to as 601 80/80 particles, [Figure 1A](#page-2-0))

![](assets/pictures/_page_2_Picture_1.jpeg)

![](assets/figures/_page_2_Figure_2.jpeg)

Figure 1. Cas9 DNA nuclease activity is hindered by nucleosomes. (A) Schematic of sgRNAs designed against the assembled 601 80/80 nucleosome substrates targeting the flanking regions, entry/exit sites, and near the nucleosomal dyad. (B) Cleavage assay comparing Cas9 cleavage on 80/80 DNA and 80/80 nucleosomes when loaded with sgRNA #3. (C) Kinetics of cleavage with sgRNA #3. (D) Comparison of the relative rates of cleavage on nucleosomes to DNA at various positions along the 80/80 nucleosome construct. The position reported is the site of cleavage by Cas9. Represented values are mean ± SEM from three replicates.

The following source data and figure supplements are available for figure 1:

Source data 1. Replicate gels of Cas9 cleavage of 80/80 601 DNA and nucleosomes with sgRNAs #2 and #6. [DOI: 10.7554/eLife.13450.004](http://dx.doi.org/10.7554/eLife.13450.004)

Source data 2. Replicate gels of Cas9 cleavage of 80/80 601 DNA and nucleosomes with sgRNAs #2 and #6. [DOI: 10.7554/eLife.13450.005](http://dx.doi.org/10.7554/eLife.13450.005)

Source data 3. Replicate gels of Cas9 cleavage of 80/80 601 DNA and nucleosomes with sgRNAs #2 and #6. [DOI: 10.7554/eLife.13450.006](http://dx.doi.org/10.7554/eLife.13450.006)

Source data 4. Replicate gels of Cas9 cleavage of 80/80 601 DNA and nucleosomes with sgRNA #5. [DOI: 10.7554/eLife.13450.007](http://dx.doi.org/10.7554/eLife.13450.007)

Source data 5. Replicate gels of Cas9 cleavage of 80/80 601 DNA and nucleosomes with sgRNA #5. [DOI: 10.7554/eLife.13450.008](http://dx.doi.org/10.7554/eLife.13450.008)

Source data 6. Replicate gels of Cas9 cleavage of 80/80 601 DNA and nucleosomes with sgRNA #1. [DOI: 10.7554/eLife.13450.009](http://dx.doi.org/10.7554/eLife.13450.009)

Source data 7. Replicate gels of Cas9 cleavage of 80/80 601 DNA and nucleosomes with sgRNA #1.

Figure 1 continued on next page

[DOI: 10.7554/eLife.13450.003](http://dx.doi.org/10.7554/eLife.13450.003)

![](assets/pictures/_page_3_Picture_1.jpeg)

Figure 1 continued

[DOI: 10.7554/eLife.13450.010](http://dx.doi.org/10.7554/eLife.13450.010)

Source data 8. Replicate gels of Cas9 cleavage of 80/80 601 DNA and nucleosomes with sgRNA #3.

[DOI: 10.7554/eLife.13450.011](http://dx.doi.org/10.7554/eLife.13450.011)

Source data 9. Replicate gels of Cas9 cleavage of 80/80 601 DNA and nucleosomes with sgRNA #3.

[DOI: 10.7554/eLife.13450.012](http://dx.doi.org/10.7554/eLife.13450.012)

Source data 10. Replicate gels of Cas9 cleavage of 80/80 601 DNA and nucleosomes with sgRNA #4.

[DOI: 10.7554/eLife.13450.013](http://dx.doi.org/10.7554/eLife.13450.013)

Source data 11. Replicate gels of Cas9 cleavage of 80/80 601 DNA and nucleosomes with sgRNA #4.

[DOI: 10.7554/eLife.13450.014](http://dx.doi.org/10.7554/eLife.13450.014)

Source data 12. Quantification of [Figure 1](#page-2-0) Cas9 cleavage gels.

[DOI: 10.7554/eLife.13450.015](http://dx.doi.org/10.7554/eLife.13450.015)

Figure supplement 1. Nucleosome positioning blocks Cas9 from binding PAM sites on DNA.

[DOI: 10.7554/eLife.13450.016](http://dx.doi.org/10.7554/eLife.13450.016)

Figure supplement 1—source data 1. -3Replicate gels of dCas9 binding to 0/0 601 DNA and nucleosomes with sgRNA #3.

[DOI: 10.7554/eLife.13450.017](http://dx.doi.org/10.7554/eLife.13450.017)

Figure supplement 1—source data 2. -3Replicate gels of dCas9 binding to 0/0 601 DNA and nucleosomes with sgRNA #3.

[DOI: 10.7554/eLife.13450.018](http://dx.doi.org/10.7554/eLife.13450.018)

Figure supplement 1—source data 3. -3Replicate gels of dCas9 binding to 0/0 601 DNA and nucleosomes with sgRNA #3.

[DOI: 10.7554/eLife.13450.019](http://dx.doi.org/10.7554/eLife.13450.019)

Figure supplement 1—source data 4. Quantification of Figure 1—figure supplement 1 gel shifts.

[DOI: 10.7554/eLife.13450.020](http://dx.doi.org/10.7554/eLife.13450.020)

([Lowary and Widom, 1998](#page-12-0)). The 601 sequence is an artificially derived sequence with high affinity for the histone octamer and has proved a valuable tool for assembling well positioning nucleosomes for biochemical studies. Using sgRNAs targeting the nucleosomal dyad, entry/exit sites, and flanking DNA, we measured the rates of Cas9 cleavage with naked 601 DNA and the 601 80/80 particles. Targeting the DNA flanking the nucleosome showed cleavage rates comparable to those of naked DNA. Cleavage rates at entry/exit sites of the nucleosome were much lower compared to naked DNA ( ~ 23–28x decrease cleavage rate vs. DNA alone) ([Figure 1B,C](#page-2-0)). Targeting near the nucleosomal dyad resulted in further inhibition of cutting by Cas9 ( ~ 1000x decrease vs. DNA alone) ([Figure 1C,D](#page-2-0)). Previous work has shown that nucleosomal DNA transiently disengages from the histone octamer, a process termed as nucleosomal DNA unpeeling or breathing. The equilibrium for DNA unpeeling gets progressively more unfavorable the closer the DNA site gets to the dyad ([Polach and Widom, 1995](#page-12-0); [Li and Widom, 2004](#page-12-0); [Luger et al., 2012](#page-12-0)). The nucleosome-mediated inhibition of Cas9 activity is more pronounced near the dyad suggesting that Cas9 cleavage occurs on DNA that is transiently disengaged from the histone octamer.

Nucleosomes block the ability of Cas9 to cleave DNA, but it is unclear at which step of Cas9 activity this occurs. Cas9 recognizes DNA target sites in a two-step process that begins with binding to the DNA protospacer adjacent motif (PAM, in this case 'NGG') through its C-terminal PAM-interacting region, followed by sequential melting of the DNA double strand and annealing of the sgRNA guide segment to the unwound target DNA strand (Figure 1—figure supplement 1A) ([Sternberg et al., 2014](#page-12-0); [Jiang et al., 2015](#page-11-0)). Complete annealing of the 20-nt guide RNA to the target DNA is required to drive a progressive conformational transformation that authorizes Cas9 to simultaneously cleave both DNA strands ([Sternberg et al., 2015](#page-12-0); [Josephs et al., 2016](#page-12-0)). Given this order of events, it is conceivable that nucleosomes can interfere with any of the steps preceding and including DNA cleavage.

To identify the point at which nucleosomes disrupt Cas9 function, we assessed binding of nuclease-dead Cas9 (dCas9) to mononucleosomal particles by an electrophoretic mobility shift assay. We performed dCas9 binding assays using 601 0/0 nucleosomal particles which are devoid of naked DNA overhangs. Binding of dCas9 pre-loaded with core targeting sgRNA with 601 0/0 nucleosomes is undetectable whereas binding to naked DNA control is still robust (Figure 1—figure supplement 1B). The presence of super shifts in the gel migration pattern suggests that multiple dCas9 molcules are engaging the same DNA substrate molecule. We investigated this further and determined that, in our binding assay, the highly transient dCas9 binding to PAMs within target DNA is observable as super shifts, likely due to a combination of a high number of PAMs on the target DNA (23 NGG

![](assets/pictures/_page_4_Picture_1.jpeg)

![](assets/figures/_page_4_Figure_2.jpeg)

**Figure 2.** Higher nucleosomal breathing dynamics enhance Cas9 cleavage. (A) Schematic illustrating nucleosome breathing and how it can enable Cas9 binding to a target in the nucleosome. (B) Cleavage assay comparing Cas9 cleavage of 601 and 5S 0/0 nucleosomes when loaded with sgRNAs targeting comparable positions at core and entry sites. (C) Quantification of (B). (D) Cas9 cleavage rates on 601 and 5S nucleosomes when targeted to core and entry sites. Values were normalized against naked DNA control rates. Represented values are mean ± SEM from three replicates. Additional gel panels shown in **Figure 2—figure supplement 1**.

DOI: 10.7554/eLife.13450.021

The following source data and figure supplement are available for figure 2:

**Source data 1.** Replicate gels of cleavage of 0/0 5S DNA and nucleosomes with sgRNA core. DOI: 10.7554/eLife.13450.022

**Source data 2.** Replicate gels of cleavage of 0/0 5S DNA and nucleosomes with sgRNA core. DOI: 10.7554/eLife.13450.023

**Source data 3.** Replicate gels of cleavage of 0/0 5S DNA and nucleosomes with sgRNA entry. DOI: 10.7554/eLife.13450.024

**Source data 4.** Replicate gels of cleavage of 0/0 5S DNA and nucleosomes with sgRNA entry. DOI: 10.7554/eLife.13450.025

**Source data 5.** Replicate gels of cleavage of 0/0 601 DNA and nucleosomes with sgRNA entry. DOI: 10.7554/eLife.13450.026

**Source data 6.** Replicate gels of cleavage of 0/0 601 DNA and nucleosomes with sgRNA entry. Figure 2 continued on next page

![](assets/pictures/_page_5_Picture_1.jpeg)

Figure 2 continued

[DOI: 10.7554/eLife.13450.027](http://dx.doi.org/10.7554/eLife.13450.027)

Source data 7. Quantification of [Figure 2](#page-4-0) Cas9 cleavage gels.

[DOI: 10.7554/eLife.13450.028](http://dx.doi.org/10.7554/eLife.13450.028)

Source data 8. Quantification of [Figure 2](#page-4-0) Cas9 cleavage gels.

[DOI: 10.7554/eLife.13450.029](http://dx.doi.org/10.7554/eLife.13450.029)

Figure supplement 1. Cas9 cleavage assay with 601 and 5S 0/0 nucleosomes.

[DOI: 10.7554/eLife.13450.030](http://dx.doi.org/10.7554/eLife.13450.030)

PAMs present in 601 0/0 sequence) and common caging effects of gel binding assays. The absence of a super shift binding pattern with 0/0 nucleosomes ([Figure 1—figure supplement 1B](#page-3-0), right) suggests that dCas9 cannot stably interact with PAMs located on nucleosomes, in a manner consistent with a recently published study ([Hinz et al., 2015](#page-11-0)).

#### Nucleosomes assembled on a native DNA sequence are permissive to Cas9 action

The artificial Widom 601 is an atypically strong nucleosome positioning sequence that shows ~ 100 fold less breathing dynamics compared to physiological nucleosome positioning sequences, such as the 5S rRNA gene ([Anderson et al., 2002](#page-11-0); [Partensky and Narlikar, 2009](#page-12-0)). To determine how Cas9 contends with nucleosomes assembled on this physiological positioning sequence, we performed cleavage experiments with nucleosomes assembled from 5S rRNA gene sequences from Xenopus borealis ([Figure 2A](#page-4-0)). Cas9-mediated cleavage at sites near the entry/exit of the nucleosome is substantially enhanced (700–fold) with 5S nucleosomes compared to 601 particles ([Figure 2B–D](#page-4-0)). In the context of 601, cutting at this site is 1000-fold slower than in naked DNA. In contrast, with 5S nucleosomes, cutting at the comparable site is only 1.5-fold slower than in naked DNA. However, Cas9 cleavage near the dyad is inhibited to a similar extent on both 5S and 601 nucleosomes, showing that the 5S-specific enhancement of Cas9 activity does not extend all the way to the nucleosomal dyad. These results support our interpretation that nucleosomal DNA breathing substantially enhances Cas9 binding to nucleosomes and demonstrate that nucleosomal DNA sequence, through its influence on nucleosome stability, can regulate Cas9 activity over a large dynamic range.

#### Nucleosome remodeling enhances Cas9 activity

We next investigated whether chromatin remodeling could enhance Cas9 activity towards chromatin substrates. Nucleosome positioning in vivo is strongly dependent on ATP-dependent chromatin remodelers, which are capable of loading, repositioning, and removing nucleosomes from the chromatin fiber. To measure how chromatin remodelers can influence Cas9 activity, we performed experiments where we pre-incubated 601 nucleosomes with remodeler enzymes prior to Cas9-mediated cleavage. For our experiments with the human ISWI-family remodeler SNF2h, we used asymmetric nucleosomes that possess flanking DNA only on the entry side (601 80/0 particles). When incubated with 601 80/0 particles, SNF2h promotes sliding of the nucleosome towards the center of the DNA molecule ([Figure 3A–B](#page-6-0), [Figure 3—figure supplement 1](#page-7-0)) ([La¨ngst et al., 1999](#page-12-0); [He et al.,](#page-11-0) [2006](#page-11-0); [Yang et al., 2006](#page-13-0)). We then performed in vitro cleavage experiments where 80/0 particles, pre-remodeled with SNF2h, were incubated with Cas9:sgRNA complex with its target site located within the nucleosome exit region. Remodeling 80/0 nucleosomes by SNF2h resulted in a strong enhancement of Cas9 cleavage to ~ 34%, showing that SNF2h slides the nucleosome enough to improve Cas9's accessibility to the target site and that Cas9 is now able to bind and cleave at a higher rate ([Figure 3A–D](#page-6-0)).

We also performed this experiment by simultaneously adding SNF2h and Cas9 and found a similar rate enhancement ([Figure 3—figure supplement 2](#page-7-0)).

While the ISWI remodeler SNF2h predominantly slides nucleosomes, remodelers from the SWI/ SNF class have multiple outcomes, which include generation of DNA loops and eviction of the histone octamer in addition to nucleosome sliding ([Rowe and Narlikar, 2010](#page-12-0); [Narlikar et al., 2001](#page-12-0); [Lorch et al., 1998](#page-12-0); [Schnitzler et al., 1998](#page-12-0); [Clapier and Cairns, 2009](#page-11-0)). To determine if the types of remodeled products generated influence Cas9 activity, we performed similar experiments using 601 80/80 particles and the yeast chromatin remodeler RSC. We find that RSC activity also dramatically

![](assets/pictures/_page_6_Picture_1.jpeg)

![](assets/figures/_page_6_Figure_2.jpeg)

Figure 3. Chromatin remodeling improves Cas9 cleavage of nucleosomal substrates. (A) Schematic of Cas9 cleavage assay with remodeling. Cas9 is presented with 601 nucleosomes either untreated or previously remodeled with SNF2h or RSC remodelers. (B) Assay comparing cleavage on untreated and remodeled 80/0 nucleosomes when Cas9 is targeted to exit site (depicted in green). These asymmetric nucleosomes are recentered by SNF2h, exposing the exit target site to Cas9 (C) Quantification of (B). (D) Cleavage rates of 80/0 nucleosomes by Cas9 relative to naked DNA, in the presence or absence of SNF2h. SNF2h improves Cas9 cleavage to ~35% of the naked DNA cleavage rate. (E) Assay comparing Cas9-mediated cleavage at entry site of 80/80 symmetric 601 nucleosomes, either untreated or previously treated with RSC remodeler. RSC can destabilize nucleosome structure and reposition nucleosomes towards the DNA ends. (F) Quantification of (E) (G) Comparison of the rates of cleavage of nucleosomes normalized to DNA control with and without the action of RSC chromatin remodeler. Mean enhancement rates of Cas9 activity by chromatin remodeling are shown. (H) Cleavage rates of 80/80 nucleosomes by Cas9 relative to naked DNA, in the presence or absence of RSC. Cas9 cleavage is substantially enhanced by RSC, attaining ~63% of the naked DNA cleavage rate. Represented values are mean ± SEM from three replicates. Additional gel panels shown in Figure 3—figure supplement 1. (I) Model of Cas9 activity in vivo in eukaryotes. Left, stable and strongly positioned nucleosomes impede Cas9 activity (downward arrows). However, nucleosomes in vivo are generally more dynamic (breathing), allowing Cas9 opportunities to target underlying DNA (center). Cas9 accessibility to nucleosomal DNA can be further enhanced by the activity of chromatin remodelers that destabilize and/or reposition nucleosomes (right).

DOI: 10.7554/eLife.13450.031

The following source data and figure supplements are available for figure 3:

**Source data 1.** Replicate gels of cleavage of 80/0 DNA and nucleosomes using sgRNA #4 with or without prior remodeling by Snf2h. DOI: 10.7554/eLife.13450.032

Figure 3 continued on next page

![](assets/pictures/_page_7_Picture_1.jpeg)

Figure 3 continued

Source Data 2. Replicate gels of cleavage of 80/0 DNA and nucleosomes using sgRNA #4 with or without prior remodeling by Snf2h.

[DOI: 10.7554/eLife.13450.033](http://dx.doi.org/10.7554/eLife.13450.033)

Source data 3. Replicate gels of cleavage of 80/0 DNA and nucleosomes using sgRNA #4 with or without prior remodeling by Snf2h.

[DOI: 10.7554/eLife.13450.034](http://dx.doi.org/10.7554/eLife.13450.034)

Source data 4. Quantification of Cas9 cleavage gels from Figure 3—source data 1–3.

[DOI: 10.7554/eLife.13450.035](http://dx.doi.org/10.7554/eLife.13450.035)

Source data 5. Replicate gels of cleavage of 80/80 DNA and nucleosomes using sgRNA 601\_2 with or without prior remodeling by RSC.

[DOI: 10.7554/eLife.13450.036](http://dx.doi.org/10.7554/eLife.13450.036)

Source data 6. Replicate gels of cleavage of 80/80 DNA and nucleosomes using sgRNA 601\_2 with or without prior remodeling by RSC.

[DOI: 10.7554/eLife.13450.037](http://dx.doi.org/10.7554/eLife.13450.037)

Source data 7. Replicate gels of cleavage of 80/80 DNA and nucleosomes using sgRNA 601\_2 with or without prior remodeling by RSC.

[DOI: 10.7554/eLife.13450.038](http://dx.doi.org/10.7554/eLife.13450.038)

Source data 8. Quantification of Cas9 cleavage gels from Figure 3—source data 5–7.

[DOI: 10.7554/eLife.13450.039](http://dx.doi.org/10.7554/eLife.13450.039)

Figure supplement 1. Cas9 cleavage assays with SNF2h and RSC chromatin remodelers.

[DOI: 10.7554/eLife.13450.040](http://dx.doi.org/10.7554/eLife.13450.040)

Figure supplement 2. Simultaneous chromatin remodeling and Cas9 cleavage of nucleosomal substrates.

[DOI: 10.7554/eLife.13450.041](http://dx.doi.org/10.7554/eLife.13450.041)

Figure supplement 2—source data 1. Gel of cleavage of 80/0 DNA and nucleosomes using sgRNA #4 with or without simultaneous remodeling by Snf2h.

[DOI: 10.7554/eLife.13450.042](http://dx.doi.org/10.7554/eLife.13450.042)

Figure supplement 3. SNF2h and RSC remodel nucleosomes prior to Cas9 cleavage.

[DOI: 10.7554/eLife.13450.043](http://dx.doi.org/10.7554/eLife.13450.043)

Figure supplement 3—source data 1. Test remodeling gel of 80/0 nucleosomes with Snf2h.

[DOI: 10.7554/eLife.13450.044](http://dx.doi.org/10.7554/eLife.13450.044)

Figure supplement 3—source data 2. Test remodeling gel of 80/80 nucleosomes with RSC.

[DOI: 10.7554/eLife.13450.045](http://dx.doi.org/10.7554/eLife.13450.045)

enhances cleavage on 601 80/80 nucleosomes when Cas9 is targeted to the entry site, negating most of the inhibitory influence of the nucleosome on Cas9 ([Figure 3E–F](#page-6-0)). These results demonstrate that two different classes of chromatin remodeling enzymes can significantly enhance Cas9 access to DNA targets normally obscured by nucleosomes.

# Discussion

Here we demonstrate, using detailed biochemical studies with a variety of nucleosomal templates, that (i) the intrinsic stability of the histone-DNA interactions, (ii) the location of the target site within the nucleosomes (nucleosome positioning), and (iii) the action of chromatin remodeling enzymes play critical roles in regulating the activity of S. pyogenes Cas9. Below we discuss the implications of our results.

Nucleosomes have been shown to inhibit the action of DNA binding factors. Recent work using nucleosomes assembled on the 601 sequence has led to the qualitatively similar conclusion that nucleosomes are refractory for Cas9 action ([Hinz et al., 2015](#page-11-0); [Horlbeck et al., 2016](#page-11-0)). The comparison here between Cas9 action on 601 nucleosomes vs. nucleosomes assembled on the native 5S sequence suggests a more refined model for how nucleosomes regulate Cas9 action. We find that Cas9 sites near the entry/exit sites of 5S nucleosomes are cleaved ~ 700-fold better than the corresponding sites within 601 nucleosomes. Given that DNA breathing occurs at least 100-fold more in 5S nucleosomes than 601 nucleosomes we propose that Cas9 gains access to nucleosomal DNA when the DNA is transiently unpeeled from the histone octamer. This model also explains why sites closer to the entry/exit sites are cut more readily by Cas9 than sites near the dyad. This is because DNA unpeeling up to the dyad is substantially less favored (100-fold) for both the 601 and 5S nucleosomes than DNA unpeeling near their respective entry/exit sites ([Anderson and Widom, 2000](#page-11-0)).

In vivo, as in vitro, the precise position of nucleosomes can greatly affect DNA factor binding. Chromatin remodeling enzymes can move nucleosomes away or towards the factor binding sites to respectively enhance or inhibit factor binding. We find that Cas9 activity can also benefit from

![](assets/pictures/_page_8_Picture_1.jpeg)

chromatin remodeling to access nucleosomal DNA, as evidenced by the strong enhancements of Cas9 cleavage resulting from the action of the chromatin remodelers SNF2h and RSC. These two remodelers produce distinct nucleosomal arrangements yet still substantially alleviate nucleosomemediated occlusion of Cas9 activity.

In combination, our data lead to a comprehensive model that reconciles both biochemical evidence and in vivo observations to explain how Cas9 is able to access nucleosomal DNA in live cells ([Figure 3I](#page-6-0)). In vivo, the majority of nucleosomes are not located on strong positioning sequences, and therefore may be permissive to Cas9 binding, especially at target sites that are readily accessible by DNA unpeeling. Chromatin remodeling activities can further provide diverse mechanisms to potentiate Cas9 activity at sites located close to the nucleosomal dyad or within more strongly positioned nucleosomes, which would otherwise be refractory to Cas9 action. We hypothesize that the combination of spontaneous DNA unpeeling and remodeling contributes to the widespread success of CRISPR-Cas9 in eukaryotic cells.

Interestingly, most applications of CRISPR-Cas9 in vivo have focused on genome engineering of protein-coding genes and other functional genomic elements associated with gene expression, which are typically associated with high rates of nucleosome remodeling ([Clapier and Cairns, 2009](#page-11-0)). It is also conceivable that Cas9 can temporarily gain access to less accessible regions of the genome during specific points of cell cycle (e.g. DNA replication), leading to sufficient DNA cleavage events to promote NHEJ-mediated mutagenesis or HDR-mediated DNA integration at appreciable rates. Recent studies on Cas9's behavior by single molecule imaging have also demonstrated that Cas9 favors more accessible euchromatin regions but is not completely excluded from transcriptionally silent heterochromatin ([Knight et al., 2015](#page-12-0)). For other CRISPR applications that require stable binding of nuclease-deficient dCas9 to DNA, such as transcriptional regulation and live-cell imaging with fluorescent dCas9, even modest nucleosome phasing could have a dramatic impact ([Gilbert et al.,](#page-11-0) [2013](#page-11-0); [Mali et al., 2013](#page-12-0); [Chen et al., 2013](#page-11-0); [Ma et al., 2015](#page-12-0)). For example, the +1 nucleosome in RNA pol II-transcribed genes is strongly positioned with phasing that dissipates gradually with each following nucleosome. Several high resolution studies conducted in parallel to our work have established that the +1 nucleosome and resulting nucleosome phasing can exert a strong influence on dCas9's DNA-binding ability for transcriptional regulation, but the effect is less striking on genome editing with Cas9 ([Horlbeck et al., 2016](#page-11-0); [Smith et al., 2016](#page-12-0)).

Our observations suggest that sgRNA design strategies that avoid targeting near the dyad of strongly phased nucleosomes are likely to be more successful than current methods. Large scale nucleosome positioning or DNA accessibility maps are now readily available and can inform CRISPR sgRNA design in order to avoid targeting regions of low accessibility ([Jiang and Pugh, 2009](#page-11-0); [Thurman et al., 2012](#page-12-0); [Wu et al., 2014](#page-13-0); [Hsieh et al., 2015](#page-11-0)). Alternatively, whole cell chromatin decondensation or de-repression using chromatin factor drugs such as HDAC or DNA methyltransferase inhibitors may be an alternative and attractive strategy for improving CRISPR-Cas9 activity towards densely compact regions of chromatin ([Haaf, 1995](#page-11-0); To´ [th et al., 2004](#page-12-0)).

# Materials and methods

#### Cas9 and sgRNA preparation

Wild-type Streptococcus pyogenes Cas9 and catalytically-inactive Cas9 (dCas9) containing D10A and H840A mutations were individually cloned into a custom pET-based expression vector encoding an N-terminal 6xHis-tag followed by a small ubiquitin-related modifier (SUMO) fusion tag and a Ulp1 protease cleavage site. Recombinant Cas9 variants were then expressed in Escherichia coli strain BL21 (DE3) (Novagen) and further purified to homogeneity as previously described ([Jiang et al.,](#page-11-0) [2015](#page-11-0)).

Single guide RNAs (sgRNAs) were prepared by in vitro run-off transcription using recombinant His-tagged T7 RNA polymerase and PCR product templates. Briefly, the DNA templates containing a T7 promoter, a 20-nt target DNA sequence (listed in [Table 1](#page-9-0)) and an optimal 78-nt sgRNA scaffold were PCR amplified using Phusion Polymerase (NEB) according to manufacturer's protocol. The following PCR products were used directly as DNA templates for in vitro RNA synthesis in 1x transcription buffer (30 mM Tris-HCl pH 8.1, 20 mM MgCl2, 2 mM spermidine, 10 mM DTT, 0.1% Triton X-100, 5 mM each NTP, and 100 $^{m}$g mL-1 T7 RNA polymerase). After incubation at 37˚C for 4–8 hr, the

![](assets/pictures/_page_9_Picture_1.jpeg)

Table 1. Spacer sequences for sgRNAs used in biochemistry experiments.

| sgRNA #          | Guide sequence                     | PAM | Target<br>strand | Figures where used                                                                                     |
|------------------|------------------------------------|-----|------------------|--------------------------------------------------------------------------------------------------------|
| 601_1            | CGAGTTCATCCCTTATGTGA               | TGG | Antisense        | Figure 1D                                                                                              |
| 601_2<br>(entry) | AATTGAGCGGCCTCGGCACC GGG Sense     |     |                  | Figure 1D, Figure 2B–D, Figure 2—figure supplement 1, Figure 3E–H, Figure 3—<br>figure supplement 1D–E |
| 601_3<br>(core)  | CCCCCGCGTTTTAACCGCCA               |     | AGG Antisense    | Figure 1B–D, Figure 1—figure supplement 1B–C, Figure 2B–D, Figure 2—figure<br>supplement 1             |
| 601_4            | GTATATATCTGACACGTGCC               | TGG | Sense            | Figure 1D                                                                                              |
| 601_5            | TCGCTGTTCAATACATGCAC               |     | AGG Sense        | Figure 1D                                                                                              |
| 601_6            | GCGACCTTGCCGGTGCCAGT CGG Antisense |     |                  | Figure 1D                                                                                              |
| 5S_1<br>(entry)  | TCTGATCTCTGCAGCCAAGC               |     | AGG Sense        | Figure 2B–E, Figure 2—figure supplement 1                                                              |
| 5S_2<br>(core)   | TATGGCCGTAGGCGAGCACA AGG Antisense |     |                  | Figure 2B–E, Figure 2—figure supplement 1                                                              |

[DOI: 10.7554/eLife.13450.046](http://dx.doi.org/10.7554/eLife.13450.046Table%201.Spacer%20sequences%20for%20sgRNAs%20used%20in%20biochemistry%20experiments.%2010.7554/eLife.13450.046sgRNA%20#Guide%20sequencePAMTarget%20strandFigures%20where%20used601_1CGAGTTCATCCCTTATGTGATGGAntisenseFigure%201D601_2%20(entry)AATTGAGCGGCCTCGGCACCGGGSenseFigure%201D,%20Figure%202B&x2013;D,%20Figure%202&x2014;figure%20supplement%201,%20Figure%203E&x2013;H,%20Figure%203&x2014;figure%20supplement%201D&x2013;E601_3%20(core)CCCCCGCGTTTTAACCGCCAAGGAntisenseFigure%201B&x2013;D,%20Figure%201&x2014;figure%20supplement%201B&x2013;C,%20Figure%202B&x2013;D,%20Figure%202&x2014;figure%20supplement%201601_4GTATATATCTGACACGTGCCTGGSenseFigure%201D601_5TCGCTGTTCAATACATGCACAGGSenseFigure%201D601_6GCGACCTTGCCGGTGCCAGTCGGAntisenseFigure%201D5S_1%20(entry)TCTGATCTCTGCAGCCAAGCAGGSenseFigure%202B&x2013;E,%20Figure%202&x2014;figure%20supplement%2015S_2%20(core)TATGGCCGTAGGCGAGCACAAGGAntisenseFigure%202B&x2013;E,%20Figure%202&x2014;figure%20supplement%201)

reactions were further treated with RNase-free DNase I (Promega) at 37˚C for 30 min to remove the DNA templates. The synthesized sgRNAs were purified by Ambion MEGAclear kit and eluted into DEPC-treated H2O for downstream experiments.

## Nucleosome reconstitution

Gradient salt dialysis was used to assemble mono-nucleosomes on DNA templates containing the 147 bp long 601 or the 5S positioning sequence from X. borealis (listed in Table 2), and labeled with fluorescein on the 5' upstream end. Histones and histone octamers were prepared as previously described ([Luger et al., 1999](#page-12-0)).

## Cas9 cleavage assays

Cleavage assays were conducted as previously described with the following modifications ([Anders and Jinek, 2014](#page-11-0)). Cas9:sgRNA complexes were reconstituted by incubating Cas9 and sgRNA for 10 min at 37˚C. Reactions contained 5 nM fluorescein labeled DNA or nucleosomes and 100 nM Cas9:sgRNA. In combined cleavage and remodeling experiments, 25 nM SNF2h or 3 nM RSC was first incubated with 5 nM naked DNA or nucleosomes for 45 min at 37˚C ([Narlikar et al.,](#page-12-0) [2001](#page-12-0)). Cleavage assays were carried out in reaction buffer (20 mM Tris-HCl pH 7.5, 70 mM KCl, 5 mM MgCl2, 5% Glycerol, and 1 mM DTT) at 25˚C. For SNF2h and RSC remodeling experiments, 0.2 mM ATP was also added. For RSC remodeling experiments, 1 mM MgCl$^{2}$ was used. Time points were quenched using stop buffer (20 mM Tris-HCl pH 7.5, 70 mM EDTA, 2% SDS, 20% glycerol, and 0.2 mg/mL xylene cyanol and bromophenol blue). Proteins were digested with 3 mg/mL of Proteinase K and incubated at 50˚C for 20 min. Samples were resolved on 1x TBE, 10% Polyacrylamide gels

Table 2. Sequences for DNA molecules used for biochemical assays (Positioning sequence highlighted in grey).

| Name      | Sequence                                                                                                                                                                                                                                                                                                                        |
|-----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 601 80/80 | CGGGATCCTAATGACCAAGGAAAGCATGATTCTTCACACCGAGTTCATCCCTTATGTGATGGACCCTATACGCGGCCGC<br>CCTGGAGAATCCCGGTGCCGagGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCC<br>CGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCTGTGCATGTATTGAAC<br>AGCGACCTTGCCGGTGCCAGTCGGATAGTGTTCCGAGCTCCCACTCTAGAGGATCCCCGGGTACCGA |
| 601 0/0   | CTGGAGAATCCCGGTGCCGagGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCCC<br>GCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCTGT                                                                                                                                                                         |
| 601 80/0  | CGGGATCCTAATGACCAAGGAAAGCATGATTCTTCACACCGAGTTCATCCCTTATGTGATGGACCCTATACGCGGCCGC<br>CCTGGAGAATCCCGGTGCCGagGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCC<br>CGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCTGT                                                                                     |
| 5S 0/0    | GGCCCGACCCTGCTTGGCTGCAGAGATCAGACGATATCGGGCACTTTCAGGGTGGTATGGCCGTAGGCGAGCACAAGGCT<br>GACTTTTCCTCCCCTTGTGCTGCCTTCTGGGGGGGGCCCAGCCGGATCCCCGGGCGAGCTCGAATT                                                                                                                                                                          |

[DOI: 10.7554/eLife.13450.047](http://dx.doi.org/10.7554/eLife.13450.047Table%202.Sequences%20for%20DNA%20molecules%20used%20for%20biochemical%20assays%20(Positioning%20sequence%20highlighted%20in%20grey).%2010.7554/eLife.13450.047NameSequence601%2080/80CGGGATCCTAATGACCAAGGAAAGCATGATTCTTCACACCGAGTTCATCCCTTATGTGATGGACCCTATACGCGGCCGCCCTGGAGAATCCCGGTGCCGagGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCTGTGCATGTATTGAACAGCGACCTTGCCGGTGCCAGTCGGATAGTGTTCCGAGCTCCCACTCTAGAGGATCCCCGGGTACCGA601%200/0CTGGAGAATCCCGGTGCCGagGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCTGT601%2080/0CGGGATCCTAATGACCAAGGAAAGCATGATTCTTCACACCGAGTTCATCCCTTATGTGATGGACCCTATACGCGGCCGCCCTGGAGAATCCCGGTGCCGagGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCTGT5S%200/0GGCCCGACCCTGCTTGGCTGCAGAGATCAGACGATATCGGGCACTTTCAGGGTGGTATGGCCGTAGGCGAGCACAAGGCTGACTTTTCCTCCCCTTGTGCTGCCTTCTGGGGGGGGCCCAGCCGGATCCCCGGGCGAGCTCGAATT)

![](assets/pictures/_page_10_Picture_1.jpeg)

for 4 hr at 140 V before visualizing using a Typhoon scanner (GE Healthcare) and quantifying with Image J (*Schneider et al., 2012*). For band quantification, background intensity was first subtracted after averaging the intensity of three areas. For cleavage gels, fraction uncleaved was determined by measuring the intensity of the uncleaved band compared to the total intensity for the lane. Similarly, fraction unbound was determined by measuring the intensity of the unbound band compared to the total intensity for the lane.

All experiments were performed in triplicate. Experiment variability is presented as the standard error of the mean, calculated by the standard deviation divided by the square root of N.

Propagation of error for Rates of Cleavage on Nucleosomes to Rates of Cleavage on DNA was calculated as follows:

$$Error = \frac{k_{Nucleosome}}{k_{DNA}} \sqrt{\left(\frac{SEM_{Nucleosomes}}{k_{Nucleosomes}}\right)^{2} + \left(\frac{SEM_{DNA}}{k_{DNA}}\right)^{2}}$$

Data were fit on Graphpad Prism using a standard one phase decay model:

$$Y = (Y_0 - Plateau)e^{-kt} + Plateau$$

where Y is the fraction of uncleaved DNA,  $Y_0$  is the value of Y at time = 0, k is the observed rate constant (min$^{-1}$) and t is time (min).

#### Native gel mobility shift assays

dCas9 and a 2x molar ratio of sgRNA were incubated for 10 min at 37°C. Various concentrations of dCas9:sgRNA complex were incubated with 20 nM naked DNA or nucleosomes in binding buffer (20 mM Tris-HCl pH 7.5, 100 mM KCl, 5 mM MgCl$_{2}$, 5% Glycerol, 1 mM DTT, and 0.02% NP-40). Samples were incubated at room temperature for 1 hr before being run on native 0.5X TBE 6% polyacrylamide gels, visualized on a Typhoon scanner, and quantified using ImageJ. Fraction unbound was measured as the intensity of all unbound species divided by the total intensity. Fraction unbound was then converted to fraction bound:

Fraction Bound = 1 - Fraction Unbound,

and binding curves were fit with:

$$Fraction Bound = \frac{\left[Cas9 : sgRNA\right]^{n}}{\left(\left[Cas9 : sgRNA\right]^{n} + K_{1/2}^{n}\right)}$$

# **Acknowledgements**

We would like to thank members of the Narlikar Lab, especially Nathan Gamarra, Coral Zhou, Kalyan Sinha, and Stephanie Johnson for providing reagents and assistance and members of the Lim lab, especially Scott Coyle, Levi Rupp, Amir Mitchell and Russell Gordley for assistance and helpful discussions during the planning and preparation of this manuscript.

## Additional information

#### Competing interests

JAD: Co-founder of Caribou Biosciences; Editas Medicine; Intellia Therapeutics. WAL: Founder of Cell Design Labs, and member of its scientific advisory board. The other authors declare that no competing interests exist.

#### **Funding**

| Funder                                                            | Grant reference number | Author      |
|-------------------------------------------------------------------|------------------------|-------------|
| Merck Fellow of the Damon<br>Runyon Cancer Research<br>Foundation | DRG-2201-14            | Fuguo Jiang |

![](assets/pictures/_page_11_Picture_1.jpeg)

| National Science Foundation        | 1244557      | Jennifer A Doudna                  |
|------------------------------------|--------------|------------------------------------|
| National Institutes of Health      | R01 DA036858 | Wendell A Lim                      |
| National Institutes of Health      | P50 GM081879 | Wendell A Lim                      |
| National Institutes of Health      | R01 GM073767 | Geeta J Narlikar                   |
| Howard Hughes Medical<br>Institute |              | Jennifer A Doudna<br>Wendell A Lim |

The funders had no role in study design, data collection and interpretation, or the decision to submit the work for publication.

#### Author contributions

RSI, Conceived of and conducted the biochemistry experiments and data analysis, Helped write this report; FJ, Generated reagents used in experiments, Edited this report; JAD, Contributed ideas and reagents, Edited this report; WAL, GJN, Co-supervised the work, Helped write this report; RA, Conceived of and conducted the work, Generated reagents, Wrote this report

# References

- Anders C, Jinek M. 2014. Chapter One In Vitro Enzymology of Cas9. In: Doudna JA, Sontheimer EJ. Methods in Enzymology. Academic Press 546:1–20. [doi: 10.1016/B978-0-12-801185-0.00001-5](http://dx.doi.org/10.1016/B978-0-12-801185-0.00001-5)
- Anderson JD, Tha˚ stro¨ m A, Widom J. 2002. Spontaneous access of proteins to buried nucleosomal DNA target sites occurs via a mechanism that is distinct from nucleosome translocation. Molecular and Cellular Biology 22: 7147–7157. [doi: 10.1128/MCB.22.20.7147-7157.2002](http://dx.doi.org/10.1128/MCB.22.20.7147-7157.2002)
- Anderson JD, Widom J. 2000. Sequence and position-dependence of the equilibrium accessibility of nucleosomal DNA target sites. Journal of Molecular Biology 296:979–987. [doi: 10.1006/jmbi.2000.3531](http://dx.doi.org/10.1006/jmbi.2000.3531)
- Chen B, Gilbert LA, Cimini BA, Schnitzbauer J, Zhang W, Li GW, Park J, Blackburn EH, Weissman JS, Qi LS, Huang B. 2013. Dynamic imaging of genomic loci in living human cells by an optimized crispr/cas system. Cell 155:1479–1491. [doi: 10.1016/j.cell.2013.12.001](http://dx.doi.org/10.1016/j.cell.2013.12.001)
- Clapier CR, Cairns BR. 2009. The biology of chromatin remodeling complexes. Annual Review of Biochemistry 78:273–304. [doi: 10.1146/annurev.biochem.77.062706.153223](http://dx.doi.org/10.1146/annurev.biochem.77.062706.153223)
- Cong L, Ran FA, Cox D, Lin S, Barretto R, Habib N, Hsu PD, Wu X, Jiang W, Marraffini LA, Zhang F. 2013. Multiplex genome engineering using crispr/cas systems. Science 339:819–823. [doi: 10.1126/science.1231143](http://dx.doi.org/10.1126/science.1231143)
- Dixon JR, Selvaraj S, Yue F, Kim A, Li Y, Shen Y, Hu M, Liu JS, Ren B. 2012. Topological domains in mammalian genomes identified by analysis of chromatin interactions. Nature 485:376–380. [doi: 10.1038/nature11082](http://dx.doi.org/10.1038/nature11082)
- Doudna JA, Charpentier E. 2014. The new frontier of genome engineering with crispr-cas9. Science 346: 1258096. [doi: 10.1126/science.1258096](http://dx.doi.org/10.1126/science.1258096)
- Ghorbal M, Gorman M, Macpherson CR, Martins RM, Scherf A, Lopez-Rubio JJ. 2014. Genome editing in the human malaria parasite plasmodium falciparum using the crispr-cas9 system. Nature Biotechnology 32:819– 821. [doi: 10.1038/nbt.2925](http://dx.doi.org/10.1038/nbt.2925)
- Gilbert LA, Larson MH, Morsut L, Liu Z, Brar GA, Torres SE, Stern-Ginossar N, Brandman O, Whitehead EH, Doudna JA, Lim WA, Weissman JS, Qi LS, a Lim W. 2013. CRISPR-mediated modular rna-guided regulation of transcription in eukaryotes. Cell 154:442–451. [doi: 10.1016/j.cell.2013.06.044](http://dx.doi.org/10.1016/j.cell.2013.06.044)
- Haaf T. 1995. The effects of 5-azacytidine and 5-azadeoxycytidine on chromosome structure and function: Implications for methylation-associated cellular processes. Pharmacology & Therapeutics 65:19–46. [doi: 10.](http://dx.doi.org/10.1016/0163-7258(94)00053-6) [1016/0163-7258\(94\)00053-6](http://dx.doi.org/10.1016/0163-7258(94)00053-6)
- He X, Fan HY, Narlikar GJ, Kingston RE. 2006. Human ACF1 alters the remodeling strategy of snf2h. The Journal of Biological Chemistry 281:28636–28647. [doi: 10.1074/jbc.M603008200](http://dx.doi.org/10.1074/jbc.M603008200)
- Hinz JM, Laughery MF, Wyrick JJ. 2015. Nucleosomes inhibit cas9 endonuclease activity in vitro. Biochemistry 54:7063–7066. [doi: 10.1021/acs.biochem.5b01108](http://dx.doi.org/10.1021/acs.biochem.5b01108)
- Horlbeck MA, Witkowsky LB, Guglielmi B, Replogle JM, Gilbert LA, Villalta JE, Torigoe SE, Tjian R, Weissman JS. 2016. Nucleosomes impede cas9 access to DNA in vivo and in vitro. eLife 5. [doi: 10.7554/eLife.12677](http://dx.doi.org/10.7554/eLife.12677)
- Hsieh TH, Weiner A, Lajoie B, Dekker J, Friedman N, Rando OJ, Friedman N. 2015. Mapping nucleosome resolution chromosome folding in yeast by micro-c. Cell 162:108–119. [doi: 10.1016/j.cell.2015.05.048](http://dx.doi.org/10.1016/j.cell.2015.05.048)
- Jiang C, Pugh BF. 2009. Nucleosome positioning and gene regulation: Advances through genomics. Nature Reviews. Genetics 10:161–172. [doi: 10.1038/nrg2522](http://dx.doi.org/10.1038/nrg2522)
- Jiang F, Zhou K, Ma L, Gressel S, Doudna JA. 2015. A cas9-guide RNA complex preorganized for target DNA recognition. Science 348:1477–1481. [doi: 10.1126/science.aab1452](http://dx.doi.org/10.1126/science.aab1452)
- Jinek M, Chylinski K, Fonfara I, Hauer M, Doudna JA, Charpentier E. 2012. A programmable dual-rna-guided DNA endonuclease in adaptive bacterial immunity. Science 337:816–821. [doi: 10.1126/science.1225829](http://dx.doi.org/10.1126/science.1225829)
- Jinek M, East A, Cheng A, Lin S, Ma E, Doudna J. 2013. RNA-programmed genome editing in human cells. eLife 2:e00471. [doi: 10.7554/eLife.00471](http://dx.doi.org/10.7554/eLife.00471)

![](assets/pictures/_page_12_Picture_1.jpeg)

- Josephs EA, Kocak DD, Fitzgibbon CJ, McMenemy J, Gersbach CA, Marszalek PE. 2016. Structure and specificity of the rna-guided endonuclease cas9 during DNA interrogation, target binding and cleavage. Nucleic Acids Research 44:2474. [doi: 10.1093/nar/gkv1293](http://dx.doi.org/10.1093/nar/gkv1293)
- Knight SC, Xie L, Deng W, Guglielmi B, Witkowsky LB, Bosanac L, Zhang ET, El Beheiry M, Masson J-B, Dahan M, Liu Z, Doudna JA, Tjian R. 2015. Dynamics of crispr-cas9 genome interrogation in living cells. Science 350: 823–826. [doi: 10.1126/science.aac6572](http://dx.doi.org/10.1126/science.aac6572)
- Li G, Widom J. 2004. Nucleosomes facilitate their own invasion. Nature Structural & Molecular Biology 11:763– 769. [doi: 10.1038/nsmb801](http://dx.doi.org/10.1038/nsmb801)
- Lorch Y, Cairns BR, Zhang M, Kornberg RD. 1998. Activated rsc-nucleosome complex and persistently altered form of the nucleosome. Cell 94:29–34. [doi: 10.1016/S0092-8674\(00\)81218-0](http://dx.doi.org/10.1016/S0092-8674(00)81218-0)
- Lowary PT, Widom J. 1998. New DNA sequence rules for high affinity binding to histone octamer and sequencedirected nucleosome positioning. Journal of Molecular Biology 276:19–42. [doi: 10.1006/jmbi.1997.1494](http://dx.doi.org/10.1006/jmbi.1997.1494)
- Luger K, Dechassa ML, Tremethick DJ. 2012. New insights into nucleosome and chromatin structure: An ordered state or a disordered affair? Nature Reviews. Molecular Cell Biology 13:436–447. [doi: 10.1038/nrm3382](http://dx.doi.org/10.1038/nrm3382)
- Luger K, Ma¨der AW, Richmond RK, Sargent DF, Richmond TJ. 1997. Crystal structure of the nucleosome core particle at 2.8 A resolution. Nature 389:251–260. [doi: 10.1038/38444](http://dx.doi.org/10.1038/38444)
- Luger K, Rechsteiner TJ, Richmond TJ. 1999. Preparation of nucleosome core particle from recombinant histones. B.-M. in Enzymology, 304:3–19. [doi: 10.1016/S0076-6879\(99\)04003-3](http://dx.doi.org/10.1016/S0076-6879(99)04003-3)
- La¨ngst G, Bonte EJ, Corona DF, Becker PB. 1999. Nucleosome movement by CHRAC and ISWI without disruption or trans-displacement of the histone octamer. Cell 97:843–852. [doi: 10.1016/S0092-8674\(00\)80797-7](http://dx.doi.org/10.1016/S0092-8674(00)80797-7)
- Ma H, Naseri A, Reyes-Gutierrez P, Wolfe SA, Zhang S, Pederson T. 2015. Multicolor CRISPR labeling of chromosomal loci in human cells. Proceedings of the National Academy of Sciences of the United States of America 112:3002–3007. [doi: 10.1073/pnas.1420024112](http://dx.doi.org/10.1073/pnas.1420024112)
- Makarova KS, Wolf YI, Alkhnbashi OS, Costa F, Shah SA, Saunders SJ, Barrangou R, Brouns SJ, Charpentier E, Haft DH, Horvath P, Moineau S, Mojica FJ, Terns RM, Terns MP, White MF, Yakunin AF, Garrett RA, van der Oost J, Backofen R, et al. 2015. An updated evolutionary classification of crispr-cas systems. Nature Reviews Microbiology 13:722–736. [doi: 10.1038/nrmicro3569](http://dx.doi.org/10.1038/nrmicro3569)
- Mali P, Aach J, Stranges PB, Esvelt KM, Moosburner M, Kosuri S, Yang L, Church GM. 2013. CAS9 transcriptional activators for target specificity screening and paired nickases for cooperative genome engineering. Nature Biotechnology 31. [doi: 10.1038/nbt.2675](http://dx.doi.org/10.1038/nbt.2675)
- Mali P, Yang L, Esvelt KM, Aach J, Guell M, DiCarlo JE, Norville JE, Church GM. 2013. RNA-guided human genome engineering via cas9. Science 339:823–826. [doi: 10.1126/science.1232033](http://dx.doi.org/10.1126/science.1232033)
- Narlikar GJ, Phelan ML, Kingston RE. 2001. Generation and interconversion of multiple distinct nucleosomal states as a mechanism for catalyzing chromatin fluidity. Molecular Cell 8:1219–1230. [doi: 10.1016/S1097-2765](http://dx.doi.org/10.1016/S1097-2765(01)00412-9) [\(01\)00412-9](http://dx.doi.org/10.1016/S1097-2765(01)00412-9)
- Narlikar GJ, Sundaramoorthy R, Owen-Hughes T. 2013. Mechanisms and functions of atp-dependent chromatinremodeling enzymes. Cell 154:490–503. [doi: 10.1016/j.cell.2013.07.011](http://dx.doi.org/10.1016/j.cell.2013.07.011)
- Olins AL, Olins DE. 1974. Spheroid chromatin units (ngr bodies). Science 183:330–332. [doi: 10.1126/science.183.](http://dx.doi.org/10.1126/science.183.4122.330) [4122.330](http://dx.doi.org/10.1126/science.183.4122.330)
- Partensky PD, Narlikar GJ. 2009. Chromatin remodelers act globally, sequence positions nucleosomes locally. Journal of Molecular Biology 391:12. [doi: 10.1016/j.jmb.2009.04.085](http://dx.doi.org/10.1016/j.jmb.2009.04.085)
- Polach KJ, Widom J. 1995. Mechanism of protein access to specific DNA sequences in chromatin: A dynamic equilibrium model for gene regulation. Journal of Molecular Biology 254:130–149. [doi: 10.1006/jmbi.1995.](http://dx.doi.org/10.1006/jmbi.1995.0606) [0606](http://dx.doi.org/10.1006/jmbi.1995.0606)
- Rowe CE, Narlikar GJ. 2010. The atp-dependent remodeler RSC transfers histone dimers and octamers through the rapid formation of an unstable encounter intermediate. Biochemistry 49:9882–9890. [doi: 10.1021/](http://dx.doi.org/10.1021/bi101491u) [bi101491u](http://dx.doi.org/10.1021/bi101491u)
- Schneider CA, Rasband WS, Eliceiri KW. 2012. NIH image to imagej: 25 years of image analysis. Nature Methods 9:671–675. [doi: 10.1038/nmeth.2089](http://dx.doi.org/10.1038/nmeth.2089)
- Schnitzler G, Sif S, Kingston RE. 1998. Human SWI/SNF interconverts a nucleosome between its base state and a stable remodeled state. Cell 94:17–27. [doi: 10.1016/S0092-8674\(00\)81217-9](http://dx.doi.org/10.1016/S0092-8674(00)81217-9)
- Smith JD, Suresh S, Schlecht U, Wu M, Wagih O, Peltz G, Davis RW, Steinmetz LM, Parts L, St Onge RP. 2016. Quantitative CRISPR interference screens in yeast identify chemical-genetic interactions and new rules for guide RNA design. Genome Biology 17:45. [doi: 10.1186/s13059-016-0900-9](http://dx.doi.org/10.1186/s13059-016-0900-9)
- Sternberg SH, LaFrance B, Kaplan M, Doudna JA. 2015. Conformational control of DNA target cleavage by crispr–cas9. Nature 527:110–113. [doi: 10.1038/nature15544](http://dx.doi.org/10.1038/nature15544)
- Sternberg SH, Redding S, Jinek M, Greene EC, Doudna JA. 2014. DNA interrogation by the CRISPR rna-guided endonuclease cas9. Nature 507:62–67. [doi: 10.1038/nature13011](http://dx.doi.org/10.1038/nature13011)
- Thurman RE, Rynes E, Humbert R, Vierstra J, Maurano MT, Haugen E, Sheffield NC, Stergachis AB, Wang H, Vernot B, Garg K, John S, Sandstrom R, Bates D, Boatman L, Canfield TK, Diegel M, Dunn D, Ebersol AK, Frum T, et al. 2012. The accessible chromatin landscape of the human genome. Nature 489:75–82. [doi: 10.1038/](http://dx.doi.org/10.1038/nature11232) [nature11232](http://dx.doi.org/10.1038/nature11232)
- Travers A, Muskhelishvili G. 2005. Bacterial chromatin. Current Opinion in Genetics & Development 15:507–514. [doi: 10.1016/j.gde.2005.08.006](http://dx.doi.org/10.1016/j.gde.2005.08.006)
- To´ th KF, Knoch TA, Wachsmuth M, Frank-Sto¨ hr M, Sto¨ hr M, Bacher CP, Mu¨ ller G, Rippe K. 2004. Trichostatin ainduced histone acetylation causes decondensation of interphase chromatin. Journal of Cell Science 117:4277– 4287. [doi: 10.1242/jcs.01293](http://dx.doi.org/10.1242/jcs.01293)

![](assets/pictures/_page_13_Picture_1.jpeg)

Vyas VK, Barrasa MI, Fink GR. 2015. A candida albicans CRISPR system permits genetic engineering of essential genes and gene families. Science Advances 1. [doi: 10.1126/sciadv.1500248](http://dx.doi.org/10.1126/sciadv.1500248)

Woodcock CLF, Safer JP, Stanchfield JE. 1976. Structural repeating units in chromatin. Experimental Cell Research 97:101–110. [doi: 10.1016/0014-4827\(76\)90659-5](http://dx.doi.org/10.1016/0014-4827(76)90659-5)

Wu Y, Zhang W, Jiang J. 2014. Genome-wide nucleosome positioning is orchestrated by genomic regions associated with dnase I hypersensitivity in rice. PLoS Genetics 10:e1004378. [doi: 10.1371/journal.pgen.1004378](http://dx.doi.org/10.1371/journal.pgen.1004378)

Yang JG, Madrid TS, Sevastopoulos E, Narlikar GJ. 2006. The chromatin-remodeling enzyme ACF is an atpdependent DNA length sensor that regulates nucleosome spacing. Nature Structural & Molecular Biology 13: 1078–1083. [doi: 10.1038/nsmb1170](http://dx.doi.org/10.1038/nsmb1170)