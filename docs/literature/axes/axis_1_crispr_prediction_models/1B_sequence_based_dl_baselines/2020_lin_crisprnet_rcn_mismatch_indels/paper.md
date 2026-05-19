**www.advancedscience.com**

# **CRISPR-Net: A Recurrent Convolutional Network Quantifies CRISPR Off-Target Activities with Mismatches and Indels**

*Jiecong Lin, Zhaolei Zhang, Shixiong Zhang, Junyi Chen, and Ka-Chun Wong\**

**The off-target effects induced by guide RNAs in the CRISPR/Cas9 gene-editing system have raised substantial concerns in recent years. Many in silico predictive models have been developed for predicting the off-target activities; however, few are capable of predicting the off-target activities with insertions or deletions between guide RNA and target DNA sequence pair. In order to fill this gap, a recurrent convolutional network named CRISPR-Net is developed for scoring the gRNA-target pairs with mismatches and indels; and a machine-learning based model named CRISPR-Net-Aggregate is also developed for aggregating the scores as the consensus off-target score for each potential guide RNA. It is demonstrated that CRISPR-Net achieves competitive performance on CIRCLE-Seq and GUIDE-seq datasets with indels and mismatches, outperforming the state-of-the-art off-target prediction methods on two independent mismatch-only datasets. The CRISPR-Net-Aggregate also surpasses a competing method on the aggregation task. Moreover, a two-stage sensitivity analysis is introduced to visualize the CRISPR-Net prediction on the gRNA-target pair of interest, demonstrating how implicit knowledge encoded in CRISPR-Net contributes to the accurate off-target activity quantification. Finally, the source code is made available at the Code Ocean repository (https://codeocean.com/capsule/9553651/tree/v1).**

J. Lin, S. Zhang, J. Chen, Prof. K.-C. Wong Department of Computer Science University of Hong Kong Hong Kong E-mail: kc.w@cityu.edu.hk Prof. Z. Zhang The Donnelly Centre for Cellular and Biomolecular Research University of Toronto Toronto M5S 3E1, Canada Prof. Z. Zhang Department of Molecular Genetics University of Toronto Toronto M5S 3E1, Canada Prof. Z. Zhang Department of Computer Science University of Toronto Toronto M5S 3E1, Canada

![](assets/pictures/_page_0_Picture_7.jpeg)

© 2020 The Authors. Published by WILEY-VCH Verlag GmbH & Co. KGaA, Weinheim. This is an open access article under the terms of the Creative Commons Attribution License, which permits use, distribution and reproduction in any medium, provided the original work is properly cited.

**DOI: 10.1002/advs.201903562**

## 1. Introduction

The CRISPR-Cas9 system is a groundbreaking tool for gene editing in various species and cell types.[1–5] The Cas9 is an RNAguided effector endonuclease protein that cleaves double-stranded DNA at the upstream of a 3-nucleotide protospacer adjacent motif (PAM) bearing sequences complementary to a 20-nucleotide segment in the guide RNA (gRNA). Its adaptability and specificity endow the gene-editing tool with great potential for repairing defective genes and editing crop genomes to boost their productivity with disease resistance.[6–9]

Although the CRISPR/Cas9 system has been demonstrated for its high-efficiency in targeted cleavages in many species such as human, mouse, and plants, many studies have reported that the Cas9 complex can wrongly bind to unintended regions (off-targets) and cleave at those unintended spots as well.[10–12] Those off-target effects of guide RNAs can lead to side effects, hindering the clinical applications of CRISPR/Cas9 system.

Therefore, it is crucial to design a robust guide RNA with high on-target cleavage efficiency and low off-target effects.[13,14]

Lin et al. summarized the off-targets induced by CRISPR gR-NAs into three cases as shown in **Figure 1**: a) off-target sites with base mismatch; b) off-target site with missing base (RNA bulge or insertion); c) off-target sites with extra base (DNA bulge or deletion).[15] The cases (b) and (c) are considered as the indel (insertion or deletion) off-target events. Recent studies have shown that the CRISPR/Cas9 system non-specifically cleaves genomic DNA sequences with base mismatches (case a) and indels (cases b and c) causing off-target effects in mammalian cells with considerable frequencies.[15] It signifies that off-target cleavage can potentially appear anywhere in genome as long as the region contains a PAM and a protospacer sequence with minor base mismatches and indels. Such sequence segments are considered as the off-target candidates, in which the candidate with detected nuclease cleavage is considered as an active off-target, while most or all of those candidates are inactive for the CRISPR-Cas9 system. Therefore, the accurate recognition and quantification of active off-target cleavage segments among a large number of candidates are required for downstream off-target effect assessment.

Many experimental techniques have been developed to assess the off-target activities of guide RNAs such as GUIDE-Seq,[16] Digenome-Seq,[17] SITE-Seq,[18] CIRCLE-Seq,[19] HTGTS,[20] and

![](assets/figures/_page_1_Figure_4.jpeg)

**Figure 1.** Three cases of off-targets induced by CRISPR guide RNA.

BLISS.[21] Although those cell-based techniques are capable of detecting CRISPR off-target mutations in an unbiased genomewide fashion, in silico methods are relatively rapid, feasible, and low-cost to quantify the off-target activities for particular gRNAs before the assays were conducted. The early in silico methods such as CCTop,[22] MIT-score,[12] and CROP-IT,[23] used hand-crafted rules, while latter methods progressed to machinelearning-based approaches such as Naive Bayes,[11] boosted regression trees,[13] support vector machine,[24] and convolutional neural network.[25] However, to the best of our knowledge, none of the existing in silico methods can quantify the off-target activities with insertions or deletions (indels) between target DNA and guide RNA sequences. Several recent studies have shown that insertions and deletions also contribute significantly to the offtarget problem,[15,26] calling for new approaches that can quantify the off-target activities of gRNA-targets with both indels and mismatches.

The hybrid convolutional and recurrent neural network was first adopted for visual recognition and description; it was then proliferated across different research fields such as natural language processing, sound event detection, and bioinformatics.[27–31] In the genomic sequence prediction task, the recurrent convolutional neural network has been well studied for discovering complex biological rules related to the proteincoding potential, predicting the function of non-coding DNA sequences, and subcellular protein locations.[32–35] Although the hybrid convolutional recurrent architectures have achieved remarkable successes in genomic sequence analysis, they have not been applied under the current context. Hence, we aim to build a deep recurrent convolutional network for modeling the off-target activities of gRNA-target with mismatches and indels. Furthermore, we performed a series of visualization techniques to uncover how base pairs in the gRNA-target contribute to the off-target activity, which is helpful to gRNA optimization in practical use.

This work provides the following contributions:

- 1. We propose an encoding scheme for converting each gRNAtarget sequence pair with indels and mismatches into a binary matrix as the suitable input for deep learning.
- 2. We develop CRISPR-Net in which a recurrent convolutional network combines Inception-based convolutional neural network and bidirectional LSTM for scoring the off-target activity of each potential gRNA-target pair with mismatches and indels
- 3. We demonstrate that CRISPR-Net not only achieves competitive performance in quantifying off-target activities with both mismatches and indels but also outperforms the current state-of-the-art off-target prediction methods on the task of mismatches-only off-target activity prediction.
- 4. We develop CRISPR-Net-Aggregate, a machine-learningbased model that aggregates the gRNA-target scores from CRISPR-Net into a single consensus off-target score.
- 5. We propose a two-stage sensitivity analysis that reveals how the sequence variations in each gRNA-target pair perturbs the CRISPR-Net off-target activity quantification. The first stage is the occlusion sensitivity analysis where each potential site is assumed to be occluded on each gRNA-target pair of interest. The second stage is a replacement sensitivity analysis by iteratively replacing each potential site in a given on- and off-target sequence pair with correct nucleotide matches (A–A, T–T, G– G, and C–C) separately.

# **Experimental Section**

### **gRNA-Target Sequence Pair Encoding Scheme**

Off-target sites of gRNA were homologous to the on-target site but with several mismatches and indels. Off-target activity of intended sites in genome could be predicted based on the sequence of gRNA and the off-target (termed gRNA-target). The goal was

21983844, 2020, 13, Downloaded from https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.201903562 by Turkey Cochrane Evidence Aid, Wiley Online Library on [01/05/2026]. See the Terms and Conditions (https://onlinelibrary.wiley.com/terms-and-conditions) on Wiley Online Library for rules of use; OA articles are governed by the applicable Creative Commons License

![](assets/figures/_page_2_Figure_4.jpeg)

**Figure 2.** Representation for the gRNA-target. The sequence pair in the purple box is an example of gRNA-target. The blue arrows illustrate how the onand off-target pair represent the gRNA-target. The mismatches in red boxes are identically represented. All information on the gRNA-target is completely preserved in the on- and off-target sequence pair.

![](assets/figures/_page_2_Figure_6.jpeg)

**Figure 3.** Seven-bit Encoding Example. The symbol "\_" denotes the DNA or RNA bulge locations. Each sequence pair is considered as a fixed length vector with the five-bit channel (A, G, C, T, \_) and two-bit direction channel. The five-bit channel preserves the nucleotides of the on-target site and the off-target site while the direction channel is designed to identify the indel and mismatch directions; for example, "00110-10" represents the mismatch "G →C" ("G" is the on-target site and "C" is the off-target site) while "00110-01" represents the mismatch "C→G"; "01001-01" represents the insertion (RNA bulge) "\_→T" and "00011-10" represents the deletion (DNA bulge) "C→\_"; "10000-00" represents the matched site "A–A."

to design and implement a model to learn the relationship between gRNA and the potential off-target sites for off-target activity prediction. In **Figure 2**, it is demonstrated that the gRNA-target sequence pair can be represented by the corresponding on- and off-target sequence pair. Since only A, G, C, and T were needed to represent each base of the on-target or off-target site, this representation of gRNA-target could avoid the redundant encoding for uracil.

The most intuitive way to convert a on- and off-target pair into the suitable convolutional neural network input was encoding each base (i.e., A, G, C, T, and Indel) of the on-target sequence and the off-target sequence with five-bit one-hot encoding with channel-wise concatenation. Therefore, every possible gRNA-target pair could be represented by a binary matrix with the size of 10 ×(length of the sequence pair). Herein, a new encoding schema was proposed which not only reduced the number of convolutional neural network's parameters, but also retained mutual information between on-target and off-target sites including mismatches, insertions, deletions, and matches.

**Figure 3** shows how the encoding scheme encodes three types of off-targets into binary matrices. First, the on-target site was used to represent the wild-type guide RNA sequence. Then, the on- and off-target DNA sequence pair was encoded with indels in addition to mismatches using a new symbol "\_" denoting

![](assets/figures/_page_3_Figure_5.jpeg)

**Figure 4.** Naïve Inception module[36].

the DNA or RNA bulge. The sequence pair was considered as a fixed length vector with five-bit channel (A, G, C, T, \_) and two-bit direction channel. The five-bit channel preserved the nucleotides of the on-target site and the off-target site. The direction channel was used to identify the insertion or deletion and the mismatch type. As exemplified in Figure 3, "00110-10" represented the mismatch "G→C" while "00110-01" represented the mismatch "C→G"; "01001-01" represented the insertion (RNA bulge) "\_→T" and "00011-10" represented the deletion (DNA bulge) "C→\_"; the one-hot vector "10000-00" represented the matched site "A–A."

Consequently, for each site pair in the gRNA-target sequence pair, the encoding schema could reduce the coding length from ten bits to seven bits, and thus the related model complexity. The following experiments show that this seven-bit encoding scheme can improve the CRISPR-Net's performance on off-target activity assessment, comparing to the ten-bit encoding scheme.

#### **Recurrent Convolutional Neural Network**

The proposed CRISPR-Net was built upon a so-called long-term recurrent convolutional neural network (LRCN).[27] The convolutional layer of LRCN is served as feature extractor while the recurrent layer was designed to recognize sequential patterns.

In the convolutional layer, the fixed-sized convolutional kernels in the original LRCN was replaced with an Inception module for CRISPR-Net, since this architecture has been shown to achieve state-of-the-art performance with relatively fast training speed in the area of computer vision. The Inception architecture was introduced in GoogLeNet[36]; it performed convolution on input with several different filter sizes (see **Figure 4**). This architecture allowed the internal layer to choose the appropriate filter size to learn the required information, even if the salient part of the input had large variation in size; besides, the parallel pooling operations in the Inception could reduce the feature map dimensionality and enable the representation to be approximately invariant to small input translation variations.

In the recurrent layer, LSTM (Long Short Term Memory) was proposed to resolve the problem of vanishing and exploding gradients rendered by traditional recurrent neural network.[37] An LSTM unit was commonly composed of a cell *c* and three gates (i.e., input gate *i*, output gate *o*, and forget gate *f*). The cell preserves the values over arbitrary time intervals and the three gates adjusts the information flow from the cell. For input sequence 〈*x*1, *x*2, …, *xT*〉, the key equations of LSTM unit used in LRCN are mathematical formulated as follows:

$$f_{t} = \sigma(W_{f}x_{t} + U_{f}h_{t-1} + b_{f})$$

$$i_{t} = \sigma(W_{i}x_{t} + U_{i}h_{t-1} + b_{i})$$

$$\tilde{c}_{t} = \tanh(W_{c}x_{t} + U_{c}h_{t-1} + b_{c})$$

$$c_{t} = i_{t} * \tilde{c}_{t} + f_{t} * c_{t-1}$$

$$o_{t} = \sigma(W_{o}x_{t} + U_{o}h_{t-1} + b_{o})$$

$$h_{t} = o_{t} * \tanh(c_{t})$$

$$z_{t} = h_{t}$$

where *W* ∈ ℝ*$^{n}$*×*$^{d}$* and *U* ∈ ℝ*$^{n}$*×*$^{n}$* are weight matrices, *xt* ∈ ℝ*$^{d}$* is the input, *ht* ∈ ℝ*$^{n}$* is the hidden state with *n* hidden units, and *zt* is the output at time *t*. The initial values are *c*$^{0}$ = 0 and *h*$^{0}$ = 0 while the operator "∗" denotes the Hadamard product and denotes the sigmoid function.

Recently, LRCNs have achieved impressive successes in the field of language tasks such as speech recognition and machine translation. Since genomic sequences are usually regarded as the "languages" of biological nature, it was intuitive to use LRCN for learning the "grammars" of the "languages"; for example, Quang et al. proposed an LRCN framework to predict the function of non-coding DNA directly from sequences[32]; Lanchantin et al. reported that their LRCN model surpassed CNN and RNN in the transcription factor binding site classification task.[33]

The advantages of the LRCN architecture for modeling genomic sequences are twofold. First, the convolutional layers of LRCN can discover useful features from sequences directly and independently, avoiding biases introduced by hand-crafted features, especially for the problems which biological mechanisms are not fully understood. Second, LSTM layers are able to actively regulate self-connecting loops to memorize the long-range information from the sequences. Next, an LRCN-based deep neural www.advancedsciencenews.com

www.advancedscience.com

![](assets/figures/_page_4_Figure_4.jpeg)

**Figure 5.** Graphical illustration of CRISPR-Net. The input of CRISPR-Net is on- and off-target sequence pair  $\langle X_{\rm on}, X_{\rm off} \rangle$  where the sequence pair is encoded into the binary matrix as the input of the Inception layer. The Inception layer consists of 10 filters with different sizes (1, 2, 3, 5). The output of Inception layer and the encoded matrix are merged by channel-wise concatenation, then passed to the recurrent layer. The bi-directional LSTM in the recurrent layer is used to learn sequential patterns. The number of each LSTM units is 15. The recurrent layer output is then passed into two non-linear dense layers after flattening, the numbers of neurons in the dense layers are 80 and 20, respectively. The output neuron uses sigmoid as the activation function, while the neurons in the other layers use ReLU as the activation functions.

network was described for quantifying the off-target activities of CRISPR guide RNAs.

### **CRISPR-Net**

Herein, an LRCN-based neural network named CRISPR-Net was proposed to predict the off-target activities. CRISPR-Net comprised an Inception-based convolutional layer and a recurrent layer with bi-directional LSTM units (BiLSTM). The outputs of BiLSTMs are passed to downstream dense layers with non-linear activations. **Figure 5** depicts the network architecture of CRISPR-Net. Remarkably, in the Inception-based convolutional layer, the pooling operations in the naïve Inception module were removed, and the encoded input was directly concatenated with the outputs of four specialized convolutional filters to preserve the spatial information of the sequence pair.

CRISPR-Net works by passing a on- and off-target sequence pair  $\langle X_{\rm on}, X_{\rm off} \rangle$  to the encoding layer to be encoded as a binary matrix  $E = [x_1, x_2, ..., x_T]$  where  $x_t \in \{0, 1\}^7$  and T denotes the sequence length, representing the mutual information of  $\langle X_{\rm on}, X_{\rm off} \rangle$ . The encoded matrix E was then fed to the Inception-based convolutional layer. The Inception-based convolutional layer comprised four convolutional filters types with dif-

ferent sizes; each filter type of filters  $F_i$  learns a specialized representation  $\Phi_{F_i}(\cdot)$  on E with ten independent convolutional filters along with the rectifier activation, to produce a fixed-size matrix  $C_i = \Phi_{F_i}(E) = [c_1, c_2, \ldots, c_T]$  where  $c_i \in \mathbb{R}^{10}$ . These four representation matrices  $\{C_1, C_2, C_3, C_4\}$  and the encoded sequence pair E were merged by channel-wise concatenation as the intermediate input  $K = [k_1, k_2, \ldots, k_T], k_i \in \mathbb{R}^{47}$  of the next-level recurrent layer for sequence learning.

In the recurrent layer, the bi-directional LSTM (B-LSTM) was a variant of the RNN which combined the outputs of two LSTMs in which one learns forward patterns from the input sequence while the other learns backward patterns from the input sequence. [38] Each forward LSTM module maps the input  $k_t$  and the previous base hidden state  $h_{t-1}$  to produce an output  $z_t^f$  and updated hidden state  $h_t$ , while the module in backward LSTM maps  $c_t$  and the following base hidden state  $h_{t+1}$  to an output  $z_t^b$ . Hence, for the intermediate input K, the B-LSTM outputs  $R = [y_1, y_2, ..., y_T]$  where  $y_t = [z_t^f, z_t^b]^T$ ,  $z_t^f$ ,  $z_t^b \in \mathbb{R}^n$ , and the number of hidden units n is 15.

The recurrent layer output *R* was then passed into two nonlinear dense layers after flattening, the numbers of neurons in the dense layers are 80 and 20, respectively. A dropout regularizer with 35% dropout rate was used in the last dense layer. Note that CRISPR-Net was trained and tested as the classifier and regressor. As a classifier, the assessment score of being active on- and

off-target sequence pair 〈*X*on, *X*off〉 was computed by taking the binary cross-entropy function to the output neuron with sigmoid activation; as a regressor, the assessment score was computed by taking the mean square loss function to the output neuron. The measured off-target activities (i.e., nuclease-cleaved sequencing reads) were re-sealed linearly to lie in [0, 1] before applying Box–Cox transformation, and the normalized values were used for training the regression model, while the gRNA-target pairs with nuclease-cleaved sequencing reads were considered as the positive samples for the classifier.

## **Visualizing CRISPR-Net Prediction Based on Sensitivity Analysis**

To elucidate how each base pair in gRNA-target influences CRISPR-Net's off-target prediction for a particular gRNA-target pair of interest, the sensitivity analysis was leveraged by perturbing each site of the gRNA-target sequence pair.

In the first stage of the sensitivity analysis, each encoded base pair of the gRNA-target was iteratively occluded with zero-valued vector, and the corresponding change of CRISPR-Net's output was regarded as the sensitivity score. The absolute value of each sensitivity score was further normalized into 0 to 1 and it was mapped back to the respective site occluded in the gRNA-target pair, indicating the base pair saliency. The saliency heat map demonstrated the effect degree of each base pair to the CRISPR-Net's prediction.

In the second stage, a replacement-based sensitivity analysis was designed to illustrate how CRISPR-Net off-target prediction changes when each site of the on and off-target pair for a specific gRNA-target was iteratively replaced by one of the match sites (A– A, T–T, G–G, and C–C). The replacement sensitivity score of the replaced site was calculated by the change of CRISPR-Net output. This analysis was particularly useful for gRNA design; it could identify the modified sites in gRNA-target pair with significant influences on the CRISPR off-target activity. The replaced site of gRNA-target with a negative sensitivity score was considered to inhibit the off-target activity while the positive sensitivity score indicates the replaced site enhances the off-target activity. Furthermore, all replacement sensitivity scores of a particular gRNAtarget pair were aggregated into a heat map for clear illustration on how each replaced site influences CRISPR-Net prediction (see Figure 14).

### **Aggregating Individual CRISPR-Net Predictions into a Single Summary Score**

CRISPR-Net was designed to assess the potential off-target activity in a particular region of the genome for a specific guide RNA. To compare several gRNAs' off-target activities, a single offtarget summary score has to be obtained for each gRNA from all of its potential gRNA-target scores across the whole genome. Listgarten et al. reorganized two datasets with Cas9/gRNA targeting on non-essential genes in cell viability screens to build and test the off-target activity summary model. Since the gRNAs in these datasets were targeted on the non-essential genes, the cells should be viable if there is not any off-target cleavage.[13]

Based on the viability data, Listgarten et al. designed and employed a machine-learning-based off-target summary model named Elevation-aggregate by analyzing the distribution of gRNA-target activities (only considering mismatches) predicted by Elevation-score. There were 23 features used for Elevationaggregate: the mean, the median, variance, standard deviation, 99th, 95th, and 90th percentiles, and the sum of off-target scores. These features were computed for each candidate of all offtargets, only genic off-targets, and only non-genic off-targets; the is-genic annotation is obtained from Ensembl.[39] They also computed extra features: the fraction of genic (and non-genic) offtargets, the fraction of targets that were genic (and non-genic), the ratio of the number of genic to non-genic targets, and the ratio of the average predictive genic score to non-genic score.[13]

As inspired and motivated from Elevation-aggregate, the extremely randomized regression trees were leveraged to aggregate the CRISPR-Net's predictions on all potential gRNA-target sequence pairs into an overall off-target score for a specific gRNA. The feature selection and the split threshold of the extremely randomized tree were completely randomized, which could reduce over-fitting. The best hyper-parameter setting that was found for the off-target aggregation task was assembling 300 extremely randomized trees whose maximum tree depth was 5, and the number of features for a split is log2(n\_features). The split criterion of each tree was mean squared error. For the unmentioned parameters, the default values of the Python Scikit-Learn package were adopted. For a fair comparison with Elevation-aggregate, 23 features were similarly computed from the distribution of the gRNA-target CRISPR-Net predictions to train the extremely randomized trees. The off-target aggregation model was named as CRISPR-Net-Aggregate.

### **Existing Off-Target Prediction Methods**

The existing off-target prediction methods were employed for two main uses. The first was to assess the activities of a given off-target region for a specific guide RNA. These methods could be summarized into two categories: one used the hand-crafted rules while the other used machine-learning-based models. The methods in the first category assessed the off-target activities using empirically determined rules such as MIT score,[12] CROP-IT score,[23] and CCTop score.[23] The rule of CCTop was built upon the experimental evidence that off-target effect would diminish if the mismatch is close to the PAM; CROP-IT score assessed the off-target effects by summing up the penalty scores of the adjacent mismatches and heuristics scores based on the mismatch positions; MIT score was built upon the experimental off-target data and reduced to one weight per position.

The first machine-learning-based method was CFD (Cutting Frequency Determination).[11] CFD is simply a Naive Bayes with two features (i.e., the position and identity of mismatch site) trained on a cleavage dataset obtained by infecting MOLM13 cells with a lentiviral library targeting the coding sequence of the human CD33 gene. This library contains thousands of guide RNAs with all possible PAM and single-nucleotide mismatches at all positions. In a benchmark study, Haeussler et al. built a large benchmark dataset comprising eight studies which detected or

![](assets/pictures/_page_6_Picture_1.jpeg)

assessed CRISPR off-target activities, to evaluate the performance of CFD and the existing hand-crafted rules.[40] Their experimental results showed that CFD outperformed the other three hand-crafted rule-based methods (i.e., CCTop score, CROP-IT score, and MIT score).

After the off-target benchmark was introduced, various machine-learning-based methods have been developed and surpassed CFD based on ROC (Receiver Operating Characteristic Curve) analysis. Listgarten et al. proposed a two-layer stacked regression model, named Elevation-score.[13] The first layer of Elevation-score used a trained boosted regression trees model to predict the off-target activity of each mismatch site in gRNAtarget pair independently hinging on four features (i.e., mismatch identity and position, mismatch identity, mismatch position, and mismatch transversion versus transition) whereas the second layer combined these predictive values for gRNA-target pairs with multiple mismatches using an L1-regularised regression combiner model. Peng et al. built an ensemble SVM classifier considering the nucleotide composition change features and position-specific binary mismatch features of gRNA-target pairs.[24]

Moreover, several studies have demonstrated the strength of deep-learning-based models on off-target activity predictions. For example, Chuai et al. present DeepCRISPR , a convolutional neural network merged with two pre-training deep convolutionary denosing encoders to predict off-target sites.[41] Liu et al. creatively introduced an attention-based convolutional neural network (i.e., AttnToMismatch\_CNN) for modeling off-target activities.[42] Another deep-learning-based model, CNN\_std, utilized multi-scale convolutional filters to learn gRNA-targets sequence patterns for off-targets classification.[25] Most recently, Alkan et al. proposed an approximate binding energy model named CRISPRoff, which consolidates the interactive energy parameters of RNA and DNA duplexes.[43] Their study indicates that CRISPRoff excelled Elevation on the computational off-target assessment.

The second use case of the existing off-target prediction methods was to aggregate the assessment of all potential off-target regions to obtain an overall assessment for a specific guide RNA. There were relatively fewer methods aiming for this case. The MIT web server provided a summary score to assess the gRNA's off-target activity based on the hand-crafted rules; CFD aggregation adopted the number of the within-gene active offtarget regions predicted by CFD as the summary score; Elevationaggregate employed the gradient boosted trees to summarize the individual gRNA-target scores into a single summary score for a specific guide RNA. Listgarten et al. compared the aforementioned off-target aggregation models on the datasets arising from gRNAs targeting non-essential genes in a viability screen, and found Elevation-aggregate achieved the best performance.

### **Modifying the Existing Off-Target Prediction Models to Handle Indels**

To the best of the authors' knowledge, there is not any published method that can quantify off-target activities with indels and mismatches. Hence, two mismatches-only off-target prediction models were modified for the comparison with CRISPR-Net. Lin et al. developed a convolutional neural network named CNN\_std for off-target prediction. The channel number was adapted from four to seven in the CNN\_std's first convolutional layer for our new encoding scheme. On the other hand, the modified CNN\_std could be trained and tested on the gRNA-target dataset with both mismatches and indels. Listgarten et al. leveraged the gradient boosted trees (Elevation-naïve) based on the hand-crafted features in the first layer of their stacked regression model for mismatch-only off-target prediction. To make use of the gradient boost trees for addressing indels off-target prediction, the gRNAtarget pairs were converted into the on and off-target pairs using the method shown in Figure 2, and the site pairs of on- and offtarget pair were extracted at every position (for example, "C–G:1" denotes a mismatch at first base pair, and "C–\_:2" denotes an indel at the second base pair), which were one-hot encoded and merged by concatenation. The gradient boosted trees could be trained and tested on both indels and mismatches dataset under this encoding scheme. The modified CNN\_std and the gradient boosted trees were evaluated alongside the CRISPR-Net on the indels off-target dataset in the experiments below.

### **Datasets**

Herein, two groups of off-target datasets were constructed for model training and validation corresponding to the two uses, each sample in the dataset was a potential target site subject to a specific gRNA. As shown in **Table 1**, the benchmark data comprise two types of gRNA-target datasets: type I contains gRNAtarget pairs with both mismatches and indels, and type II contains gRNA-target pairs with only mismatches.

In the first type, dataset I/1 contains gRNA-target sequence pairs with both mismatches and indels from ten different guide RNAs, of which 7371 active off-targets (430 of them with indels) were experimentally validated with CIRCLE-seq.[19] CIRCLEseq (circularization for in vitro reporting of cleavage effects by sequencing) was a highly sensitive and unbiased method for identifying the genome-wide off-targets of CRISPR-Cas9 nucleases. The CIRCLE-seq-validated off-targets were twofold: 340 offtargets with one-bp indel and up-to-three-bp mismatches and 7031 off-targets with up-to-six-bp mismatches. Given the gRNA sequences, Cas-Offinder,[45] a versatile tool that searches for potential off-target sites of Cas9 RNA-guided endonucleases, was adopted to obtain 577 578 inactive off-target sites in the genome associated with the two types as mentioned above. It was noted that the potential off-target search of Cas-Offinder is not approximate. For example, if a search was defined where up to six mismatches were tolerated, then all such sites across the genome would be returned. CRISPR-Net and two modified offtarget prediction models were evaluated under leave-one-gRNAout cross-validation (LOGOCV) on the dataset I/1. Each round of LOGOCV involved partitioning the whole dataset into two nonoverlapping subsets, training the model on nine gRNAs, and testing the remaining one gRNA. To further evaluate the models, the models were trained on dataset I/1 and they were tested on an independent dataset II/2. The dataset I/2 was comprised of 60 GUIDE-Seq validated off-targets (13 of them with indels)

![](assets/pictures/_page_7_Picture_3.jpeg)

**Table 1.** Dataset details in the first group.

| Type / No. | Technique                   | Total   | Validated Off-targets | Guide RNAs | With Indel              | Literature<br>Tasi et al.[19] |  |
|------------|-----------------------------|---------|-----------------------|------------|-------------------------|-------------------------------|--|
| I / 1      | CIRCLE-Seq                  | 584 949 | 7371                  | 10         | Yes                     |                               |  |
| I / 2      | GUIDE-Seq                   | 213 943 | 60                    | 6          | Yes                     | Listgarten et al.[13]         |  |
| II / 1     | Protein knockout detection  | 4853    | 2273                  | 65         | Doench et al.[11]<br>No |                               |  |
| II / 2     | PCR, Digenome-Seq and HTGTS | 10 129  | 354                   | 19         | No                      | Haeussler et al.[40]          |  |
| II / 3     | SITE-Seq                    | 217 733 | 3767                  | 9          | No                      | Cameron et al.[18]            |  |
| II / 4     | GUIDE-Seq                   | 294 534 | 52                    | 9          | No                      | Tasi et al.[16]               |  |
| II / 5     | GUIDE-Seq                   | 95 829  | 54                    | 5          | No                      | Kleinstiver et al.[44]        |  |
| II / 6     | GUIDE-Seq                   | 383 463 | 56                    | 22         | No                      | Listgarten et al.[13]         |  |

among 213 943 gRNA-target sequence pairs from six different guide RNAs.

In the second type, there were six independent gRNA-target datasets with only mismatches: dataset II/1 was constructed by Doench et al. with 4,853 gRNAs targeting on human coding sequence CD33; dataset II/2 is a off-target benchmark dataset built by Haeussler et al.[40]; dataset II/3 contains 3,767 positive offtarget sites from 9 gRNAs validated by SITE-Seq[18]; dataset I/4, I/5 and I/6 are all GUIDE-Seq validated gRNA-target datasets while they are from three different literatures (i.e., Tasi et al.,[16] Listgarten et al.,[13] and Kleinstiver et al.[44]). Note that Listgarten et al., Cameron et al., and Kleinstiver et al. only provided the active off-target sites. Thus, Cas-Offinder was utilized[45] to obtain all possible off-targeting sites with up-to-six-mismatches in the human genome and the corresponding datasets were constructed.

To evaluate the aggregation of off-target activity, Listgarten et al. made use of two datasets with guide RNAs targeting non-essential genes in viability screens—the Avana and GeCKO libraries.[13] Similarly, 5059 guide RNAs targeting on 878 nonessential genes were assembled from Avana library (named Avana-Agg) and 5178 guide RNAs targeting on 875 non-essential genes from GeCKOv2 library (named GeCKO-Agg) from Doench et al.,[11] of which the overlapped gRNAs was removed for redundancy removal. For each guide RNA in both Avana-Agg and GeCKO-Agg dataset, Cas-Offinder was used to find all potential off-targets (near-match CRISPR-Cas9 targets) in the human genome (hg38) with up to 6 bp mismatches or up to 3 bp mismatches with 1 bp indels. The viability screens of both libraries were performed in A375 melanoma cells. The targeting genes in both datasets were from a curated set of 927 non-essential genes.[46]

### **Experiments**

The experiments in this work were conducted in two phases: the first phase was designed to validate CRISPR-Net on quantifying the individual gRNA-target pair's off-target activity; the second phase was to evaluate the CRISPR-Net-Aggregate with the stateof-the-art off-target aggregation method—Elevation-aggregate.

In phase one, four steps of validation experiments were performed based on ROC (Receiver Operating Characteristic curve) and PR (Precision-Recall) analysis. First, four different versions of CRISPR-Net comprising the CRISPR-Net-Regressor and CRISPR-Net-Classifier were evaluated with two encoding schemes respectively alongside two existing modified models (modified CNN\_std and gradient boosted regression trees) on CIRCLE-Seq dataset (Dataset I/1 in Table 1) under leave-onegRNA-out cross-validation. Then, four different CRISPR-Net and two modified models were trained on CIRCLE-Seq dataset and they were tested on GUIDE-Seq dataset (Dataset I/2 in Table 1).

Second, CRISPR-Nets were compared with the five existing off-target prediction models (i.e., Elevation-score, Ensemble SVM, AttenToMismatch\_CNN, CNN\_std, and CRISPRoff) on an independent mismatches-only GUIDE-Seq dataset (i.e., Dataset II/5), while CFD was served as a baseline model. Note that two CRISPR-Nets were trained, Elevation-score and CNN\_std on the same dataset (i.e., Dataset II/1, II/2, and II/3, named Elevation dataset) which were constructed by Listgarten et al.[13] Both AttnToMismatch\_CNN and Ensemble SVM were well-established models provided by their authors that were already trained on a larger dataset including Elevation dataset. Besides, since CRISPRoff could only predict the off-target sites with PAM of NGG, NAG or NGA, the samples were removed without any of those PAMs in testing datasets just for CRISPRoff during ROC and PR analyses.

Third, two datasets (i.e., Dataset I/1 and II/4) were explored along with the training datasets (Elevation dataset) in step two on how they influence the prediction and generalisation performance of CRISPR-Net. Note that the off-target sites of dataset I/1 and II/4 were detected by SITE-Seq and CIRCLE-Seq which are the recently published biomedical approaches for identifying genome-wide CRISPR/Cas9 off-target mutations. Furthermore, the generalization of CRISPR-Net on dataset II/5 (which was not used to train any of those models) was investigated by comparing with six aforementioned existing methods.

Last step in phase one experiment was to compare CRISPR-Net with another state-of-the-art method—DeepCRISPR which utilized four binary epigenetic features (CTCF binding information, H3K4me3 position information, chromatin-opening information, and DNA methylation information) along with gRNAtarget sequence content for modelling off-target effects. Similarly, CRISPR-Net and CNN\_std were augmented with the same epigenetic features by concatenating seven-bit channels with those four-bit epigenetic channels for each off-target in the encoding layer. Then, two off-target datasets from HEK295 and K562 cell types constructed by Chuai et al. were used[41] for models testing. For a fair comparison, modified CRSIPR-Net, CNN\_std, and DeepCRISPR were trained on the dataset with the same epoches

![](assets/pictures/_page_8_Picture_1.jpeg)

from one cell type and tested them on the other. Besides, the same encoding schema was used to enable CNN\_std to handle epigenetics features for model comparison.

In phase two, the off-target scores of all gRNA-target pairs belonging to 10,237 gRNAs in Avana-Agg and GeCKO-Agg dataset were obtained from CRISPR-Net prediction. Then, CRISPR-Net-Aggregate was trained on the Avana-Agg gRNA-target scores and it was evaluated with Elevation-aggregate on GeCKO-Agg dataset (which were not used to train Elevation-score or CRISPR-Net-Aggregate). Note that gRNA-target scores from GeCKO-Agg dataset used by Elevation-aggregate were predicted by Elevationscore. Since each gRNA in GeCKO-Agg dataset had two replicas targeting one same non-essential gene, the weighted spearman correlation between predictive overall off-target scores and the two replicas' cell viability (log2 fold-change) and their average viability for comparing the performance of Elevation-aggregate and CRISPR-Net-Aggregate was calculated. Furthermore, CRISPR-Net-Aggregate was trained on the CRISPR-Net predicted scores with the only mismatch in Avana-Agg dataset for comparison with CRISPR-Net-Aggregate trained on the off-target scores of gRNA-target pairs with mismatches and indels.

The CRISPR-Net models in this paper were trained using Adam optimizer (learning rate is 0.0001) with the batch size of 10 000. Since the labels of all gRNA-target dataset were imbalanced, bootstrapping sampling was performed to balance the samples in each batch (that is, 5000 positive samples and 5000 negative samples in one batch) when training the models. Surprisingly, it was found that bootstrapping-sampling batch training improved the regressor's performance while decreased the classifier's performance. Therefore, only bootstrapping-sampling batch training was performed on regression models in the experiment. Besides, all CRISPR-Nets were implemented using Keras 2.2.4 with TensorFlow 1.12.0 backend.[47,48] An NVIDIA Titan XP GPU was adopted for training the models. The gradient boosted regression trees[49] were implemented in Python 3.6 using the sci-kit learn 0.20.0 library. The source code of DeepCRISPR, AttnToMismatch\_CNN, CRISPRoff, CFD, CNN\_std, ensemble SVM, and Elevation were provided by their authors.

# **Results**

In this section, we first reported the performance of CRISPR-Nets and two modified off-target models (i.e., Modified CNN\_std and gradient boosted tree) on CIRCLE-Seq dataset (Dataset I/1) and GUIDE-Seq dataset (Dataset I/2) based on ROC and PR analyses. Second, we demonstrated that CRISPR-Net outperforms six existing off-target prediction models on the dataset with only mismatches off-targets (Dataset II/5). Third, we presented the performance of CRISPR-Nets trained on different datasets and reported the optimal combination (i.e., Dataset I/1, II/1, II/2, II/3, and II/4), we also showed that the CRISPR-Net trained on the optimal combined dataset has excellent generalization on another independent testing dataset (Dataset II/6) comparing to the existing off-target predictors. Fourthly, we demonstrated that CRISPR-Net augmented by four epigenetics features surpassed the state-of-the-art methods, DeepCRISPR on two datasets from HEK293 and K562 cell types respectively. Then, we reported the gRNA off-target summarization performance of CRISPR-NetAggregate and Elevation-aggregate on the GeCKO dataset. Finally, we used a case study to show how our sensitivity analysis visualizes CRISPR-Net prediction for a particular gRNA-target pair.

### **Evaluation of CRISPR-Nets on Indel-and-Mismatch gRNA-Target Prediction**

In this evaluation, four different types of CRISPR-Nets were evaluated alongside two modified off-target prediction models under leave-one-gRNA-out cross-validation on the CIRCLE-Seq dataset (Dataset I/1 in Table 1) comprising ten different guide RNA target sites with both indels and mismatches. In each fold of LOGOCV, the sequence pairs related to the same target site are used as testing data while the remaining sequence pairs are used for model training. As **Figure 6** shows, all of the proposed CRISPR-Nets outperformed two modified models in terms of both AUROC (Area under ROC curve) and AUPRC (Area under PRC curve), with an average AUROC of 0.964 and an average AUPRC of 0.428. The deep-learning-based model, the modified CNN\_std, achieved better performance (AUROC=0.934, AUPRC=0.288) than gradient Boosted trees (AUROC=0.845, AUPRC=0.071). Among the proposed CRISPR-Nets, the CRISPR-Net classifier with the 7 bit encoding scheme achieved the highest AUROC (0.969) and the highest AUPRC (0.477). We found that the performance gap among those models is more pronounced in PRC curve than the ROC curve.

Furthermore, we trained the abovementioned six models on the CIRCLE-Seq dataset and tested them on a GUIDE-Seq dataset (Dataset I/2 in Table 1) which comprises the gRNA-target pairs with both indels and mismatches from six different gRNAs. As shown in **Figure 7**, four CRISPR-Nets and the modified CNN\_std achieved very close performance on ROC curve, while the CRISPR-Net classifier with the seven-bit encoding scheme achieved the highest AUPRC (0.254) among six models with 7.6% improvement of the CRISPR-Net regressor with seven-bit encoding scheme which still achieved the second highest AUPRC (0.178).

In summary, for both ROC and PRC analysis, our proposed deep-learning-based models outperformed the gradient boosted trees significantly on both GUIDE-Seq and CIRCLE-Seq datasets. In ROC analysis, the deep-learning-based models achieved very similar performance with average AUROC of 0.974 on two independent datasets, while in PRC analysis, our proposed CRISPR-Nets outperformed the modified CNN\_std, with the average AUPRC improvement of 14% and 11.3% on CIRCLE-Seq and GUIDE-Seq datasets respectively. Moreover, we found that the seven-bit encoding scheme improved both CIRSRP-Net-Regressor and CRISPR-Net-Classifier, especially on PRC analysis, with the average AUROC improvement of 0.4% and average AUPRC improvement of 4.5% on the two independent datasets. The CRISPR-Net-classifier with 7-bit encoding scheme achieved the best performance with AUPRC of 0.477 and 0.254 on both CIRCLE-Seq and GUIDE-Seq datasets, with a significant improvement of 18.9% and 13.4% compared with the modified CNN\_std. Note that the number of CRISPR-Net parameters with seven-bit encoding scheme (both regressor and classifier) is fewer than the CRISPR-Net with the ten-bit encod-

![](assets/figures/_page_9_Figure_4.jpeg)

**Figure 6.** Comparison of four versions of CRISPR-Net and two modified off-target prediction methods in CIRCLE-Seq dataset (Datasaet I/1 in Table 1) with mean ROC (left) and mean PRC (right) curves.

![](assets/figures/_page_9_Figure_6.jpeg)

**Figure 7.** Comparison of four versions of CRISPR-Net and two modified off-target prediction methods in GUIDE-Seq dataset (Datasaet I/2 in Table 1) with ROC (left) and PRC (right) curves.

ing scheme, which is beneficial for reducing the model training time especially on the large off-target datasets with hundreds of thousands of gRNA-target pairs. Therefore, we used both CRISPR-Net-Classifier and CRISPR-Net-Regressor with 7 bit encoding scheme to compare with the existing off-target prediction models on the mismatch-only datasets in the next experiment.

### **Evaluation of CRISPR-Net on Mismatches-only gRNA-Target Prediction**

In the second experiment, we compared CRISPR-Net-Regressor and CRISPR-Net-Classifier with the six existing off-target prediction methods on independent GUIDE-Seq data (Dataset II/5 in Table 1) which contains the gRNA-target pairs with mismatches from five different gRNAs. As the ROC and PR curves shown in **Figure 8**, two CRISPR-Net models outperformed six existing methods on Dataset II/5 with the average AUROC of 0.987 and AUPRC of 0.294. CRISPR-Net-Classifier achieved the highest AUROC (0.991) and AUPRC (0.323), with a significiant AUPRC improvement of 16% compared to the Elevation-score which achieved the best performance (AUPRC = 0.163) among the exsiting models on PRC analysis. Moreover, we believe that the quantity and quality of training data can affect model performance. Thus, we explored different training datasets to improve CRISPR-Net-classifier in the next experiment.

#### **Evaluation of CRISPR-Nets with Different Training Datasets**

In this scenario, we investigated how two recently published off-targets datasets (in which active off-targets were detected by CIRCLE-Seq and SITE-Seq) along with the training data (ttermed Elevation dataset) in Section 3.2 influence CRISPR-Net's predictive performance. We generated seven

![](assets/figures/_page_10_Figure_4.jpeg)

**Figure 8.** Comparison of CRISPR-Nets and six competing off-target prediction methods in Dataset II/5 with ROC (left) and PRC (right) curves. CRISPR-Nets were trained on Dataset II/1, II/2, and II/4.

![](assets/figures/_page_10_Figure_6.jpeg)

**Figure 9.** Comparison of CRISPR-Nets trained on different dataset on Datasaet II/5 with ROC (left) and PRC (right) curves. The performance of Elevationscore was served as benchmark.

combinations (see **Figure 9**) from those three datasets to train CRISPR-Net and test on the GUIDE-Seq dataset (Dataset II/5), and Elevation-score was served as the benchmark. All CRISPR-Nets showed comparable excellent performance in ROC analysis as shown in Figure 9, each of them reached an AUROC higher than 0.99. While the precision-recall curves in Figure 9 depicted that the PR performance of CRISPR-Nets are wildly different, the model trained on Elevation dataset achieved better PR performace (AUPRC = 0.323) than those trained on SITE-Seq (AUPRC = 0.181) or CIRCLE-Seq dataset (AUPRC = 0.184) alone, since the Elevation dataset contains relatively more diverse gRNAs. The CRISPR-Net trained on three combined datasets (i.e., CIRCLE-Seq, SITE-Seq and Elevation dataset) achieved the highest AUPRC (0.329) with a slight improvement (0.6%) on Elevation dataset (AUPRC = 0.323). Note that all CRISPR-Nets surpassed Elevation-score based on the AUROC and AUPRC analyses.

The CRISPR-Net trained on the Elevation dataset, and the one trained on three combined datasets achieved the most competitive performance among all seven models on Dataset II/5. To further test the generalization of these two well-trained CRISPR-Nets, we compared them with six existing methods on another independent testing dataset (i.e., Dataset II/6), which includes 22 different gRNAs. **Figure 10** depicts that CRISPR-Net (CRISPR-Net\*\*) trained on three combined datasets outperformed the other six existing off-target prediction models, reaching the highest AUROC of 0.995 and the highest AUPRC of 0.317. Remarkably, as shown in the precision-recall curves in Figure 10, CRISPR-Net\*\* had a significant advantage over the second-best method with an 18.3% improvement in AUPRC.

As summarized in **Table 2**, CRISPR-Net trained on SITE-Seq, CIRCLE-Seq and Elevation dataset (i.e., Dataset I/1, II/1, II2, II/3 and II/4) is robust with superior performance on two

![](assets/figures/_page_11_Figure_5.jpeg)

**Figure 10.** Comparison of CRISPR-Nets and six competing off-target prediction methods in Datasaet II/6 with ROC (left) and PRC (right) curves. CRISRP-Net\* was trained on Elevation dataset (i.e., Dataset II/1, II/2 and II/4 in Table 1), and CRISPR-Net\*\* was train on SITE-Seq, CIRCLE-Seq and Elevation datasets.

**Table 2.** Performance comparison between CRISPR-Net and six state-of-the-art machine-learning-based off-target prediction models.

| Dataset                | Metric | Off-target prediction methods |                 |           |              |         |                     |       |  |
|------------------------|--------|-------------------------------|-----------------|-----------|--------------|---------|---------------------|-------|--|
|                        |        | CRISPR-Net                    | Elevation-score | CRISPRoff | Ensemble SVM | CNN_std | AttenToMiamatch_CNN | CFD   |  |
| Dataset II/5           | AUROC  | 0.995                         | 0.979           | 0.973     | 0.982        | 0.956   | 0.961               | 0.925 |  |
| Kleinstiver et al.[44] | AUPRC  | 0.329                         | 0.163           | 0.104     | 0.113        | 0.115   | 0.069               | 0.066 |  |
| Dataset II/6           | AUROC  | 0.995                         | 0.987           | 0.990     | 0.979        | 0.976   | 0.963               | 0.963 |  |
| Listgarten et al.[13]  | AUPRC  | 0.317                         | 0.078           | 0.046     | 0.048        | 0.034   | 0.025               | 0.030 |  |

independent testing data (i.e., Dataset II/5 and II/6), comparing with six existing off-target prediction models. CRISPR-Net, Elevation-score, CRISPRoff, and Ensemble SVM had comparably excellent performance on off-target prediction as measured by AUROC. However, CRISPR-Net significantly surpassed the other models based on AUPRC measurement, with the improvement of 16.6% and 23.9% over the second-best model on Dataset II/5 and II/6 respectively. Our two independent tests demonstrate that CRISPR-Net has great generalization on off-target activity prediction for different gRNAs.

### **Evaluation of CRISPR-Net with Epigenetic Features**

We next investigated off-target predictive performance of CRISPR-Net with four epigenetic features (CTCF, DNase, H3K4me3, and RRBS) through comparing with a modified CNN\_std and DeepCRISPR on two off-target datasets from HEK293T and K562 cell types. As **Figure 11** depicts, CRISPR-Net trained on K562 datasets achieved the highest AUROC of 0.803 and AUPRC of 0.115 among three models on HEK293T off-target dataset. Nevertheless, CRISPR-Net remarkably outperformed the other two models on K562 off-target dataset, reaching AUROC of 0.981 and AUPRC of 0.328 (see **Figure 12**). As a summary, CRISPR-Net with both sequence content and epigenetic features is competitive with superior performance for both ROC and PR analyses comparing to the state-of-the-art deep-learning-based model DeepCRISPR, achiving the average AUROC of 0.892 and AUPRC of 0.221.

### **Evaluation of CRISPR-Net-Aggregate on gRNA Off-Target Scoring**

In the last experiment, we evaluated our CRISPR-Net-Aggregate alongside Elevation-aggregate on the task of assigning genomewide normalized scores to gRNAs. On the GeCKO dataset which comprises 5178 guide RNAs targeting on 875 non-essential genes in A375 melanoma cells, we found that CRISPR-Net-Aggregate outperformed Elevation-aggregate on two replicas and the average in weighted spearman correlation, yielding a significant improvement as depicted in **Figure 13**. Moreover, we used CRISPR-Net-aggregate to summarise the offtarget scores of gRNA-target with the only mismatch predicted by CRISPR-Net. As Figure 6 shows, the mismatchonly CRISPR-Net-aggregate also achieved better performance than Elevation-score. In this case, CRISPR-Net-Aggregate and Elevation-aggregate used the same gRNA-target pairs to assess overall off-target effects of the gRNAs in the GeCKO dataset. The observations indicate that CRISPR-Net-Aggregate has a distinct advantage over Elevation-aggregate in assessing the offtarget activities of different gRNAs. Moreover, we found that CRISPR-Net-Aggregate using both indel and mismatch gRNAtargets pairs as input achieved better performance than that using mismatch gRNA-target pairs only, which indicates the indels

![](assets/figures/_page_12_Figure_5.jpeg)

**Figure 11.** Comparison of CRISPR-Net, DeepCRISPR and CNN\_std on HEK293T off-target dataset with ROC (left) and PRC (right) curves.

![](assets/figures/_page_12_Figure_7.jpeg)

**Figure 12.** Comparison of CRISPR-Net, DeepCRISPR and CNN\_std on K562 off-target dataset with ROC (left) and mean PRC (right) curves.

are relevant and should not be ignored in the off-target problem indeed.

### **Visualization of CRISPR-Net on the gRNA-target Prediction**

In this experiment, we employed sensitivity analysis on CRISPR-Net-Classifier with seven-bit encoding scheme which has achieved the best performance in the above two experiments, to uncover how it predicts the off-target activity given a candidate gRNA-target pair. In the experiment of leave-one-gRNA-out cross-validation on CIRCLE-Seq dataset, CRISPR-Net achieved an average true positive rate of 0.9 with an average false positive rate of 0.1 at the threshold of 0.003. Thus, we consider that every gRNA-target with CRISPR-Net predictive score higher than 0.003 is an active off-target.

Herein, we used this CRISPR-Net to assess an active off-target site (GGCACTGCTGCTAGAGGTGCAGG) located on chromosome 1 from 201 067 360 to 201 067 383 with 598 nucleasecleaved sequencing reads detected by CIRCLE-Seq, of which gRNA is designed to target Site 4 (GGCACTGCGGCTGGAG- GTGGAGG). We note that all gRNA-target pairs from Site 4 were not used for training in this case. There are three mismatch sites in this gRNA-target pair comprising "G–T:9" (denote the mismatch site "G–T" at position 9), "A–G:13", and "C–G:20". The predictive off-target score of this gRNA-target pair is 0.113 surpassing the threshold value.

The diagram in **Figure 14**a demonstrates the occluded-based sensitivity scores of each site in the gRNA-target. Each base pair of the gRNA-target is iteratively occluded with zero-valued vector; the sensitivity score of each site represents the corresponding change of CRISPR-Net's output by occluding. The line chart in Figure 14a indicates the sensitivity score of each site. Moreover, we normalized the absolute value of each sensitivity score and mapped each back to the respective site occluded in the gRNAtarget pair to generate a saliency heat map (in Figure 14a).

As the diagram shows, one negative and two positive peaks occur at the match site at the fifth position and two mismatch sites at positions 13 and 20, respectively. The heapmap shows that the mismatch site "G–A:13" has the highest heat value, which means it has the most significant impact on the CRISPR-Net prediction

![](assets/figures/_page_13_Figure_4.jpeg)

**Figure 13.** Comparison of CRISPR-Net-Aggregate and Elevation-aggregate on GeCKO dataset with weighted Spearman correlation.

among all sites of the gRNA-target. The significant-high sensitivity score at location 13 indicates that the off-target activity of this site will increase remarkably if the mismatch in location 13 (7 bp from the PAM ) is removed. This phenomenon is expected since CRISPR-Cas9 depends explicitly on the gRNA seed sequence of the PAM-proximal region.[50] Therefore, when we occluded the mismatch at location 13, the CRISPR-Net prediction results in a high positive sensitivity score. In contrast, the mismatch at location 9 (11 bp from the PAM) had a small sensitivity score because its location is distal to the PAM and at the blurred edge of the seed region. Interestingly, we found that the mismatch site "G–T:9" has little effect on the predictive score that a match site "C–C:5" obtained a higher heat value than "G–T:9". Moreover, we found that some match sites near the mismatch sites with positive peak have high heat value (see the heat map in Figure 14a). The observations imply that some match sites of gRNA-target contribute to the off-target activity, which explained the enhanced performance of our CRISPR-Net over Elevation-score (only consider the mismatch sites) from another side.

The bar charts in Figure 14b–e show the replacement-based sensitivity analysis results, each of which demonstrates the change of CRISPR-Net predictive score caused by replacing each site iteratively with one of the match sites (A–A, T–T, G–G, and C–C). We then aggregated the sensitivity scores from those four bar charts into a heatmap for clear demonstration. As the four bar charts show, the mismatch site "A–G:13", replaced by "A– A" leads to the highest increase in the CRISPR-Net predictive score, and the match site "G–G:16" replaced by "C–C" causes the highest decrease. Furthermore, we found some interesting relationships between occluded-based analysis and the replacementbased analysis comparing Figure 14a,f: the mismatch site "A– G:13", which incurs a positive peak in occluded-based analysis, in the replacement-based analysis, obtains improvement no matter what kind of the match site it is changed into; the match site "C–C:5" revealed the exact opposite trend. This situation also appears on another peak site, "G–C:20". However, for another mismatch site "T–G:6" whose sensitivity score is nearly zero in occluded-based analysis, the respective predictive score is increased when this site was replaced by "A–A," while we replaced it with "G–G," the respective predictive score decreased. These observations imply that occluded-based analysis can be complemented by replacement-based analysis. The replacement-based analysis uncovers how base-pair substitution in gRNA-target influences the predictive off-target activity, which should be helpful to gRNA optimization.

# **Discussion and Conclusions**

In this study, we propose CRISPR-Net, a recurrent convolutional network that quantifies the off-target activities of gRNA-target pairs with indels and mismatches. We also present CRISPR-Net-Aggregate, which aggregates the predictive gRNA-target scores into a single off-target summary score to assess each gRNA. Since the exact mechanism of off-target activity remained unclear, hand-crafted features for modeling may lead to information loss. We designed a new encoding scheme that converts each gRNA-target sequence pair into a binary matrix as the input for deep learning. Since there is not any published method that can assess off-target activities with indels and mismatches, we modified the state-of-the-arts mismatches-only off-target prediction models for performance comparison with CRISPR-Net. In our

![](assets/figures/_page_14_Figure_4.jpeg)

**Figure 14.** An example of two-stage sensitivity analysis on CRISPR-Net prediction. The *x*-axis shows the sites of the gRNA-target pair (PAM is on the right), while the *y*-axis is the change of CRISPR-Net prediction (Sensitivity score) caused by site perturbation of gRNA-target. The diagram in figure (a) demonstrates the occluded-based sensitivity scores of each site in the gRNA-target, and the heap map in figure (a) shows the absolute sensitivity scores which imply the site impact on the CRISPR-Net prediction. Note that the zero sensitivity score is the red dashed line. The bar charts in figure (b) to (e) show the replacement-based sensitivity analysis results, each of which demonstrates the change in CRISPR-Net predictive score caused by replacing each site iteratively with one of the match sites (A–A, T–T, G–G, and C–C). These sensitivity scores from four bar charts are aggregated into a heat map in figure (f).

![](assets/pictures/_page_15_Picture_3.jpeg)

experiments, CRISPR-Net not only achieved competitive performance on both CIRCLE-Seq and GUIDE-Seq datasets, outperforming two modified off-target scoring models in ROC and PRC analysis but also surpassed six available competing off-target scoring models on two independent GUIDE-Seq datasets. CRISPR-Net with four epigenetic features also outperformed DeepCRISPR on off-target testing datasets form two cell types. Furthermore, CRISPR-Net-Aggregate outperformed significantly the state-of-the-art model, Elevation-aggregate, in weighted Spearman correlation for the task of summary scoring on the GeCKO dataset with 5178 guide RNAs.

The main contribution of our study is to present CRISPR-Net as the first-of-its-kind model that quantifies the off-target activities with mismatches and indels between target DNA and guide RNA sequences. Our experiments demonstrated the utility of a hybrid convolutional recurrent neural network for the off-target problem. Such hybrid architecture of CRISPR-Net can learn the essential features and the sequential combination between these features simultaneously. In a previous study, Kim et al. used convolutional neural network to predict the efficiency of guide RNA and achieved success.[51] Since the optimal convolutional kernel size for gRNA off-target activity prediction remains unknown, we employed the Inception module (i.e., four convolutional kernels of different sizes) in the convolutional layer to capture features in different ranges. In addition, we designed and employed a twostage sensitivity analysis for visualizing the implicit knowledge encoded in CRISPR-Net prediction, in order to unveil how each site of gRNA-target pair contributes to the predictive off-target activity. Our visualization provides novel insights and platforms for guide RNA optimization with broad impacts.

Together, CRISPR-Net and CRISPR-Net-Aggregate provide an integrated in silico tool to quantify the off-target activities of CRISPR guide RNAs. As additional off-target assays and the corresponding datasets become available, both CRISPR-Net and CRISPR-Net-Aggregate are expected to be enhanced. We will also carefully investigate the flanking genomic sequences for modeling off-target activities in the future. We believe that such an intelligent tool can contribute to the CRISPR-Cas9 guide RNA design in a rigorous manner.

# **Acknowledgements**

The authors would like to thank Prof. Luca Pinello and his colleagues in Massachusetts General Hospital and Harvard Medical School for their valuable discussions on computational off-target dectection. The authors also thank the two anonymous reviewers for their constructive comments. The work described in this paper was substantially supported by two grants from the Research Grants Council of the Hong Kong Special Administrative Region [CityU 11203217] and [CityU 11200218], one grant from the Health and Medical Research Fund, the Food and Health Bureau, The Government of the Hong Kong Special Administrative Region [07181426], and the funding from Hong Kong Institute for Data Science (HKIDS) at City University of Hong Kong. The work described in this paper was partially supported by a grant from City University of Hong Kong (CityU 11202219).

# **Conflict of Interest**

The authors declare no conflict of interest.

# **Keywords**

CRISPR systems, deep learning, off-target activities

Received: December 10, 2019 Revised: April 23, 2020 Published online: May 20, 2020

- [1] L. Cong, F. A. Ran, D. Cox, S. Lin, R. Barretto, N. Habib, P. D. Hsu, X. Wu, W. Jiang, L. A. Marraffini, F. Zhang, *Science* **2013**, *339*, 819.
- [2] M. A. Moreno-Mateos, J. P. Fernandez, R. Rouet, C. E. Vejnar, M. A. Lane, E. Mis, M. K. Khokha, J. A. Doudna, A. J. Giraldez, *Nat. Commun.* **2017**, *8*, 2024.
- [3] J. F. Hultquist, J. Hiatt, K. Schumann, M. J. McGregor, T. L. Roth, P. Haas, J. A. Doudna, A. Marson, N. J. Krogan, *Nat. Protoc.* **2018**, 1.
- [4] F. A. Ran, P. D. Hsu, C. Y. Lin, J. S. Gootenberg, S. Konermann, A. E. Trevino, D. A. Scott, A. Inoue, S. Matoba, Y. Zhang, F. Zhang, *Cell* **2013**, *154*, 1380.
- [5] J. A. Doudna, E. Charpentier, *Science* **2014**, *346*, 1258096.
- [6] H. Ma, N. Marti-Gutierrez, S. W. Park, J. Wu, Y. Lee, K. Suzuki, A. Koski, D. Ji, T. Hayama, R. Ahmed, H. Darby, C. Van Dyken, Y. Li, E. Kang, A. R. Park, D. Kim, S.-T. Kim, J. Gong, Y. Gu, X. Xu, D. Battaglia, S. A. Krieg, D. M. Lee, D. H. Wu, D. P. Wolf, S. B. Heitner, J. C. I. Belmonte, P. Amato, J.-S. Kim, S. Kaul, et al., *Nature* **2017**, *548*, 413.
- [7] D. B. T. Cox, R. J. Platt, F. Zhang, *Nat. Med.* **2015**, *21*, 121.
- [8] R. S. Shapiro, A. Chavez, C. B. Porter, M. Hamblin, C. S. Kaas, J. E. DiCarlo, G. Zeng, X. Xu, A. V. Revtovich, N. V. Kirienko, Y. Wang, G. M. Church, J. J. Collins, *Nat. Microbiol.* **2018**, *3*, 73.
- [9] N. J. Kramer, M. S. Haney, D. W. Morgens, A. Joviˇci´c, J. Couthouis, A. Li, J. Ousey, R. Ma, G. Bieri, C. K. Tsui, Y. Shi, N. T. Hertz, M. Tessier-Lavigne, J. K. Ichida, M. C. Bassik, A. D. Gitler, *Nat. Genet.* **2018**, *50*, 603.
- [10] Y. Fu, J. A. Foden, C. Khayter, M. L. Maeder, D. Reyon, J. K. Joung, J. D. Sander, *Nat. Biotechnol.* **2013**, *31*, 822.
- [11] J. G. Doench, N. Fusi, M. Sullender, M. Hegde, E. W. Vaimberg, K. F. Donovan, I. Smith, Z. Tothova, C. Wilen, R. Orchard, H. W. Virgin, J. Listgarten, D. E. Root, *Nat. Biotechnol.* **2016**, *34*, 184.
- [12] P. D. Hsu, D. A. Scott, J. A. Weinstein, F. A. Ran, S. Konermann, V. Agarwala, Y. Li, E. J. Fine, X. Wu, O. Shalem, T. J. Cradick, L. A. Marraffini, G. Bao, F. Zhang, *Nat. Biotechnol.* **2013**, *31*, 827.
- [13] J. Listgarten, M. Weinstein, B. P. Kleinstiver, A. A. Sousa, J. K. Joung, J. Crawford, K. Gao, L. Hoang, M. Elibol, J. G. Doench, N. Fusi, *Nat. Biomed. Eng.* **2018**, *2*, 38.
- [14] S. Zhang, X. Li, Q. Lin, K. C. Wong, *Bioinformatics* **2018**.
- [15] Y. Lin, T. J. Cradick, M. T. Brown, H. Deshmukh, P. Ranjan, N. Sarode, B. M. Wile, P. M. Vertino, F. J. Stewart, G. Bao, *Nucleic Acids Res.* **2014**, *42*, 7473.
- [16] S. Q. Tsai, Z. Zheng, N. T. Nguyen, M. Liebers, V. V. Topkar, V. Thapar, N. Wyvekens, C. Khayter, A. J. Iafrate, L. P. Le, M. J. Aryee, J. K. Joung, *Nat. Biotechnol.* **2015**, *33*, 187.
- [17] D. Kim, S. Bae, J. Park, E. Kim, S. Kim, H. R. Yu, J. Hwang, J. I. Kim, J. S. Kim, *Nat. Methods* **2015**, *12*, 237.
- [18] P. Cameron, C. K. Fuller, P. D. Donohoue, B. N. Jones, M. S. Thompson, M. M. Carter, S. Gradia, B. Vidal, E. Garner, E. M. Slorach, E. Lau, L. M. Banh, A. M. Lied, L. S. Edwards, A. H. Settle, D. Capurso, V. Llaca, S. Deschamps, M. Cigan, J. K. Young, A. P. May, *Nat. Methods* **2017**, *14*, 600.
- [19] S. Q. Tsai, N. T. Nguyen, J. Malagon-Lopez, V. V. Topkar, M. J. Aryee, J. K. Joung, *Nat. Methods* **2017**, *14*, 607.
- [20] R. L. Frock, J. Hu, R. M. Meyers, Y. J. Ho, E. Kii, F. W. Alt, *Nat. Biotechnol.* **2015**, *33*, 179.

![](assets/pictures/_page_16_Picture_3.jpeg)

- [21] W. X. Yan, R. Mirzazadeh, S. Garnerone, D. Scott, M. W. Schneider, T. Kallas, J. Custodio, E. Wernersson, Y. Li, L. Gao, Y. Federova, B. Zetsche, F. Zhang, M. Bienko, N. Crosetto, *Nat. Commun.* **2017**, *8*, 15058.
- [22] M. Stemmer, T. Thumberger, M. del Sol Keyer, J. Wittbrodt, J. L. Mateo, *PloS one* **2015**, *10*, e0124633.
- [23] R. Singh, C. Kuscu, A. Quinlan, Y. Qi, M. Adli, *Nucleic Acids Res.* **2015**, *43*, e118.
- [24] H. Peng, Y. Zheng, Z. Zhao, T. Liu, J. Li, *Bioinformatics* **2018**, *34*, i757.
- [25] J. Lin, K. C. Wong, *Bioinformatics* **2018**, *34*, i656.
- [26] K. R. Anderson, M. Haeussler, C. Watanabe, V. Janakiraman, J. Lund, Z. Modrusan, J. Stinson, Q. Bei, A. Buechler, C. Yu, S. R. Thamminana, L. Tam, M. A. Sowick, T. Alcantar, N. O'Neil, J. Li, L. Ta, L. Lima, M. Roose-Girma, X. Rairdan, S. Durinck, S. Warming, *Nat. Methods* **2018**, *15*, 512.
- [27] J. Donahue, L. A. Hendricks, S. Guadarrama, M. Rohrbach, S. Venugopalan, K. Saenko, T. Darrell, in: *Proceedings of the IEEE conference on computer vision and pattern recognition*, **2015**, pp. 2625–2634.
- [28] S. Venugopalan, H. Xu, J. Donahue, M. Rohrbach, R. Mooney, K. Saenko, in: *Proceedings of the 2015 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, **2015**, pp. 1494–1504.
- [29] A. Telenti, C. Lippert, P. C. Chang, M. DePristo, *Hum. Mol. Genet.* **2018**, *27*, R63.
- [30] D. Tang, B. Qin, T. Liu, in: *Proceedings of the 2015 conference on empirical methods in natural language processing*, **2015**, pp. 1422–1432.
- [31] G. Trigeorgis, F. Ringeval, R. Brueckner, E. Marchi, M. A. Nicolaou, B. Schuller, S. Zafeiriou, in: *Acoustics, Speech and Signal Processing (ICASSP), 2016 IEEE International Conference on*, **2016**, pp. 5200– 5204.
- [32] D. Quang, X. Xie, *Nucleic Acids Res.* **2016**, *44*, e107.
- [33] J. Lanchantin, R. Singh, B. Wang, Y. Qi, in: *PACIFIC SYMPOSIUM ON BIOCOMPUTING 2017*, **2017**, pp. 254–265.
- [34] J. J. Almagro Armenteros, C. K. Sønderby, S. K. Sønderby, H. Nielsen, O. Winther, *Bioinformatics* **2017**, *33*, 3387.

- [35] S. T. Hill, R. Kuintzle, A. Teegarden, E. Merrill III, P. Danaee, D. A. Hendrix, *Nucleic Acids Res.* **2018**, *46*, 8105.
- [36] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, A. Rabinovich, in: *Proceedings of the IEEE conference on computer vision and pattern recognition*, **2015**, pp. 1–9.
- [37] F. A. Gers, J. Schmidhuber, F. Cummins, **1999**.
- [38] A. Graves, J. Schmidhuber, *Neural Networks* **2005**, *18*, 602.
- [39] A. Yates, W. Akanni, M. R. Amode, D. Barrell, K. Billis, D. Carvalho-Silva, C. Cummins, P. Clapham, S. Fitzgerald, L. Gil, J. Lin, Z. Zhang, S. Zhang, J. Chen, K.-C. Wong, *Nucleic Acids Res.* **2015**, *44*, D710.
- [40] M. Haeussler, K. Schönig, H. Eckert, A. Eschstruth, J. Mianné, J. B. Renaud, S. Schneider-Maunoury, A. Shkumatava, L. Teboul, J. Kent, J. S. Joly, J. P. Concordet, *Genome Biol.* **2016**, *17*, 148.
- [41] G. Chuai, H. Ma, J. Yan, M. Chen, N. Hong, D. Xue, C. Zhou, C. Zhu, K. Chen, B. Duan, F. Gu, S. Qu, D. Huang, J. Wei, Q. Liu, *Genome Biol.* **2018**, *19*, 80.
- [42] Q. Liu, L. X. Di He, *PLoS Comput. Biol.* **2019**, *15*.
- [43] F. Alkan, A. Wenzel, C. Anthon, J. H. Havgaard, J. Gorodkin, *Genome Biol.* **2018**, *19*, 1.
- [44] B. P. Kleinstiver, V. Pattanayak, M. S. Prew, S. Q. Tsai, N. Nguyen, Z. Zheng, J. K. Joung, *Nature* **2015**.
- [45] S. Bae, J. Park, J. S. Kim, *Bioinformatics* **2014**, *30*, 1473.
- [46] T. Hart, K. R. Brown, F. Sircoulomb, R. Rottapel, J. Moffat, *Mol. Syst. Biol.* **2014**, *10*.
- [47] M. Abadi, P. Barham, J. Chen, Z. Chen, A. Davis, J. Dean, M. Devin, S. Ghemawat, G. Irving, M. Isard, M. Kudlur, J. Levenberg, R. Monga, S. Moore, D. G. Murray, B. Steiner, P. Tucker, V. Vasudevan, P. Warden, M. Wicke, Y. Yu, X. Zheng, in: *OSDI*, **2016**, pp. 265–283.
- [48] F. Chollet, J. Lin, Z. Zhang, S. Zhang, J. Chen, K.-C. Wong, Astrophysics Source Code Library **2018**.
- [49] J. H. Friedman, *Annals of statistics*, **2001**, pp. 1189–1232.
- [50] H. Manghwar, B. Li, X. Ding, A. Hussain, K. Lindsey, X. Zhang, S. Jin, *Adv. Sci.* **2020**, 1902312.
- [51] H. K. Kim, S. Min, M. Song, S. Jung, J. W. Choi, Y. Kim, S. Lee, S. Yoon, H. H. Kim, *Nat. Biotechnol.* **2018**, *36*, 239.