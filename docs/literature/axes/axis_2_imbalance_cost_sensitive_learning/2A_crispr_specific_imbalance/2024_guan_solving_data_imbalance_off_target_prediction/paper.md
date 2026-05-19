![](assets/pictures/_page_0_Picture_1.jpeg)

Contents lists available at [ScienceDirect](www.sciencedirect.com/science/journal/00104825)

# Computers in Biology and Medicine

journal homepage: [www.elsevier.com/locate/compbiomed](https://www.elsevier.com/locate/compbiomed)

![](assets/pictures/_page_0_Picture_5.jpeg)

![](assets/pictures/_page_0_Picture_6.jpeg)

# A systematic method for solving data imbalance in CRISPR off-target prediction tasks

Zengrui Guan , Zhenran Jiang \*

*School of Computer Science and Technology, East China Normal University, Shanghai, 200062, China* 

ARTICLE INFO

*Keywords:*  CRISPR/Cas9 system Data imbalance Off-target prediction

#### ABSTRACT

Accurately identifying potential off-target sites in the CRISPR/Cas9 system is crucial for improving the efficiency and safety of editing. However, the imbalance of available off-target datasets has posed a major obstacle in enhancing prediction performance. Despite several prediction models have been developed to address this issue, there remains a lack of systematic research on handling data imbalance in off-target prediction. This article systematically investigates the data imbalance issue in off-target datasets and explores numerous methods to process data imbalance from a novel perspective. First, we highlight the impact of the imbalance problem on offtarget prediction tasks by determining the imbalance ratios present in these datasets. Then, we provide a comprehensive review of various sampling techniques and cost-sensitive methods to mitigate class imbalance in off-target datasets. Finally, systematic experiments are conducted on several state-of-the-art prediction models to illustrate the impact of applying data imbalance solutions. The results show that class imbalance processing methods significantly improve the off-target prediction capabilities of the models across multiple testing datasets. The code and datasets used in this study are available at [https://github.com/gzrgzx/CRISPR\\_Data\\_Im](https://github.com/gzrgzx/CRISPR_Data_Imbalance)  [balance](https://github.com/gzrgzx/CRISPR_Data_Imbalance).

## 1. Introduction

The CRISPR/Cas9 system, as the third-generation gene-editing technology, has been widely used in the biomedical field due to its specificity and ease of use [\[1,2](#page-12-0)]. However, a major challenge in its practical implementation is the occurrence of off-target effects associated with the CRISPR/Cas9 system [[3](#page-12-0),[4](#page-12-0)]. Off-target events can lead to detrimental mutations, which may cause damage or cell death [[5](#page-12-0),[6](#page-12-0)]. Therefore, accurately identifying potential off-target sites is crucial to improve the efficiency of gene editing [7–[9\]](#page-12-0).

During the past years, several experimental methods have been developed to detect off-target effects. These include Digenome-Seq [\[10](#page-12-0)], GUIDE-seq [11], SITE-Seq [12], CIRCLE-Seq [\[13](#page-12-0)], HTGTS [\[14](#page-12-0)], BLISS [15], and CHANGE-Seq [\[16](#page-12-0)]. Although these experimental methods can unbiasedly identify off-target sites throughout the entire genome, they pose challenges in terms of operational complexity and high cost. In contrast, using computational methods to predict potential off-target sites in the CRISPR/Cas9 system can significantly enhance the efficiency of gene editing.

Early computational methods mainly depended on mathematical

statistics. For example, CROP-IT evaluates off-target effects by calculating penalty scores for adjacent mismatches and heuristic scores based on mismatch positions [\[17](#page-12-0)]. MIT scores identify potential off-targets by calculating mismatch positions in the gRNA-DNA complex, considering only the location and quantity of guide RNA (gRNA) mismatch positions [18]. CCTop scores detect off-targets by assessing whether mismatches are close to the Protospacer Adjacent Motif (PAM) region where mismatches near PAM leads to weakened off-target effects [19]. Recent advances in machine learning have produced many methods that significantly improve the accuracy of off-target prediction. CFD is a simple Naive Bayes method that identifies off-targets based on mismatch positions and types in the gRNA-DNA sequence [20]. Abadi et al. [\[21](#page-12-0)] developed CRISTA, which uses a random forest regression model to predict the probability of off-target. Elevation, proposed by Listgarten et al. [\[22](#page-12-0)], predicts off-target activity using a machine learning score aggregation model. Chuai et al. [\[23](#page-12-0)] introduced the DeepCRISPR model, which integrates on-target and off-target prediction of sgRNA into a deep learning framework. Liu et al. [\[24](#page-12-0)] proposed AttnTo-Mismatch\_CNN, a model that combines transformer modules with Convolutional Neural Networks (CNN) to predict off-target sites. The

*E-mail address:* [zrjiang@cs.ecnu.edu.cn](mailto:zrjiang@cs.ecnu.edu.cn) (Z. Jiang).

$^{\*}$ Corresponding author.

CRISPR\_Net model by Lin et al. [\[25](#page-12-0)] is based on a Long Short-Term Recurrent Convolutional Neural Network (LRCN), where the convolutional layer acts as a feature extractor and the recurrent layer recognizes sequence patterns. Sun et al. [26] developed CRISPR-M with a new multi-view deep learning model to predict the sgRNA off-target effects for target sites containing indels and mismatches.

However, the techniques for detecting off-targets throughout the entire genome typically identify fewer authentic off-target sites than the total number of possible nucleotide mismatch positions. This extreme data imbalance makes training traditional machine learning models challenging, as highly imbalanced data may introduce bias, leading to the model quickly achieving high accuracy on majority samples. However, the model may often perform poorly in minority samples, which are truly of interest.

To solve the problem of data imbalance in model training, Chuai et al. [\[23](#page-12-0)] and Liu et al. [24] employed oversampling to reduce the bias introduced by data imbalance, constructing balanced mini-batches by repeatedly sampling minority class samples with majority class samples during each training iteration. Zhang et al. [27] conducted experiments by randomly undersampling majority class samples to eliminate the effect of data imbalance on the model. However, both random oversampling and undersampling have significant drawbacks. For example, oversampling repeats minority class samples, potentially amplifying noise and causing model overfitting. Undersampling discards most of the majority class samples, which leads to data waste and may introduce significant deviation in the model.

In addition, Zhang et al. [28] used data augmentation in their experiments by rotating the matrix formed by gRNA-DNA sequence pairs. Although this method increases the number of off-target samples by four times, it also has two potential problems. Firstly, the new matrices obtained by rotation may be meaningless and may not effectively represent the information about gRNA-DNA sequence pairs. Secondly, the dataset is extremely imbalanced, with an imbalance ratio of 1:250. Even if the minority samples are quadrupled, the imbalance problem still exists. Gao et al. [\[29](#page-12-0)] also emphasized addressing data imbalance in experiments. They suggested using SMOTE (Synthetic Minority Over-sampling Technique), a method for generating synthetic samples, to effectively alleviate data imbalance. However, they did not propose a systematic solution to the problem of data imbalance in the CRISPR off-target prediction tasks. In this study, our contributions are as follows:

- (1) This paper presents an overview of data imbalance-solving methods and highlights the effect of class imbalance by analyzing the imbalance ratio in available off-target datasets.
- (2) The effect of various methods to address data imbalance, including both data-level and cost-sensitive methods, is systematically evaluated. The feasibility of these imbalance methods in the off-target prediction is further analyzed.
- (3) Systematic experiments have been conducted on multiple offtarget prediction models, demonstrating that most data imbalance methods can improve off-target prediction performance without altering the underlying models. The experimental results indicate that Focal loss, as a cost-sensitive function-based method, can achieve better performance in solving data imbalance problem.

## 2. Materials and methods

### 2.1. Datasets

To perform the experiments systematically, we collected two sets of CRISPR off-target datasets designated for model training and testing, respectively. Further, we curated them based on the context of related studies. The off-target datasets usually consist of three parts: sgRNA, OtDNA, and label. sgRNA indicates the 23bp guide RNA sequence, OtDNA indicates the 23bp target DNA sequence, the label indicates whether gRNA and DNA occur off-target,1 indicates off-target, and 0 means no off-target.

As shown in Table 1, we divided the eight datasets into two groups: the first group was used to train the models, while the second group was used to evaluate the generalization ability of the model. This dataset division method was established to maintain consistency with the approach used by Lin et al. [30]. It is worth noting that CRISPR\_Net was trained on the first group of datasets. Dataset I/1, obtained by Doench et al. [20], comes from protein knockout experiments on the human coding sequence CD33. Dataset I/2, from Haeussler et al. [31], contains 52 validated off-targets from 19 gRNAs. Dataset I/3 is derived from Cameron et al. [\[12](#page-12-0)], who identified 3767 validated off-targets through SITE-Seq. Datasets I/4, II/1, and II/2 contain validated off-targets obtained by GUIDE-seq technology from different studies. Validation off-targets in Datasets II/3 and II/4 were obtained by combining multiple detection methods (such as GUIDE-seq, Digenome-Seq, HTGTS, BLESS, and IDLVs) from two different cell types (293-related cell lines with 18 sgRNAs and K562 with 12 sgRNAs). Datasets I/1, I/2, I/3, I/4, II/1, and II/2 were contributed by Lin et al. [[30](#page-12-0)], while datasets II/3 and II/4 were contributed by Chuai et al. [23].

As shown in Table 1, we calculated the imbalance ratio in these datasets. The calculation of the imbalance ratio is shown in formula 1:

$$Imbalance\ ratio = \frac{Validated\ Off-targets}{Total} \tag{1}$$

where "Total" is the total number of samples in the dataset, and "Validated Off-targets" represents the quantity of experimentally validated off-targets. The introduction of the imbalance ratio aims to highlight the extreme imbalance in the off-target datasets. We observed that the imbalance rate of all data sets except data set I/1 was greater than 1:50, and most data sets exceeded 1:200. It is worth noting that the imbalance ratios of datasets II/1 and II/2 are the most severe, reaching 1:6847 and 1:1774, respectively. This makes it difficult for the model to predict a small proportion of off-target samples. To identify more potential offtarget sites, we need to develop effective methods to solve the imbalance in the available off-target datasets.

### 2.2. The selection of off-target prediction models

To comprehensively compare the performance of different methods of handling data imbalance, we select the models that provide publicly available source code and show advanced performance in their papers.

**Table 1**  The details of the datasets used in this study.

| Type/<br>No. | Technique                                                   | Validated<br>Off-targets | Total   | Imbalance<br>ratio | Guide<br>RNAs | Refs |
|--------------|-------------------------------------------------------------|--------------------------|---------|--------------------|---------------|------|
| I/1          | Protein<br>knockout<br>detection                            | 2273                     | 4853    | 1:2.14             | 65            | [20] |
| I/2          | PCR,<br>Digenome<br>Seq and<br>HTGTS                        | 52                       | 10,129  | 1:194.79           | 19            | [31] |
| I/3          | SITE-Seq                                                    | 3767                     | 217,733 | 1:57.80            | 9             | [12] |
| I/4          | GUIDE-seq                                                   | 354                      | 294,534 | 1:832.02           | 9             | [11] |
| II/1         | GUIDE-seq                                                   | 56                       | 383,463 | 1:6847.55          | 22            | [22] |
| II/2         | GUIDE-seq                                                   | 54                       | 95,829  | 1:1774.61          | 5             | [32] |
| II/3         | GUIDE-seq,<br>Digenome<br>Seq, HTGTS,<br>BLESS and<br>IDLVs | 536                      | 132,914 | 1:247.97           | 18            | [23] |
| II/4         | GUIDE-seq,<br>Digenome<br>Seq, HTGTS,<br>BLESS and<br>IDLVs | 120                      | 20,319  | 1:169.33           | 12            | [23] |

The selected models include CRISPR\_Net, CNN\_std, CrisprDNT, CRISP-R\_IP, CnnCrispr and DL-CRISPR.

- (1) CRISPR\_Net: CRISPR\_Net is constructed using a Long Short-Term Recurrent Convolutional Neural Network (LRCN) [25]. The convolutional layers in LRCN serve as feature extractors, while the recurrent layers identify sequence patterns. CRISPR\_Net has achieved excellent performance on numerous datasets.
- (2) CNN\_std: CNN\_std is one of the earliest models used for CRISPR off-target prediction [30]. The model consists of multi-layer CNNs and linear layers, which utilize CNN to extract information from gRNA-DNA sequence pairs thoroughly.
- (3) CRISPR\_IP: The CRISPR\_IP model is the first to simultaneously use CNN, LSTM, and Attention modules [\[27](#page-12-0)]. CRISPR\_IP demonstrates competitive performance of off-target prediction by introducing a novel encoding method.
- (4) CrisprDNT: CrisprDNT is the latest off-target prediction model with excellent performance [33]. CrisprDNT combines CNN, LSTM, and Transformer architectures and proposes a new anti-noise model to reduce the impact of data noise.
- (5) CnnCrispr: The CnnCrispr [\[34](#page-12-0)] is the first to use an unsupervised GloVe model to train embedding vectors and embed sequence information into a new input matrix, and uses LSTM and CNN to extract features to predict off-target.
- (6) DL-CRISPR: DL-CRISPR [28] uses CNN to predict off-target, but for the first time, DL-CRISPR flips the matrix composed of gRNA-DNA to achieve data enhancement.

### 2.3. Methods of resolving data imbalance

In recent years, various machine learning methods have been used to address the imbalance problem [35]. In general, these learning methods typically solve the issue from two perspectives: modifying the training data to reduce imbalance and adjusting the model's underlying learning or decision processes to increase sensitivity to minority groups [[36,37](#page-12-0)]. Fig. 1 illustrates a schematic diagram for class imbalance handling in off-target prediction.

Method1 involves pre-processing the training dataset before model training to modify its distribution and transform the imbalanced dataset into a more balanced one. This category includes three data processing methods: oversampling, undersampling, and hybrid-sampling. Then, the

![](assets/figures/_page_2_Figure_12.jpeg)

**Fig. 1.** The flowchart of data imbalance resolution methods in off-target prediction. The methods are divided into two types: data-level methods (Method1) and cost-sensitive methods (Method2).

balanced datasets obtained through different sampling methods are fed into the off-target prediction model for training, effectively reducing the potential bias that may occur when training the model on imbalanced datasets.

Method2, known as cost-sensitive methods, is typically used during model training. Unlike data sampling methods, cost-sensitive methods do not alter the original distribution of the training data. Instead, adjustments are made during the model's learning or decision processes to increase the importance of minority classes. In cost-sensitive learning, the assignment of penalties through a cost matrix for each class increases the cost of misclassifying minority class samples, thereby reducing the likelihood of learning model making errors on these key samples.

As shown in Fig. 1, this article provides a comprehensive overview of class imbalance solutions, including both data-level and cost-sensitive methods.

#### 2.3.1. Data-level methods

Oversampling is a widely used technique to increase the number of data points of a minority class via synthetic generation. Specifically, as shown in [Fig. 2](#page-3-0), oversampling includes random oversampling, SMOTE, ADASYN (Adaptive Synthetic), KMeansSMOTE, and SVMSMOTE. Undersampling includes methods such as random undersampling, ENN (Edited Nearest Neighbors), NearMiss, NearMiss\_2, and TomekLinks. Hybrid sampling includes two methods: SMOTETomek and SMOTEENN.

Random oversampling is the most common oversampling technique. This technique involves randomly duplicating minority class samples to increase the size of the dataset. Specifically, by randomly selecting minority class samples and integrating them into the original dataset, a balanced dataset is achieved with an equal distribution of positive and negative class samples. Typical applications of this method can be observed in DeepCRISPR [\[23](#page-12-0)] and CnnCrisp [34]. However, it must be admitted that repetitive sampling of identical instances may lead to significant overfitting.

To address this concern, an advanced oversampling technique known as SMOTE has been developed [\[38](#page-12-0)]. SMOTE works by interpolating between samples in the original dataset to generate new instances. [Fig. 3](#page-3-0) elucidates the synthetic generation process used by SMOTE, where synthetic samples are interpolated between two existing samples. For any given sample points *xi* and *xj*, a new synthetic sample point *xnew* can be obtained using formula 2:

$$\mathbf{x}_{new} = \mathbf{x}_i + \alpha * (\mathbf{x}_j - \mathbf{x}_i), \alpha \in [0, 1] \tag{2}$$

SMOTE generates non-overlapping new samples by using linear interpolation between any two data points. To do this, it selects the K nearest neighbors (KNN) for interpolation. Gao et al. [\[29](#page-12-0)] confirm that using SMOTE to generate new samples effectively improves the accuracy of off-target prediction models.

ADASYN (Adaptive Synthetic) [\[39](#page-12-0)], works in a similar way to SMOTE, but differs in the selection of original samples for interpolation. The number of synthetic samples generated by ADASYN is proportional to the number of neighboring samples that do not belong to the same class. ADASYN dynamically adjusts the sample weight based on the density distribution of minority and majority class samples to determine the number of new samples to be generated.

[Fig. 4](#page-4-0) provides a visual representation of the data distribution after applying different oversampling methods. We thank Lemaître et al. [\[40](#page-13-0)] for providing the visualization tool. It is observed that both the SMOTE and ADASYN methods showed a trend of 'points forming lines'. However, unlike SMOTE, ADASYN generates more new samples in regions with a greater diversity of classes. KMeansSMOTE shows a clear clustering pattern in the new samples generated, while SVMSMOTE produces a clear separation line between minority classes. In [Fig. 4,](#page-4-0) the overlap between the copied new sampling points and the original sampling points in the two-dimensional plane makes it almost impossible to detect the differences before and after random oversampling. In

![](assets/figures/_page_3_Figure_2.jpeg)

**Fig. 2.** Data-level methods for addressing class imbalance.

![](assets/figures/_page_3_Figure_4.jpeg)

**Fig. 3.** The visualization process of the oversampling method (SMOTE) and undersampling method (NearMiss and TomekLinks).

addition, a clear trend of 'points connected by lines' can be observed between the sample points obtained through SMOTE and ADASYN, consistent with their respective interpolation methods. Additionally, ADASYN differs from SMOTE by generating more new sample points in regions with greater category diversity.

In addition to SMOTE, we have also introduced two variants of the SMOTE algorithm: KMeansSMOTE and SVMSMOTE. As the name suggests, KMeansSMOTE first used KMeans clustering and then oversampled the dataset using SMOTE. In contrast, SVMSMOTE used a Support Vector Machine (SVM) classifier to identify support vectors (samples near the decision boundary) before synthesizing samples using SMOTE. Both of these methods are proposed to address the problem of SMOTE generating new sample points indiscriminately without properly considering outliers. [Fig. 4](#page-4-0) shows that the new samples generated by KMeansSMOTE exhibit significant clustering. Similarly, the new sample points generated by SVMSMOTE show different decision boundaries between different classes.

## ● Undersampling

Undersampling, also known as downsampling or sub-sampling, is an effective technique used in addressing the class imbalance problem. Random undersampling is the most widely used technique, which entails randomly selecting samples from the majority class and retaining a subset to reduce its proportion in the original dataset.

Although random undersampling is more effective than random oversampling in some cases, a major drawback of this method is that it discards potentially valuable data. To mitigate this limitation, researchers have proposed modification methods for judicious data selection. One such algorithm is NearMiss, which employs two heuristic rules for sample selection. In a binary classification scenario where the majority class samples are to be downsampled and the minority class samples belong to a hostile class, NearMiss-1 selects the positive sample with the smallest average distance to the N nearest negative samples. Fig. 3 shows NearMiss-1, where the positive sample connected by the green dashed line is selected because its average distance to the three nearest negative samples is the smallest (0.72). NearMiss-2, which operates as a two-stage algorithm, first retains M nearest neighbors for each negative sample and selects the positive sample with the largest average distance to N nearest negative samples. [Fig. 5](#page-5-0) visualizes the distribution of data points after undersampling using the NearMiss algorithm, highlighting the removal of numerous samples from the majority class that were initially close to the minority class.

TomekLinks is another popular undersampling algorithm, which identifies pairs of samples from different classes that are mutual nearest neighbors. As shown in [Fig. 5,](#page-5-0) undersampling via TomekLinks does not necessarily achieve inter-class balance; instead, it cleans up the dataset by eliminating some outliers to simplify the classification problem.

Edited Nearest Neighbor (ENN) is similar to TomekLinks, which uses the nearest neighbor algorithm to clean up the dataset by removing samples that are inconsistent with the nearest neighbor relationship. For each sample in the class slated for undersampling, ENN calculates its nearest neighbors and only retains the sample if most or all of its nearest neighbors belong to the same class.

## ● Hybrid-sampling

Hybrid sampling is an innovative method that combines

![](assets/figures/_page_4_Figure_2.jpeg)

**Fig. 4.** The sample distributions from five oversampling techniques.

oversampling and undersampling techniques. In the previous section, we discussed how SMOTE introduces new sample points between edge outliers and inliers. However, these samples are occasionally considered noise and may affect the classification results. To alleviate this issue, data cleaning techniques similar to TomekLinks and ENN can be employed for noise removal. By combining these two categories of technology (oversampling first, then data cleaning), a hybrid sampling technique emerges that integrates both oversampling and undersampling methods. Two common hybrid sampling techniques are SMOTETomek (SMOTE + TomekLinks) and SMOTEENN (SMOTE +

ENN). Specifically, SMOTEENN shows superior efficacy in eliminating noise samples compared to SMOTETomek.

Compared with the sample distribution obtained using the SMOTE method in Fig. 4, we observed that SMOTETomek removed a small portion of noise points, but the effect was not significant. On the other hand, SMOTEENN removes more noise samples, resulting in a clear decision boundary between samples of different classes. The cleaning process of SMOTEENN can significantly enhance the model's performance.

As shown in [Fig. 6,](#page-6-0) we describe the distribution of sample points after

![](assets/figures/_page_5_Figure_2.jpeg)

**Fig. 5.** The sample distributions result from five undersampling methods.

applying hybrid sampling techniques. Compared with the distribution of sample points obtained by the SMOTE method in [Fig. 4](#page-4-0), it is evident that SMOTETomek eliminates some noise points from a small portion of the samples. However, the effect is not significant. On the contrary, SMO-TEENN successfully removes more noise samples, resulting in a discernible decision boundary between different classes after the cleaning process. In this case, the performance of the model can be significantly improved.

#### 2.3.2. Cost-sensitive methods

Compared to data sampling methods, cost-sensitive methods that handle class imbalance do not alter the inherent distribution of training data. On the contrary, they make adjustments during the model learning or decision process to increase the importance of the minority class. This is primarily achieved by adjusting the loss function in the original model. Many traditional models typically use cross-entropy loss in their loss functions. Therefore, we replace it with two alternative loss functions: Focal Loss and GHM Loss.

![](assets/figures/_page_6_Figure_2.jpeg)

Fig. 6. Visualization diagram of the sample distributions after utilizing two hybrid sampling techniques (SMOTETomek and SMOTEENN).

Focal Loss introduced by Lin et al. [41], is mainly designed to solve the problem of imbalance in the number of easy and hard samples. The calculation of Focal Loss is shown in formula 3:

$$FL = \begin{cases} -\alpha * (1-p)^{\gamma} * \log(p), & \text{if } y = 1\\ -(1-\alpha) * p^{\gamma} * \log(1-p), & \text{if } y = 0 \end{cases} \tag{3}$$

where p represents the model's predicted probability of a sample being class 1,  $\alpha$  can suppress the imbalance between positive and negative samples,  $\gamma$  provides control over the imbalance between easy and hard samples. For instance, if we set  $\gamma$  to 2, when y=1, and p=0.968, indicating that this sample is an easy-to-classify sample because the model predicts a high probability for class 1, this time  $(1-0.968)^2 \approx 0.001$ , the loss is reduced by a factor of 1000.

GHM (Gradient Harmonized Margin) Loss represents an improvement over Focal Loss [36]. According to Li et al. [42], the Focal Loss tends to excessively focus on samples that are particularly difficult to classify, which may potentially reduce the overall accuracy of the model. This approach not only suppresses easily classifiable samples but also those that are particularly challenging to classify.

### 2.4. Performance evaluation

In this study, we primarily evaluate the performance of the model using PR\_AUC (Area Under the Precision-Recall Curve), a commonly used evaluation metric in imbalanced data. In contrast, we did not adopt the ROC-AUC (Area Under the Receiver Operating Characteristic Curve)

![](assets/figures/_page_6_Figure_10.jpeg)

Fig. 7. Experiments results of five oversampling methods (Random Oversampling, SMOTE, ADASYN, KMeansSMOTE, and SVMSMOTE) on six off-target prediction models: CRISPR\_Net, CRISPR\_IP, CrisprDNT, CNN\_std, CnnCrispr, and DL-CRISPR. The models were uniformly trained on the I-type dataset.

for off-target prediction tasks. The basic principle of this decision is that our experiments show that the models generally get high ROC\_AUC scores, many of which exceed 0.99. In this case, the effectiveness of ROC\_AUC as an evaluation metric is limited.

## 3. Results and discussion

### 3.1. Performance comparison of oversampling methods

To demonstrate the effectiveness of oversampling methods, we conducted comparative experiments on six off-target prediction models: CRISPR\_Net, CRISPR\_IP, CrisprDNT, CNN\_std, CnnCrispr, and DL-CRISPR using five oversampling techniques: random oversampling, SMOTE, ADASYN, KMeansSMOTE, and SVMSMOTE. The models were uniformly trained on the I-type dataset. We recorded the PR\_AUC values on the testing datasets (II/1, II/2, II/3, II/4). For example, "CRISP-R\_Net\_SMOTE" refers to the CRISPR\_Net model trained using the SMOTE oversampling technique, and similar naming conventions apply to other models. In our experiments, all five techniques were applied to the same off-target prediction model, and we also trained the model without any sampling for comparison.

As shown in [Fig. 7](#page-6-0), for the CRISPR\_Net model, the PR\_AUC values on datasets II/1, II/3, and II/4 increased by 58.8 %, 1.7 %, and 75.4 %, respectively, after applying the five oversampling techniques. The SVMSMOTE method yielded the best results on datasets II/1 and II/4, while dataset II/3 showed improvement with the KMeansSMOTE technique. However, dataset II/2 did not show improved performance after applying oversampling methods, indicating the limitations of oversampling techniques and should be applied judiciously based on specific circumstances.

In contrast, CRISPR\_IP and CrisprDNT showed varying degrees of improvement after applying oversampling techniques. For CRISPR\_IP, the most significant PR\_AUC improvements on datasets II/1, II/2, II/3, and II/4 were 36.3 %, 43.9 %, 123 %, and 33.7 %, respectively. Notably, the dataset II/3 showed the most significant improvement, with CRISPR\_IP consistently showing improvement on this dataset regardless of the oversampling technique used. This validates the efficacy of oversampling techniques in addressing class imbalance issues in offtarget prediction. Although CrisprDNT did not achieve as high an improvement as CRISPR\_IP, it still shows significant enhancement, with PR\_AUC values increasing by 80.2 % and 43.7 % on the II/1 and II/4 datasets, respectively. We also performed experiments on three models CNN\_std, CnnCrispr, and DL-CRISPR, where CnnCrispr was missing experiments via the SVMSMOTE oversampling method as SVMSMOTE produces data that is not pre-trained by the Glove module in the CnnCrispr model resulting in an unavailability of output for the model. From [Fig. 7,](#page-6-0) we find that these three models are less effective on testing datasets II/1 and II/2 due to their limited predictive power, and it is not meaningful to compare model performance on these two testing datasets. On testing datasets II/3 and II/4, we observe a significant improvement in the PR\_AUC values after some oversampling method. This proves that even if the performance of the off-target prediction model is limited, the use of a suitable oversampling method can improve the prediction performance of the model.

### 3.2. Performance comparison of undersampling methods

To evaluate the performance of undersampling methods, we also selected six off-target prediction models—CRISPR\_Net, CRISPR\_IP, CrisprDNT, CNN\_std, CnnCrispr, and DL-CRISPR on four Type II datasets (II/1, II/2, II/3, II/4). We used five classical undersampling methods, namely random undersampling, NearMiss, NearMiss\_2, TomekLinks, and ENN, to investigate the effect of undersampling methods. The results of PR\_AUC curve is shown in Fig. 8 and Supplementary Fig. S1.

The results for dataset II/1 show that CRISPR\_Net and CrisprDNT achieved the highest PR\_AUC improvements of 53.6 % and 61.4 %, respectively, after applying TomekLinks and random undersampling. In contrast, CRISPR\_IP did not show any significant improvement after

![](assets/figures/_page_7_Figure_12.jpeg)

**Fig. 8.** Experiments results of using five undersampling methods, Random Undersampling, ENN (Edited Nearest Neighbors), NearMiss, NearMiss\_2, and TomekLinks. These methods were applied to three off-target prediction models: CRISPR\_Net, CRISPR\_IP, and CrisprDNT. The models were uniformly trained on the I-type dataset.

undersampling, suggesting that the removal of data by undersampling may lead to a partial loss of information, hindering the model's ability to obtain sufficient training.

For the II/2, II/3, and II/4 datasets, CRISPR\_Net, CRISPR\_IP, and CrisprDNT all showed significant improvements. On the II/2 dataset, the highest PR\_AUC improvements for CRISPR\_Net, CRISPR\_IP, and CrisprDNT were 39.6 %, 40.2 %, and 1 %, respectively. Random undersampling, ENN, and TomekLinks achieved favorable results on these two models. On the II/3 dataset, the TomekLinks undersampling technique performed the best on CRISPR\_Net, CRISPR\_IP, and CrisprDNT, with PR\_AUC improvements of 3.7 %, 87.6 %, and 26.4 %, respectively. On the II/4 dataset, the NearMiss\_2 method yielded the best results for CRISPR\_Net and CRISPR\_IP, with PR\_AUC improvements of 33.3 % and 118 %, respectively, while CrisprDNT performed best with the TomekLinks method, with a PR\_AUC improvement of 39.0 %, and an improvement of 25.4 % with the NearMiss\_2 method.

Similarly, we performed experiments on three models CNN\_std, CnnCrispr, and DL-CRISPR as shown in Supplementary Fig. S1. For the two models CNN\_std and DL-CRISPR, although the model performance is improved after the undersampling method, the improvement is not significant due to the limited prediction capability of the model itself. In contrast, the CnnCrispr model shows significant improvement after the undersampling method, especially the PR\_AUC values on testing datasets II/2, II/3, and II/4. In addition, we found that using NearMiss\_2 undersampling method for CnnCrispr model achieved better improvement on all four testing datasets, which indicates that NearMiss\_2 undersampling technique is a relatively stable method.

To further compare the performance of oversampling and undersampling methods, we compiled the results of CNN\_std, CRISPR\_Net, CRISPR\_IP, and CrisprDNT on datasets II/1, II/2, II/3, and II/4. Using a boxplot, we visualized the PR\_AUC values in Fig. 9. Observing the PR\_AUC values in Fig. 9, we found that the oversampling methods generally outperform the undersampling methods on datasets II/3 and II/4, but CRISPR\_IP is an exception on dataset II/4. However, the oversampling and undersampling methods have advantages on datasets II/1 and II/2. Regardless of the dataset or model, more stable results can be obtained by using different oversampling methods than by using undersampling methods.

### 3.3. Performance of hybrid sampling and cost-sensitive methods

To evaluate the performance of hybrid sampling and cost-sensitive methods, we select two hybrid sampling methods, SMOTETomek and SMOTEENN, along with two cost-sensitive methods, Focal Loss and GHM, for experiments on testing datasets II/1, II/2, II/3, and II/4. We choose CRISPR\_Net, CRISPR\_IP, CrisprDNT, CNN\_std, CnnCrispr and DL-CRISPR as the base models.

[Fig. 10](#page-9-0) shows that the Focal loss method performs well on different models and testing datasets, except for the CNN\_std model, which is mediocre on testing dataset II/3, which may be related to the poorer prediction performance due to the simpler structure of the CNN\_std model. For CRISPR\_Net, Focal Loss achieved the best results on testing datasets II/1, II/2, and II/4, with PR\_AUC improvements of 153 %, 89.6 %, and 117 %, respectively. Although it performed less effectively on the testing dataset II/3, the significant PR\_AUC improvements demonstrated the feasibility of Focal Loss in addressing class imbalance issues in offtarget datasets. Similarly, Focal Loss showed superior performance on CRISPR\_IP and CrisprDNT models. For CRISPR\_IP, it achieved the best results on testing datasets II/1, II/2, and II/3, with PR\_AUC improvements of 40.8 %, 54.8 %, and 93.1 %, respectively, and performed slightly worse than GHM on testing dataset II/4. When applied to CrisprDNT, Focal Loss achieved the best results on testing datasets II/1, II/2, and II/4, with PR\_AUC improvements of 137 %, 6.8 %, and 4.2 %, respectively. Although the magnitude of improvement was lower than that of CRISPR\_Net and CRISPR\_IP, this could be attributed to the already high off-target prediction accuracy of CrisprDNT. For both CnnCrispr and DL-CRISPR models, the Focal loss method also achieves competitive results. The Focal loss method outperforms the original model on testing datasets II/1, II/2, II/3, and II/4. Furthermore, Focal loss outperformed most oversampling and undersampling techniques on

![](assets/figures/_page_8_Figure_10.jpeg)

![](assets/figures/_page_8_Figure_11.jpeg)

![](assets/figures/_page_8_Figure_12.jpeg)

![](assets/figures/_page_8_Figure_13.jpeg)

**Fig. 9.** Performance comparison of oversampling and undersampling methods on different models. We used boxplots to illustrate the experimental results of four offtarget prediction models, namely CNN\_std, CRISPR\_Net, CRISPR\_IP, and CrisprDNT, on the testing datasets II/1, II/2, II/3, and II/4.

![](assets/figures/_page_9_Figure_2.jpeg)

**Fig. 10.** Results of two hybrid sampling methods (SMOTETomek and SMOTEENN) and two cost-sensitive methods (Focal Loss and GHM) on the testing datasets II/1, II/2, II/3, and II/4. The models were uniformly trained on the class I dataset. For example, "CRISPR\_Net\_SMOTETomek" refers to the model trained on CRISPR\_Net using the SMOTETomek hybrid sampling technique, and similar naming conventions apply to the other models.

most testing datasets, making it a promising method for addressing class imbalance issues in off-target prediction tasks.

Finally, we show the results of the four models (CRISPR\_Net, CRISPR\_IP, CrisprDNT, and CNN\_std) on the testing datasets II/1 and II/ 2 using the hybrid sampling method and the cost-sensitive method, respectively in [Figs. 11 and 12.](#page-10-0) The number of validated off-target sites in testing datasets II/1 and II/2 was 56 and 54, respectively. We have obtained the probabilities of predicting validated off-target sites as positive samples for SMOTETomek, SMOTEENN, Focal Loss, and GHM. In the visualizations, red indicates a higher probability of the model predicting a sample as a positive sample, signifying the successful prediction of the validated off-target site. Conversely, blue indicates a lower probability of predicting a sample as a positive sample, suggesting potential difficulty in identifying the validated off-target site. Notably, the visualization for CNN\_std generally showed a blue hue, indicating its poor off-target prediction ability. In addition, SMOTEENN performed well in the visualizations for the other three models with an overall red hue, indicating a high probability of correctly identifying many validated off-target sites. On the other hand, GHM performed poorly mainly showing a blue or white hue indicating a low probability of correctly identifying validated off-target sites. Therefore, it is reasonable for researchers to analyze the statistical results from multiple perspectives when selecting a suitable sampling technique or cost-sensitive method for their models.

### 3.4. Statistical significance of the data imbalance methods

The four data unbalance methods, namely oversampling,

undersampling, hybrid sampling and cost-sensitive function, were utilized to conduct experiments on six off-target prediction models. The results demonstrated that the appropriate data unbalance analysis method can effectively enhance the model's performance. For the offtarget prediction task, the number of validated off-target sites that can be predicted by the model is crucial. Therefore, after employing different data imbalance resolution methods, the statistical significance of the number of validations off-target sites predicted by the statistical model was analyzed. We counted the number of validated off-target sites predicted by the two best-performing models, CRISPR\_Net and CrisprDNT, on testing datasets II/3 and II/4. CRISPR\_Net and CrisprDNT employ multiple methods for resolving the data imbalance, and we separately counted the number of validated off-target sites predicted by the models after employing each of these methods, with a selection of four methods with the highest number of predicted validated off-target sites for visual presentation, as shown in [Figs. 13 and 14](#page-11-0).

The test datasets II/3 and II/4 have 536 and 120 validated off-target sites, respectively. As shown in [Fig. 13,](#page-11-0) the CrisprDNT model predicted the highest number of validations off-target sites, 409, 400, 382, and 369, respectively, after using four methods: NearMiss, NearMiss\_2, undersampling, and oversampling. On the other hand, the CRISPR-Net model predicted the highest number of validations off-target sites for four undersampling methods (NearMss, SMOTEENN, and ENN). On the other hand, the CRISPR-Net model predicted the highest number of offtarget sites, which were 359, 345, 280, and 240, respectively. [Fig. 14](#page-12-0)  shows the statistical results of the model on test dataset II/4, where both the CrisprDNT and CRISPR-Net models were validated for off-target detection using the NearMss method, predicting the highest number of

![](assets/figures/_page_10_Figure_2.jpeg)

**Fig. 11.** Visualizations results of hybrid sampling and cost-sensitive methods on testing dataset II/1, which contained 56 validated off-target sites.

validated off-target sites, with 82 and 64, respectively. In addition, CrisprDNT achieved good results on undersampling, NearMiss\_2, and ADASYN methods.

We found that the number of validated off-target sites predicted by the statistical model was better than that predicted by the oversampling method after using the undersampling method. We analyzed that this may be due to the oversampling method producing some erroneous samples, which affects the robustness of the model. In addition, we found that NearMass has achieved competitive results in predicting the number of validated off-target sites, which is consistent with previous experimental results.

## 4. Conclusion

The identification of potential off-target sites poses a significant challenge in the application of the CRISPR-Cas9 system. The issue of class imbalance within off-target datasets has constrained the performance of existing prediction models. To address the class imbalance issue, this paper systematically investigates the data imbalance problem within current off-target datasets and proposes various solutions. These methods included oversampling, undersampling, mixed sampling, and

cost-sensitive methods. Results from these methods generally demonstrate an improvement in the prediction performance of original models for off-target sites. Particularly, the cost-sensitive methods and focal loss methods show promise for future research.

However, the research also indicates that none of the proposed methods for addressing data imbalance are universally applicable to all off-target prediction models or capable of achieving optimal results across all datasets. Even the effective and stable focal loss approach may not be suitable for some datasets. This variability is attributed to the inherent distribution of the datasets and the predictive capabilities of the models. Therefore, researchers should design appropriate methods based on the characteristics of the models and datasets. In the future, we will consider more deep reinforcement learning methods to enhance efficiency of off-target prediction [\[43](#page-13-0)].

# **CRediT authorship contribution statement**

**Zengrui Guan:** Writing – review & editing, Writing – original draft, Methodology, Formal analysis, Data curation. **Zhenran Jiang:** Writing – original draft, Supervision, Project administration, Methodology.

![](assets/figures/_page_11_Figure_2.jpeg)

**Fig. 12.** Visualizations results of the hybrid sampling and cost-sensitive methods on testing dataset II/2, which contained 54 validated off-target sites.

![](assets/figures/_page_11_Figure_4.jpeg)

**Fig. 13.** Statistical results of the number of validated off-target sites predicted by CrisprDNT and CRISPR\_Net models on testing dataset II/3.

![](assets/figures/_page_12_Figure_2.jpeg)

**Fig. 14.** Statistical results of the number of validated off-target sites predicted by CrisprDNT and CRISPR\_Net models on testing dataset II/4.

## **Declaration of competing interest**

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## **Acknowledgement**

This work was partially supported by grants from the National Key R&D Program of China (2019YFA0802800 and 2019YFA0110802).

## **Appendix A. Supplementary data**

Supplementary data to this article can be found online at [https://doi.](https://doi.org/10.1016/j.compbiomed.2024.108781)  [org/10.1016/j.compbiomed.2024.108781.](https://doi.org/10.1016/j.compbiomed.2024.108781)

## **References**

- [1] [Z. Zhang, A.E. Baxter, D. Ren, K. Qin, Z. Chen, S.M. Collins, et al., Efficient](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref1)  [engineering of human and mouse primary cells using peptide-assisted genome](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref1) [editing, Nat. Biotechnol. 42 \(2\) \(2024\) 305](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref1)–315.
- [2] [Y. Zhao, D. Tabet, D. Rubio Contreras, et al., Genome-scale mapping of DNA](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref2) [damage suppressors through phenotypic CRISPR-Cas9 screens, Mol. Cell. 83 \(15\)](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref2) [\(2023\) 2792](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref2)–2809.
- [3] [I. Tasan, H. Zhao, Targeting specificity of the CRISPR/Cas9 system, ACS Synth.](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref3) [Biol. 6 \(9\) \(2017\) 1609](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref3)–1613.
- [4] [G.H. Chuai, Q.L. Wang, Q. Liu, In silico meets in vivo: towards computational](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref4)  [CRISPR-based sgRNA design, Trends Biotechnol. 35 \(1\) \(2017\) 12](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref4)–21.
- [5] C. Jim´[enez, N. Crosetto, Discovering CRISPR](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref5)–cas off-target breaks, Nat. Methods [20 \(5\) \(2023\) 641](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref5)–642.
- [6] [S. Cancellieri, J. Zeng, L.Y. Lin, et al., Human genetic diversity alters off-target](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref6)  [outcomes of therapeutic gene editing, Nat. Genet. 55 \(1\) \(2023\) 34](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref6)–43.
- [7] H.H. Wessels, A. Stirn, A. M´[endez-Mancilla, E.J. Kim, S.K. Hart, D.A. Knowles, et](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref7)  [al., Prediction of on-target and off-target activity of CRISPR-Cas13d guide RNAs](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref7)  [using deep learning, Nat. Biotechnol. 42 \(4\) \(2024\) 628](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref7)–637.
- [8] [O. Yaish, M. Asif, Y. Orenstein, A systematic evaluation of data processing and](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref8) [problem formulation of CRISPR off-target site prediction, Briefings Bioinf. 23 \(5\)](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref8)  [\(2022\) bbac157](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref8).
- [9] [M. Toufikuzzaman, M.A. Hassan Samee, M. Sohel Rahman, CRISPR-DIPOFF: an](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref9)  [interpretable deep learning approach for CRISPR Cas-9 off-target prediction,](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref9) [Briefings Bioinf. 25 \(2\) \(2024\) bbad530.](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref9)
- [10] [D. Kim, S. Bae, J. Park, et al., Digenome-seq: genome-wide profiling of CRISPR-](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref10)[Cas9 off-target effects in human cells, Nat. Methods 12 \(3\) \(2015\) 237](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref10)–243.
- [11] [S.Q. Tsai, Z. Zheng, N.T. Nguyen, et al., GUIDE-seq enables genome-wide profiling](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref11)  [of off-target cleavage by CRISPR-Cas nucleases, Nat. Biotechnol. 33 \(2\) \(2015\)](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref11)  187–[197.](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref11)
- [12] [P. Cameron, C.K. Fuller, P.D. Donohoue, et al., Mapping the genomic landscape of](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref12)  CRISPR–[Cas9 cleavage, Nat. Methods 14 \(6\) \(2017\) 600](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref12)–606.
- [13] [S.Q. Tsai, N.T. Nguyen, J. Malagon-Lopez, et al., CIRCLE-seq: a highly sensitive in](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref13)  vitro screen for genome-wide CRISPR–[Cas9 nuclease off-targets, Nat. Methods 14](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref13) [\(6\) \(2017\) 607](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref13)–614.
- [14] [R.L. Frock, J. Hu, R.M. Meyers, et al., Genome-wide detection of DNA double](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref14)[stranded breaks induced by engineered nucleases, Nat. Biotechnol. 33 \(2\) \(2015\)](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref14)  179–[186.](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref14)

- [15] [W.X. Yan, R. Mirzazadeh, S. Garnerone, et al., BLISS is a versatile and quantitative](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref15)  [method for genome-wide profiling of DNA double-strand breaks, Nat. Commun. 8](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref15)  [\(1\) \(2017\) 1](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref15)–9.
- [16] [C.R. Lazzarotto, N.L. Malinin, Y. Li, et al., CHANGE-seq reveals genetic and](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref16) epigenetic effects on CRISPR–[Cas9 genome-wide activity, Nat. Biotechnol. 38 \(11\)](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref16)  [\(2020\) 1317](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref16)–1327.
- [17] [R. Singh, C. Kuscu, A. Quinlan, et al., Cas9-chromatin binding information enables](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref17)  [more accurate CRISPR off-target prediction, Nucleic Acids Res. 43 \(18\) \(2015\)](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref17) [e118](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref17).
- [18] [P.D. Hsu, D.A. Scott, J.A. Weinstein, et al., DNA targeting specificity of RNA-guided](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref18)  [Cas9 nucleases, Nat. Biotechnol. 31 \(2013\) 827](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref18)–832.
- [19] [M. Stemmer, T. Thumberger, M. del Sol Keyer, et al., CCTop: an intuitive, flexible](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref19)  [and reliable CRISPR/Cas9 target prediction tool, PLoS One 10 \(4\) \(2015\)](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref19) [e0124633](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref19).
- [20] [J.G. Doench, N. Fusi, M. Sullender, et al., Optimized sgRNA design to maximize](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref20)  [activity and minimize off-target effects of CRISPR-Cas9, Nat. Biotechnol. 34 \(2\)](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref20) [\(2016\) 184](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref20)–191.
- [21] [S. Abadi, W.X. Yan, D. Amar, et al., A machine learning approach for predicting](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref21) [CRISPR-Cas9 cleavage efficiencies and patterns underlying its mechanism of](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref21)  [action, PLoS Comput. Biol. 13 \(10\) \(2017\) e1005807.](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref21)
- [22] [J. Listgarten, M. Weinstein, B.P. Kleinstiver, et al., Prediction of off-target activities](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref22)  [for the end-to-end design of CRISPR guide RNAs, Nat. Biomed. Eng. 2 \(1\) \(2018\)](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref22)  38–[47.](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref22)
- [23] [G. Chuai, H. Ma, J. Yan, et al., DeepCRISPR: optimized CRISPR guide RNA design](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref23)  [by deep learning, Genome Biol. 19 \(2018\) 1](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref23)–18.
- [24] [Q. Liu, D. He, L. Xie, Prediction of off-target specificity and cell-specific fitness of](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref24)  [CRISPR-Cas System using attention boosted deep learning and network-based gene](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref24)  [feature, PLoS Comput. Biol. 15 \(10\) \(2019\) e1007480](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref24).
- [25] [J. Lin, Z. Zhang, S. Zhang, et al., CRISPR-Net: a recurrent convolutional network](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref25)  [quantifies CRISPR off-target activities with mismatches and indels, Adv. Sci. 7 \(13\)](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref25)  [\(2020\) 1903562.](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref25)
- [26] [J. Sun, J. Guo, J. Liu, CRISPR-M: predicting sgRNA off-target effect using a](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref26) [Multiview deep learning network, PLoS Comput. Biol. 20 \(3\) \(2024\) e1011972.](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref26)
- [27] [Z.R. Zhang, Z.R. Jiang, Effective use of sequence information to predict CRISPR-](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref27)[Cas9 off-target, Comput. Struct. Biotechnol. J. 20 \(2022\) 650](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref27)–661.
- [28] [Y. Zhang, Y. Long, R. Yin, et al., DL-CRISPR: a deep learning method for off-target](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref28)  [activity prediction in CRISPR/Cas9 with data augmentation, IEEE Access 8 \(2020\)](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref28)  [76610](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref28)–76617.
- [29] [Y. Gao, G. Chuai, W. Yu, et al., Data imbalance in CRISPR off-target prediction,](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref29)  [Briefings Bioinf. 21 \(4\) \(2020\) 1448](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref29)–1454.
- [30] [J. Lin, K.C. Wong, Off-target predictions in CRISPR-Cas9 gene editing using deep](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref30) [learning, Bioinformatics 34 \(17\) \(2018\) i656](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref30)–i663.
- [31] M. Haeussler, K. Schonig, ¨ [H. Eckert, et al., Evaluation of off-target and on-target](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref31) [scoring algorithms and integration into the guide RNA selection tool CRISPOR,](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref31)  [Genome Biol. 17 \(1\) \(2016\) 1](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref31)–12.
- [32] [B.P. Kleinstiver, M.S. Prew, S.Q. Tsai, et al., Engineered CRISPR-Cas9 nucleases](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref32) [with altered PAM specificities, Nature 523 \(7561\) \(2015\) 481](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref32)–485.
- [33] [Z.R. Guan, Z.R. Jiang, Transformer-based anti-noise models for CRISPR-Cas9 off](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref33)[target activities prediction, Briefings Bioinf. 24 \(3\) \(2023\) bbad127.](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref33)
- [34] [Q. Liu, X. Cheng, G. Liu, et al., Deep learning improves the ability of sgRNA off](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref34)[target propensity prediction, BMC Bioinf. 21 \(1\) \(2020\) 1](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref34)–15.
- [35] [J.M. Johnson, T.M. Khoshgoftaar, Survey on deep learning with class imbalance,](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref35)  [Journal of Big Data 6 \(1\) \(2019\) 1](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref35)–54.
- [36] [J.L. Leevy, T.M. Khoshgoftaar, R.A. Bauder, et al., A survey on addressing high](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref36)[class imbalance in big data, Journal of Big Data 5 \(1\) \(2018\) 1](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref36)–30.
- [37] [H. Ali, M.N.M. Salleh, R. Saedudin, et al., Imbalance class problems in data mining:](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref37)  [a review, Indonesian Journal of Electrical Engineering and Computer Science 14](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref37) [\(3\) \(2019\) 1560](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref37)–1571.
- [38] [N.V. Chawla, K.W. Bowyer, L.O. Hall, et al., SMOTE: synthetic minority over](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref38)[sampling technique, J. Artif. Intell. Res. 16 \(2002\) 321](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref38)–357.
- [39] [H. He, Y. Bai, E.A. Garcia, et al., ADASYN: adaptive synthetic sampling approach](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref39) [for imbalanced learning\[C\]. 2008 IEEE International Joint Conference on Neural](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref39)

- [Networks \(IEEE World Congress on Computational Intelligence\), IEEE, 2008,](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref39) [pp. 1322](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref39)–1328.
- [40] [G. Lemaître, F. Nogueira, C.K. Aridas, Imbalanced-learn: a python toolbox to tackle](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref40)  [the curse of imbalanced datasets in machine learning, J. Mach. Learn. Res. 18 \(17\)](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref40)  [\(2017\) 1](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref40)–5.
- [41] [T.Y. Lin, P. Goyal, R. Girshick, et al., Focal loss for dense object detection\[C\],](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref41)  [Proceedings of the IEEE international conference on computer vision \(2017\)](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref41) [2980](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref41)–2988.
- [42] [B. Li, Y. Liu, X. Wang, Gradient harmonized single-stage detector\[C\], Proc. AAAI](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref42) [Conf. Artif. Intell. 33 \(2019\) 8577](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref42)–8584.
- [43] [J. Yang, A.A.S. Soltan, D.W. Eyre, et al., Algorithmic fairness and bias mitigation](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref43)  [for clinical machine learning with deep reinforcement learning, Nat. Mach. Intell. 5](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref43)  [\(2023\) 884](http://refhub.elsevier.com/S0010-4825(24)00866-7/sref43)–894.