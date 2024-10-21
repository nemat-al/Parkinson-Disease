# Parkinson-Disease

## Index
 [Introduction](#Introduction)
1. [Dataset description](#1-dataset-description)
2. [DataClass](#2-dataclass)
3. [Baseline models](#3-baseline-models)
4. [Initial Models]()

4.1. [Motif Search](#4.1-motif-search)

4.2. [Classification](#4.2-classification)

4.3. [Clustering](#4.3-clustering)

4.4. [Models on sequence data](#4.4-models-on-sequence-data)

4.5. [Applying FWHM scaling and adding other features for RFC](#4.5-applying-fwhm-scaling-and-adding-other-features-for-rfc)

4.6. [RFClassifier](#4.6-fclassifier)

5. [Explinable Models](#18explinable-models)

6. [The Hybrid models](#19the-hybrid-models)

----
### Introduction
Parkinson’s disease (PD) patients suffer from abnormal gait patterns. Therefore, monitoring and analysis of skeletal movements can aid in PD diagnosis. Several machine learning based models
were developed to automate the differentiation of abnormal gait from normal gait, which can serve as a tool for diagnosis and monitoring the effect of PD treatment. This work aspires to find
more complex structures in the time series gait data to introduce more efficient predictive modeling. The input to our algorithm is the gait in Parkinson’s Disease dataset maintained by
Physionet. The Dataset contains a time series of vertical ground reaction force (VGRF) as gait measurements from 93 patients with Parkinson’s Disease and 73 healthy controls collected during
walking at a normal phase.

Parkinson's disease (PD), a highly concerning neurodegenerative disorder that more than 10 million people are living with it worldwide. Symptoms pf Parkinson
s disease, from the source [[What is Parkinson's Disease Parkinson's NebraskaUR](https://parkinsonsnebraska.org/understanding%20parkinsons%20disease/)] :
![image](https://github.com/user-attachments/assets/ef9c1849-d3bf-4127-a431-0ee2738b879b)


### 1. Dataset Description

For the later tasks, the following data set was used.
The database contains measures of gait from 93 patients with idiopathic PD, and 73 healthy controls.
The dataset contains:
1. Vertical ground reaction force records of subjects as they walked for approximately 2 minutes
on level ground.

  • The file contains the measures from 8 sensors for each foot.

  • Each individual walks for 2 minutes, records are taken at 100 samples per second.

  Thus, we have 12000 record for each 2 mins walk.

2. Demographics file contains demographic information, measures of disease severity and other
related measures.

The following image shows the sensors underneath each foot[1]:

![image](https://github.com/user-attachments/assets/5881be73-44d8-4cae-b2e1-71555c27a01d)


### [2. DataClass](https://github.com/nemat-al/Parkinson-Disease/tree/main/DataClass)

The dataclass contains read the data, segment it, scale it and iterpolate it.

The dataset is multidimensional time series data with periodic structure, the repeated pattern in the signals are slightly different which is used as the main subject of the study to perform classification with Machine Learning models

![image](https://github.com/user-attachments/assets/752389f5-591d-4eeb-916f-540c90908e75)



### [3. Baseline models](https://github.com/nemat-al/Parkinson-Disease/blob/main/Baseline_models.ipynb)

Applying random forest classifier on the statics from the raw signal data AFTER being filtered.

### 4. Initial Models

#### [4.1 Motif Search](https://github.com/nemat-al/Parkinson-Disease/tree/main/Motif%20Search)

In the [first](https://github.com/nemat-al/Parkinson-Disease/blob/main/Motif%20Search/Motif_Search.ipynb) file, tried to apply motif identification on different features from the time series dataset, it appeared that the shape of the dataset (pressure - no pressure) is what resulted as motif and that is not useful.

In the [second](https://github.com/nemat-al/Parkinson-Disease/blob/main/Motif%20Search/Motif_Search_with_filtering.ipynb) file, Tried to filter the data, but still having the same problem.

#### [4.2 Classification](https://github.com/nemat-al/Parkinson-Disease/blob/main/Classification.ipynb)

| Task                       | features                               | Accuracy      |
| -------------              | -------------                          | ------------- |
| Severity Detection         | Univariate classification (L2 sensor)  | 0.39          |
| Parkinson’s Classification | Univariate classification (L2 sensor)  | 0.71          |
| Severity Detection         | Multivariate classification            | 0.38          |
| Parkinson’s Classification | Multivariate classification            | 0.82          |

#### [4.3 Clustering](https://github.com/nemat-al/Parkinson-Disease/tree/main/Clustering)

Clustering for fait time series dataset did not result in promising results. 
The [first](https://github.com/nemat-al/Parkinson-Disease/blob/main/Clustering/Clustering_PD_VGF_Gait_Stances.ipynb) file tried to cluster the data from both lef and right feet. The [second](https://github.com/nemat-al/Parkinson-Disease/blob/main/Clustering/Clustering_left_stances.ipynb) file applies the clustering on data from the left foot only. The [third](https://github.com/nemat-al/Parkinson-Disease/blob/main/Clustering/Clustering_right_stances.ipynb) file applies the clustering on data from the right foot only.

#### [4.4 Models on sequence data](https://github.com/nemat-al/Parkinson-Disease/blob/main/Models_Sequences_data.ipynb)

Data Class + Applying FWHM on accumalated forces fro the right foot + applying models on the sequences.

#### [4.5 Applying FWHM scaling and adding other features for RFC](https://github.com/nemat-al/Parkinson-Disease/blob/main/fwhm_scaling_RFC.ipynb)
Applying FWHM scaling and adding other features (stride time, max heel strike, max toe strike) for RFCs

#### 4.6 RFClassifier

- [RFClassifier](https://github.com/nemat-al/Parkinson-Disease/tree/main/RFClassifier) Class for training, predicting, scoring the results with Random Forest Classifier.
- [RFC models](https://github.com/nemat-al/Parkinson-Disease/blob/main/RFCmodels.ipynb)
In this code file, we use [class data](https://github.com/nemat-al/Parkinson-Disease/tree/main/DataClass) and [RFC class](https://github.com/nemat-al/Parkinson-Disease/tree/main/RFClassifier) to apply the previous different models on the data.
(SUMMING UP)
1. RFC basemodel using statics from raw data.
2. RFC models on interpolated scaled stances with FWHM algorithm, and additional features
3. RFC on scaled stances with 3 extra features and statics
4. RFC on scaled stances with 6 extra features and statics
5. RFC model on scaled stances from the right & left foot with 6 extra features each
6. RFC model on scaled stances from the right & left foot with 6 extra features each and basemodels
7. RFC model on scaled stances from 16 sensor from the right & left foot with 6 extra features each
8. RFC model on scaled stances from 16 sensor from the right & left foot with the sum of all sensors and with 6 extra features each
9. RFC model on scaled stances from 16 sensor from the right & left foot and the sum of all sensors and with 6 extra features each foot and the statics from base models

| Model                      | Input data                                                    | Accuracy      | Precision      | Recall      | F1      |
| -------------              | -------------                                                 | ------------ | ------------ | ------------ | ------------ |
| RFC n_est=200     | Statics on filtered raw data from each sensor  | 0.8329  |0.8716 |0.9084          |0.8844          |
| RFC n_est=200     | Right foot related: [ Interpolated Scaled stances, 3 features]  | 0.7092 |0.7708  |0.8176 |0.7897 |
| RFC n_est=200     | Right foot related: [ Interpolated Scaled stances, 6 features]| 0.7341| 0.7896 |0.8370|0.8815|

### Result of experiemnts so far

![image](https://github.com/user-attachments/assets/9351963b-e7eb-43ae-a295-9725cd86fbd8)


### [5. Hybrid models](https://github.com/nemat-al/Parkinson-Disease/tree/main/Hybrid_Models)
- [Trying hybrid models](https://github.com/nemat-al/Parkinson-Disease/blob/main/Hybridmodel.ipynb)
trying different hybrid models for right stances and 3 features.

- [Hybrid model](https://github.com/nemat-al/Parkinson-Disease/blob/main/Hybridmodels.ipynb)
Hybrid model class and a train it on for right stances and 3 features and on for right stances and 6 features.
Two final notebooks for training hybrid model on all data with/without statics for binary classification and severity Detection. 

The propesd methodology:

![image](https://github.com/user-attachments/assets/2bb69869-4c9f-48d9-a0e5-c7c1beed2461)

The model architecture for Severity Detection Multiclass Classification

![image](https://github.com/user-attachments/assets/557b291e-39f6-4902-a184-129fec8ae94d)

### [6.Explinable Models](https://github.com/nemat-al/Parkinson-Disease/tree/main/Explaining)
[First file](https://github.com/nemat-al/Parkinson-Disease/blob/main/Explaining/Explinability.ipynb): Finding what features are more important on models trained on statics and extracted features

[Second file](https://github.com/nemat-al/Parkinson-Disease/blob/main/Explaining/Explinability_Continue.ipynb): Showing the most important features in different plots.

As a result of model explaining and feature analysis, the most important features are
- Maximum force at heel strike.
- Maximum force from the accumulative signal.
- Swing time interval.
- Maximum force at toe off (Most different for people with higher levels of PD).
  ![image](https://github.com/user-attachments/assets/869e7238-8aae-4d99-9b5e-f3aa435d2f2c)


### Short summary of results/findings

This research explores Parkinson's disease symptoms and their impact on gait analysis. Using a public dataset, this study proposes a methodology to classify neurological states based on multi-dimensional time series by dividing the data into its repetitive pattern components. This approach improves the accuracy of machine learning models for Parkinson's disease diagnosis and severity detection, achieving an accuracy score of 99.69% and 99.73%, respectively. The study highlight important spatiotemporal features, such as the maximum force experienced at toe off, for accurately diagnosing and monitoring Parkinson's disease. Overall, this study demonstrates the potential of using segmented signals from time series datasets along with extracted features to improve the accuracy of Parkinson's Disease diagnosis and severity detection.


----
References:
[1] Abdulhay E. et al. Gait and tremor investigation using machine learning techniques for the diagnosis of Parkinson disease // Future Generation
Computer Systems. Elsevier BV, 2018 . Vol. 83 . P. 366 373
[] G. Gilmore, A. Gouelle, M. B. Adamson, M. Pieterman, and M. Jog, “Forward and backward walking in Parkinson disease: A factor analysis,” Gait & Posture, vol. 74, pp. 14–19, Oct. 2019, doi: 10.1016/J.GAITPOST.2019.08.005.
