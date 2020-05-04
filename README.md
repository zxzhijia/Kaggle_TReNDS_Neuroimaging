# Kaggle project: TReNDS Neuroimaging

A simple and well-designed structure is essential for Machine Learning / Deep Learning projects. Template is from: https://github.com/MrGemy95/Tensorflow-Project-Template. 

## project info

link: https://www.kaggle.com/c/trends-assessment-prediction

Deadline: 

Merger Deadline: June 22, 2020

Entry Deadline: June 22, 2020

End Date (Final Submission Deadline): June 29, 2020 11:59 PM UTC

## notebooks:

https://www.kaggle.com/aerdem4/rapids-svm-on-trends-neuroimaging

score: 0.159

interesting cuML python API runs ML on GPU

## related papers:

https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0066572

## more details:

Neuroimaging

Goals: 

Prediction of target variables, generalization to different sites/scanners

Data Interpretation:

Target data: [age, domain1_var1, domain1_var2, domain2_var1, domain2_var2], continuous variable

How features are obtained: scICA, GIG-ICA

sMRI: high spatial resolution, providing structural/anatomical info in the brain.
fMRI: low spatial resolution, record brain activity (Blood Oxygen Level Dependent signals) along time.

features: SBM loadings — subject-level weights, gray matter concentration maps from sMRI

					    loading.csv.
					    
					    gray matter volume may be affected by person’s age or hobbies (e.g., play tetris)

				             197 regions in the brain?

		FNC matrices — subject-level cross correlation values among 53 component timecourses (time series signals)

					    in fnc.csv file: 53 * 52 / 2 = 1378 features

		component SM — subject-level 3D images of 53 spatial networks

					    SM_features — 53 3D spatial maps for train samples (possibly the key features)

fMRI_mask — Variations among participants, movement artifacts

ICN_numbers — names for different networks, (each network corresponds to a index in the connectivity matrix)

Possible Bias in Data:

Different sites: 1. Quality of measured data due to different devices

			 2. Different sample population

			 3. Different conditions when taking the measurements, e.g., time, weather, etc. 

Missing values, especially for domain 1 — require data imputation techniques to be applied here

Pre-analysis of the data

Correlation among target variables — domain 1 variable have high correlations

Stratification technique? — to represent the original dataset, to deal with the missing values?

Statistical Analysis:

FNC correlations:

across the participants, the correlation between FNC matrices and target values

study of univariate/multi-variate effects

elbow points for explained variance and PCA technique

Structural (grey matter volume) correlations

High resolution may not always help, and also waste time to process data
