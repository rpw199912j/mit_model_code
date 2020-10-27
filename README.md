# Metal-Insulator Transition Classifiers
This repository contains the code and data used in constructing the thermally-driven metal-insulator transition (MIT) classifiers, which are 3
binary classifiers: a Metal _vs._ non-Metal model, an Insulator _vs._ non-Insulator model and an MIT _vs._ non-MIT model.

**Check out our [preprint paper](https://arxiv.org/abs/2010.13306) on arXiv:**

> Georgescu, A. B.; Ren, P.; Toland, A. R.; Olivetti, E. A.; Wagner, N.; Rondinelli, J. M. 
> A Database and Machine Learning Model to Identify Thermally Driven Metal-Insulator Transition Compounds. 
> arXiv:2010.13306 [cond-mat] 2020.


# Table of Content
- [Model Description](#model-description)
  * [Research Question](#research-question)
  * [Training Algorithm](#training-algorithm)
  * [A Word of Caution](#a-word-of-caution)
- [General Workflow](#general-workflow)
  * [1. Data Preparation](#1-data-preparation)
    + [1.1 Getting CIF files](#11-getting-cif-files)
    + [1.2 Generate ionization lookup dataframe](#12-generate-ionization-lookup-dataframe)
    + [1.3 Generate features using the cif files](#13-generate-features-using-the-cif-files)
    + [1.4 Clean up the data](#14-clean-up-the-data)
  * [2. Model Building](#2-model-building)
    + [2.1 Tune the XGBoost model](#21-tune-the-xgboost-model)
    + [2.2 Evaluate performance and save models](#22-evaluate-performance-and-save-models)
    + [2.3 Select important features and iterate](#23-select-important-features-and-iterate)
  * [3. Deploy & Serve Models](#3-deploy--serve-models)
- [Demo Notebooks](#demo-notebooks)
  * [generate_lookup_table.ipynb](#generate_lookup_tableipynb)
  * [generate_compound_features.ipynb](#generate_compound_featuresipynb)
  * [EDA_and_data_cleaning.ipynb](#EDA_and_data_cleaningipynb)
  * [model_building_and_eval.ipynb](#model_building_and_evalipynb)
  * [pipeline_demo.ipynb (**Make a prediction right in your web browser!**)](#pipeline_demoipynb)
  * [Supporting notebooks](#supporting-notebooks)
    + [model_comparison.ipynb](#model_comparisonipynb)
    + [shap_analysis.ipynb](#shap_analysisipynb)
    + [test_featurizer_sub_functions.ipynb](#test_featurizer_sub_functionsipynb)
    + [handbuilt_featurizer_benchmark.ipynb](#handbuilt_featurizer_benchmarkipynb)
    + [dataset_visualization.ipynb](#dataset_visualizationipynb)

# Model Description
## Research Question
The research question of this project is whether a machine learning classification model can predict temperature-driven metal-insulator 
transition behavior based on a series of compositional and structural descriptors/features of a given compound.

## Training Algorithm
The training algorithm or the model type chosen for this task is an [XGBoost](https://xgboost.readthedocs.io/en/latest/) 
tree classifier implemented in the Python programming language. XGBoost models have helped won numerous 
Kaggle competitions and have been shown to perform well on classification tasks. For this research project, if you wonder why we chose XGBoost over other
model types and why binary classification over multi-class classification, you can refer to [this section](#model_comparisonipynb). The takeaway is that XGBoost is consistently among the best performing model types
and that it is faster to train compared to other models with comparable performance. The performance across all model types on binary classifications is also
better than that on multi-class classifications.

## A Word of Caution
Since the vast majority of the training data comes from oxides and there are not that many well-documented oxides that
exhibit MIT behavior, the training dataset as a result is quite small for machine learning standards
(228 observations / rows). Thus, the models, especially with a high dimensional feature set, can easily overfit
and there is an ongoing effort to expand and find new MIT materials to add to the dataset. Thus, as we continue to expand
our dataset, the models trained on the dataset are also subject to change over the course of time.

**We strongly encourage people to contribute temperature-driven MIT materials that aren't already included in our dataset. 
Please include your name, institution, the CIF file and reference publications in your email and send them to 
Professor [James M. Rondinelli](mailto:jrondinelli@northwestern.edu).**

**You can also suggest new MIT material(s) by opening an issue with the `New MIT material` template.**

# General Workflow
## 1. Data Preparation
### 1.1 Getting CIF files
The CIF files are obtained through online databases such as [ICSD database](https://icsd.products.fiz-karlsruhe.de/en), 
[Springer Materials](https://materials.springer.com/) 
and [Materials Project](https://materialsproject.org/) in addition to a few hand generated ones. The vast majority of CIF files are
high-quality experimental structures files from the ICSD database, with a few from the Springer and Materials
Project databases.

**Note**: Unfortunately, we can not directly share the collected CIF files due to copyright concerns. However, you can find the material ID of the 
compounds included in our dataset [here](data/processed/csv_version/IMT_Classification_Dataset_Full_Feature_Set_v9.csv) 
(you should look at the `struct_file_path` column to find the IDs). Should you have access, you can use those IDs 
to download CIF files from ICSD, Springer and Materials Project. 
You will find 4 suffixes in `struct_file_path` which correspond to 4 sources as follows.

|Suffix|Source|
|:-----|:-----|
|CollCode|ICSD|
|SD|Springer Materials|
|MP|Materials Project|
|HandGenerated|Generated by hand based on publications|

### 1.2 Generate ionization lookup dataframe
This step creates an ionization lookup table that is used in the subsequent featurization process.

### 1.3 Generate features using the CIF files
A total of 164 compositional and structural features are generated using a combination of [matminer](https://hackingmaterials.lbl.gov/matminer/) and our
in-house handbuilt featurizers. These features then undergo further processing and selection down the pipeline.

### 1.4 Clean up the data
After a brief exploratory data anaylsis, it is found that the raw output from the featurizers contains features with missing values, 
zero-variance (i.e. the feature value is the same for all compounds) and high linear correlation (greater than 0.95). Therefore, the data cleaning process is carried out in the following order:

- Drop rows / compounds with at least 6 missing features
- Impute missing values by column / feature
    - Group the dataset by the assigned label (one of metal, insulator or MIT)
    - Calculate the median value for each feature within each label group
    - Fill the missing values with the corresponding medians of each label group
- Remove features with zero variance
- Remove features with high linear correlation
    - Find features with linear correlation greater than 0.95
    - Drop one of the two features in each pair of highly correlated features

After data cleaning, the dataset now has 103 (102 numeric & 1 one-hot-encoded categorical with 2 levels) features 
remaining and will be referred to as [the full feature](data/processed/csv_version/IMT_Classification_Dataset_Full_Feature_Set_v9.csv) set from now on.

## 2. Model Building
The model building process follows an iterative approach. During the first iteration, the cleaned-up full feature set is fed into
the classifiers, trained and then evaluated. Then with the help of [SHAP values](https://github.com/slundberg/shap) and domain knowledge,
features with high importance are selected and used as input to the second iteration of model training and evaluation.

### 2.1 Tune the XGBoost model
The training process starts with hyperparameter tuning with grid search cross validation. The default parameter search grid 
for the XGBClassifier is as follows.

| Parameter       | Search space |
|:-------------   |:-------------|
| n_estimators    | [10, 20, 30, 40, 80, 100, 150, 200] |
| max_depth       | [2, 3, 4, 5]      |
| learning_rate   | np.logspace(-3, 2, num=6)     |
| subsample       | [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]|
| scale_pos_weight| [num_of_negative_class / num_of_positive_class] |
| base_score      | [0.3, 0.5, 0.7]      |
| random_state    | [seed]   |

The scoring metric during tuning is [f1_weighted](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html). 
The best tuned parameters are then stored for model evaluation,

### 2.2 Evaluate performance and save models
Due to the scarcity of training examples, stratified 5-fold cross validation (cv) is used to evaluate model performance instead of a hold-out test set.
There are 4 evaluation metrics used:
 
1. [precision_weighted](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)
2. [recall_weighted](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)
3. [roc_auc](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
4. [f1_weighted](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)

Since the cross validation splits depend on the random seed, a list of 10 seeds (integers from 0 to 9) are used to take into account
the variation in model performance due to different splits from different seeds. For each seed, a stratified 5-fold cv is carried out, from which
the median / mean values for the metrics are obtained. With 10 seeds, there are 10 median / mean values for each metric and finally a median / mean value
is calculated from those 10 values, along with the interquartile range / standard deviation respectively. Essentially, the values reported are either
a median of medians by default or an average of averages should you choose so.

After model evaluation, the models are trained on the entire dataset (228 compounds with the full feature set) with the best parameters and then stored. 


### 2.3 Select important features and iterate
Using the stored models, a SHAP analysis is carried out to find the most important features. These important features are further screened
using domain knowledge. Currently, 10 features are selected to create a [reduced feature set](data/processed/csv_version/IMT_Classification_Dataset_Reduced_Feature_Set_v9.csv). 
This feature selection step mainly serves to prevent overfitting.

With this reduced feature set, the entire model building process is repeated and the models are re-tuned, re-evaluated and 
re-trained on the reduced feature set.

## 3. Deploy & Serve Models
The trained classifiers are made available to the larger materials science community through Jupyter notebooks hosted via
the [Binder](https://mybinder.readthedocs.io/en/latest/) service. One can immediately upload a CIF file and easily make a prediction using our classifiers directly
in the web browser.

The models served on the Binder server are by default based on the reduced feature set.

# Demo Notebooks
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rpw199912j/mit_model_code/master?urlpath=lab/tree/notebooks/)

There are several [Jupyter notebooks](notebooks) 
available for easier result replication and demonstration purposes. You can immediately launch interactive versions of these
notebooks in your web browser by clicking on the binder icon above or clicking on the subsection titles below.

**Note**: Any changes made on the server will not be saved unless you download a copy of the notebook onto your local machine.

You can replicate the workflow by using the notebooks in the following order. 

## [generate_lookup_table.ipynb](https://mybinder.org/v2/gh/rpw199912j/mit_model_code/master?urlpath=lab/tree/notebooks/generate_lookup_table.ipynb)
This notebook generates the ionization energy lookup spreadsheet.

## [generate_compound_features.ipynb](https://mybinder.org/v2/gh/rpw199912j/mit_model_code/master?urlpath=lab/tree/notebooks/generate_compound_features.ipynb)
This notebook allows you to generate features for all the structures. As mentioned before, since we cannot share the structure files, 
running this notebook will not work due to the absence of CIF files.

## [EDA_and_data_cleaning.ipynb](https://mybinder.org/v2/gh/rpw199912j/mit_model_code/master?urlpath=lab/tree/notebooks/EDA_and_data_cleaning.ipynb)
This notebook presents an exploratory data analysis along with a data cleaning process on the output dataset from _generate_compound_features.ipynb_.

## [model_building_and_eval.ipynb](https://mybinder.org/v2/gh/rpw199912j/mit_model_code/master?urlpath=lab/tree/notebooks/model_building_and_eval.ipynb)
This notebook contains the code that tunes, trains and evaluates the models along with a SHAP analysis on models trained with the full feature set. It is NOT recommended to train the models directly
on the Binder server since it is a very memory intensive process (it will also take a very long time to train!). The Binder container by default has 2GB of RAM and if the memory
limit is exceeded, there is a possibility that the kernel will restart and you'll have to start over. 
That being said, you are welcome to download the repository onto your local machine and play around with the model parameters and selection.


## [pipeline_demo.ipynb](https://mybinder.org/v2/gh/rpw199912j/mit_model_code/master?urlpath=lab/tree/notebooks/pipeline_demo.ipynb)
This notebook demonstrates the prediction pipeline through which a prediction is made on a new structure that is not
included in the original training set. You can even **upload your own CIF structure** and get a prediction! 
If you just want to play around with the trained models or make a prediction on 
a structure of your own choice, you can start here.

## Supporting notebooks
### [model_comparison.ipynb](https://mybinder.org/v2/gh/rpw199912j/mit_model_code/master?urlpath=lab/tree/notebooks/model_comparison.ipynb)
This notebook answers the question of "Why should one choose XGBoost over some other models?" by comparing the classification performance of 6 model types on the full feature set across 4 classification tasks.
 The model types are as follows.

|Model type|Description|
|:---|:---|
|[DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)|Naive models that are always random guessing (baseline performance)|
|[LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)|Linear classifiers with L2 regularization|
|[DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)|Generic decision tree classifiers|
|[RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)|Ensemble decision tree classifiers|
|[GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)|Gradient-boosting tree classifiers|
|[XGBoostClassifier](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier)|Extreme gradient-boosting tree classifiers|

The 4 classification tasks are:

1. Metal vs. non-Metals (Insulators + MITs)
2. Insulator vs. non-Insulators (Metals + MITs)
3. MIT vs. non-MIT (Metals + Insulators)
4. Multi-class classification

The metrics and evaluation method are the same as the [process](#22-evaluate-performance-and-save-models) mentioned earlier. The comparison results are summarized in [this table](https://github.com/rpw199912j/mit_model_code/blob/master/data/processed/csv_version/model_metrics_comparison_with_raw.csv).
A [summary plot](plots/model_comparison_boxplot.pdf) is also provided for easier interpretation.

### [shap_analysis.ipynb](https://mybinder.org/v2/gh/rpw199912j/mit_model_code/master?urlpath=lab/tree/notebooks/shap_analysis.ipynb)
This notebook presents a brief SHAP analysis on models trained with the reduced feature set.

### [test_featurizer_sub_functions.ipynb](https://mybinder.org/v2/gh/rpw199912j/mit_model_code/master?urlpath=lab/tree/notebooks/test_featurizer_sub_functions.ipynb)
This is a brief [tutorial 
notebook](notebooks/test_featurizer_sub_functions.ipynb)
that explains some of the sub-functions in the 
[compound_featurizer.py](data/compound_featurizer.py)
file.

### [handbuilt_featurizer_benchmark.ipynb](https://mybinder.org/v2/gh/rpw199912j/mit_model_code/master?urlpath=lab/tree/notebooks/handbuilt_featurizer_benchmark.ipynb)
This notebooks provides a benchmark of how "good" the handbuilt featurizer is against values from 
[Table 2 & 3](data/torrance_tables/torrance_tabulated.xlsx) 
of [Torrance et al](https://www.sciencedirect.com/science/article/abs/pii/0921453491905346).

### [dataset_visualization.ipynb](https://mybinder.org/v2/gh/rpw199912j/mit_model_code/master?urlpath=lab/tree/notebooks/dataset_visualization.ipynb)
This notebooks contains visualization plots to be included in the paper.

