# Metal-insulator transition model
This repository contains the code and data used in constructing the metal-insulator transition classifier. 

# Table of Content
- [Model Description](https://github.com/rpw199912j/mit_model_code#model-description)
  * [Research Question](https://github.com/rpw199912j/mit_model_code#research-question)
  * [Training Algorithm](https://github.com/rpw199912j/mit_model_code#training-algorithm)
  * [A Word of Caution](https://github.com/rpw199912j/mit_model_code#a-word-of-caution)
- [General Workflow (Work in progress)](https://github.com/rpw199912j/mit_model_code#general-workflow-work-in-progress)
  * [1. Data Preparation](https://github.com/rpw199912j/mit_model_code#1-data-preparation)
    + [1.1 Getting CIF files](https://github.com/rpw199912j/mit_model_code#11-getting-cif-files)
    + [1.2 Generate ionization lookup dataframe](https://github.com/rpw199912j/mit_model_code#12-generate-ionization-lookup-dataframe)
    + [1.3 Generate features using the cif files](https://github.com/rpw199912j/mit_model_code#13-generate-features-using-the-cif-files)
    + [1.4 Clean up the data](https://github.com/rpw199912j/mit_model_code#14-clean-up-the-data)
  * [2. Model Building](https://github.com/rpw199912j/mit_model_code#2-model-building)
    + [2.1 Tune the XGBoost model](https://github.com/rpw199912j/mit_model_code#21-tune-the-xgboost-model)
    + [2.2 Evaluate performance and save models](https://github.com/rpw199912j/mit_model_code#22-evaluate-performance-and-save-models)
    + [2.3 Select important features and iterate](https://github.com/rpw199912j/mit_model_code#23-select-important-features-and-iterate)
  * [3. Deploy & Serve Models](https://github.com/rpw199912j/mit_model_code#3-deploy--serve-models)
- [Demo notebooks (Work in progress)](https://github.com/rpw199912j/mit_model_code#demo-notebooks-work-in-progress)
  * [generate_compound_features.ipynb](https://github.com/rpw199912j/mit_model_code#generate_compound_featuresipynb)
  * [generate_lookup_table.ipynb](https://github.com/rpw199912j/mit_model_code#generate_lookup_tableipynb)
  * [EDA_and_data_cleaning.ipynb](https://github.com/rpw199912j/mit_model_code#EDA_and_data_cleaningipynb)
  * [model_building_and_eval.ipynb](https://github.com/rpw199912j/mit_model_code#model_building_and_evalipynb)
  * [pipeline_demo.ipynb (**Make a prediction right in your web browser!**)](https://github.com/rpw199912j/mit_model_code#pipeline_demoipynb)
  * [Supporting notebooks](https://github.com/rpw199912j/mit_model_code#supporting-notebooks)
    + [test_featurizer_sub_functions.ipynb](https://github.com/rpw199912j/mit_model_code#test_featurizer_sub_functionsipynb)
    + [handbuilt_featurizer_benchmark.ipynb](https://github.com/rpw199912j/mit_model_code#handbuilt_featurizer_benchmarkipynb)


# Model Description
## Research Question
The research question of this project is whether a machine learning classification model can predict metal-insulator 
transition (MIT) behavior based on a series of compositional and structural descriptors / features of a given compound.

## Training Algorithm
The training algorithm or the model type chosen for this task is an [XGBoost](https://xgboost.readthedocs.io/en/latest/) 
tree classifier implemented in the Python programming language. This XGBoost type of models has helped won numerous 
Kaggle competitions and has been shown to perform well on classification tasks.

## A Word of Caution
Since the vast majority of the training data comes from oxides and there are not that many well-documented oxides that
exhibit MIT behavior, the training dataset as a result is quite small for machine learning standards
(229 observations / rows). Thus, the models, especially with a high dimensional feature set, can easily overfit
and there is an ongoing effort to expand and find new MIT materials to add to the dataset.

# General Workflow (Work in progress)
## 1. Data Preparation
### 1.1 Getting CIF files
The CIF files are obtained through the [ICSD database](https://icsd.products.fiz-karlsruhe.de/en), 
[Springer Materials](https://materials.springer.com/) 
and [Materials Project](https://materialsproject.org/). The vast majority of CIF files are
high-quality experimental structures files from the ICSD database, with a few from the Springer and Materials
Project databases.

**Note**: Unfortunately, we can not share the collected CIF files directly due to copyright concerns. However, you can find the material ID of the 
compounds included in our dataset [here](https://github.com/rpw199912j/mit_model_code/blob/master/data/processed/IMT_Classification_Dataset_Processed_v9.xlsx) 
(you should look at the `struct_file_path` column to find the IDs). If you have access, you can use those IDs to download CIF files from ICSD & Springer.

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

After data cleaning, the dataset now has 103 (102 numeric & 1 one-hot-encoded categorical with 2 levels) features remaining and will be referred to as the full feature set from now on.

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
| scale_pos_weight| num_of_negative_class / num_of_positive_class |
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

Since the cross validation splits depend on the random seed, a list of 10 seeds (integers from 0-9) are used to take into account
the variation in model performance due to different splits from different seeds. For each seed, a stratified 5-fold cv is carried out, from which
the median / mean values for the metrics are obtained. With 10 seeds, there are 10 median / mean values for each metric and finally a median / mean value
is calculated from those 10 values, along with the interquartile range / standard deviation respectively. Essentially, the values reported are either
a median of medians or an average of averages.

After model evaluation, the models are trained on the entire dataset with the best parameters and then stored. 


### 2.3 Select important features and iterate
Using the stored models, a SHAP analysis is carried out to find the most important features. These important features are further screened
using domain knowledge. Currently, 10 features are selected to create a reduced feature set. This feature selection step 
mainly serves to prevent overfitting.

With this reduced feature set, the entire model building process is repeated and the models are re-tuned, re-evaluated and 
re-trained on the reduced feature set.

## 3. Deploy & Serve Models
The trained classifiers are made available to the larger materials science community through Jupyter notebooks hosted via
the [Binder](https://mybinder.readthedocs.io/en/lates) service. One can immediately upload a CIF file and easily make a prediction using our classifiers directly
in the web browser.

The models served on the Binder server are by default based on the reduced feature set.

# Demo notebooks (Work in progress)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rpw199912j/mit_model_code/master?urlpath=lab/tree/notebooks/)

There are several [Jupyter notebooks](https://github.com/rpw199912j/mit_model_code/tree/master/notebooks) 
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
This notebook contains the code that tunes, trains and evaluates the models along with the SHAP analysis. It is NOT recommended to train the models directly
on the Binder server since it is a very memory intensive process. The Binder container by default has 2GB of RAM and if the memory
limit is exceeded, there is a possibility that the kernel will restart and you'll have to start over. 
That being said, you are welcome to download the repository onto your local machine and play around with the model parameters and selection.


## [pipeline_demo.ipynb](https://mybinder.org/v2/gh/rpw199912j/mit_model_code/master?urlpath=lab/tree/notebooks/pipeline_demo.ipynb)
This notebook demonstrates the prediction pipeline through which a prediction is made on a new structure that is not
included in the original training set. If you just want to play around with the trained models or make a prediction on 
a structure of your own choice, you can start here.

## Supporting notebooks
### [test_featurizer_sub_functions.ipynb](https://mybinder.org/v2/gh/rpw199912j/mit_model_code/master?urlpath=lab/tree/notebooks/test_featurizer_sub_functions.ipynb)
This is a brief [tutorial 
notebook](https://github.com/rpw199912j/mit_model_code/blob/master/notebooks/test_featurizer_sub_functions.ipynb)
that explains some of the sub-functions in the 
[compound_featurizer.py](https://github.com/rpw199912j/mit_model_code/blob/master/data/compound_featurizer.py)
file.

### [handbuilt_featurizer_benchmark.ipynb](https://mybinder.org/v2/gh/rpw199912j/mit_model_code/master?urlpath=lab/tree/notebooks/handbuilt_featurizer_benchmark.ipynb)
This notebooks provides a benchmark of how "good" the handbuilt featurizer is against values from 
[Table 2 & 3](https://github.com/rpw199912j/mit_model_code/blob/master/data/torrance_tables/torrance_tabulated.xlsx) 
of [Torrance et al](https://www.sciencedirect.com/science/article/abs/pii/0921453491905346).

