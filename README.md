# Metal-insulator transition model
This repository contains the code and data used in constructing metal-insulator transition classifier. 
Original cif structure files can be identified in 
[Structures.zip](https://github.com/rpw199912j/mit_model_code/blob/master/data/Structures.zip)

# Table of Content
- [Model Description](https://github.com/rpw199912j/mit_model_code#model-description)
  * [Research Question](https://github.com/rpw199912j/mit_model_code#research-question)
  * [Training Algorithm](https://github.com/rpw199912j/mit_model_code#training-algorithm)
  * [A Word of Caution](https://github.com/rpw199912j/mit_model_code#a-word-of-caution)
- [General Workflow (Work in progress)](https://github.com/rpw199912j/mit_model_code#general-workflow-work-in-progress)
  * [1. Data Preparation](https://github.com/rpw199912j/mit_model_code#1-data-preparation)
    + [1.1 Getting cif files](https://github.com/rpw199912j/mit_model_code#11-getting-cif-files)
    + [1.2 Generate features using the cif files](https://github.com/rpw199912j/mit_model_code#12-generate-features-using-the-cif-files)
    + [1.3 Select useful features and clean up the data](https://github.com/rpw199912j/mit_model_code#13-select-useful-features-and-clean-up-the-data)
    + [1.4 Generate ionization lookup dataframe](https://github.com/rpw199912j/mit_model_code#14-generate-ionization-lookup-dataframe)
  * [2. Model Building](https://github.com/rpw199912j/mit_model_code#2-model-building)
    + [2.1 Train the XGBoost model](https://github.com/rpw199912j/mit_model_code#21-train-the-xgboost-model)
    + [2.2 Evaluate model performance](https://github.com/rpw199912j/mit_model_code#22-evaluate-model-performance)
  * [3. Deploy & Serve Models](https://github.com/rpw199912j/mit_model_code#22-evaluate-model-performance)
- [Demo notebooks (Work in progress)](https://github.com/rpw199912j/mit_model_code#demo-notebooks-work-in-progress)
  * [generate_compound_features.ipynb](https://github.com/rpw199912j/mit_model_code#generate_compound_featuresipynb)
  * [generate_lookup_table.ipynb](https://github.com/rpw199912j/mit_model_code#generate_lookup_tableipynb)
  * [EDA_and_data_cleaning.ipynb](https://github.com/rpw199912j/mit_model_code#EDA_and_data_cleaningipynb)
  * [model_building_and_eval.ipynb](https://github.com/rpw199912j/mit_model_code#model_building_and_evalipynb)
  * [pipeline_demo.ipynb](https://github.com/rpw199912j/mit_model_code#pipeline_demoipynb)
  * [Supporting notebooks](https://github.com/rpw199912j/mit_model_code#supporting-notebooks)
    + [test_featurizer_sub_functions.ipynb](https://github.com/rpw199912j/mit_model_code#test_featurizer_sub_functionsipynb)
    + [handbuilt_featurizer_benchmark.ipynb](https://github.com/rpw199912j/mit_model_code#handbuilt_featurizer_benchmarkipynb)


# Model Description
## Research Question
The research question of this project is whether a machine learning classification model can predict metal-insulator 
transition (MIT) behavior  based on a series of compositional and structural descriptors / features of a given compound.

## Training Algorithm
The training algorithm or the model type chosen for this task is an [XGBoost](https://xgboost.readthedocs.io/en/latest/) 
tree classifier implemented in the Python programming language. This XGBoost type of models has helped won numerous 
Kaggle competitions and has been shown to perform well on classification tasks.

## A Word of Caution
Since the vast majority of the training data comes from oxides and there are not that many well-documented oxides that
exhibit MIT behavior, the training dataset as a result is quite small for machine learning standards
(around 200 observations). Thus, the models, especially with a high dimensional feature set, can easily overfit
and there is an ongoing effort to expand and find new MIT materials to add to the dataset.

# General Workflow (Work in progress)
## 1. Data Preparation
### 1.1 Getting cif files

### 1.2 Generate features using the cif files

### 1.3 Select useful features and clean up the data

### 1.4 Generate ionization lookup dataframe

## 2. Model Building
### 2.1 Train the XGBoost model

### 2.2 Evaluate model performance

## 3. Deploy & Serve Models


# Demo notebooks (Work in progress)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rpw199912j/mit_model_code/master?urlpath=lab/tree/notebooks/)

There are several [notebooks](https://github.com/rpw199912j/mit_model_code/tree/master/notebooks) 
available for easier result replication and demonstration purposes. You can immediately launch interactive versions of these
notebooks in your web browser by clicking on the binder icon above or clicking on the subsection titles below.

You can replicate the workflow by using the notebooks in the following order.

## [generate_compound_features.ipynb](https://mybinder.org/v2/gh/rpw199912j/mit_model_code/master?urlpath=lab/tree/notebooks/generate_compound_features.ipynb)
This notebook allows you to generate features for all the structures contained in the [Structures.zip](https://github.com/rpw199912j/mit_model_code/blob/master/data/Structures.zip).

## [generate_lookup_table.ipynb](https://mybinder.org/v2/gh/rpw199912j/mit_model_code/master?urlpath=lab/tree/notebooks/generate_lookup_table.ipynb)
This notebook generates the ionization energy lookup spreadsheet.

## [EDA_and_data_cleaning.ipynb](https://mybinder.org/v2/gh/rpw199912j/mit_model_code/master?urlpath=lab/tree/notebooks/EDA_and_data_cleaning.ipynb)
This notebook presents an exploratory data analysis along with a data cleaning process on the output dataset from _generate_compound_features.ipynb_.

## [model_building_and_eval.ipynb](https://mybinder.org/v2/gh/rpw199912j/mit_model_code/master?urlpath=lab/tree/notebooks/model_building_and_eval.ipynb)
This notebook contains the code that tunes, trains and evaluates the models.

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

