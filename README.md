# Metal-insulator transition model
Code and data used in constructing metal-insulator transition classifier. 
Original cif structure files can be identified in 
[Structures.zip](https://github.com/rpw199912j/mit_model_code/blob/master/data/Structures.zip)

## Model Description
### Research Question
The research question of this project is whether a machine learning classification model can predict metal-insulator 
transition (MIT) behavior  based on a series of compositional and structural descriptors / features of a given compound.

### Training Algorithm
The training algorithm or the model type chosen for this task is an [XGBoost](https://xgboost.readthedocs.io/en/latest/) 
tree classifier implemented in the Python programming language. This XGBoost type of models has helped won numerous 
Kaggle competitions and has been shown to perform well on classification tasks.

### A Word of Caution
Since the vast majority of the training data comes from oxides and there are not that many well-documented oxides that
exhibit MIT behavior, the training dataset as a result is quite small for machine learning standards
(around 200 observations). Thus, the models, especially with a high dimensional feature set, can easily overfit
and there is an ongoing effort to expand and find new MIT materials to add to the dataset.

## General Workflow (Work in progress)
### 1. Data Preparation
#### 1.1 Getting cif files

#### 1.2 Generate features using the cif files

#### 1.3 Select useful features and clean up the data

#### 1.4 Generate ionization lookup dataframe

### 2. Model Building
#### 2.1 Train the XGBoost model

#### 2.2 Evaluate model performance

### 3. Deploy & Serve Models



## Demo notebooks (Work in progress)

There are several [notebooks](https://github.com/rpw199912j/mit_model_code/tree/master/notebooks) 
available for easier result replication and demonstration purposes. You can find a brief [tutorial 
notebook](https://github.com/rpw199912j/mit_model_code/blob/master/notebooks/test_featurizer_sub_functions.ipynb)
that demostrate some of the sub-functions in the 
[compound_featurizer.py](https://github.com/rpw199912j/mit_model_code/blob/master/data/compound_featurizer.py)
file, or you can immediately start a Docker containerized version of the notebook by clicking on the Binder icon below.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rpw199912j/mit_model_code/master?urlpath=lab/tree/notebooks/test_featurizer_sub_functions.ipynb)
