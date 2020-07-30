# Metal-insulator transition model
Code and data used in constructing metal-insulator transition classifier. 
Original cif structure files can be identified in 
[Structures.zip]("https://github.com/rpw199912j/mit_model_code/blob/master/data/Structures.zip")

## General Workflow (Work in progress)
### Getting cif files

### Generate features using the cif files

### Select useful features and clean up the data

### Train and evaluate the XGBoost model

## Demo notebooks (Work in progress)

There are several [notebooks](https://github.com/rpw199912j/mit_model_code/tree/master/notebooks) 
available for easier result replication and demonstration purposes. You can find a brief [tutorial 
notebook](https://github.com/rpw199912j/mit_model_code/blob/master/notebooks/test_featurizer_sub_functions.ipynb)
that demostrate some of the sub-functions in the 
[compound_featurizer.py](https://github.com/rpw199912j/mit_model_code/blob/master/data/compound_featurizer.py)
file, or you can immediately start a Docker containerized version of the notebook by clicking on the Binder icon below.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rpw199912j/mit_model_code/master?urlpath=lab/tree/notebooks/test_featurizer_sub_functions.ipynb)
