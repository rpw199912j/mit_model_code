import numpy as np
import pandas as pd
import pymatgen as mg

# %% define filepath constant
TABLE_PATH = "./torrance_tabulated.xlsx"

# %% read in the tables
# read in the tabulated data of closed_shell oxides in table 2 in the original paper
closed_shell_oxides = pd.read_excel(TABLE_PATH, sheet_name="table_2")
# rename beta-Ga2O3 as Ga2O3
closed_shell_oxides.loc[closed_shell_oxides.formula == "b-Ga2O3", "formula"] = "Ga2O3"

# read in the tabulated data of all the other oxides in table 3 in the original paper
all_other_oxides = pd.read_excel(TABLE_PATH, sheet_name="table_3")
# rename beta-Mn2O3, beta-MnO2, alpha-Fe2O3 as Mn2O3, MnO2, Fe2O3
all_other_oxides.replace({"formula": {"b-Mn2O3": "Mn2O3", "b-MnO2": "MnO2", "a-Fe2O3": "Fe2O3"}}, inplace=True)

# combine the two dataframe together
all_oxides = pd.concat([closed_shell_oxides, all_other_oxides], ignore_index=True)
# select the relevant columns
all_oxides = all_oxides[["formula", "v", "iv", "iv_p1"]]


# %% parse the chemical formula
def get_elem_of_interest(formula_str):
    """Get the element of interest from the chemical formula"""
    # get all the elements
    elem_lst = mg.Composition(formula_str).elements
    # grab the second last element from the right
    elem_of_interest = elem_lst[-2]
    # return the element symbol in string format
    return elem_of_interest.symbol


# %%
# create a new column "element" with the element of interest
all_oxides["element"] = all_oxides["formula"].apply(get_elem_of_interest)
# drop the "formula" column and rearrange the order of columns
all_oxides = all_oxides[["element", "v", "iv", "iv_p1"]]
# drop duplicate rows
all_oxides.drop_duplicates(inplace=True)
# sort by element symbol and oxidation states
all_oxides.sort_values(by=["element", "v"], inplace=True, ignore_index=True)
