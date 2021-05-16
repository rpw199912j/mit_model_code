import os
import pandas as pd
import pymatgen as mg
from glob import glob
from zipfile import ZipFile
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score

# %% set up path constant
STRUCT_FOLDER_PATH = "../data/torrance_tables/benchmark_structures"

# %% check if the benchmark_structures.zip is unzipped
if not os.path.isdir(STRUCT_FOLDER_PATH):
    # if still zipped, unzip the folder containing all the cif files
    with ZipFile("".join([STRUCT_FOLDER_PATH, ".zip"])) as struct_files:
        struct_files.extractall(path="../data/torrance_tables/")


# %%
def initialize_benchmark_df_helper(file_path):
    """Helper function for initialize_benchmark_df()"""
    # read in the original structure
    struct = mg.Structure.from_file(file_path)
    # get the primitive cell
    struct = struct.get_primitive_structure()
    # add oxidation states using the guess routine in Pymatgen
    struct.add_oxidation_state_by_guess()
    # make a supercell of 2a, 2b, 2c
    struct.make_supercell([2, 2, 2])
    return {"formula": struct.composition.reduced_formula,
            "structure_oxid": struct}


def initialize_benchmark_df():
    """Return a dataframe containing all the cif files in the target directory"""
    # get all the cif file paths as a list
    cif_file_paths = glob(STRUCT_FOLDER_PATH + "/*.cif")
    # for each file path, read in the structure and get its reduced formula
    cif_lst_dict = [initialize_benchmark_df_helper(file) for file in cif_file_paths]

    return pd.DataFrame(cif_lst_dict)


# %%
def process_benchmark_df(df_input):
    """Take in the featurized the benchmark dataframe and clean it up"""
    # select the relevant columns
    df_output = df_input[["formula", "avg_mx_dists", "avg_mm_dists", "iv", "iv_p1",
                          "v_m", "v_x", "est_hubbard_u", "est_charge_trans"]]
    # rename the column names to match those found in torrance_tabulated.xlsx
    df_output = df_output.rename(columns={"avg_mx_dists": "d_mo", "avg_mm_dists": "d_mm", "v_x": "v_o",
                                          "est_hubbard_u": "hubbard", "est_charge_trans": "charge_transfer"})
    # drop rows containing NA values, sort by the formula and reindex the dataframe
    return df_output.dropna().sort_values("formula").reset_index(drop=True)


# %%
def process_torrance_df(df_torr, df_bench):
    """
    Take in the processed benchmark dataframe and the unprocessed torrance dataframe.
    Then match the two dataframes
    """
    # drop the irrelevant columns
    df_torr = df_torr.drop(columns=["spacegroup_symbol", "spacegroup_number", "ref", "_em/s", "mu",
                                    "optical_bandgap", "class_label", "v"])
    # remove duplicate rows
    df_torr = df_torr.drop_duplicates(ignore_index=True)
    # use left join to select rows only present in the benchmark dataframe
    df_torr = df_bench[["formula"]].join(df_torr.set_index("formula"), on="formula", sort=True)
    return df_torr.reset_index(drop=True)


# %% calcuate rmse, explained variance and R^2 values for each feature
def evaluate_performance(df_true, df_pred):
    """
    Evaluate the performance of the handbuilt featurizer
    through RMSE, explained variance score and R^2 values
    """
    rmse = mean_squared_error(df_true, df_pred, multioutput="raw_values", squared=False)
    var_explained = explained_variance_score(df_true, df_pred, multioutput="raw_values")
    r2_scores = r2_score(df_true, df_pred, multioutput="raw_values")
    return pd.DataFrame(list(zip(df_pred.columns, rmse, var_explained, r2_scores)),
                        columns=["feature", "rmse", "var_explained", "r_squared"])
