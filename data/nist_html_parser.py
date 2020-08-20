import os
import pymatgen as mg
from zipfile import ZipFile


# %% set up path constant
HTML_FOLDER_PATH = "../data/ionization_energy/raw_nist_html"

# %% check if the raw_nist_html.zip is unzipped
if not os.path.isdir(HTML_FOLDER_PATH):
    # if still zipped, unzip the folder containing all the html files
    with ZipFile("".join([HTML_FOLDER_PATH, ".zip"])) as html_files:
        html_files.extractall(path="../data/ionization_energy/")


# %%
def clean_specie_name(specie_str: str) -> str:
    """
    Clean up the species name from '<element>\xa0<roman number for oxidation state>'
    to '<element>'
    """
    return specie_str.split("\xa0")[0]


def clean_energy_str(energy_str: str) -> float:
    """Remove '[]' or '()' from the energy string"""
    energy_str = str(energy_str)
    trans_dict = {"[": "", "]": "", "(": "", ")": "", " ": ""}
    trans_table = energy_str.maketrans(trans_dict)
    return float(energy_str.translate(trans_table))


def get_ionization_lookup(df_input):
    """Convert the input dataframe into a lookup dataframe of ionization energies"""
    # get the column names
    columns = df_input.columns
    # get the first three column names
    specie_name, ion_charge, ionization_energy = columns[:3]
    # clean up the dataframe using the two helper functions
    df_input[specie_name] = df_input[specie_name].apply(clean_specie_name)
    df_input[ionization_energy] = df_input[ionization_energy].apply(clean_energy_str)

    # select the first 3 columns and rename them
    df_input = df_input.iloc[:, :3].rename(columns={specie_name: "element",
                                                    ion_charge: "v",
                                                    ionization_energy: "iv_p1"})

    # shift the iv_p1 columns down by one row and create a new column iv from it
    df_input["iv"] = df_input["iv_p1"].shift()
    # reorder the columns
    df_input = df_input[["element", "v", "iv", "iv_p1"]]
    # get the element
    element = mg.Element(df_input["element"][0])
    # return the rows with 0 < v < max_oxidation_state
    return df_input[(df_input.v > 0) & (df_input.v <= element.max_oxidation_state)]
