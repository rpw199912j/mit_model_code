import json
import pandas as pd
import pymatgen as mg
from pymatgen.io.cif import CifWriter

# %% define filepath constant
TABLE_PATH = "./torrance_tabulated.xlsx"
API_PATH = "./mp_api_key.txt"
STRUCT_PATH = "./benchmark_structures/"

# %% read in the tables
# read in the tabulated data in table 2 and table 3
all_oxides_dict = pd.read_excel(TABLE_PATH, sheet_name=["table_2", "table_3"])
# merge the two dataframe
all_oxides = pd.concat(all_oxides_dict, ignore_index=True)
# rename beta-Ga2O3, beta-Mn2O3, beta-MnO2, alpha-Fe2O3 as Ga2O3, Mn2O3, MnO2, Fe2O3
all_oxides.replace({"formula": {"b-Ga2O3": "Ga2O3", "b-Mn2O3": "Mn2O3", "b-MnO2": "MnO2", "a-Fe2O3": "Fe2O3"}},
                   inplace=True)

# %% query the materials project
# select the "formula" & "spacegroup_number" columns for query and remove duplicated rows
mp_query_df = all_oxides[["formula", "spacegroup_number"]].drop_duplicates(ignore_index=True)

# load in the api key
# NOTE: If you want to run this script yourself,
# you need to include your own Materials Project API key in a JSON format as a txt file under this directory
with open(API_PATH) as f:
    api_key = json.load(f)["api_key"]

# initialize a list to store all the structures
struct_lst = []

with mg.MPRester(api_key) as m:
    for formula, spacegroup_number in zip(mp_query_df.formula, mp_query_df.spacegroup_number):
        try:
            # query structures based on the pretty formula and spacegroup number
            struct = m.query(criteria={"pretty_formula": formula, "spacegroup.number": spacegroup_number},
                             properties=["structure"])[0]["structure"]
        except IndexError:
            # if there isn't an exact, get the pretty formula
            struct = formula

        struct_lst.append(struct)

# %% examine the query result
# check which compounds don't have an exact match in Materials Project
unmatched_struct = [structure for structure in struct_lst if isinstance(structure, str)]
print("Number of unmatched structures: {}\n {}".format(len(unmatched_struct), unmatched_struct))
# there are 15 out of 90 unmatched structures

# get the compound structures with a match
matched_struct = [structure for structure in struct_lst if isinstance(structure, mg.Structure)]

# write the structures as CIF files
for structure in matched_struct:
    pretty_formula = structure.composition.reduced_formula
    cif = CifWriter(structure)
    cif.write_file(STRUCT_PATH + pretty_formula + ".cif")
