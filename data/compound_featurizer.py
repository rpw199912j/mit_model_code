import os
import difflib
import itertools
import numpy as np
import pandas as pd
import pymatgen as mg
from glob import glob
from zipfile import ZipFile
from collections import OrderedDict
from typing import Union, Iterable, Tuple, Dict, List
from pymatgen.analysis.ewald import EwaldSummation
from matminer.featurizers.conversions import StrToComposition, CompositionToOxidComposition, StructureToOxidStructure
from matminer.featurizers.composition import ElementProperty, OxidationStates
from matminer.featurizers.structure import EwaldEnergy, GlobalInstabilityIndex, StructuralHeterogeneity

# %% set up path constant
STRUCTURE_PATH = "../data/Structures"

# %% check if the Structure.zip is unzipped
if not os.path.isdir(STRUCTURE_PATH):
    # if still zipped, unzip the folder containing all the cif files
    with ZipFile("".join([STRUCTURE_PATH, ".zip"])) as structures:
        structures.extractall(path=STRUCTURE_PATH)


# %%
def generate_formula_from_structure_helper(filepath: str, label: int):
    """
    The helper function for generate_formula_from_structure()
    Generate formula and assign class label from a given cif file

    :param filepath: String, the filepath of the cif structure file
    :param label: Integer, 0 for Metal; 1 for Insulator; 2 for MIT
    :return: Dictionary, a dictionary containing the compound formula, label and filepath
    """
    # read in the cif file
    struct = mg.Structure.from_file(filepath)
    # get the reduced formula
    formula = struct.composition.reduced_formula
    # construct a row with compound formula, label, filepath and Pymatgen structure object
    return {"Compound": formula, "Label": label, "struct_file_path": filepath, "structure": struct}


def generate_formula_from_structure():
    """Return a Pandas DataFrame with each row having the compound formula, label and filepath"""
    subfolder_lst = ["../data/Structures/Metals/*.cif",
                     "../data/Structures/Insulators/*.cif",
                     "../data/Structures/MIT_materials/*/*.cif"]

    # get all the filepaths in each subfolder
    cif_files = [file for subfolder in subfolder_lst for file in glob(subfolder)]
    # form a list of class labels, each class label should repeat for the number of compounds in each class
    labels = [class_label for class_label in [0, 1, 2]
              for dummy_i in range(len(glob(subfolder_lst[class_label])))]
    # generate a list of dictionaries, with each dictionary corresponding to one compound
    data_dict = [generate_formula_from_structure_helper(cif_file, label)
                 for cif_file, label in zip(cif_files, labels)]

    return pd.DataFrame(data_dict)


# # get the initial dataframe
# df = generate_formula_from_structure()
# # manually correct the Pymatgen naming from TiPbO3 to PbTiO3
# df.Compound[df.Compound == "TiPbO3"] = "PbTiO3"


def get_struct(compound_formula: str, df_input: pd.DataFrame, struct_type: str = "structure") -> mg.Structure:
    """
    Return the Pymatgen structure of the given compound.

    :param compound_formula: String, the compound's chemical formula
    :param df_input: Pandas DataFrame, the dataframe where the structure is stored
    :param struct_type: String, "structure" or "structure_oxid"
    :return: Pymatgen structure
    """
    try:
        struct_output = df_input.loc[df_input.Compound == compound_formula, struct_type].values[0]
    # if the formula has no exact in the input dataframe,
    # return an error message along with the closest matches if there is any
    except IndexError:
        raise Exception("The structure does not exist in this dataframe. The closest matches are {}.".
                        format(difflib.get_close_matches(compound_formula, df_input.Compound)))
    return struct_output


# helper function to read in new data that may not present in the original data set
def read_new_struct(structure: mg.Structure):
    """
    Read in new structure data and return an initial Pandas DataFrame

    :param structure: String, structure file path or Pymatgen Structure
    :return: Pandas DataFrame
    """
    if isinstance(structure, str):
        struct = mg.Structure.from_file(structure)
    elif isinstance(structure, mg.Structure):
        struct = structure

    formula = struct.composition.reduced_formula
    compound_dict = {"Compound": formula, "structure": struct}
    return pd.DataFrame([compound_dict])


# %% generate composition based features

def composition_featurizer(df_input: pd.DataFrame) -> pd.DataFrame:
    """Return a Pandas DataFrame with all compositional features"""

    # generate the "composition" column
    df_comp = StrToComposition().featurize_dataframe(df_input, col_id="Compound")
    # generate features based on elemental properites
    ep_featurizer = ElementProperty.from_preset(preset_name="magpie")
    ep_featurizer.featurize_dataframe(df_comp, col_id="composition", inplace=True)
    # generate the "composition_oxid" column based on guessed oxidation states
    CompositionToOxidComposition(return_original_on_error=True).featurize_dataframe(
        # ignore errors from non-integer stoichiometries
        df_comp, "composition", ignore_errors=True, inplace=True
    )
    # generate features based on oxidation states
    os_featurizer = OxidationStates()
    os_featurizer.featurize_dataframe(df_comp, "composition_oxid", ignore_errors=True, inplace=True)
    # remove compounds with predicted oxidation states of 0
    return df_comp[df_comp["minimum oxidation state"] != 0]


# df_with_comp_features = composition_featurizer(df)
# print(df_with_comp_features.shape)

# %% generate structure based features
def structure_featurizer(df_input: pd.DataFrame) -> pd.DataFrame:
    """Return a Pandas DataFrame with all structural features"""

    # generate the "structure_oxid" column
    df_struct = StructureToOxidStructure().featurize_dataframe(df_input, col_id="structure",
                                                               ignore_errors=True)
    # generate features based on EwaldEnergy
    ee_featurizer = EwaldEnergy()
    ee_featurizer.featurize_dataframe(df_struct, col_id="structure_oxid",
                                      ignore_errors=True, inplace=True)
    # generate features based on the variance in bond lengths and atomic volumes (slow to run)
    sh_featurizer = StructuralHeterogeneity()
    sh_featurizer.featurize_dataframe(df_struct, col_id="structure",
                                      ignore_errors=True, inplace=True)
    # calculate global instability index
    gii_featurizer = GlobalInstabilityIndex()
    gii_featurizer.featurize_dataframe(df_struct, col_id="structure_oxid",
                                       ignore_errors=True, inplace=True)
    # rename the column from "global instability index" to "gii"
    df_struct.rename(columns={"global instability index": "gii"}, inplace=True)
    return df_struct


# test_struct = get_struct("BaTiO3", df)
# test_df = read_new_struct(test_struct)
# test_df = structure_featurizer(test_df)
# print(test_df)

# %% generate handcrafted features
def parse_element(structure: mg.Structure) -> Dict:
    """
    Group the elements into metals and non-metals,
    also pick out the metal element with the highest electronegativity

    :param structure: Pymatgen Structure
    :return: Dictionary, {"non-metals": [list of non-metal elements],
                          "all_metals": [list of all metal elements],
                          "most_electro_neg_metal": Pymatgen Element,
                          "other_metals": [metal elements except most_electro_neg_metal]}
    """
    # create a list with all the elements
    element_lst = structure.composition.element_composition.elements
    # create a set with all the non-metals (faster lookup time during the next step)
    non_metal_set = {element for element in element_lst if not element.is_metal}
    # create a dictionary with only the metal elements' symbol and electronegativity
    metal_dict = {element: element.X for element in element_lst if element not in non_metal_set}
    # sort the metal dictionary by the electronegativity in descending order
    metal_dict = {key: metal_dict[key] for key in sorted(metal_dict, key=metal_dict.get, reverse=True)}
    # get the element symbol with the highest electronegativity
    try:
        most_electro_neg_metal = max(metal_dict, key=metal_dict.get)
        all_metals = list(metal_dict.keys())
        metal_dict.pop(most_electro_neg_metal)
        other_metals = list(metal_dict.keys())
    except ValueError:
        # if no metal present, only populate the "non-metals" key
        return {"non_metals": list(non_metal_set), "all_metals": None,
                "most_electro_neg_metal": None, "other_metals": None}
    return {"non_metals": list(non_metal_set),
            "all_metals": all_metals,
            "most_electro_neg_metal": most_electro_neg_metal,
            "other_metals": other_metals}


def find_metal_old(structure):
    """Find metal species by electronegativity ranking.
    Args:
        structure: Pymatgen Structure object
    Returns:
        metal: str, name of relevant metal element
    """
    anions = ['O', 'F', 'N', 'S', 'Se']
    try:
        metal = str(structure.composition.element_composition.elements[-2])
        if metal in anions:  # If there are two anions return next element
            metal = str(structure.composition.element_composition.elements[-3])
    except:
        return None
    return metal


def metal_equal(structure):
    try:
        new_method = parse_element(structure)["most_electro_neg_metal"].symbol
    except AttributeError:
        new_method = None
    old_method = find_metal_old(structure)
    return pd.Series([structure.composition.reduced_formula, new_method != old_method, new_method, old_method],
                     index=["compound", "different", "metal_from_new", "metal_from_old"])


# difference_df = df["structure"].apply(func=metal_equal)
# difference_df = difference_df[difference_df.different == True].drop(columns="different")


# %% Create featurizers to calculate M-X, M-M, X-X bond distance (M: metal, X: non_metal)
def get_elem_info(structure: mg.Structure, makesupercell: bool = True) -> Tuple[Dict, Dict, mg.Structure]:
    """
    Helper function for calc_mx/mm/xx_distances()
    Get information on the element composition and the site indices for all element of a structure

    :param structure: Pymatgen Structure, the structure of a compound
    :param makesupercell: Boolean, whether or not to make a supercell
    :return: Tuple, (a dictionary with key-value pair - element: [list of indices],
                     a dictionary with key-value pair - element_type: [list of elements],
                     Pymatgen Structure)
    """
    # create a copy of the original structure
    structure = structure.copy()
    # create a list with all the elements
    elem_lst = structure.composition.element_composition.elements
    # initialize a dictionary with each element as the key and an empty list as the value
    elem_indices = {element: [] for element in elem_lst}
    # classify all elements into a metal group and a non-metal group
    elem_group = parse_element(structure)

    # iterate over all sites
    for i, site in enumerate(structure.sites):
        # record the site index for the all the element(s) on this site
        for element in site.species.element_composition.elements:
            elem_indices[element].append(i)

    # in the case that an element only appears in one site, make a supercell (inplace)
    if makesupercell and (np.sum([len(lst) == 1 for lst in elem_indices.values()]) > 0):
        # make a supercell of a'=2a, b'=b, c'=c
        structure.make_supercell(scaling_matrix=[2, 1, 1])
        elem_indices, elem_group, structure = get_elem_info(structure)

    return elem_indices, elem_group, structure


def get_elem_distances(structure, elem_1, elem_indices, elem_2=None, threshold_val=1, only_unique=False):
    """
    Take in a structure and two elements (can be the same elements),
    calculate the bond lengths between those two elements in the given structure

    :param structure: Pymatgen Structure, the structure of a compound
    :param elem_1: Pymatgen Element, the first element
    :param elem_indices: a dictionary with key-value pair - element: [list of indices]
    :param elem_2: Pymatgen Element, the second element
    :param threshold_val: the threshold value (default: 1) to calculate distance between elements in Angstrom
    :param only_unique: Boolean, if True, only return the unique values
    :return: Float, the calculated element distance
    """
    # if the second element is filled out and is distinct from elem_1
    if elem_2 and (elem_1 != elem_2):
        # get all cartesion pairwise combinations for the indices of elem_1 and elem_2 (no duplicates)
        pairwise_indices = itertools.product(elem_indices[elem_1],
                                             elem_indices[elem_2])
    else:  # if the second element is not specified or is the same as the first element
        # get pairwise combinations indices
        pairwise_indices = itertools.combinations(elem_indices[elem_1], 2)
    # get distance for every pair of indices
    distances = [structure.get_distance(i, j) for i, j in pairwise_indices]
    # determine the minimum distance and
    # gets all element distances in the range [min_dist, min_dist+0.01]
    # return distances
    return choose_min(distances, threshold_val, only_unique)


def choose_min(lst: List, threshold: float = 1, unique_only: bool = False):
    """
    Helper function for get_elem_distances()
    Take a list and return the calculated minimum values (see return for details)

    :param lst: List or a list-like object such as numpy.array
    :param threshold: Float, the threshold value (default: 1) to add the minimum number in lst
    :param unique_only: Boolean, if True, only return the unique values
    :return: Float, the selected minimum values in the range [min_value, min_value+threshold]
    """
    # convert the original list into a numpy array
    lst = np.array(lst)
    # if the input is empty, return 0
    if lst.size == 0:
        return np.array([0])

    # get all the unique values in the list
    unique_lst = np.unique(lst)
    # get the maximum cutoff value
    max_cutoff = np.min(unique_lst) + threshold
    # if the minimum is exactly zero, max_cutoff would be exactly threshold
    # this would usually happen when two elements occupy the same site, giving rise to 0 distance
    if max_cutoff == threshold:
        # drop all the zero values
        return choose_min(lst[lst != 0], threshold, unique_only)

    if unique_only:
        # create a condition mask (list of bool) to select all values <= min_value + threshold in the unique list
        cond_mask_unique = unique_lst <= max_cutoff
        min_values = unique_lst[cond_mask_unique]
    else:
        # create a condition mask (list of bool) to select all values <= min_value + threshold in the original list
        cond_mask_orig = lst <= max_cutoff
        min_values = lst[cond_mask_orig]

    return min_values


def calc_mx_dists(structure, cutoff=1, return_unique=False):
    """
    Take a structure and return all metal-non_metal bond lengths in a dictionary

    :param structure: Pymatgen Structure, the structure of a compound
    :param cutoff: Float, a threshold value (default: 1) used to determine the bond lengths
    :param return_unique: Boolean, if True, only return the unique values
    :return: Dictionary, {"metal_symbol-non_metal_symbol": the corresponding bond length}
    """
    element_indices, element_group, structure = get_elem_info(structure)
    # if there are both metal and non_metal elements in the structure
    if element_group["all_metals"] and element_group["non_metals"]:
        # create a cartesian product of all metal and non_metal elements
        pairwise_elem = itertools.product(element_group["all_metals"],
                                          element_group["non_metals"])
        # calcute bond distance for every pair of elements
        mx_dists = {"{}-{}".format(metal.symbol, non_metal.symbol)
                    : get_elem_distances(structure, metal, element_indices, non_metal, cutoff, return_unique)
                    for metal, non_metal in pairwise_elem}
        return mx_dists
    # if there is no metal or no non_metal, return None
    return None


def calc_mm_dists(structure, cutoff=1, return_unique=False):
    """
    Take a structure and return all metal-metal bond lengths in a dictionary

    :param structure: Pymatgen Structure, the structure of a compound
    :param cutoff: Float, a threshold value (default: 1) used to determine the bond lengths
    :param return_unique: Boolean, if True, only return the unique values
    :return: Dictionary, {"metal_symbol-metal_symbol": the corresponding bond length}
    """
    element_indices, element_group, structure = get_elem_info(structure)
    # if there is metal in the structure
    if element_group["all_metals"]:
        # create all possible combinations of metal elements (self included)
        pairwise_elem = itertools.combinations_with_replacement(element_group["all_metals"], 2)
        # calcute bond distance for every pair of elements
        mm_dists = {"{}-{}".format(metal_1.symbol, metal_2.symbol)
                    : get_elem_distances(structure, metal_1, element_indices, metal_2, cutoff, return_unique)
                    for metal_1, metal_2 in pairwise_elem}
        return mm_dists

    return None


def calc_xx_dists(structure, cutoff=1, return_unique=False):
    """
    Take a structure and return all non_metal-non_metal bond lengths in a dictionary

    :param structure: Pymatgen Structure, the structure of a compound
    :param cutoff: Float, a threshold value (default: 1) used to determine the bond lengths
    :param return_unique: Boolean, if True, only return the unique values
    :return: Dictionary, {"non_metal_symbol-non_metal_symbol": the corresponding bond length}
    """
    element_indices, element_group, structure = get_elem_info(structure)
    # since we are only looking for non_metal elements,
    # there is no need to check if there is metal present in the structure
    # if there is non_metal in the structure
    if element_group["non_metals"]:
        pairwise_elem = itertools.combinations_with_replacement(element_group["non_metals"], 2)

        xx_dists = {"{}-{}".format(non_metal_1.symbol, non_metal_2.symbol)
                    : get_elem_distances(structure, non_metal_1, element_indices, non_metal_2, cutoff, return_unique)
                    for non_metal_1, non_metal_2 in pairwise_elem}
        return xx_dists

    return None


# %% test cases
# test_struct = get_struct("Ca2AlFeO5", df)
# # test_struct = get_struct("Sr2La2Fe2O7.84", df)
# print(calc_mm_dists(test_struct, cutoff=1, return_unique=True))
# print(calc_mx_dists(test_struct, cutoff=1, return_unique=True))
# print(calc_xx_dists(test_struct, cutoff=1, return_unique=True))


# %% define a function that can parse element pair string
def parse_elem_pair(elem_pair_str: str):
    """Parse a string containing a pair of element symbol and return two Pymatgen Element objects"""
    elem_pair_str = elem_pair_str.split("-")
    return tuple([mg.Element(elem_str) for elem_str in elem_pair_str])


# print(parse_elem_pair("Ti-O"))
# print(parse_elem_pair("Ti-Ti"))
# print(parse_elem_pair("O-O"))


# %%
def classify_mm_pairs(elem_pair_lst: Union[list, Iterable]):
    """
    Classify all metal-metal pairs into three categories.
    The following represents how relevant the category is with decreasing relevance

    1. transition metal and transition metal
    2. transition metal and non-transition metal
    3. non-transition metal and non-transition metal

    :param elem_pair_lst: List or iterables, all metal element pairs. e.g. ["Ti-Ti", "Ba-Ba"]
    :return: Ordered Dictionary, {"trans_trans": {set of elem pairs in string format},
                                  "trans_non_trans": {set of elem pairs in string format},
                                  "non_trans_non_trans": {set of elem pairs in string format}}
    """
    # parse all the element pairs from strings to tuples with two Pymatgen Elements
    elem_pairs_parsed = [parse_elem_pair(elem_pair) for elem_pair in elem_pair_lst]

    # initialize the empty sets
    trans_trans = set()
    trans_non_trans = set()
    non_trans_non_trans = set()

    for elem_pair_str, (elem_1, elem_2) in zip(elem_pair_lst, elem_pairs_parsed):
        # find all pairs where both metals are transition metals
        if elem_1.is_transition_metal and elem_2.is_transition_metal:
            trans_trans = trans_trans.union({elem_pair_str})
        # find all pairs where one of the two metals is a transition metal
        elif elem_1.is_transition_metal or elem_2.is_transition_metal:
            trans_non_trans = trans_non_trans.union({elem_pair_str})
        # find all pairs where none of the two metals is a transition metal
        else:
            non_trans_non_trans = non_trans_non_trans.union({elem_pair_str})

    return OrderedDict({"trans_trans": trans_trans,
                        "trans_non_trans": trans_non_trans,
                        "non_trans_non_trans": non_trans_non_trans})


def classify_mx_pairs(elem_pair_lst: Union[list, Iterable]):
    """
    Classify all metal-non_metal pairs into three categories.
    The following represents how relevant the category is with decreasing relevance

    1. transition metal and oxygen
    2. transition metal and non_oxygen non-metal, non-transition metal and oxygen
    3. non-transition metal and non_oxygen non-metal

    :param elem_pair_lst: List or iterables, all metal-non_metal element pairs. e.g. ["Ti-O", "Ba-O"]
    :return: Ordered Dictionary, {"trans_oxy": {set of elem pairs in string format},
                                  "one_trans_or_one_oxy": {set of elem pairs in string format},
                                  "non_trans_non_oxy": {set of elem pairs in string format}}
    """
    # parse all the element pairs from strings to tuples with two Pymatgen Elements
    elem_pairs_parsed = [parse_elem_pair(elem_pair) for elem_pair in elem_pair_lst]

    # initialize the empty sets
    trans_oxy = set()
    one_trans_or_one_oxy = set()
    non_trans_non_oxy = set()

    for elem_pair_str, (elem_1, elem_2) in zip(elem_pair_lst, elem_pairs_parsed):
        # if elem_1 is non_metal and elem_2 is a metal, swap the elements
        if (not elem_1.is_metal) and elem_2.is_metal:
            elem_1, elem_2 = elem_2, elem_1
        # find all pairs where the metal is a transition metal and the non-metal is oxygen
        if elem_1.is_transition_metal and (elem_2 == mg.Element("O")):
            trans_oxy = trans_oxy.union({elem_pair_str})
        # find all pairs where one element is a transition metal or one element is oxygen
        elif elem_1.is_transition_metal or (elem_2 == mg.Element("O")):
            one_trans_or_one_oxy = one_trans_or_one_oxy.union({elem_pair_str})
        # find all pairs where the metal is non-transition metal and the non-metal is not oxygen
        else:
            non_trans_non_oxy = non_trans_non_oxy.union({elem_pair_str})

    return OrderedDict({"trans_oxy": trans_oxy,
                        "one_trans_or_one_oxy": one_trans_or_one_oxy,
                        "non_trans_non_oxy": non_trans_non_oxy})


def classify_xx_pairs(elem_pair_lst: Union[list, Iterable]):
    """
    Classify all non_metal-non_metal pairs into three categories.
    The following represents how relevant the category is with decreasing relevance

    1. oxygen and oxygen
    2. oxygen and non_oxygen non_metal
    3. non_oxygen non_metal and non_oxygen non_metal

    :param elem_pair_lst: List or iterables, all non_metal-non_metal element pairs. e.g. ["Ti-Ti", "Ba-Ba"]
    :return: Ordered Dictionary, {"oxy_oxy": {set of elem pairs in string format},
                                  "oxy_non_oxy": {set of elem pairs in string format},
                                  "non_oxy_non_oxy": {set of elem pairs in string format}}
    """
    # parse all the element pairs from strings to tuples with two Pymatgen Elements
    elem_pairs_parsed = [parse_elem_pair(elem_pair) for elem_pair in elem_pair_lst]

    # initialize the empty sets
    oxy_oxy = set()
    oxy_non_oxy = set()
    non_oxy_non_oxy = set()

    for elem_pair_str, (elem_1, elem_2) in zip(elem_pair_lst, elem_pairs_parsed):
        # find all pairs where both non_metals are oxygen
        if (elem_1 == mg.Element("O")) and (elem_2 == mg.Element("O")):
            oxy_oxy = oxy_oxy.union({elem_pair_str})
        # find all pairs where one of the two non_metals is an oxygen
        elif (elem_1 == mg.Element("O")) or (elem_2 == mg.Element("O")):
            oxy_non_oxy = oxy_non_oxy.union({elem_pair_str})
        # find all pairs where none of the two non_metals is an oxygen
        else:
            non_oxy_non_oxy = non_oxy_non_oxy.union({elem_pair_str})

    return OrderedDict({"oxy_oxy": oxy_oxy,
                        "oxy_non_oxy": oxy_non_oxy,
                        "non_oxy_non_oxy": non_oxy_non_oxy})


# %%
# test_struct = get_struct("Ca2AlFeO5", df)
# print(classify_mm_pairs(calc_mm_dists(test_struct, return_unique=True).keys()))
# print(classify_mx_pairs(calc_mx_dists(test_struct, return_unique=True).keys()))
# print(classify_xx_pairs(calc_xx_dists(test_struct, return_unique=True).keys()))


# %%
def return_most_relevant_pairs(pair_clf_dictionary: OrderedDict, cumulative: int = None):
    """
    Return the element pair(s) that are most relevant.
    The specific relevance rules can be seen in the classify_mm/mx/xx_pairs() functions

    :param pair_clf_dictionary: Ordered Dictionary, the relevance of different types of bond
    :param cumulative: None or Integer (1-3), if Integer, will return all element pairs up until the cumulative level
    :return: Set, all the element pairs in strings form. e.g {"Ti-Ti", "Ti-O"}
    """
    cumulative_set = set()
    ordered_sets = pair_clf_dictionary.values()

    if cumulative:
        if cumulative > 3 or cumulative < 0:
            raise Exception("Please choose a cumulative level between 1 and 3")

        ordered_sets = list(ordered_sets)
        index = 0
        while index < cumulative:
            elem_pair_set = ordered_sets[index]
            cumulative_set = cumulative_set.union(elem_pair_set)
            index += 1

        return cumulative_set

    for elem_pair_set in ordered_sets:
        if elem_pair_set:
            return elem_pair_set


# %%
# test_struct = get_struct("Ca2AlFeO5", df)
# test_mm_dict = classify_mm_pairs(calc_mm_dists(test_struct, return_unique=True).keys())
# test_mx_dict = classify_mx_pairs(calc_mx_dists(test_struct, return_unique=True).keys())
# test_xx_dict = classify_xx_pairs(calc_xx_dists(test_struct, return_unique=True).keys())
# return_most_relevant_pairs(test_mm_dict)


def return_relevant_dists(structure_oxid: mg.Structure, calc_funcs: Tuple, cutoff_val: float = 1,
                          return_unique: bool = False, cumulative_level: int = None):
    """
    Return the most relevant metal-metal distance. e.g. only return distances between transition metals

    :param structure_oxid: Pymatgen Structure, the input structure
    :param calc_funcs: Tuple, containing the calc_mm/mx/xx_dists and classify_mm/mx/xx_pairs functions
    :param cutoff_val: Float, the value that determine the range of distance to keep from the minimum distance
    :param return_unique: Boolean, if True, only return unique distance values
    :param cumulative_level: None or Integer, if Integer (1-3), the value determines the number of pairs to keep
    :return: Numpy Array, all the relevant distances
    """
    # get the functions
    calc_dists, classify_pairs = calc_funcs
    # get all the pairwise distances as a dictionary
    pairwise_dists = calc_dists(structure_oxid, cutoff_val, return_unique)
    # classify all element pairs
    pairwise_clf = classify_pairs(pairwise_dists.keys())
    # get the relevant element pairs and only keep those selected
    relevant_pairs_set = return_most_relevant_pairs(pairwise_clf, cumulative_level)
    relevant_dists = {elem_pair: dists for elem_pair, dists in pairwise_dists.items()
                      if elem_pair in relevant_pairs_set}
    # return the relevant distances
    return np.concatenate(list(relevant_dists.values()))


# return_relevant_dists(test_struct, (calc_mm_dists, classify_mm_pairs))


def return_relevant_mm_dists(structure_oxid: mg.Structure, **kwargs):
    """A wrapper function around the generic return_relevant_dists() to only calculate metal-metal distances"""
    return return_relevant_dists(structure_oxid, (calc_mm_dists, classify_mm_pairs), **kwargs)


def return_relevant_mx_dists(structure_oxid: mg.Structure, **kwargs):
    """A wrapper function around the generic return_relevant_dists() to only calculate metal-non_metal distances"""
    return return_relevant_dists(structure_oxid, (calc_mx_dists, classify_mx_pairs), **kwargs)


def return_relevant_xx_dists(structure_oxid: mg.Structure, **kwargs):
    """A wrapper function around the generic return_relevant_dists() to only calculate non_metal-non_metal distances"""
    return return_relevant_dists(structure_oxid, (calc_xx_dists, classify_xx_pairs), **kwargs)


# return_relevant_mm_dists(test_struct, cumulative_level=3)
# return_relevant_mx_dists(test_struct, cumulative_level=3)
# return_relevant_xx_dists(test_struct, cumulative_level=3)


# %% calculate the maximum Madelung potential for each element in a structure
def calc_elem_max_potential(structure_oxid: mg.Structure, full_list=False, check_vesta=False):
    """
    Return the maximum Madelung potential for all elements in a structure.

    :param structure_oxid: Pymatgen Structure with oxidation states for each site
    :param full_list: Boolean, if True, return all the site potentials associated with the element
    :param check_vesta: Boolean, if True, convert all site potentials' units from V to e/Angstrom
    :return: Dictionary: {Pymatgen Element: maximum Madelung potential for that element}
                      or {Pymatgen Element: list of all Madelung potentials for that element}
    """
    # set the conversion factor if the potential values needs to be compared with the ones from VESTA
    # conversion value obtained from page 107 of https://jp-minerals.org/vesta/archives/VESTA_Manual.pdf
    if check_vesta:
        vesta_conversion = 14.39965
    else:
        vesta_conversion = 1

    # define a dictionary that stores the oxidation states for all the element
    elem_charge_lookup = {specie.element: specie.oxi_state for specie in structure_oxid.composition.elements}
    # if there is only one element, then ewald summation will not work
    if len(elem_charge_lookup) == 1:
        return {elem: None for elem in elem_charge_lookup.keys()}

    # initialize the Ewald Summation object in order to calculate Ewald site energy
    ews = EwaldSummation(structure_oxid)
    # obtain the site indices where each element occupies
    elem_indices, *_ = get_elem_info(structure_oxid, makesupercell=False)
    # get the site potential using indices
    # the Ewald site energy is converted to site potential using V=2E/q
    # TODO: need to resolve division by zero error message
    site_potentials = {elem: [2 * ews.get_site_energy(index) / (elem_charge_lookup[elem] * vesta_conversion)
                              for index in indices]
                       for elem, indices in elem_indices.items()}
    if full_list:
        return site_potentials

    return {elem: max(potentials) for elem, potentials in site_potentials.items()}


# calc_elem_max_potential(test_struct, full_list=True, check_vesta=True)

# %%
def choose_max_potential(elem_lst, potential_dict):
    """Return the maximum potential in a list of elements"""
    # if the element list is empty, return None
    if not elem_lst:
        return None
    # if the structure only contains 1 element, return None
    if len(potential_dict) == 1:
        return None

    # initialize the maximum potential with negative infinity
    max_potential = float('-inf')
    for elem in elem_lst:
        # get the element potential
        elem_potential = potential_dict[elem]
        # find the maximum potential in the list
        # by only updating the maximum potential if there is a value greater than the current one
        if elem_potential > max_potential:
            max_potential = elem_potential

    return max_potential


# %%
def return_relevant_potentials(structure_oxid: mg.Structure, **kwargs):
    """
    Return the relevant potentials for the metal site and non_metal site with the following relevance order

    1. transition metal and oxygen
    2. transition metal and non-oxygen
    3. non-transition metal and oxygen
    4. non-transition metal and non-oxygen

    :param structure_oxid: Pymatgen Structure, the input structure
    :return: Tuple, the maximum potentials (unit: V) for the metal site and non-metal site
    """
    # get the maximum potentials for all the elements in the structure
    max_potentials = calc_elem_max_potential(structure_oxid, **kwargs)
    # get the classification of each element into metal and non_metals
    _, elem_group, _ = get_elem_info(structure_oxid, makesupercell=False)
    # get all the metals
    all_metals = elem_group["all_metals"]
    # if there are metals, find the transition metals
    if all_metals:
        trans_metals = [metal for metal in all_metals if metal.is_transition_metal]
    else:
        trans_metals = []
    # find all the non_metals
    non_metals = elem_group["non_metals"]

    # find the corresponding max potentials in each group except oxygen
    all_metals_max = choose_max_potential(all_metals, max_potentials)
    trans_metals_max = choose_max_potential(trans_metals, max_potentials)
    non_metals_max = choose_max_potential(non_metals, max_potentials)

    # if there is oxygen, find the corresponding potential
    try:
        oxygen_max = max_potentials[mg.Element("O")]
    except KeyError:
        oxygen_max = None

    # return potential tuples based on the hierarchy
    # if transition metal and oxygen both exist, return their potentials
    if trans_metals_max and oxygen_max:
        return trans_metals_max, oxygen_max
    # if only transition metal exists, return the transition metal and non_oxygen non_metal potentials
    elif trans_metals_max and (not oxygen_max):
        return trans_metals_max, non_metals_max
    # if only oxygen exists, return the non_transition metal and oxygen potentials
    elif (not trans_metals_max) and oxygen_max:
        return all_metals_max, oxygen_max
    # if neither transition metal nor oxygen exists, return the non_transition metal and non_oxygen potentials
    else:
        return all_metals_max, non_metals_max


# %%
# test_struct = get_struct("Si", df)
# return_relevant_potentials(test_struct, check_vesta=True)

# test_struct = get_struct("Sr2SnO4", df)
# return_relevant_potentials(test_struct, check_vesta=True)
