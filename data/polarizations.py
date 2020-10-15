import re
import numpy as np
import pandas as pd
import pymatgen as mg
from collections import defaultdict
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


class Polarizations:
    """
    A representation of Shannon and Fischer 2016 polarizabilities
    """

    def __init__(self, struct, polarizability_df):
        if isinstance(struct, str):
            try:
                struct = mg.Structure.from_file(struct)
            except FileNotFoundError:
                raise Exception("Please provide a valid filepath or pymatgen structure")

        self.structure = struct
        self.elements = self.structure.composition.element_composition.to_data_dict['elements']
        self.lookup_df_dict = polarizability_df
        self.cat_pols = self.lookup_df_dict["parsed"]
        self.an_pols = self.lookup_df_dict["anions"]

    def get_spacegroup_symbol(self):
        struct = self.structure
        space_group_analyzer = SpacegroupAnalyzer(struct)

        return space_group_analyzer.get_space_group_symbol()

    def get_molar_volume(self):
        """
        Get molar volume for calculating refractive indices later as well as for calculating anion volume

        Returns
            volume per formula unit as float
        """
        structure = self.structure
        volume = structure.volume
        _, units = structure.composition.get_reduced_composition_and_factor()
        unit_volume = volume / units

        return unit_volume

    def ion_class(self):
        """
        Classifies an element as cation or anion

        Returns
            classed: list with anion and cations
        """
        elements = self.elements.copy()
        max_eneg = 0
        for i in range(len(elements)):
            if mg.Element(elements[i]).X > max_eneg:
                max_eneg = mg.Element(elements[i]).X
                max_i = i
        anion = elements.pop(max_i)
        cations = elements
        classed = [anion, cations]
        return classed

    def get_an_vol(self):
        """
        Calculate molar volume per anion
        """

        structure = self.structure

        unit_volume = self.get_molar_volume()
        reduced_form = structure.composition.element_composition.to_reduced_dict
        anion, cations = self.ion_class()

        try:
            num_anions = np.float(reduced_form[anion])
            an_vol = unit_volume / num_anions
        except ZeroDivisionError:
            print(''.join(re.split(r'\D', reduced_form[::-1])[::-1]))
            return None

        return an_vol

    def get_coordinations(self):
        """
        Inputs
            struct: a pymatgen structure object

        Returns
            coordination_numbers: a dictionary containing each species' name and average coordination
            number rounded to nearest integer
        """
        struct = self.structure
        vnn = CrystalNN()
        coordination_numbers = {}
        stoich = defaultdict(int)
        for site in struct.as_dict()['sites']:
            elem = site['species'][0]['element']
            stoich[elem] += 1
        for spec in stoich.keys():
            coordination_numbers[spec] = 0.0
        for atom in range(len(struct)):
            try:
                cn = vnn.get_cn(struct, atom, use_weights=False)
                coordination_numbers[struct[atom].species.elements[0].element.symbol] += cn
            except:
                return None
        for spec in coordination_numbers:
            coordination_numbers[spec] = coordination_numbers[spec] / stoich[spec]
            coordination_numbers[spec] = int(round(coordination_numbers[spec]))

        return coordination_numbers

    def get_valences(self):
        """
        Uses Pymatgen to obtain likely valence states of every element in structure

        Returns
            vals: dictionary of average valence state for every element in composition
        """

        struct = self.structure
        bv = BVAnalyzer()
        try:
            valences = bv.get_valences(struct)
            struct = bv.get_oxi_state_decorated_structure(struct)
        except:
            return None

        if isinstance(valences[0], list):
            valences = [item for sublist in valences for item in sublist]

        stoich = defaultdict(int)
        for site in struct.as_dict()['sites']:
            elem = site['species'][0]['element']
            stoich[elem] += 1

        vals = {}

        for spec in stoich.keys():
            vals[spec] = 0.0

        for atom in range(len(struct)):
            try:
                vals[struct.as_dict()['sites'][atom]['species'][0]['element']] += valences[atom]
            except Exception as e:
                print("Trouble with {}".format(struct.formula))
                print('Do you have partial occupancies?')
                return None

        for spec in vals:
            vals[spec] = vals[spec] / stoich[spec]
            vals[spec] = int(round(vals[spec]))

        return vals

    def get_ion_params(self):
        """
        Reads in a pandas dataframe with 'structs' and 'valences' columns.

        Returms a dictionary with all ions' polarizability parameters.
        """
        params = {}

        # Extract formal valence dictionary from csv file
        valences = self.get_valences()
        if not valences:
            return None

        # Determine anion by finding element with highest electronegativity
        # Assumes homoanionicity
        anion, cations = self.ion_class()

        coordinations = self.get_coordinations()
        try:
            for key in coordinations.keys():
                new_key = re.sub(r'[^A-Za-z]+', '', key)
                coordinations[new_key] = coordinations.pop(key)
        except:
            return None

        for i in range(len(cations)):
            cat_elem = cations[i]
            cat_kn = coordinations[cat_elem]
            cat_val = valences[cat_elem]
            cat_pol_params = self.cat_pols.loc[(self.cat_pols['Ion'] == cat_elem) &
                                               (self.cat_pols['coordination'] == cat_kn) &
                                               (self.cat_pols['valence'] == cat_val)]
            params[cat_elem] = cat_pol_params

        an_pol_params = self.an_pols.loc[self.an_pols['Ion'] == anion]
        params[anion] = an_pol_params
        return params

    def get_cat_pol(self, element, ion_params):
        try:
            cat_pol = ion_params[element]['α(D) (Å3)'].values[0]
            return cat_pol
        except Exception as e:
            return None

    def get_an_pol(self, element, an_vol, ion_params):
        try:
            an_pol_params = ion_params[element]
            an_pol = an_pol_params['α°_'] * 10 ** (-an_pol_params['N_o'] / an_vol ** 1.2)
            return an_pol
        except:
            return None

    def get_compound_pol(self):
        """
        Get the total polarizability for the compound in Å^3
        """
        valences = self.get_valences()
        an_vol = self.get_an_vol()
        ion_params = self.get_ion_params()
        if not valences or not ion_params:
            return None

        struct = self.structure
        elem_comp = struct.composition.element_composition.to_reduced_dict

        # Determine anion by finding element with highest electronegativity
        # Assumes homoanionicity
        anion, cations = self.ion_class()

        total_pol = 0
        for elem in valences:
            if elem in cations:
                cat_pol = self.get_cat_pol(elem, ion_params)

                if cat_pol is None:
                    print('A cation\'s polarizability is missing in {}'.format(struct.formula))
                    return None
                total_pol += cat_pol * elem_comp[elem]

            if elem == anion:
                an_pol = self.get_an_pol(elem, an_vol, ion_params)

                if an_pol is None or an_pol.empty:
                    print('An anion\'s polarizability is missing in {}'.format(struct.formula))
                    return None

                total_pol += an_pol * elem_comp[elem]

        return float(total_pol)

    def get_ref_index(self):
        """
        Predict refractive index based on Eq. 4b of Shannon&Fischer_2016
        """
        total_pol = self.get_compound_pol()
        molar_volume = self.get_molar_volume()
        if not total_pol:
            return None
        ref_index = np.sqrt((4 * np.pi * total_pol) / ((2.26 - 4 * np.pi / 3) * total_pol + molar_volume) + 1)
        return ref_index
