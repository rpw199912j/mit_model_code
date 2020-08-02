import time
import requests

# the request URL
POST_URL = "https://physics.nist.gov/cgi-bin/ASD/ie.pl"

# the element lists (all metals + metalloids) to find the ionization energy for
elem_lst = ["Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
            "Cu", "Zn", "Ga", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
            "In", "Sn", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
            "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb",
            "Bi", "Po", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es",
            "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Cn", "Fl",
            "B", "Si", "Ge", "As", "Sb", "Te", "At"]

# intialize the data form to post
payload = {'encodedlist': 'XXT2',
           'spectra': None,
           'units': '1',
           'format': '0',
           'order': '0',
           'sp_name_out': 'on',
           'ion_charge_out': 'on',
           'e_out': '0',
           'unc_out': 'on',
           'biblio': 'on',
           'submit': 'Retrieve Data'}

# let the scraper pretend that it's a web browser
headers = {
    'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 Safari/537.36"
}

# iterate over all 93 elements
# would take 15.5 minutes to complete
for element in elem_lst:
    # fill in the element for request
    payload["spectra"] = element
    # post the request
    response = requests.request("POST", POST_URL, headers=headers, data=payload)
    # save the response as an HTML file
    with open("../data/ionization_energy/raw_nist_html/{}.html".format(element), mode="wb") as html_file:
        html_file.write(response.content)
    # pause execution for 10 second
    time.sleep(10)
    # print out the element name after each iteration
    print("{} recorded".format(element))
