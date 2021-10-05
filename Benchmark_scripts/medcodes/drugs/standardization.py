import json
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
pd.options.mode.chained_assignment = None 

class Drug(object):
    """
    Base Class for Drug Object
    ==========================

    Parameters
    ----------
    drug_id : str
        drug name
    id_type : str
        Can be one of: 'name','ndc','rxcui','smiles','iupac'


    Attributes
    ----------
    generic_name : str
        generic name of drug (e.g., clopidogrel)
    brand_name : str
        brand name of drug  (e.g., plavix)
    pharm_class : list of str
        pharmacological classes of the drug. Can be PE (physiological), MOA (mechanism of action),
        or CS (chemical structure)
    route : list of str
        possible routes of administration
    ndc : int
        National Drug Code (NDC) identifier
    rxcui : str
        RxCui identifier
    """
    def __init__(self, drug_id, id_type):
        if id_type not in ['name', 'ndc', 'smiles']:
            raise ValueError("id_type must be one of: 'name', 'smiles', 'ndc'")
        if not isinstance(drug_id, str):
            raise TypeError("drug_id must be a string")
        self.drug_id = drug_id
        self.id_type = id_type
        if id_type == 'ndc':
            self.ndc = self.drug_id

        self.name = None
        self.ndc = None
        self.smiles = None
        self.iupac = None
        self.rxcui = None
        self.cid = None
        self.inchikey = None

        if id_type == 'ndc':
            self.ndc = drug_id
        if id_type == 'smiles':
            self.smiles = drug_id
        if id_type == 'name':
            self.name = drug_id
        if id_type == 'iupac':
            self.iupac = drug_id
        if id_type == 'cid':
            self.cid = drug_id
        if id_type == 'inchikey':
            self.inchikey = drug_id

        if self.name:
            _drug_name_validator(self.name)

    def get_smiles(self, canonical=True):
        """
        Gets SMILES for drug of interest. 

        Parameters
        ----------
        canonical : bool
            Detemrines whether to get canonical or isomeric SMILES. Default is set to True.
            If False, retrieves isomeric SMILES.
        """
        if self.id_type == 'smiles':
            self.smiles = self.drug_id
        else:
            _pubchem_id_type_checker(self.id_type)
            r = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{self.id_type}/{self.drug_id}/property/CanonicalSMILES,IsomericSMILES/json")
            response = r.json()
            data = response['PropertyTable']['Properties'][0]
            smiles_type = 'CanonicalSMILES'
            if not canonical:
                smiles_type = 'IsomericSMILES'
            self.smiles = data[f'{smiles_type}']
        return self.smiles

    def get_iupac(self):
        """Get IUPAC name for drug of interest. Uses PubChem API."""
        if self.id_type == 'iupac':
            self.iupac = self.drug_id
        else:
            _pubchem_id_type_checker(self.id_type)
            r = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{self.id_type}/{self.drug_id}/property/iupacname/json")
            response = r.json()
            data = response['PropertyTable']['Properties'][0]
            self.iupac = data['IUPACName']
        return self.iupac

    def get_inchikey(self):
        """Gets InChiKey for the drug of interest. Uses PubChem API."""
        if not self.inchikey:
            _pubchem_id_type_checker(self.id_type)
            r = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{self.id_type}/{self.drug_id}/property/InChIKey/json")
            response = r.json()
            data = response['PropertyTable']['Properties'][0]
            self.inchikey = data['InChIKey']
        return self.inchikey

    def describe(self):
        """Provides descriptive sumamry of drug of interest."""
        if not self.name:
            raise ValueError("Sorry! id_type must be 'name'")
        r = requests.get(f"https://api.fda.gov/drug/ndc.json?search=brand_name:{self.drug_id}")
        response = r.json()
        data = response['results'][0]
        self.brand_name = data['brand_name']
        self.generic_name = data['generic_name']
        self.active_ingredients = [i['name'] for i in data['active_ingredients']]
        self.pharm_class = get_pharm_class(self.drug_id, as_df=False)
        self.route = data['route']
        self.ndc = data['product_ndc']
        self.product_type = data['product_type']

        print(f"Generic name: {self.generic_name}")
        print(f"Brand name: {self.brand_name}")
        print(f"Active ingredients: {self.active_ingredients}")
        print(f"Routes of administration: {self.route}")
        print(f"Pharmacologic Classes: {self.pharm_class}")
        print(f"NDC: {self.ndc}")
        print(f"Product type: {self.product_type}")

    def get_pharm_class(self, as_df=True):
        if not self.name:
            raise ValueError("Sorry! id_type must be 'name'")
        self.pharm_class = get_pharm_class(self.name, as_df=False)
        print(f"There are {len(self.pharm_class)} pharmacologic classes.")
        return get_pharm_class(self.drug_id, as_df=as_df)

    def get_atc(self, as_df=True):
        return get_atc(self.drug_id, self.id_type, as_df=as_df)

    def get_mesh(self, as_df=True):
        return get_mesh(self.drug_id, self.id_type, as_df=as_df)

def get_pharm_class(drug_name, as_df=True):
    """
    Gets pharmacological classes of a drug using FDA API.
    
    Parameters
    ----------
    drug_name : str
        brand or generic name of drug
    as_df : bool
        determines output of function

    Returns
    -------
    pd.DataFrame or list    
    """
    pharm_class = []
    r = requests.get(f"https://api.fda.gov/drug/ndc.json?search=brand_name:{drug_name}")
    response = r.json()
    data = response['results'][0]
    if 'pharm_class' in data:
        pharm_class += data['pharm_class']
    terms = ['moa', 'cs', 'pe', 'epc']
    for t in terms:
        try:
            pharm_class += data['openfda'][f'pharm_class_{t}']
        except:
            pass
    output = pharm_class
    if as_df:
        class_names = []
        class_types = []
        for i in pharm_class:
            cn, ct = _parse_pharm_class(i)
            class_names.append(cn)
            class_types.append(ct)
        output = pd.DataFrame({'class_name': class_names, 'class_type': class_types})
    return output

def get_mesh(drug_id, id_type, as_df=True):
    """
    Gets MeSH terms for a given drug.

    Parameters
    ----------
    drug_id : str
    id_type : str
    as_df : bool

    Returns
    -------
    pd.DataFrame or list
    """
    if id_type not in ['name', 'rxcui']:
        raise ValueError("Sorry! This method only works for drug names.")
    mesh_terms = []
    mesh_id = []
    try:
        path = "https://rxnav.nlm.nih.gov/REST/rxclass/class/"
        if id_type == 'name':
            drug_id = drug_id.replace(' ', '+')
            r = requests.get(path + f"byDrugName.json?drugName={drug_id}&relaSource=MESH")
        if id_type == 'rxcui':
            r = requests.get(path + f"byRxcui.json?rxcui={drug_id}&relaSource=MESH")
        response = r.json()
        all_concepts = response['rxclassDrugInfoList']['rxclassDrugInfo']
        for i in all_concepts:
            mesh_id.append(i['rxclassMinConceptItem']['classId'])
            mesh_terms.append(i['rxclassMinConceptItem']['className'])
    except:
        pass
    output = list(set(mesh_terms))
    if as_df:
        output = pd.DataFrame({'mesh_id': mesh_id, 'mesh_term':mesh_terms})
        output = output.drop_duplicates()
    return output

def get_atc(drug_id, id_type, as_df=True):
    """
    Gets ATC id's for a given drug.

    Parameters
    ----------
    drug_id : str
        drug id
    id_type : str
        type of id. Can be either rxcui or name.
    as_df : bool
        determines output of function. If True, outputs a dataframe.
        Default is set to True.

    Returns
    -------
    pd.DataFrame or list
    """
    if id_type not in ['rxcui', 'name']:
        raise ValueError("id_type must be either rxcui or name")

    atc_class_id = []
    atc_class_name = []
    try:
        path = 'https://rxnav.nlm.nih.gov/REST/rxclass/class/'
        if id_type == 'name':
            drug_id = drug_id.replace(' ', '+')
            r = requests.get(path + f"byDrugName.json?drugName={drug_id}&relaSource=ATC")
        if id_type == 'rxcui':
            r = requests.get(path + f"byRxcui.json?rxcui={drug_id}&relaSource=ATC")
        response = r.json()
        concept_groups = response['rxclassDrugInfoList']['rxclassDrugInfo']
        for i in concept_groups:
            atc_class_name.append(i['rxclassMinConceptItem']['className'])
            atc_class_id.append(i['rxclassMinConceptItem']['classId'])
    except Exception:
        pass
    output = list(set(atc_class_id))
    if as_df:
        output = pd.DataFrame({'atc_id': atc_class_id, 'description': atc_class_name})
        output = output.drop_duplicates()
    return output


def get_rxcui(drug_id, id_type):
    """
    Gets RxCUI for a given drug.
    
    Parameters
    ----------
    drug_id : str

    id_type : str
        type of id. Can be either rxcui or name.

    Returns
    -------
    str    
    """
    if id_type not in ['name', 'ndc']:
        raise ValueError("Sorry! id_type must be either 'name' or 'ndc'")
    try:
        path = 'https://rxnav.nlm.nih.gov/REST/rxcui.json'
        if id_type == 'name':
            r = requests.get(path + f"?name={drug_id}")
        if id_type == 'ndc':
            r = requests.get(path + f"?idtype=NDC&id={drug_id}")
        response = r.json()
        rxcui_list = response['idGroup']['rxnormId']
        if len(rxcui_list) > 1:
            print(f"There is more than one RxCUI for drug {id_type} {drug_id}")
            rxcui = rxcui_list
        if len(rxcui_list) == 1:
            rxcui = rxcui_list[0]
    except Exception:
        pass
    return rxcui

def spelling_suggestions(drug_name):
    """
    Provides list of spelling suggestions for a given drug name. Uses the RxNorm API.
    """
    if not isinstance(drug_name, str):
        raise TypeError("drug_name must be a string.")
    r = requests.get(f"https://rxnav.nlm.nih.gov/REST/spellingsuggestions.json?name={drug_name}")
    response = r.json()
    suggestions = response['suggestionGroup']['suggestionList']['suggestion']
    return suggestions

def _parse_pharm_class(text):
    text_list = text.split(' ')
    class_type = text_list.pop()
    class_type = class_type.strip('[]')
    term = ' '.join(text_list)
    return term, class_type

def _pubchem_id_type_checker(id_type):
    if id_type not in ['name', 'iupac', 'cid', 'inchikey', 'smiles']:
        raise ValueError("id_type must be one of 'name', 'iupac', 'cid', 'inchikey', 'smiles'")

def _drug_name_validator(drug_name):
    fda_fail = _test_fda_api(drug_name)
    pubchem_fail = _test_pubchem_api(drug_name)
    if fda_fail and pubchem_fail:
        suggestions = spelling_suggestions(drug_name)
        raise ValueError(f"Drug name not found. Here are some suggestions: {suggestions}")

def _test_fda_api(drug_name):
    api_fail = False
    r = requests.get(f"https://api.fda.gov/drug/ndc.json?search=brand_name:{drug_name}")
    response = r.json()
    if 'error' in response:
        fda_fail = True
    return api_fail

def _test_pubchem_api(drug_name):
    api_fail = False
    r = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/property/json")
    response = r.json()
    if 'Fault' in response:
        api_fail = True
    return api_fail
