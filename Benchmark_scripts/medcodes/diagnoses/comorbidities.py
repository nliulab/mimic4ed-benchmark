"""
Comorbidtiies
=============
ICD (International Classification of Diseases) is a classification system that standardizes
diagnoses into alphanumeric codes. There are two variants of ICD: 1) ICD-9, an earlier version
which contains 13,000 codes, and 2) ICD-10, a more recent version with over 68,000 codes.

The comorbidity functions in this module are used to cluster diagnoses into comorbidity groups using
comorbidity mapping indices such as Elixhauser, Charlson, or a custom mapper.
"""

import pandas as pd

from medcodes.diagnoses._mappers import comorbidity_mappers, icd9cm, icd10

icd9_codes = icd9cm.keys()
icd10_codes = icd10.keys()

def _check_icd_inputs(icd_code, icd_version):
    """Checks that icd_code input is the correct format."""
    if icd_version not in [9,10]:
        raise ValueError("icd_version must be either 9 or 10. Default is set to 9.")
    if not isinstance(icd_code, str):
        raise TypeError("icd_code must be a string.")
    if (icd_version==10 and icd_code not in icd10_codes):
        raise ValueError(f"{icd_code} is not a recognized ICD-10 code.")
    if (icd_version==9 and icd_code not in icd9_codes):
        raise ValueError(f"{icd_code} is not a recognized ICD-9CM code.")

def _format_icd_code(icd_code):
    """Removes punctuation from icd_code string."""
    icd_code = icd_code.replace(".", "")
    icd_code = icd_code.strip()
    return icd_code

def _check_custom_map(custom_map):
    """Checks that vals of custom_map dict are dictionaries."""
    if not isinstance(custom_map, dict):
        raise TypeError("custom_map must be a dictionary.")
    for k, val in custom_map.items():
        if not isinstance(val, list):
            raise TypeError(f'{k} values must be a list')

def charlson(icd_code, icd_version=9):
    """
    Identifies relevant Charlson comorbidities for a ICD code of interest.
    Uses Charlson comorbidity index mappings as defined by Quan et al. [1].

    Parameters
    ----------
    icd_code : str
        ICD code
    icd_version : str
        Can be either 9 or 10

    Returns
    -------
    list
        Charlson comorbidities

    Note
    ----
    This function supports ICD-9CM and ICD-10.

    References
    ----------
    [1] Quan H, Sundararajan V, Halfon P, et al. Coding algorithms for 
    defining Comorbidities in ICD-9-CM and ICD-10 administrative data. 
    Med Care. 2005 Nov; 43(11): 1130-9.
    """
    # try:
    #     _check_icd_inputs(icd_code=icd_code, icd_version=icd_version)
    # except ValueError as e:
    #     print("Warning: ", str(e))
    icd_code = _format_icd_code(icd_code=icd_code)

    mapper = comorbidity_mappers[f'charlson_{icd_version}']

    comorbidities = []

    for k, val in mapper.items():
        if icd_code.startswith(tuple(val)):
            comorbidities.append(k)
    return comorbidities

def elixhauser(icd_code, icd_version=9):
    """
    Identifies relevant Elixhauser comorbidities for a given ICD code.
    Uses Elixhauser comorbidity index mappings as defined by Quan et al. [1].

    Parameters
    ----------
    icd_code : str
        ICD diagnosis code
    icd_version : str
        Version of ICD. Can be either 9 or 10.
    
    Returns
    -------
    list
        Elixhauser comorbidities

    Note
    ----
    This function supports ICD-9CM and ICD-10.

    References
    ----------
    [1] Quan H, Sundararajan V, Halfon P, et al. Coding algorithms for 
    defining Comorbidities in ICD-9-CM and ICD-10 administrative data. 
    Med Care. 2005 Nov; 43(11): 1130-9.
    """
    # try:
    #     _check_icd_inputs(icd_code=icd_code, icd_version=icd_version)
    # except ValueError as e:
    #     print("Warning: ", str(e))
    icd_code = _format_icd_code(icd_code=icd_code)
    mapper = comorbidity_mappers[f'elixhauser_{icd_version}']

    comorbidities = []
    for k, val in mapper.items():
        if icd_code.startswith(tuple(val)):
            comorbidities.append(k)
    return comorbidities

def custom_comorbidities(icd_code, icd_version, custom_map):
    """
    Applies custom mapping to ICD code.

    Parameters
    ----------
    icd_code : str
        International Classification of Diseases (ICD) code
    icd_version : int
        Version of ICD. Can be either 9 or 10.
    custom_map : dict
        A customized mapper that defines one group of 
        multiple groups of ICD codes.
    
    Returns
    -------
    list
        Custom comorbidities for the ICD code of interest.

    Note
    ----
    This function supports ICD-9CM and ICD-10.

    Example
    -------
    >>> custom_map = {'stroke': ['33']}
    >>> icd_code = '33010'
    >>> custom_comorbidities(icd_code=icd_code, icd_version=9, custom_map=custom_map)
    """
    _check_icd_inputs(icd_code=icd_code, icd_version=icd_version)
    icd_code = _format_icd_code(icd_code=icd_code)
    _check_custom_map(custom_map)

    comorbidities = []
    for k, val in custom_map.items():
        if icd_code.startswith(tuple(val)):
            comorbidities.append(k)
    return comorbidities

def comorbidities(icd_codes, icd_version=9, mapping='elixhauser', custom_map=None):
    """
    Parameters
    ----------
    icd_codes : list
        List of ICD codes
    icd_version : int
        Version of ICD codes. Can be either 9 or 10. 
        Note that version 9 refers to ICD-9CM.
    mapping : str
        Type of comorbiditiy mapping. Can be one of 'elixhauser', 
        'charlson', 'custom'. If custom mapping is desired, the mapper must
        be specified in `custom_map`.
    custom_map : dict
        Custom mapper dictionary. Used when mapping is set to 'custom'.

    Returns
    -------
    pd.DataFrame
        Dataframe with columns `icd_code`, `description`, `comorbidity`.
    
    Note
    ----
    This function supports ICD-9CM and ICD-10.

    References
    ----------
    [1] Quan H, Sundararajan V, Halfon P, et al. Coding algorithms for 
    defining Comorbidities in ICD-9-CM and ICD-10 administrative data. 
    Med Care. 2005 Nov; 43(11): 1130-9.
    """
    if mapping not in ['elixhauser', 'charlson', 'custom']:
        raise ValueError("mappign must be one of 'elixhauser', 'charlson', 'custom'")

    if custom_map:
        if not isinstance(custom_map, dict):
            raise TypeError("custom_map must be a dictionary")

    all_comorbidities = []
    descriptions = []
    for icd_code in icd_codes:
        c = None
        if mapping == 'custom':
            c = custom_comorbidities(icd_code, icd_version, custom_map)
        if mapping == 'elixhauser':
            c = elixhauser(icd_code, icd_version)
        if mapping == 'charlson':
            c = charlson(icd_code, icd_version)
        all_comorbidities.append(c)
        d = None
        if icd_version == 9:
            d = icd9cm[icd_code]
        if icd_version == 10:
            d = icd10[icd_code]
        descriptions.append(d)
    comorbidities_table = pd.DataFrame({'icd_code': icd_codes, 
                                        'description': descriptions, 
                                        f'{mapping.lower()}_comorbidity': all_comorbidities})

    return comorbidities_table


