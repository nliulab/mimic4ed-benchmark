from pandas import DataFrame

from medcodes.drugs._mappers import ATC_LV1, ATC_LV2, ATC_LV3, ATC_LV4, ATC_LV5

def atc_classification(atc_id):
    """
    Gets information for a given ATC id.
    
    Parameters
    ----------
    atc_id : str
        ATC code for a given drug

    Returns
    -------
    pd.DataFrame
        A dataframe with descriptions of ATC levels 1-5.

    References
    ----------
    [1] https://www.whocc.no/atc_ddd_index/
    """
    if not isinstance(atc_id, str):
        raise ValueError("ATC code must be a string.")

    atc_id = list(atc_id.upper())

    lv1_code = atc_id[0]
    lv2_code = ''.join(atc_id[0:3])
    lv3_code = ''.join(atc_id[0:4])
    lv4_code = ''.join(atc_id[0:5])
    lv5_code = None
    if len(atc_id) == 6:
        lv5_code = ''.join(atc_id)
    codes = [lv1_code, lv2_code, lv3_code, lv4_code, lv5_code]

    lv1_desc = ATC_LV1[lv1_code]
    lv2_desc = ATC_LV2[lv2_code]
    lv3_desc = ATC_LV3[lv3_code]
    lv4_desc = ATC_LV4[lv4_code]
    lv5_desc = None
    if len(atc_id) == 6:
        lv5_desc = ATC_LV5[lv5_code]
    descriptions = [lv1_desc, lv2_desc, lv3_desc, lv4_desc, lv5_desc]

    atc_info = DataFrame({
        'level': [1, 2, 3, 4, 5],
        'code': codes,
        'description': descriptions})
    return atc_info