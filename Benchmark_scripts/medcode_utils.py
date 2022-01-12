import pandas as pd
from datetime import timedelta
from medcodes import charlson, elixhauser
from medcodes.diagnoses.icd_conversion import convert_9to10_list, convert_10to9_list

cci_var_map = {
    'myocardial infarction': 'cci_MI',
    'congestive heart failure': 'cci_CHF',
    'peripheral vascular disease': 'cci_PVD',
    'cerebrovascular disease': 'cci_Stroke',
    'dementia': 'cci_Dementia',
    'chronic pulmonary disease': 'cci_Pulmonary',
    'rheumatic disease': 'cci_Rheumatic',
    'peptic ulcer disease': 'cci_PUD',
    'mild liver disease': 'cci_Liver1',
    'diabetes without chronic complication': 'cci_DM1',
    'diabetes with chronic complication': 'cci_DM2',
    'hemiplegia or paraplegia': 'cci_Paralysis',
    'renal disease': 'cci_Renal',
    'malignancy': 'cci_Cancer1',
    'moderate or severe liver disease': 'cci_Liver2',
    'metastatic solid tumor': 'cci_Cancer2',
    'AIDS/HIV': 'cci_HIV'
}

eci_var_map = {
'congestive heart failure' : 'eci_CHF',
'cardiac arrhythmias' : 'eci_Arrhythmia',
'valvular disease' : 'eci_Valvular',
'pulmonary circulation disorders' : 'eci_PHTN',
'peripheral vascular disorders' : 'eci_PVD',
'hypertension, complicated' : 'eci_HTN1',
'hypertension, uncomplicated' : 'eci_HTN2',
'paralysis' : 'eci_Paralysis',
'other neurological disorders' : 'eci_NeuroOther',
'chronic pulmonary disease' : 'eci_Pulmonary',
'diabetes, complicated' : 'eci_DM1',
'diabetes, uncomplicated' : 'eci_DM2',
'hypothyroidism' : 'eci_Hypothyroid',
'renal failure' : 'eci_Renal',
'liver disease' : 'eci_Liver',
'peptic ulcer disease excluding bleeding' : 'eci_PUD',
'AIDS/HIV' : 'eci_HIV',
'lymphoma' : 'eci_Lymphoma',
'metastatic cancer' : 'eci_Tumor2',
'solid tumor without metastasis' : 'eci_Tumor1',
'rheumatoid arthritis' : 'eci_Rheumatic',
'coagulopathy' : 'eci_Coagulopathy',
'obesity' : 'eci_Obesity',
'weight loss' : 'eci_WeightLoss',
'fluid and electrolyte disorders' : 'eci_FluidsLytes',
'blood loss anemia' : 'eci_BloodLoss',
'deficiency anemia' : 'eci_Anemia',
'alcohol abuse' : 'eci_Alcohol',
'drug abuse' : 'eci_Drugs',
'psychoses' : 'eci_Psychoses',
'depression' : 'eci_Depression'
}

map_dict = {'charlson': cci_var_map,
            'elixhauser': eci_var_map}
empty_map_vector = {k:{v:0 for key, v in map_i.items()} for k, map_i in map_dict.items()}


def commorbidity_set(icd, version, mapping='charlson'):
    c_list = []
    if mapping == 'charlson':
        for i, c in enumerate(icd):
            c_list.extend(charlson(c, version[i]))
    elif mapping == 'elixhauser':
        for i, c in enumerate(icd):
            c_list.extend(elixhauser(c, version[i]))
    else:
        raise ValueError(
            f"{mapping} is not a recognized mapping. It must be \'charlson\' or \'elixhauser\'")
    return set(c_list)


def commorbidity_dict(icd, version, mapping='charlson'):
    map_set = commorbidity_set(icd, version, mapping)
    map_vector = empty_map_vector[mapping].copy()
    for c in map_set:
        map_vector[map_dict[mapping][c]] = 1
    return map_vector


def diagnosis_with_time(df_diagnoses, df_admissions):
    df_diagnoses_with_adm = pd.merge(df_diagnoses, df_admissions.loc[:, [
                                     'hadm_id', 'subject_id', 'dischtime']], on=['subject_id', 'hadm_id'], how='left')
    df_diagnoses_with_adm.loc[:, 'dischtime'] = pd.to_datetime(
        df_diagnoses_with_adm.loc[:, 'dischtime'])
    df_diagnoses_sorted = df_diagnoses_with_adm.sort_values(
        ['subject_id', 'dischtime']).reset_index()
    return df_diagnoses_sorted

def encode_icd_to_index(codes, icd_encode_mapping):
    encoded_list = []
    for code in codes:
        encoded_list.append(icd_encode_mapping[code])
    return encoded_list

def icd_list(df_edstays, df_diagnoses, df_admissions, timerange, version=9, digit3=False):
    timerange = timedelta(days=timerange)
    icd_set = set({})
    df_diagnoses_sorted = diagnosis_with_time(df_diagnoses, df_admissions)
    j_start = 0
    j_end = 0
    prev_subject = None
    diagnoses = []
    stay_ids = []
    # Loop through ED visits
    for i, row in df_edstays.iterrows():
        if i % 10000 == 0:
            print('Process: %d/%d' % (i, len(df_edstays)), end='\r')
        # If new subject, find the start and end index of same subject in sorted admission df
        stay_ids.append(row['stay_id'])
        if row['subject_id'] != prev_subject:
            j_start = j_end
            while j_start < len(df_diagnoses_sorted) and df_diagnoses_sorted['subject_id'][j_start] < row['subject_id']:
                j_start += 1
            j_end = j_start

            while j_end < len(df_diagnoses_sorted) and df_diagnoses_sorted['subject_id'][j_end] == row['subject_id']:
                j_end += 1
            prev_subject = row['subject_id']
        # Count number of previous admissions within the time range
        icd_list = []
        version_list = []
        for j in range(j_start, j_end):
            if row['intime'] > df_diagnoses_sorted['dischtime'][j] and row['intime']-df_diagnoses_sorted['dischtime'][j] <= timerange:
                icd_list.append(df_diagnoses_sorted.loc[j, 'icd_code'])
                version_list.append(df_diagnoses_sorted.loc[j, 'icd_version'])
        if version==10:
            icd_list = set(convert_9to10_list(icd_list, version_list))
        else:
            icd_list = set(convert_10to9_list(icd_list, version_list, digit3=digit3))
        diagnoses.append(icd_list)
        icd_set.update(icd_list)
    
    icd_encode_mapping = {code:i for i, code in enumerate(icd_set)}

    diagnoses_record = []
    for i, codes in enumerate(diagnoses):
        index_codes = encode_icd_to_index(codes, icd_encode_mapping)    
        diagnoses_record.append({'stay_id':stay_ids[i], 'icd_list': codes, 'icd_encoded_list':index_codes})
    df_icd_list = pd.DataFrame.from_records(diagnoses_record)
    return df_icd_list, icd_encode_mapping


def commorbidity(df_master, df_diagnoses, df_admissions, timerange):
    timerange = timedelta(days=timerange)

    df_diagnoses_sorted = diagnosis_with_time(df_diagnoses, df_admissions)
    #df_master = df_master.sort_values(['subject_id', 'stay_id'])

    j_start = 0
    j_end = 0
    prev_subject = None
    diagnoses = []
    versions = []
    stay_ids = []
    # Loop through ED visits
    for i, row in df_master.iterrows():
        stay_ids.append(row['stay_id'])
        if i % 10000 == 0:
            print('Process: %d/%d' % (i, len(df_master)), end='\r')
        # If new subject, find the start and end index of same subject in sorted admission df
        if row['subject_id'] != prev_subject:
            j_start = j_end
            while j_start < len(df_diagnoses_sorted) and df_diagnoses_sorted['subject_id'][j_start] < row['subject_id']:
                j_start += 1
            j_end = j_start

            while j_end < len(df_diagnoses_sorted) and df_diagnoses_sorted['subject_id'][j_end] == row['subject_id']:
                j_end += 1
            prev_subject = row['subject_id']
        # Count number of previous admissions within the time range
        icd_list = []
        version_list = []
        for j in range(j_start, j_end):
            if row['intime'] > df_diagnoses_sorted['dischtime'][j] and row['intime']-df_diagnoses_sorted['dischtime'][j] <= timerange:
                icd_list.append(df_diagnoses_sorted.loc[j, 'icd_code'])
                version_list.append(df_diagnoses_sorted.loc[j, 'icd_version'])
        diagnoses.append(icd_list)
        versions.append(version_list)
    
    cci_eci=[]
    for i, code in enumerate(diagnoses):
        cci_eci.append({'stay_id':stay_ids[i], **commorbidity_dict(code, versions[i], mapping='charlson'), **commorbidity_dict(code, versions[i], mapping='elixhauser')})
    df_cci_eci = pd.DataFrame.from_records(cci_eci)
    df_master = pd.merge(df_master, df_cci_eci, on='stay_id', how='left')
    return df_master
