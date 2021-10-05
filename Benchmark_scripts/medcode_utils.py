import pandas as pd
from datetime import timedelta
from medcodes import charlson, elixhauser
from medcodes.diagnoses._mappers import elixhauser_charlson
map_dict = {'charlson':list(elixhauser_charlson.charlson_codes_v9.keys()),'elixhauser':list(elixhauser_charlson.elixhauser_codes_v9.keys())}
map_index = {k:{c:i for i,c in enumerate(map_d)} for k,map_d in map_dict.items()}
empty_map_vector = {k:[0 for _ in map_i] for k,map_i in map_index.items()}
    

def commorbidity_set(icd, version, mapping='charlson'):
    c_list = []
    if mapping=='charlson':
        for i, c in enumerate(icd):
            c_list.extend(charlson(c, version[i]))
    elif mapping=='elixhauser':
        for i, c in enumerate(icd):
            c_list.extend(elixhauser(c, version[i]))
    else:
        raise ValueError(f"{mapping} is not a recognized mapping. It must be \'charlson\' or \'elixhauser\'")
    return set(c_list)

def commorbidity_vector(icd, version, mapping='charlson'):
    map_set = commorbidity_set(icd, version, mapping)
    map_vector = empty_map_vector[mapping].copy()
    for c in map_set:
        map_vector[map_index[mapping][c]]=1
    return map_vector

def diagnosis_with_time(df_diagnoses, df_admissions):
    df_diagnoses_with_adm = pd.merge(df_diagnoses, df_admissions.loc[:,['hadm_id', 'subject_id','dischtime']], on = ['subject_id','hadm_id'], how='left')
    df_diagnoses_with_adm.loc[:,'dischtime'] = pd.to_datetime(df_diagnoses_with_adm.loc[:,'dischtime'])
    df_diagnoses_sorted = df_diagnoses_with_adm.sort_values(['subject_id', 'dischtime']).reset_index()
    return df_diagnoses_sorted

def commorbidity(df_master, df_diagnoses, df_admissions, timerange):
    timerange = timedelta(days=timerange)

    df_diagnoses_sorted = diagnosis_with_time(df_diagnoses, df_admissions)
    #df_master = df_master.sort_values(['subject_id', 'stay_id'])

    j_start = 0
    j_end = 0
    prev_subject=None
    diagnoses = []
    versions = []
    # Loop through ED visits
    for i, row in df_master.iterrows():
        if i % 10000 == 0:
            print('Process: %d/%d' % (i, len(df_master)), end='\r')
        # If new subject, find the start and end index of same subject in sorted admission df
        if row['subject_id'] != prev_subject:
            j_start=j_end
            while j_start<len(df_diagnoses_sorted) and df_diagnoses_sorted['subject_id'][j_start] < row['subject_id']:
                j_start+=1
            j_end = j_start
            
            while j_end<len(df_diagnoses_sorted) and df_diagnoses_sorted['subject_id'][j_end] == row['subject_id']:
                j_end+=1
            prev_subject=row['subject_id']
        # Count number of previous admissions within the time range
        icd_list = []
        version_list = []
        for j in range(j_start, j_end):
            if row['intime']>df_diagnoses_sorted['dischtime'][j] and row['intime']-df_diagnoses_sorted['dischtime'][j]<=timerange:
                icd_list.append(df_diagnoses_sorted.loc[j, 'icd_code'])
                version_list.append(df_diagnoses_sorted.loc[j, 'icd_version'])
        diagnoses.append(icd_list)
        versions.append(version_list)



    cci= []
    eci= []
    for i, code in enumerate(diagnoses):
        cci.append(commorbidity_vector(code, versions[i], mapping='charlson'))
        eci.append(commorbidity_vector(code, versions[i], mapping='elixhauser'))

    df_master.loc[:,'cci'] = cci
    df_master.loc[:,'eci'] = eci
    return df_master