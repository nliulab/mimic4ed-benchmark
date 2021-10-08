import pandas as pd
from datetime import timedelta
import re
import os


def read_edstays_table(edstays_table_path):
    df_edstays = pd.read_csv(edstays_table_path)
    df_edstays['intime'] = pd.to_datetime(df_edstays['intime'])
    df_edstays['outtime'] = pd.to_datetime(df_edstays['outtime'])
    return df_edstays

def read_patients_table(patients_table_path):
    df_patients = pd.read_csv(patients_table_path)
    df_patients['dod'] = pd.to_datetime(df_patients['dod'])
    return df_patients

def read_admissions_table(admissions_table_path):
    df_admissions = pd.read_csv(admissions_table_path)
    df_admissions =  df_admissions[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime','ethnicity', 'edregtime','edouttime', 'insurance']]
    df_admissions['admittime'] = pd.to_datetime(df_admissions['admittime'])
    df_admissions['dischtime'] = pd.to_datetime(df_admissions['dischtime'])
    df_admissions['deathtime'] = pd.to_datetime(df_admissions['deathtime'])
    return df_admissions

def read_icustays_table(icustays_table_path):
    df_icu = pd.read_csv(icustays_table_path)
    df_icu['intime'] = pd.to_datetime(df_icu['intime'])
    df_icu['outtime'] = pd.to_datetime(df_icu['outtime'])
    return df_icu

def read_triage_table(triage_table_path):
    df_triage = pd.read_csv(triage_table_path)
    vital_rename_dict = {vital: '_'.join(['triage', vital]) for vital in ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain', 'acuity']}
    df_triage.rename(vital_rename_dict, axis=1, inplace=True)
    return df_triage

def read_diagnoses_table(diagnoses_table_path):
    df_diagnoses = pd.read_csv(diagnoses_table_path)
    return df_diagnoses

def read_vitalsign_table(vitalsign_table_path):
    df_vitalsign = pd.read_csv(vitalsign_table_path)
    vital_rename_dict = {vital: '_'.join(['ed', vital]) for vital in
                         ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'rhythm', 'pain']}
    df_vitalsign.rename(vital_rename_dict, axis=1, inplace=True)
    return df_vitalsign

def read_pyxis_table(pyxis_table_path):
    df_pyxis = pd.read_csv(pyxis_table_path)
    return df_pyxis

def merge_edstays_patients_on_subject(df_edstays,df_patients):
    df_edstays = pd.merge(df_edstays, df_patients[['subject_id', 'anchor_age', 'gender', 'anchor_year','dod']], on = ['subject_id'], how='left')
    return df_edstays

def merge_edstays_admissions_on_subject(df_edstays ,df_admissions):
    df_edstays = pd.merge(df_edstays,df_admissions, on = ['subject_id', 'hadm_id'], how='left')
    return df_edstays

def merge_edstays_triage_on_subject(df_master ,df_triage):
    df_master = pd.merge(df_master,df_triage, on = ['subject_id', 'stay_id'], how='left')
    return df_master

def add_age(df_master):
    df_master['in_year'] = df_master['intime'].dt.year
    df_master['age'] = df_master['in_year'] - df_master['anchor_year'] + df_master['anchor_age']
    #df_master.drop(['anchor_age', 'anchor_year', 'in_year'],axis=1, inplace=True)
    return df_master

def add_inhospital_mortality(df_master):
    inhospital_mortality = df_master['dod'].notnull() & ((df_master['intime'] <= df_master['dod']) & (df_master['dischtime'] >= df_master['dod']))
    df_master['outcome_inhospital_mortality'] = inhospital_mortality
    return df_master

def add_ed_mortality(df_master):
    ed_mortality = df_master['deathtime'].notnull() & ((df_master['intime'] <= df_master['deathtime']) & (df_master['outtime'] >= df_master['deathtime']))
    df_master['ed_death'] = ed_mortality
    return df_master

def add_before_ed_mortality(df_master):
    before_ed_mortality = df_master['deathtime'].notnull() & (df_master['intime'] > df_master['deathtime'])
    df_master['before_ed_mortality'] = before_ed_mortality
    return df_master

def add_ed_los(df_master):
    ed_los = df_master['outtime'] - df_master['intime']
    df_master['ed_los'] = ed_los
    return df_master


def add_outcome_icu_transfer(df_master, df_icustays, icu_transfer_timerange):
    timerange_delta = timedelta(hours = icu_transfer_timerange)
    df_master_icu = pd.merge(df_master,df_icustays[['subject_id', 'hadm_id', 'intime']], on = ['subject_id', 'hadm_id'], how='left', suffixes=('','_icu'))
    time_diff = (df_master_icu['intime_icu']- df_master_icu['outtime'])
    df_master_icu['time_to_icu_transfer'] = time_diff
    df_master_icu[''.join(['outcome_icu_transfer_', str(icu_transfer_timerange), 'h'])] = time_diff <= timerange_delta
    # df_master_icu.drop(['intime_icu', 'time_to_icu_transfer'],axis=1, inplace=True)
    return df_master_icu


def generate_past_ed_visits(df_master, past_ed_visits_timerange):
    #df_master = df_master.sort_values(['subject_id', 'intime']).reset_index()
    
    timerange_delta = timedelta(days=past_ed_visits_timerange)
    n_ed = [0 for _ in range(len(df_master))]
    
    # Loop through the sorted ED visits
    for i, row in df_master.iterrows():
        if i % 10000 == 0:
            print('Process: %d/%d' % (i, len(df_master)), end='\r')
    # count number of subject's ED visits before the current iteration
        j=i-1
        while j>=0 and df_master['subject_id'][j] == row['subject_id'] and row['intime']-df_master['intime'][j]<=timerange_delta:
            n_ed[i]+=1
            j-=1

    df_master.loc[:,''.join(['n_ed_', str(past_ed_visits_timerange), "d"])] = n_ed

    return df_master

def generate_past_admissions(df_master, df_admissions, past_admissions_timerange):
    sorted_df = df_admissions[df_admissions['subject_id'].isin(df_master['subject_id'].unique().tolist())][['subject_id', 'admittime']].copy()
    
    sorted_df.loc[:,'admittime'] = pd.to_datetime(sorted_df['admittime'])
    sorted_df.sort_values(['subject_id', 'admittime'], inplace=True)
    sorted_df.reset_index(drop=True, inplace=True)

    timerange_delta = timedelta(days=past_admissions_timerange)

    j_start = 0
    j_end = 0
    prev_subject=None
    n_adm = [0 for _ in range(len(df_master))]
    # Loop through ED visits
    for i, row in df_master.iterrows():
        if i % 10000 == 0:
            print('Process: %d/%d' % (i, len(sorted_df)), end='\r')
        # If new subject, find the start and end index of same subject in sorted admission df
        if row['subject_id'] != prev_subject:
            j_start=j_end
            while j_start<len(sorted_df) and sorted_df['subject_id'][j_start] < row['subject_id']:
                j_start+=1
            j_end = j_start
            while j_end<len(sorted_df) and sorted_df['subject_id'][j_end] == row['subject_id']:
                j_end+=1
            prev_subject=row['subject_id']
        # Count number of previous admissions within the time range
        for j in range(j_start, j_end):
            if row['intime']>sorted_df['admittime'][j] and row['intime']-sorted_df['admittime'][j]<=timerange_delta:
                n_adm[i]+=1

    df_master.loc[:,''.join(['n_hosp_', str(past_admissions_timerange), "d"])] = n_adm

    return df_master


def generate_past_icu_visits(df_master, df_icustays, past_icu_visits_timerange):
    sorted_df = df_icustays[df_icustays['subject_id'].isin(df_master['subject_id'].unique().tolist())][['subject_id', 'intime']].copy()
    sorted_df.sort_values(['subject_id', 'intime'], inplace=True)
    sorted_df.reset_index(drop=True, inplace=True)

    timerange_delta = timedelta(days=past_icu_visits_timerange)
    j_start = 0
    j_end = 0
    prev_subject=None
    n_icu = [0 for _ in range(len(df_master))]
    # Loop through ED visits
    for i, row in df_master.iterrows():
        if i % 10000 == 0:
            print('Process: %d/%d' % (i, len(sorted_df)), end='\r')
        # If new subject, find the start and end index of same subject in sorted admission df
        if row['subject_id'] != prev_subject:
            j_start=j_end
            while j_start<len(sorted_df) and sorted_df['subject_id'][j_start] < row['subject_id']:
                j_start+=1
            j_end = j_start
            while j_end<len(sorted_df) and sorted_df['subject_id'][j_end] == row['subject_id']:
                j_end+=1
            prev_subject=row['subject_id']
        # Count number of previous admissions within the time range
        for j in range(j_start, j_end):
            if row['intime']>sorted_df['intime'][j] and row['intime']-sorted_df['intime'][j]<=timerange_delta:
                n_icu[i]+=1

    df_master.loc[:,''.join(['n_icu_', str(past_icu_visits_timerange), "d"])] = n_icu

    return df_master


def generate_future_ed_visits(df_master, next_ed_visit_timerange):
    n_stays = len(df_master)
    time_of_next_ed_visit = [float("NaN") for _ in range(n_stays)]
    time_to_next_ed_visit = [float("NaN") for _ in range(n_stays)]
    outcome_ed_revisit = [False for _ in range(n_stays)]

    timerange_delta = timedelta(days = next_ed_visit_timerange)

    curr_subject=None
    next_subject=None

    for i, row in df_master.iterrows():
        curr_subject = row['subject_id']
        next_subject= df_master['subject_id'][i+1] if i< (n_stays-1) else None
        
        if curr_subject == next_subject:
            curr_outtime = row['outtime']
            next_intime = df_master['intime'][i+1]
            next_intime_diff = next_intime - curr_outtime

            time_of_next_ed_visit[i] = next_intime
            time_to_next_ed_visit[i] = next_intime_diff
            outcome_ed_revisit[i] = next_intime_diff < timerange_delta 

    df_master.loc[:,'next_ed_visit_time'] = time_of_next_ed_visit
    df_master.loc[:,'next_ed_visit_time_diff'] = time_to_next_ed_visit
    df_master.loc[:,''.join(['outcome_ed_revisit_', str(next_ed_visit_timerange), "d"])] = outcome_ed_revisit

    return df_master

def encode_chief_complaints(df_master, complaint_dict):

    holder_list = []
    complaint_colnames_list = list(complaint_dict.keys())
    complaint_regex_list = list(complaint_dict.values())

    for i, row in df_master.iterrows():
        curr_patient_complaint = str(row['chiefcomplaint'])
        curr_patient_complaint_list = [False for _ in range(len(complaint_regex_list))]
        complaint_idx = 0

        for complaint in complaint_regex_list:
            if re.search(complaint, curr_patient_complaint, re.IGNORECASE):
                curr_patient_complaint_list[complaint_idx] = True
            complaint_idx += 1
        
        holder_list.append(curr_patient_complaint_list)
    
    df_encoded_complaint = pd.DataFrame(holder_list, columns = complaint_colnames_list)

    df_master = pd.concat([df_master,df_encoded_complaint], axis=1)
    return df_master

def merge_vitalsign_info_on_edstay(df_master, df_vitalsign, options=[]):
    df_vitalsign.sort_values('charttime', inplace=True)

    grouped = df_vitalsign.groupby(['stay_id'])

    for option in options:
        method = getattr(grouped, option, None)
        assert method is not None, "Invalid option. " \
                                   "Should be a list of values from 'max', 'min', 'median', 'mean', 'first', 'last'. " \
                                   "e.g. ['median', 'last']"
        df_vitalsign_option = method(numeric_only=True)
        df_vitalsign_option.rename({name: '_'.join([name, option]) for name in
                                    ['ed_temperature', 'ed_heartrate', 'ed_resprate', 'ed_o2sat', 'ed_sbp', 'ed_dbp', 'ed_pain']},
                                   axis=1,
                                   inplace=True)
        df_master = pd.merge(df_master, df_vitalsign_option, on=['subject_id', 'stay_id'], how='left')

    return df_master

def merge_med_count_on_edstay(df_master, df_pyxis):
    df_pyxis_fillna = df_pyxis.copy()
    df_pyxis_fillna['gsn'].fillna(df_pyxis['name'], inplace=True)
    grouped = df_pyxis_fillna.groupby(['stay_id'])
    df_medcount = grouped['gsn'].nunique().reset_index().rename({'gsn': 'n_med'}, axis=1)
    df_master = pd.merge(df_master, df_medcount, on='stay_id', how='left')
    return df_master

def merge_medrecon_count_on_edstay(df_master, df_medrecon):
    df_medrecon_fillna = df_medrecon.copy()
    df_medrecon_fillna['gsn'].fillna(df_medrecon['name'])
    grouped = df_medrecon_fillna.groupby(['stay_id'])
    df_medcount = grouped['gsn'].nunique().reset_index().rename({'gsn': 'n_medrecon'}, axis=1)
    df_master = pd.merge(df_master, df_medcount, on='stay_id', how='left')
    return df_master
