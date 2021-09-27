import pandas as pd
from datetime import timedelta
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
    return df_triage


def merge_edstays_patients_on_subject(df_edstays ,df_patients):
    df = pd.merge(df_edstays, df_patients[['subject_id', 'anchor_age', 'gender', 'anchor_year','dod']], on = ['subject_id'], how='left')
    return df

def merge_edstays_admissions_on_subject(df_edstays ,df_admissions):
    df_edstays = pd.merge(df_edstays,df_admissions, on = ['subject_id', 'hadm_id'], how='left')
    return df_edstays

def add_age(df_edstays):
    df_edstays['in_year'] = df_edstays['intime'].dt.year
    df_edstays['age'] = df_edstays['in_year'] - df_edstays['anchor_year'] + df_edstays['anchor_age']
    df_edstays.drop(['anchor_age', 'anchor_year', 'in_year'],axis=1, inplace=True)
    return df_edstays

def add_inhospital_mortality(df_edstays):
    inhospital_mortality = df_edstays['dod'].notnull() & ((df_edstays['admittime'] <= df_edstays['dod']) & (df_edstays['dischtime'] <= df_edstays['dod']))
    df_edstays['inhospital_mortality'] = inhospital_mortality
    return df_edstays

def generate_past_ed_visits(df_edstays):
    df_edstays = df_edstays.sort_values(['subject_id', 'intime']).reset_index()
    
    timerange = timedelta(days=365)
    n_ed = [0 for _ in range(len(df_edstays))]
    
    # Loop through the sorted ED visits
    for i, row in df_edstays.iterrows():
        if i % 10000 == 0:
            print('Process: %d/%d' % (i, len(df_edstays)), end='\r')
    # count number of subject's ED visits before the current iteration
        j=i-1
        while j>=0 and df_edstays['subject_id'][j] == row['subject_id'] and row['intime']-df_edstays['intime'][j]<=timerange:
            n_ed[i]+=1
            j-=1

    df_edstays.loc[:,'n_ed'] = n_ed

    return df_edstays

def generate_past_admissions(df_edstays, df_admissions):
    sorted_df = df_admissions[df_admissions['subject_id'].isin(df_edstays['subject_id'].unique().tolist())][['subject_id', 'admittime']].copy()
    
    sorted_df.loc[:,'admittime'] = pd.to_datetime(sorted_df['admittime'])
    sorted_df.sort_values(['subject_id', 'admittime'], inplace=True)
    sorted_df.reset_index(drop=True, inplace=True)

    timerange = timedelta(days=365)

    j_start = 0
    j_end = 0
    prev_subject=None
    n_adm = [0 for _ in range(len(df_edstays))]
    # Loop through ED visits
    for i, row in df_edstays.iterrows():
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
            if row['intime']>sorted_df['admittime'][j] and row['intime']-sorted_df['admittime'][j]<=timerange:
                n_adm[i]+=1

    df_edstays.loc[:,'n_adm'] = n_adm

    return df_edstays


def generate_past_icu_visits(df_edstays, df_icustays):
    sorted_df = df_icustays[df_icustays['subject_id'].isin(df_edstays['subject_id'].unique().tolist())][['subject_id', 'intime']].copy()
    sorted_df.sort_values(['subject_id', 'intime'], inplace=True)
    sorted_df.reset_index(drop=True, inplace=True)

    timerange = timedelta(days=365)
    j_start = 0
    j_end = 0
    prev_subject=None
    n_icu = [0 for _ in range(len(df_edstays))]
    # Loop through ED visits
    for i, row in df_edstays.iterrows():
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
            if row['intime']>sorted_df['intime'][j] and row['intime']-sorted_df['intime'][j]<=timerange:
                n_icu[i]+=1

    df_edstays.loc[:,'n_icu'] = n_icu

    return df_edstays

def merge_edstays_triage_on_subject(df_edstays ,df_triage):
    df_edstays = pd.merge(df_edstays,df_triage, on = ['subject_id', 'stay_id'], how='left')
    return df_edstays
