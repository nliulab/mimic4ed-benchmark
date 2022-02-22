import argparse
from helpers import *
from medcode_utils import commorbidity, extract_icd_list

parser = argparse.ArgumentParser(description='Extract per-subject data from MIMIC-III CSV files.')

parser.add_argument('--mimic4_path', type=str, help='Directory containing the main MIMIC-IV subdirectories: core, ed, hosp, icu, ed',required=True)
#parser.add_argument('mimic4_path', type=str, help='Directory containing MIMIC-III CSV files.')
parser.add_argument('--output_path', type=str, help='Output directory for "master_dataset.csv"', required=True)
parser.add_argument('--icu_transfer_timerange', type=int, help='Timerange in hours for ICU transfer outcome', default=12)
parser.add_argument('--next_ed_visit_timerange', type=int, help='Timerange in days days for next ED visit outcome', default=3)

args, _ = parser.parse_known_args()

mimic_iv_path = args.mimic4_path
output_path = args.output_path
icu_transfer_timerange = args.icu_transfer_timerange 
next_ed_visit_timerange = args.next_ed_visit_timerange 

mimic_iv_core_path = os.path.join(mimic_iv_path, 'core')
mimic_iv_hosp_path = os.path.join(mimic_iv_path , 'hosp')   
mimic_iv_icu_path = os.path.join(mimic_iv_path, 'icu')
mimic_iv_ed_path = os.path.join(mimic_iv_path, 'ed')

icu_filename_dict = {"chartevents":"chartevents.csv","datetimeevents":"datetimeevents.csv","d_items":"d_items.csv","icustays":"icustays.csv","inputevents":"inputevents.csv","outputevents":"outputevents.csv","procedureevents":"procedureevents.csv"}
core_filename_dict = {"patients":"patients.csv", "admissions":"admissions.csv", "transfers":"transfers.csv"}
hosp_filename_dict = {"d_hcpcs":"d_hcpcs.csv","d_icd_diagnoses":"d_icd_diagnoses.csv","d_labitems":"d_labitems.csv","emar":"emar.csv","hcpcsevents":"hcpcsevents.csv","microbiologyevents":"microbiologyevents.csv","poe":"poe.csv","prescriptions":"prescriptions.csv","services":"services.csv","diagnoses_icd":"diagnoses_icd.csv","d_icd_procedures":"d_icd_procedures.csv","drgcodes":"drgcodes.csv","emar_detail":"emar_detail.csv","labevents":"labevents.csv","pharmacy":"pharmacy.csv","poe_detail":"poe_detail.csv","procedures_icd":"procedures_icd.csv"}
ed_filename_dict = {'diagnosis':'diagnosis.csv', 'edstays':'edstays.csv',  'medrecon':'medrecon.csv',  'pyxis':'pyxis.csv',  'triage':'triage.csv',  'vitalsign':'vitalsign.csv'}

complaint_dict = {"chiefcom_chest_pain" : "chest pain", "chiefcom_abdominal_pain" : "abdominal pain|abd pain", 
"chiefcom_headache" : "headache|lightheaded", "chiefcom_shortness_of_breath" : "breath", "chiefcom_back_pain" : "back pain", "chiefcom_cough" : "cough", 
"chiefcom_nausea_vomiting" : "nausea|vomit", "chiefcom_fever_chills" : "fever|chill", "chiefcom_syncope" :"syncope", "chiefcom_dizziness" : "dizz"}

## Reading main tables
df_edstays = read_edstays_table(os.path.join(mimic_iv_ed_path, ed_filename_dict['edstays']))
df_patients = read_patients_table(os.path.join(mimic_iv_core_path, core_filename_dict['patients']))
df_admissions = read_admissions_table(os.path.join(mimic_iv_core_path, core_filename_dict["admissions"]))
df_icustays = read_icustays_table(os.path.join(mimic_iv_icu_path, icu_filename_dict['icustays']))
df_triage = read_triage_table(os.path.join(mimic_iv_ed_path, ed_filename_dict['triage']))
df_vitalsign = read_vitalsign_table(os.path.join(mimic_iv_ed_path, ed_filename_dict['vitalsign']))
df_pyxis = read_pyxis_table(os.path.join(mimic_iv_ed_path, ed_filename_dict['pyxis']))
df_medrecon = read_pyxis_table(os.path.join(mimic_iv_ed_path, ed_filename_dict['medrecon']))

## Read data here for ICD.
df_diagnoses = read_diagnoses_table(os.path.join(mimic_iv_hosp_path, hosp_filename_dict['diagnoses_icd']))


## Merging patients -> merging admissions -> merging triage -> master
df_master = merge_edstays_patients_on_subject(df_edstays ,df_patients)
df_master = merge_edstays_admissions_on_subject(df_master ,df_admissions)

## Adding age, mortality and ICU transfer outcome
df_master = add_age(df_master)
df_master = add_inhospital_mortality(df_master)
df_master = add_inhospital_mortality(df_master)
df_master = add_ed_mortality(df_master)
df_master = add_before_ed_mortality(df_master)
df_master = add_ed_los(df_master)
df_master = add_outcome_icu_transfer(df_master, df_icustays, icu_transfer_timerange)
df_master['outcome_hospitalization'] = ~pd.isnull(df_master['hadm_id'])
df_master['outcome_critical'] = df_master['outcome_inhospital_mortality'] | df_master[''.join(['outcome_icu_transfer_', str(icu_transfer_timerange), 'h'])]

# Sort Master table for further process
df_master = df_master.sort_values(['subject_id', 'intime']).reset_index()

# Filling subjects NA ethnicity, takes ~17s
df_master = fill_na_ethnicity(df_master)

## Generate past ED visits
df_master = generate_past_ed_visits(df_master, timerange=30)
df_master = generate_past_ed_visits(df_master, timerange=90)
df_master = generate_past_ed_visits(df_master, timerange=365)

## Oucome:  future ED revisit variables
df_master = generate_future_ed_visits(df_master, next_ed_visit_timerange)

# Generate past admissions
df_master = generate_past_admissions(df_master, df_admissions, timerange=30)
df_master = generate_past_admissions(df_master, df_admissions, timerange=90)
df_master = generate_past_admissions(df_master, df_admissions, timerange=365)

## Generate past icu visits
df_master  = generate_past_icu_visits(df_master, df_icustays, timerange=30)
df_master  = generate_past_icu_visits(df_master, df_icustays, timerange=90)
df_master  = generate_past_icu_visits(df_master, df_icustays, timerange=365)

## Generate numeric timedelta variables
df_master = generate_numeric_timedelta(df_master)

## Mergining with triage table, Comment: revise the variable names? triage_*
df_master = merge_edstays_triage_on_subject(df_master, df_triage) ## note change to merge master 

## Encoding 10 chief complaints
df_master = encode_chief_complaints(df_master, complaint_dict)

# This function takes about 10 min
df_master = commorbidity(df_master, df_diagnoses, df_admissions, timerange = 356*5)
extract_icd_list(df_edstays, df_diagnoses, df_admissions, output_path, timerange = 356*5, version = 'v9')
extract_icd_list(df_edstays, df_diagnoses, df_admissions, output_path, timerange = 356*5, version = 'v9_3digit')
extract_icd_list(df_edstays, df_diagnoses, df_admissions, output_path, timerange = 356*5, version = 'v10')

# Merging with vitalsign info
df_master = merge_vitalsign_info_on_edstay(df_master, df_vitalsign, options=['last'])

# Merging with Medication info
df_master = merge_med_count_on_edstay(df_master, df_pyxis)
df_master = merge_medrecon_count_on_edstay(df_master, df_medrecon)

# Output master_dataset
df_master.to_csv(os.path.join(output_path, 'master_dataset.csv'), index=False)