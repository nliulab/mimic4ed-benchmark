import argparse
import pandas as pd
import os
from helpers import *
from sklearn.impute import SimpleImputer

parser = argparse.ArgumentParser(description='Extract per-subject data from MIMIC-III CSV files.')

parser.add_argument('--master_dataset_path', type=str, help='Directory containing "master_dataset.csv"',required=True)
parser.add_argument('--output_path', type=str, help='Output directory for filtered and processed "train.csv" and "test.csv"',required=True)

args, _ = parser.parse_known_args()

master_dataset_path = args.master_dataset_path
output_path = args.output_path

# from mimic-extract
vitals_valid_range = {
    'temperature': {'outlier_low': 14.2, 'valid_low': 26, 'valid_high': 45, 'outlier_high':47},
    'heartrate': {'outlier_low': 0, 'valid_low': 0, 'valid_high': 350, 'outlier_high':390},
    'resprate': {'outlier_low': 0, 'valid_low': 0, 'valid_high': 300, 'outlier_high':330},
    'o2sat': {'outlier_low': 0, 'valid_low': 0, 'valid_high': 100, 'outlier_high':150},
    'sbp': {'outlier_low': 0, 'valid_low': 0, 'valid_high': 375, 'outlier_high':375},
    'dbp': {'outlier_low': 0, 'valid_low': 0, 'valid_high': 375, 'outlier_high':375},
    'pain': {'outlier_low': 0, 'valid_low': 0, 'valid_high': 10, 'outlier_high':10},
    'acuity': {'outlier_low': 1, 'valid_low': 1, 'valid_high': 5, 'outlier_high':5},
}


# Reading master_dataset.csv
df_master = pd.read_csv(os.path.join(master_dataset_path, 'master_dataset.csv'))

# Filtering Age < 18
df_master = df_master[df_master['age'] >= 18]

# Outlier detection and removal
df_master = convert_temp_to_celcius(df_master)
df_master = remove_outliers(df_master, vitals_valid_range)

# Split train and test
df_train = df_master.sample(frac=0.8,random_state=10) #random state is a seed value
df_test = df_master.drop(df_train.index)

#print(df_train.head())

# Missing value imputation
df_missing_stats = df_train.isnull().sum().to_frame().T
df_missing_stats.loc[1] = df_missing_stats.loc[0] / len(df_master)
df_missing_stats.index = ['no. of missing values', 'percentage of missing values']
#print(df_missing_stats)

vitals_cols = [col for col in df_master.columns if len(col.split('_')) > 1 and 
                                                   col.split('_')[1] in vitals_valid_range]

imputer = SimpleImputer(strategy='mean')
df_train[vitals_cols] = imputer.fit_transform(df_train[vitals_cols])
df_test[vitals_cols] = imputer.transform(df_test[vitals_cols])

# Adding Score values for train and test
add_triage_MAP(df_test) # add an extra variable MAP
add_score_CCI(df_test)
add_score_CART(df_test)
add_score_REMS(df_test)
add_score_NEWS(df_test)
add_score_NEWS2(df_test)
add_score_MEWS(df_test)

add_triage_MAP(df_train) # add an extra variable MAP
add_score_CCI(df_train)
add_score_CART(df_train)
add_score_REMS(df_train)
add_score_NEWS(df_train)
add_score_NEWS2(df_train)
add_score_MEWS(df_train)

# Output train and test csv
df_train.to_csv(os.path.join(output_path, 'train.csv'), index=False)
df_test.to_csv(os.path.join(output_path, 'test.csv'), index=False)